"""P3.2 (two-box) — end-to-end CPU-worker → files → GPU-trainer flow.

Runs the worker and trainer in one process against temp dirs; the only thing
that differs on real hardware is that ``rsync``/``scp`` moves the shard and
weight files between the two boxes instead of them sharing a directory. Asserts
the loop closes: cold-start init matches the trainer's published v0 (so first
shards are on-policy), shards are produced and consumed, PPO advances the weight
version, and the weights actually change.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from train.harness.config import TrainConfig
from train.harness.policy_stub import SkeletonPolicy
from train.harness.rollout_worker import LocalCollector, run_worker
from train.harness.shard import (
    latest_weights_version,
    list_shards,
    load_weights_into,
    read_shard,
)
from train.harness.trainer import run_trainer


def _cfg():
    # use_skeleton=True: this test exercises the shard/weights *plumbing* (cold-start
    # parity, produce/consume, version advance), which is model-agnostic. The P3.1
    # stub keeps it fast and lets the SkeletonPolicy weight-compat references below
    # match what build_policy(cfg) constructs. The real AgarPolicy is the default
    # for the two-box CLI and is covered by tests/test_model.py.
    return TrainConfig(
        games_per_worker=4, rollout_len=6, policy_seats=tuple(range(8)),
        device="cpu", use_skeleton=True,
    )


def test_cold_start_init_matches_across_boxes():
    """Worker (no weights yet) and trainer both seed from base_seed ⇒ identical
    version-0 policy ⇒ first shards are on-policy."""
    cfg = _cfg()
    torch.manual_seed(cfg.base_seed)
    a = SkeletonPolicy(entity_hidden=cfg.entity_hidden, trunk_hidden=cfg.trunk_hidden)
    torch.manual_seed(cfg.base_seed)
    b = SkeletonPolicy(entity_hidden=cfg.entity_hidden, trunk_hidden=cfg.trunk_hidden)
    for k in a.state_dict():
        torch.testing.assert_close(a.state_dict()[k], b.state_dict()[k])


def test_local_collector_produces_valid_rollout():
    cfg = _cfg()
    torch.manual_seed(cfg.base_seed)
    policy = SkeletonPolicy(entity_hidden=cfg.entity_hidden, trunk_hidden=cfg.trunk_hidden)
    col = LocalCollector(cfg, policy, worker_id=0)
    buf, last_values = col.collect_rollout()

    assert buf.R == cfg.games_per_worker * len(cfg.policy_seats)
    assert last_values.shape == (buf.R,)
    assert np.all(np.isfinite(buf.reward))
    assert np.any(buf.reward != 0.0)          # Δmass reward flows
    assert np.any(buf.active)                  # someone acted
    # reward only where a row acted (active is this-step; reward completes next step,
    # so at least the reward's support ⊆ active-somewhere)
    assert np.all(np.isfinite(last_values))


def test_end_to_end_worker_then_trainer(tmp_path):
    cfg = _cfg()
    shards = str(tmp_path / "shards")
    weights = str(tmp_path / "weights")
    consumed = str(tmp_path / "consumed")

    # CPU box: cold-start (v0), write 4 shards
    n = run_worker(cfg, worker_id=0, shard_out=shards, weights_in=weights,
                   num_rollouts=4, log_every=100)
    assert n == 4
    assert len(list_shards(shards)) == 4
    # each shard is on-policy v0 and carries the right rows
    s0 = read_shard(list_shards(shards)[0])
    assert s0.meta["policy_version"] == 0
    assert s0.buffer.R == cfg.games_per_worker * len(cfg.policy_seats)

    # snapshot the trainer's v0 weights for a change-check
    torch.manual_seed(cfg.base_seed)
    ref = SkeletonPolicy(entity_hidden=cfg.entity_hidden, trunk_hidden=cfg.trunk_hidden)
    key = next(iter(ref.state_dict()))
    v0_weight = ref.state_dict()[key].clone()

    # GPU box: consume 2 shards/update -> 2 updates
    final = run_trainer(cfg, shard_in=shards, weights_out=weights, consumed_dir=consumed,
                        num_updates=2, shards_per_update=2, idle_timeout_s=2.0,
                        poll_interval_s=0.1, log_every=100)

    assert final == 2
    assert latest_weights_version(weights) == 2
    assert len(list_shards(shards)) == 0                 # all consumed
    assert len(os.listdir(consumed)) == 4

    # weights actually changed from v0 (PPO did something)
    trained = SkeletonPolicy(entity_hidden=cfg.entity_hidden, trunk_hidden=cfg.trunk_hidden)
    load_weights_into(trained, weights)
    assert not torch.equal(trained.state_dict()[key], v0_weight)


def test_trainer_resumes_from_existing_weights(tmp_path):
    """Re-running the trainer on a dir that already has checkpoints continues from
    the latest version instead of restarting at v0."""
    cfg = _cfg()
    shards = str(tmp_path / "shards")
    weights = str(tmp_path / "weights")
    consumed = str(tmp_path / "consumed")

    # first run: 2 updates → v2
    run_worker(cfg, 0, shard_out=shards, weights_in=weights, num_rollouts=4, log_every=100)
    v_first = run_trainer(cfg, shards, weights, consumed, num_updates=2,
                          shards_per_update=2, idle_timeout_s=2.0, poll_interval_s=0.1, log_every=100)
    assert v_first == 2

    # capture the resumed weights to prove it loads them (not a fresh seed init)
    resumed = SkeletonPolicy(entity_hidden=cfg.entity_hidden, trunk_hidden=cfg.trunk_hidden)
    load_weights_into(resumed, weights)
    key = next(iter(resumed.state_dict()))
    trained_weight = resumed.state_dict()[key].clone()

    # more data, then re-run the SAME trainer command → resumes at v2, continues to v4
    run_worker(cfg, 1, shard_out=shards, weights_in=weights, num_rollouts=4, log_every=100)
    v_second = run_trainer(cfg, shards, weights, consumed, num_updates=2,
                           shards_per_update=2, idle_timeout_s=2.0, poll_interval_s=0.1, log_every=100)
    assert v_second == 4  # continued from 2, not restarted at 0→2

    # --fresh restarts from a seeded v0 (weights reset to the seed init)
    v_fresh = run_trainer(cfg, shards, weights, consumed, num_updates=0,
                          shards_per_update=2, idle_timeout_s=0.5, poll_interval_s=0.1,
                          log_every=100, fresh=True)
    fresh_pol = SkeletonPolicy(entity_hidden=cfg.entity_hidden, trunk_hidden=cfg.trunk_hidden)
    load_weights_into(fresh_pol, weights)
    torch.manual_seed(cfg.base_seed)
    seed_pol = SkeletonPolicy(entity_hidden=cfg.entity_hidden, trunk_hidden=cfg.trunk_hidden)
    torch.testing.assert_close(fresh_pol.state_dict()[key], seed_pol.state_dict()[key])


def test_worker_loads_published_weights(tmp_path):
    """A worker started after the trainer has published picks up the latest
    version (the hot-reload path uses the same loader)."""
    cfg = _cfg()
    weights = str(tmp_path / "weights")
    torch.manual_seed(cfg.base_seed)
    from train.harness.shard import write_weights
    p = SkeletonPolicy(entity_hidden=cfg.entity_hidden, trunk_hidden=cfg.trunk_hidden)
    write_weights(weights, p, version=0, arch_meta={})
    write_weights(weights, p, version=7, arch_meta={})

    fresh = SkeletonPolicy(entity_hidden=cfg.entity_hidden, trunk_hidden=cfg.trunk_hidden)
    assert load_weights_into(fresh, weights) == 7
