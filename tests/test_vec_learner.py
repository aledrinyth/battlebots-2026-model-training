"""P3.2 — vectorized multiprocess actor↔learner integration.

The acceptance ("sustained throughput ≥50% of projection for ≥30 min without
deadlock or memory growth; GPU inference batch ≥512") is a *rig-scale* property
that can't run in CI on a laptop with no GPU. What CI *can* prove — and what
these tests assert — is that the whole pipeline is correct and stable at small
scale: real OS processes exchange obs over shared memory, the barrier protocol
never deadlocks, games advance, the Δmass reward flows into the buffer, PPO
updates run, the actor processes are genuinely torch-free, and memory is flat
across updates. The rig run itself is the ``--runslow`` throughput test plus the
manual command in ``notes/P3.2-vectorized-actors.md``.

Spawn note: children must import ``train``; we put the repo root on ``PYTHONPATH``
(inherited by spawned processes) so the test is invocation-independent.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")  # train-side integration test; skipped in the bot venv

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = _REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

from train.harness.config import TrainConfig  # noqa: E402
from train.harness.learner import VecLearner, run_training  # noqa: E402


def test_batch_rows_arithmetic():
    cfg = TrainConfig(num_actors=64, policy_seats=tuple(range(8)))
    assert cfg.batch_rows == 512  # the ≥512 inference-batch target on the rig
    # min_infer_batch gate rejects an under-sized topology
    with pytest.raises(ValueError):
        TrainConfig(num_actors=4, policy_seats=(0,), min_infer_batch=512).validate()


def test_small_run_advances_without_deadlock():
    cfg = TrainConfig(
        num_actors=2, num_updates=2, rollout_len=4, policy_seats=tuple(range(8))
    )
    learner = VecLearner(cfg)
    stats = learner.train()

    assert stats.updates == 2
    assert stats.rounds == 2 * (cfg.rollout_len + 1)  # +1 bootstrap round/update
    assert stats.batch_rows == 16
    assert not stats.deadlocked
    assert stats.agent_steps > 0
    for k in ("loss", "policy_loss", "value_loss", "entropy"):
        assert np.isfinite(stats.last_metrics[k])


def test_reward_signal_reaches_the_buffer():
    """The dense Δmass reward must actually populate the buffer (not stay zero) —
    a longer rollout so mass changes accumulate."""
    cfg = TrainConfig(
        num_actors=2, num_updates=1, rollout_len=24, policy_seats=tuple(range(8)),
        scripted_names=("greedy_food",),
    )
    learner = VecLearner(cfg)
    learner.train()
    # After the last update the buffer holds that rollout; some transitions must
    # carry non-zero reward, and rewards must be finite.
    assert np.any(learner.buffer.reward != 0.0)
    assert np.all(np.isfinite(learner.buffer.reward))
    # active steps and rewards line up: a reward only on rows that acted.
    acted = learner.buffer.active
    rewarded = learner.buffer.reward != 0.0
    assert np.all(rewarded[~acted] == False)  # noqa: E712


def test_actor_process_is_torch_free():
    """The actor module (and its full import graph) must never pull in torch —
    the match-legal / spawn-fast invariant. Checked in a clean subprocess."""
    code = (
        "import sys; import train.harness.vec_actor as m; "
        "assert 'torch' not in sys.modules, "
        "'torch leaked into the actor import graph'; print('ok')"
    )
    env = dict(os.environ, PYTHONPATH=_REPO)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


@pytest.mark.slow
def test_throughput_and_memory_stability():
    """Longer run: throughput is non-trivial and RSS does not grow across updates
    (the leak guard). Baseline is taken *after* the first update so torch's
    one-time warmup allocation is excluded."""
    cfg = TrainConfig(
        num_actors=4, num_updates=6, rollout_len=32, policy_seats=tuple(range(8)),
    )
    learner = VecLearner(cfg)

    # Instrument RSS per update by wrapping the public run: we re-run with a
    # duration and sample via the stats the learner already records.
    stats = learner.train()
    assert stats.updates == 6
    assert not stats.deadlocked
    assert stats.rounds_per_s > 0.0
    # Leak guard: RSS growth *after the first update* (torch warmup excluded) must
    # be small — the trajectory buffer is fixed-size, so steady state is flat.
    assert stats.rss_warm_kb > 0
    growth_mb = stats.rss_growth_post_warmup_kb / 1024
    assert growth_mb < 64, f"post-warmup RSS grew {growth_mb:.0f} MB over 5 updates"
