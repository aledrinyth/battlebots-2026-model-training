"""Trainer (P3.2, two-box) — GPU-box learner over shard files.

Runs on the **GPU box (B200)**. It owns the policy + optimizer, watches an input
directory for trajectory **shards** (produced by ``rollout_worker`` on the CPU
box and copied over with ``rsync``/``scp``), runs GAE + PPO on the GPU, and
publishes versioned **weight files** the CPU box copies back. The PPO math and
GAE are the *same* ``learner.ppo_update`` / ``RolloutBuffer.compute_gae`` as the
single-box path — only the data source (files, not a live shared-memory slab)
differs.

Staleness: because actors generate data with an older policy than the one being
updated, each shard carries the ``policy_version`` it was collected under. The
trainer logs the lag (``current_version − shard_version``) and can drop shards
older than ``max_staleness`` (PPO tolerates mild lag via its clip ratio;
``--max-staleness`` bounds it). ``--shards-per-update`` merges several small
worker shards along the row axis into one large-batch update (to reach the
≥512-row inference/train batch cheaply).

Cold start: the trainer seeds the policy with ``torch.manual_seed(cfg.base_seed)``
and publishes it as version 0 *before* consuming any shards, so a cold worker's
seeded init matches version 0 and its first shards are on-policy.
"""

from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np
import torch

from train.harness.config import TrainConfig
from train.harness.shard import (
    Shard,
    archive_shard,
    clear_weights,
    latest_weights_version,
    list_shards,
    load_weights_into,
    merge_shards,
    prune_weights,
    read_shard,
    write_weights,
)
from train.model import MODEL_VERSION, build_policy
from train.ppo import ppo_update


def _arch_meta(cfg: TrainConfig) -> dict:
    from bot.adapters.act_v1 import HEADING_BINS
    from bot.adapters.obs_v1 import OBS_VERSION

    # Record everything ``build_policy`` reads, so a checkpoint reconstructs the
    # exact architecture (P3.3: real AgarPolicy by default, or the stub).
    return {
        "model_version": MODEL_VERSION,
        "use_skeleton": cfg.use_skeleton,
        "model_entity_hidden": cfg.model_entity_hidden,
        "model_trunk_hidden": cfg.model_trunk_hidden,
        "entity_hidden": cfg.entity_hidden,
        "trunk_hidden": cfg.trunk_hidden,
        "heading_bins": HEADING_BINS,
        "obs_version": OBS_VERSION,
    }


def run_trainer(
    cfg: TrainConfig,
    shard_in: str,
    weights_out: str,
    consumed_dir: str,
    num_updates: int = 0,
    duration_s: float = 0.0,
    shards_per_update: int = 1,
    max_staleness: int = 0,
    keep_weights: int = 5,
    poll_interval_s: float = 1.0,
    idle_timeout_s: float = 0.0,
    log_every: int = 1,
    fresh: bool = False,
) -> int:
    """Consume shards → PPO updates → publish weights. Returns the final version.

    Stops at ``num_updates`` / ``duration_s`` / ``idle_timeout_s`` (whichever
    first; 0 = that axis is unbounded).

    **Resume:** if ``weights_out`` already holds checkpoints (a prior run), the
    trainer loads the latest and **continues** from that version — so re-running
    the same command after a kill picks up where it left off. Pass ``fresh=True``
    to ignore existing weights and restart from a seeded ``v0``. (The optimizer
    state is not checkpointed, so Adam's moments re-warm over the first few
    updates — a negligible transient.)
    """
    cfg.validate()
    torch.manual_seed(cfg.base_seed)
    device = torch.device(cfg.device)
    policy = build_policy(cfg).to(device)  # AgarPolicy by default; stub if cfg.use_skeleton
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    arch = _arch_meta(cfg)

    resume_v = None if fresh else latest_weights_version(weights_out)
    if resume_v is not None:
        load_weights_into(policy, weights_out)
        version = resume_v
        print(f"[resume] loaded weights v{version} from {weights_out}; continuing", flush=True)
    else:
        if fresh:
            # wipe old checkpoints so the seeded v0 is genuinely the latest version
            # (else a stale higher-version file would shadow it for loaders).
            n = clear_weights(weights_out)
            if n:
                print(f"[fresh] cleared {n} existing checkpoint(s); restarting at v0", flush=True)
        version = 0
        write_weights(weights_out, policy, version, arch)  # publish v0 for cold workers

    updates = 0
    dropped = 0
    t0 = time.monotonic()
    last_activity = t0

    while True:
        if num_updates and updates >= num_updates:
            break
        if duration_s and (time.monotonic() - t0) >= duration_s:
            break

        paths = list_shards(shard_in)
        if len(paths) < shards_per_update:
            if idle_timeout_s and (time.monotonic() - last_activity) >= idle_timeout_s:
                break
            time.sleep(poll_interval_s)
            continue

        batch_paths = paths[:shards_per_update]
        shards: List[Shard] = [read_shard(p) for p in batch_paths]
        buffer, last_values, min_version = merge_shards(shards)
        staleness = version - min_version

        if max_staleness and staleness > max_staleness:
            for p in batch_paths:
                archive_shard(p, consumed_dir + "_stale")
            dropped += len(batch_paths)
            continue

        gae = buffer.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)
        metrics = ppo_update(policy, optimizer, buffer, gae.advantages, gae.returns, cfg, device)
        version += 1
        updates += 1
        last_activity = time.monotonic()

        write_weights(weights_out, policy, version, arch)
        prune_weights(weights_out, keep_weights)
        for p in batch_paths:
            archive_shard(p, consumed_dir)

        if updates % log_every == 0:
            el = time.monotonic() - t0
            print(
                f"[upd {updates}] v={version} rows={buffer.R} staleness={staleness} "
                f"dropped={dropped} loss={metrics.get('loss', 0):+.4f} "
                f"pl={metrics.get('policy_loss', 0):+.4f} vl={metrics.get('value_loss', 0):.4f} "
                f"ent={metrics.get('entropy', 0):.3f} kl={metrics.get('approx_kl', 0):+.4f} "
                f"upd/s={updates/el:.2f}",
                flush=True,
            )
    return version


def _main() -> None:
    p = argparse.ArgumentParser(description="P3.2 two-box trainer (GPU box)")
    p.add_argument("--shard-in", required=True, help="dir to read trajectory shards from")
    p.add_argument("--weights-out", required=True, help="dir to write learner weights to")
    p.add_argument("--consumed-dir", default=None, help="dir to move consumed shards to")
    p.add_argument("--updates", type=int, default=0, help="stop after N updates (0=unbounded)")
    p.add_argument("--duration", type=float, default=0.0, help="wall-clock cap (s); 0=off")
    p.add_argument("--shards-per-update", type=int, default=1)
    p.add_argument("--max-staleness", type=int, default=0, help="drop shards older than N versions (0=off)")
    p.add_argument("--keep-weights", type=int, default=5)
    p.add_argument("--idle-timeout", type=float, default=0.0, help="stop after N idle s (0=off)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore existing weights and restart from v0 (default: resume)")
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--policy-seats", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--seed", type=int, default=20260709)
    args = p.parse_args()

    consumed = args.consumed_dir or (args.shard_in.rstrip("/") + "_consumed")
    cfg = TrainConfig(
        rollout_len=args.rollout,
        device=args.device,
        policy_seats=tuple(args.policy_seats),
        base_seed=args.seed,
    )
    final = run_trainer(
        cfg, args.shard_in, args.weights_out, consumed,
        num_updates=args.updates, duration_s=args.duration,
        shards_per_update=args.shards_per_update, max_staleness=args.max_staleness,
        keep_weights=args.keep_weights, idle_timeout_s=args.idle_timeout,
        fresh=args.fresh,
    )
    print(f"trainer finished at weights version {final}")


if __name__ == "__main__":
    _main()


__all__ = ["run_trainer"]
