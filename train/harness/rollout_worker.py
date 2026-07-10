"""Rollout worker (P3.2, two-box) — CPU-box experience generator.

Runs on the **CPU actor box**. Unlike the single-box ``VecLearner`` (where a
central process does batched inference and the actors only step the env), the
two-box worker runs the policy **locally on CPU** — a forward for a 300–800k-param
net is sub-millisecond, so there is no need to ship per-step observations to the
GPU box. Instead the worker collects **whole rollouts** and writes them as shard
files (:mod:`train.harness.shard`); you ``rsync``/``scp`` those to the GPU box.
Between rollouts it hot-reloads any newer weights file the GPU box has produced
(and you have copied back), so its policy tracks the learner with a small lag.

``LocalCollector`` is the in-process twin of the single-box collection loop: it
vectorizes ``games_per_worker`` in-process :class:`~train.harness.game_runner.GameRunner`
games into fixed-slot rows, runs one local batched forward per round, records the
trajectory into a :class:`~train.harness.rollout.RolloutBuffer`, and computes the
GAE bootstrap ``last_values``. The reward/rollover semantics are identical to the
single-box path (shared ``GameRunner``), so shards train the same as live data.

Cold start: with no weights file present the worker builds a policy from
``torch.manual_seed(cfg.base_seed)`` — the *same* seeded init the trainer
publishes as version 0 — so the first (version-0) shards are on-policy.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Tuple

import numpy as np
import torch

from train.harness.collate import build_batched_obs
from train.harness.config import TrainConfig
from train.harness.game_runner import GameRunner
from train.harness.rollout import RolloutBuffer
from train.harness.shard import latest_weights_version, load_weights_into, write_shard
from train.model import build_policy
from train.ppo import sample_actions


def _scripted_by_seat(cfg: TrainConfig) -> dict:
    names = cfg.scripted_names or ("dont_move",)
    mapping, j = {}, 0
    for pid in range(8):
        if pid in cfg.policy_seats:
            continue
        mapping[pid] = names[j % len(names)]
        j += 1
    return mapping


class LocalCollector:
    """In-process vectorized collection with local CPU inference (one worker)."""

    def __init__(self, cfg: TrainConfig, policy: torch.nn.Module, worker_id: int = 0) -> None:
        self.cfg = cfg
        self.policy = policy  # torch, CPU
        self.seats = tuple(cfg.policy_seats)
        self.S = len(self.seats)
        self.G = cfg.games_per_worker
        self.R = self.G * self.S
        scripted = _scripted_by_seat(cfg)
        base = cfg.base_seed + worker_id * 10_000_019
        self.runners = [
            GameRunner(
                stream_index=g,
                policy_seats=self.seats,
                scripted_by_seat=scripted,
                base_seed=base,
                mass_scale=cfg.mass_scale,
                reward_clip=cfg.reward_clip,
                food_shaping_coef=cfg.food_shaping_coef,
            )
            for g in range(self.G)
        ]
        self.buffer = RolloutBuffer(cfg.rollout_len, self.R, policy.hidden_size)
        self.hidden = torch.zeros(self.R, policy.hidden_size)
        # P3.5 temporally-extended exploration (training-only; off unless configured).
        self.persistence = None
        if cfg.explore_persist_prob > 0.0:
            from train.harness.explore import ActionPersistence
            self.persistence = ActionPersistence(
                self.R, cfg.explore_persist_prob, cfg.explore_persist_max,
                cfg.base_seed + worker_id * 10_000_019)
        # P3.5 Track C: oracle-guided bursts (training-only; off unless configured).
        self.oracle = None
        if cfg.explore_oracle_prob > 0.0:
            from train.harness.explore import OracleGuidance
            self.oracle = OracleGuidance(
                self.R, cfg.explore_oracle_prob, cfg.explore_oracle_max,
                cfg.base_seed + worker_id * 10_000_019)

    def _gather(self):
        obs = np.zeros((self.R, self.buffer.obs.shape[-1]), dtype=np.float32)
        active = np.zeros(self.R, dtype=bool)
        is_reset = np.zeros(self.R, dtype=bool)
        pr = np.zeros(self.R, dtype=np.float32)
        pd = np.zeros(self.R, dtype=np.float32)
        pa = np.zeros(self.R, dtype=np.float32)
        for g, runner in enumerate(self.runners):
            f = runner.observe()
            lo, hi = g * self.S, (g + 1) * self.S
            obs[lo:hi] = f.obs
            active[lo:hi] = f.active > 0.5
            is_reset[lo:hi] = f.is_reset > 0.5
            pr[lo:hi], pd[lo:hi], pa[lo:hi] = f.prev_reward, f.prev_done, f.prev_acted
        return obs, active, is_reset, pr, pd, pa

    def _scatter_step(self, heading: np.ndarray, split: np.ndarray) -> None:
        for g, runner in enumerate(self.runners):
            lo, hi = g * self.S, (g + 1) * self.S
            runner.step(heading[lo:hi], split[lo:hi])

    def collect_rollout(self) -> Tuple[RolloutBuffer, np.ndarray]:
        """Collect one ``rollout_len``-step rollout; return ``(buffer, last_values)``."""
        self.buffer.reset()
        self.buffer.set_hidden0(self.hidden)
        pending = False
        with torch.no_grad():
            for t in range(self.cfg.rollout_len):
                obs, active, is_reset, pr, pd, pa = self._gather()
                if pending:
                    self.buffer.set_prev_transition(t - 1, pr, pd, pa)
                batched = build_batched_obs(obs, is_reset, "cpu")
                hl, sl, value, self.hidden = self.policy(batched, self.hidden)
                heading, split, logp = sample_actions(hl, sl)
                if self.persistence is not None:
                    heading, split, logp = self.persistence.apply(
                        hl, sl, heading, split, logp, is_reset)
                if self.oracle is not None:  # applied last: the oracle wins on its rows
                    heading, split, logp = self.oracle.apply(
                        hl, sl, heading, split, logp, is_reset, obs)
                h_np, s_np = heading.numpy(), split.numpy()
                self.buffer.add(t, obs, is_reset, active, h_np, s_np, logp.numpy(), value.numpy())
                self._scatter_step(h_np, s_np)
                pending = True

            # bootstrap round: complete step T-1's reward + V(s_T); action discarded
            obs, active, is_reset, pr, pd, pa = self._gather()
            self.buffer.set_prev_transition(self.cfg.rollout_len - 1, pr, pd, pa)
            batched = build_batched_obs(obs, is_reset, "cpu")
            hl, sl, value, hidden_after = self.policy(batched, self.hidden)
            heading, split, _ = sample_actions(hl, sl)
            self._scatter_step(heading.numpy(), split.numpy())
            self.hidden = hidden_after.detach()
        return self.buffer, value.numpy()


def run_worker(
    cfg: TrainConfig,
    worker_id: int,
    shard_out: str,
    weights_in: str,
    num_rollouts: int = 0,
    duration_s: float = 0.0,
    log_every: int = 5,
    compress: bool = False,
) -> int:
    """Collect rollouts → shard files, hot-reloading weights between them.

    Runs until ``num_rollouts`` shards are written or ``duration_s`` elapses
    (whichever first; 0 = unbounded on that axis). Returns the number of shards
    written.
    """
    cfg.validate()
    # Single-threaded per worker: many workers (≈ one per core) each spawning BLAS
    # threads would oversubscribe the box catastrophically. Inference here is one
    # tiny forward per round, so one thread is plenty. (Also honor OMP_NUM_THREADS=1
    # in the environment for the engine/numpy side.)
    torch.set_num_threads(1)
    torch.manual_seed(cfg.base_seed)  # identical init to the trainer's version 0
    policy = build_policy(cfg)  # AgarPolicy by default; SkeletonPolicy if cfg.use_skeleton
    policy.eval()
    version = load_weights_into(policy, weights_in) or 0
    collector = LocalCollector(cfg, policy, worker_id)

    seq = 0
    t0 = time.monotonic()
    while (num_rollouts == 0 or seq < num_rollouts):
        if duration_s and (time.monotonic() - t0) >= duration_s:
            break
        buffer, last_values = collector.collect_rollout()
        path = write_shard(
            shard_out, worker_id, seq, buffer, last_values, policy_version=version,
            extra_meta={"games": collector.G, "seats": collector.S}, compress=compress,
        )
        seq += 1
        # hot-reload newer weights (produced by the GPU box, copied back here)
        newv = latest_weights_version(weights_in)
        if newv is not None and newv > version:
            load_weights_into(policy, weights_in)
            version = newv
        if seq % log_every == 0:
            el = time.monotonic() - t0
            print(
                f"[worker {worker_id}] shards={seq} rows/shard={collector.R} "
                f"policy_v={version} rollouts/s={seq/el:.2f} last={path.split('/')[-1]}",
                flush=True,
            )
    return seq


def _main() -> None:
    p = argparse.ArgumentParser(description="P3.2 two-box rollout worker (CPU box)")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--shard-out", required=True, help="dir to write trajectory shards")
    p.add_argument("--weights-in", required=True, help="dir to read learner weights from")
    p.add_argument("--games", type=int, default=8, help="in-process games per worker")
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--rollouts", type=int, default=0, help="stop after N shards (0=unbounded)")
    p.add_argument("--duration", type=float, default=0.0, help="wall-clock cap (s); 0=off")
    p.add_argument("--policy-seats", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--seed", type=int, default=20260709)
    p.add_argument("--compress", action="store_true",
                   help="gzip shards (smaller over the network; costs worker CPU)")
    args = p.parse_args()

    cfg = TrainConfig(
        games_per_worker=args.games,
        rollout_len=args.rollout,
        policy_seats=tuple(args.policy_seats),
        base_seed=args.seed,
        device="cpu",
    )
    n = run_worker(cfg, args.worker_id, args.shard_out, args.weights_in,
                   num_rollouts=args.rollouts, duration_s=args.duration, compress=args.compress)
    print(f"worker {args.worker_id} wrote {n} shards to {args.shard_out}")


if __name__ == "__main__":
    _main()


__all__ = ["LocalCollector", "run_worker"]
