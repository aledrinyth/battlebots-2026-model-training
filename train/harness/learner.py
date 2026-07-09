"""Central learner (P3.2) — batched GPU inference + PPO over N actor processes.

This is the "single GPU inference+learner process" half of the P3.1-chosen
custom substrate. It:

1. spawns ``cfg.num_actors`` :func:`~train.harness.vec_actor.actor_process`
   workers (numpy/CPU-only) and hands them shared-memory slabs;
2. every round, synchronizes on a two-hit barrier, reads the whole obs slab as
   one contiguous ``(R, OBS_WIDTH)`` array, runs **one** batched forward
   (``R = num_actors × seats`` rows — the ≥512 inference batch of the P3.2
   acceptance when scaled), samples actions, and writes them back;
3. accumulates ``rollout_len`` steps into a :class:`~train.harness.rollout.RolloutBuffer`,
   completing each step's one-round-late reward, then computes GAE and runs a
   recurrent-PPO update (clip 0.2, γ=0.999, λ=0.95, BPTT = ``rollout_len``);
4. reports sustained throughput and RSS so the acceptance ("≥50% of projection
   for ≥30 min without deadlock or memory growth") is *measured*, not asserted.

The network here is still the P3.1 :class:`~train.harness.policy_stub.SkeletonPolicy`
stub; **P3.3 swaps in the real deep-set model + asymmetric critic** and formalizes
``train/model.py`` / ``train/ppo.py``. The PPO update below is the minimal correct
version that proves the buffer→update path closes and reduces loss (its overfit
sanity is P3.3's full acceptance). Nothing about the transport, the buffer, or
the GAE changes when the network is replaced.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import time
from dataclasses import dataclass, field
from threading import BrokenBarrierError
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from train.harness.collate import build_batched_obs
from train.harness.config import TrainConfig
from train.harness.rollout import RolloutBuffer
from train.harness.shm import META_FIELDS, SlabSet
from train.harness.vec_actor import actor_process
from train.model import assert_actor_param_budget, build_policy, count_actor_parameters
from train.ppo import evaluate_actions, ppo_update, sample_actions

# The PPO step + BPTT re-eval + action sampling now live in ``train.ppo`` (P3.3
# formalization). Re-exported here — and aliased to their historical private
# names — so existing importers (``from train.harness.learner import ppo_update,
# sample_actions, _evaluate``) keep working unchanged.
_evaluate = evaluate_actions

_MI = {name: i for i, name in enumerate(META_FIELDS)}


# --- RSS probe (no psutil dependency) ---------------------------------------

def _rss_kb(pids: List[int]) -> int:
    """Summed resident-set size (KB) of the given pids via ``ps`` (macOS/Linux).

    One batched ``ps -p pid,pid,…`` call — probing 100+ actor pids one
    subprocess at a time cost multiple seconds *per update* on a many-core rig.
    ``ps`` prints the rows it finds even when some pids are gone (nonzero exit),
    so dead pids just drop out of the sum.
    """
    if not pids:
        return 0
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)],
            capture_output=True, text=True, timeout=15,
        )
        return sum(int(line.split()[0]) for line in out.stdout.splitlines() if line.strip())
    except Exception:
        return 0


# --- stats ------------------------------------------------------------------

@dataclass
class TrainStats:
    """Evidence the P3.2 loop ran: throughput, batch size, memory stability."""

    updates: int = 0
    rounds: int = 0
    agent_steps: int = 0            # active rows summed over rounds
    batch_rows: int = 0            # constant inference batch width (R)
    wall_s: float = 0.0
    rss_start_kb: int = 0
    rss_warm_kb: int = 0           # RSS after update 1 (excludes torch warmup) — leak baseline
    rss_end_kb: int = 0
    rss_peak_kb: int = 0
    deadlocked: bool = False
    last_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def rounds_per_s(self) -> float:
        return self.rounds / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def agent_steps_per_s(self) -> float:
        return self.agent_steps / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def rss_growth_kb(self) -> int:
        return self.rss_end_kb - self.rss_start_kb

    @property
    def rss_growth_post_warmup_kb(self) -> int:
        """RSS growth after the first update — the actual leak signal (torch's
        one-time warmup allocation is charged to update 1, not to a leak)."""
        base = self.rss_warm_kb or self.rss_start_kb
        return self.rss_end_kb - base

    def summary(self) -> str:
        return (
            f"updates={self.updates} rounds={self.rounds} "
            f"batch_rows={self.batch_rows} "
            f"rounds/s={self.rounds_per_s:.1f} agent_steps/s={self.agent_steps_per_s:.0f} "
            f"rss_start={self.rss_start_kb//1024}MB rss_warm={self.rss_warm_kb//1024}MB "
            f"rss_end={self.rss_end_kb//1024}MB "
            f"rss_growth_post_warmup={self.rss_growth_post_warmup_kb//1024}MB "
            f"deadlocked={self.deadlocked}"
        )


# --- the learner ------------------------------------------------------------

class VecLearner:
    """Owns the policy + actor processes + shared slabs; runs the training loop."""

    def __init__(
        self,
        cfg: TrainConfig,
        policy: Optional[nn.Module] = None,
        *,
        on_update: Optional[Callable[["VecLearner", int, Dict[str, float]], None]] = None,
        global_update0: int = 0,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # Opt-in learner-side torch thread cap (env-only; no config/hash impact).
        # torch sizes its intra-op pool to every core by default, which on a
        # 100+-core CPU box adds per-op sync overhead to the small rollout
        # forwards and competes with the actor processes for cores. Set e.g.
        # TRAIN_TORCH_THREADS=16 to cap it; unset leaves torch's default.
        _tt = os.environ.get("TRAIN_TORCH_THREADS")
        if _tt:
            torch.set_num_threads(max(1, int(_tt)))
        #: Optional post-update hook (P3.4): called after every completed PPO
        #: update as ``on_update(self, global_update, metrics)`` where
        #: ``global_update = global_update0 + local_update`` (so a resumed run
        #: keeps counting up). The :class:`~train.harness.run.RunManager` uses it
        #: to append the metrics curve and checkpoint on a wall-clock cadence.
        #: ``None`` (the default) leaves the P3.2/P3.3 loop byte-for-byte
        #: unchanged. Settable after construction too.
        self.on_update = on_update
        #: Offset added to the local update index for the hook + logs on resume.
        self.global_update0 = int(global_update0)
        # P3.3: the real AgarPolicy (deep-set + GRU(256) + asymmetric critic) is
        # the default; ``cfg.use_skeleton`` selects the P3.1 wiring stub. Both
        # honor the same ``policy(BatchedObs, hidden)`` contract, so the loop below
        # is unchanged. Print + assert the actor-path parameter budget (300k–800k)
        # when we build the real network — the playbook P3.3 acceptance.
        self.policy = (policy if policy is not None else build_policy(cfg)).to(self.device)
        if policy is None and not cfg.use_skeleton:
            n = assert_actor_param_budget(self.policy)
            print(f"[model] AgarPolicy actor-path params={n:,} (budget 300k–800k)", flush=True)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

        self.R = cfg.batch_rows
        self.seats = cfg.seats_per_actor
        self.buffer = RolloutBuffer(cfg.rollout_len, self.R, self.policy.hidden_size)

        # P3.5: temporally-extended exploration (training-only; off unless configured).
        self.persistence = None
        if cfg.explore_persist_prob > 0.0:
            from train.harness.explore import ActionPersistence
            self.persistence = ActionPersistence(
                self.R, cfg.explore_persist_prob, cfg.explore_persist_max, cfg.base_seed)
        # P3.5 Track C: oracle-guided bursts (training-only; off unless configured).
        self.oracle = None
        if cfg.explore_oracle_prob > 0.0:
            from train.harness.explore import OracleGuidance
            self.oracle = OracleGuidance(
                self.R, cfg.explore_oracle_prob, cfg.explore_oracle_max, cfg.base_seed)

        self._ctx = mp.get_context("spawn")
        self._slabs: Optional[SlabSet] = None
        self._procs: List[mp.process.BaseProcess] = []
        self._barrier = None
        self._stop = None
        self._error = None

    # -- process lifecycle ---------------------------------------------------

    def _scripted_by_seat(self) -> Dict[int, str]:
        names = self.cfg.scripted_names or ("dont_move",)
        mapping: Dict[int, str] = {}
        j = 0
        for pid in range(8):
            if pid in self.cfg.policy_seats:
                continue
            mapping[pid] = names[j % len(names)]
            j += 1
        return mapping

    def _spawn(self) -> None:
        self._slabs = SlabSet.create(self.cfg.num_actors, self.seats)
        names = self._slabs.names()
        self._barrier = self._ctx.Barrier(self.cfg.num_actors + 1)
        self._stop = self._ctx.Event()
        self._error = self._ctx.Event()
        scripted = self._scripted_by_seat()

        # Each actor is a single-stream numpy worker; a default BLAS pool per
        # actor (cores-wide, ×num_actors processes) oversubscribes a many-core
        # box into scheduler thrash. Pin the children to 1 BLAS thread via env
        # inherited at spawn (the two-box rollout_worker does the same). Only
        # unset vars are pinned — an explicit user setting wins — and the
        # parent's env is restored after the starts (the learner's own torch
        # pool is untouched; it was sized at import).
        _thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
        _saved = {v: os.environ.get(v) for v in _thread_vars}
        for v in _thread_vars:
            os.environ.setdefault(v, "1")

        for a in range(self.cfg.num_actors):
            p = self._ctx.Process(
                target=actor_process,
                args=(
                    a,
                    tuple(self.cfg.policy_seats),
                    scripted,
                    self.cfg.base_seed,
                    self.cfg.mass_scale,
                    self.cfg.reward_clip,
                    names,
                    self._barrier,
                    self._stop,
                    self._error,
                    self.cfg.barrier_timeout_s,
                    self.cfg.food_shaping_coef,
                ),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

        for v, old in _saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old

    def _shutdown(self) -> None:
        if self._stop is not None:
            self._stop.set()
        # Release actors blocked at the next barrier-1, then let them exit.
        if self._barrier is not None:
            try:
                self._barrier.wait(self.cfg.barrier_timeout_s)
            except (BrokenBarrierError, Exception):
                try:
                    self._barrier.abort()
                except Exception:
                    pass
        for p in self._procs:
            p.join(timeout=self.cfg.barrier_timeout_s)
            if p.is_alive():
                p.terminate()
        if self._slabs is not None:
            self._slabs.close()

    # -- barrier helpers -----------------------------------------------------

    def _sync(self) -> None:
        """One barrier hit; raises RuntimeError on break (deadlock/actor crash)."""
        try:
            self._barrier.wait(self.cfg.barrier_timeout_s)
        except BrokenBarrierError as exc:
            raise RuntimeError("barrier broken (actor crash or timeout)") from exc

    # -- one exchange round --------------------------------------------------

    def _read_slab(self):
        meta = self._slabs.meta.reshape(self.R, -1)
        obs = self._slabs.obs.reshape(self.R, -1).copy()  # copy: slab is overwritten next round
        active = meta[:, _MI["active"]] > 0.5
        is_reset = meta[:, _MI["is_reset"]] > 0.5
        prev_reward = meta[:, _MI["prev_reward"]].copy()
        prev_done = meta[:, _MI["prev_done"]].copy()
        prev_acted = meta[:, _MI["prev_acted"]].copy()
        return obs, active, is_reset, prev_reward, prev_done, prev_acted

    def _forward_and_write(self, obs, is_reset, hidden):
        batched = build_batched_obs(obs, is_reset, self.device)
        with torch.no_grad():
            hl, sl, value, new_hidden = self.policy(batched, hidden)
            heading, split, logp = sample_actions(hl, sl)
            if self.persistence is not None:
                heading, split, logp = self.persistence.apply(
                    hl, sl, heading, split, logp, is_reset)
            if self.oracle is not None:  # applied last: the oracle wins on its rows
                heading, split, logp = self.oracle.apply(
                    hl, sl, heading, split, logp, is_reset, obs)
        act = self._slabs.act.reshape(self.R, -1)
        act[:, 0] = heading.cpu().numpy().astype(np.float32)
        act[:, 1] = split.cpu().numpy().astype(np.float32)
        return heading, split, logp, value, new_hidden

    # -- main loop -----------------------------------------------------------

    def train(self) -> TrainStats:
        stats = TrainStats(batch_rows=self.R)
        self._spawn()
        pids = [p.pid for p in self._procs] + [__import__("os").getpid()]
        stats.rss_start_kb = _rss_kb(pids)
        stats.rss_peak_kb = stats.rss_start_kb

        t0 = time.monotonic()
        hidden = torch.zeros(self.R, self.policy.hidden_size, device=self.device)
        try:
            while stats.updates < self.cfg.num_updates:
                if self.cfg.duration_s and (time.monotonic() - t0) >= self.cfg.duration_s:
                    break
                self.buffer.reset()
                self.buffer.set_hidden0(hidden)
                pending_slot: Optional[int] = None

                # ---- collect T stored steps ----
                for t in range(self.cfg.rollout_len):
                    self._sync()  # barrier 1: actors filled
                    obs, active, is_reset, pr, pd, pa = self._read_slab()
                    if pending_slot is not None:
                        self.buffer.set_prev_transition(pending_slot, pr, pd, pa)
                    heading, split, logp, value, hidden = self._forward_and_write(obs, is_reset, hidden)
                    self.buffer.add(
                        t, obs, is_reset, active,
                        heading.cpu().numpy(), split.cpu().numpy(),
                        logp.cpu().numpy(), value.cpu().numpy(),
                    )
                    pending_slot = t
                    stats.rounds += 1
                    stats.agent_steps += int(active.sum())
                    self._sync()  # barrier 2: actors step

                # ---- bootstrap round T: last reward + V(s_T); action discarded ----
                self._sync()
                obs, active, is_reset, pr, pd, pa = self._read_slab()
                self.buffer.set_prev_transition(self.cfg.rollout_len - 1, pr, pd, pa)
                _, _, _, value, hidden_after = self._forward_and_write(obs, is_reset, hidden)
                last_values = value.cpu().numpy()
                stats.rounds += 1
                self._sync()

                # ---- GAE + PPO ----
                gae = self.buffer.compute_gae(last_values, self.cfg.gamma, self.cfg.gae_lambda)
                metrics = ppo_update(
                    self.policy, self.optimizer, self.buffer,
                    gae.advantages, gae.returns, self.cfg, self.device,
                )
                # Direct reward visibility (P3.5 diagnostics): ``eats`` counts
                # active steps with positive Δmass this rollout (a food eat —
                # decay-only steps are ≤ 0), so "is reward flowing?" is read
                # straight off the log instead of inferred from value_loss.
                rew = self.buffer.reward[self.buffer.active]
                metrics["mean_reward"] = float(rew.mean()) if rew.size else 0.0
                metrics["eats"] = float((rew > 0).sum())
                stats.updates += 1
                stats.last_metrics = metrics
                hidden = hidden_after.detach()

                if self.on_update is not None:
                    # P3.4 hook: logging + wall-clock checkpointing live here so the
                    # transport loop stays oblivious to run management. Runs between
                    # rollouts (actors idle at barrier-1), so a checkpoint write does
                    # not race the slabs.
                    self.on_update(self, self.global_update0 + stats.updates, metrics)

                rss = _rss_kb(pids)
                stats.rss_peak_kb = max(stats.rss_peak_kb, rss)
                if stats.updates == 1:
                    stats.rss_warm_kb = rss
                if stats.updates % self.cfg.log_every == 0:
                    el = time.monotonic() - t0
                    print(
                        f"[upd {stats.updates}/{self.cfg.num_updates}] "
                        f"rounds={stats.rounds} rounds/s={stats.rounds/el:.1f} "
                        f"batch={self.R} loss={metrics.get('loss', 0):+.4f} "
                        f"pl={metrics.get('policy_loss', 0):+.4f} vl={metrics.get('value_loss', 0):.4f} "
                        f"ent={metrics.get('entropy', 0):.3f} kl={metrics.get('approx_kl', 0):+.4f} "
                        f"eats={metrics.get('eats', 0):.0f} mr={metrics.get('mean_reward', 0):+.5f} "
                        f"rss={rss//1024}MB",
                        flush=True,
                    )
        except RuntimeError as exc:
            stats.deadlocked = "barrier broken" in str(exc)
            raise
        finally:
            stats.wall_s = time.monotonic() - t0
            stats.rss_end_kb = _rss_kb(pids)
            stats.rss_peak_kb = max(stats.rss_peak_kb, stats.rss_end_kb)
            self._shutdown()
        return stats


def run_training(cfg: TrainConfig, policy: Optional[nn.Module] = None) -> TrainStats:
    """Convenience entry point: build a :class:`VecLearner` and run it."""
    return VecLearner(cfg, policy=policy).train()


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="P3.2 vectorized actor↔learner training")
    p.add_argument("--actors", type=int, default=4)
    p.add_argument("--updates", type=int, default=4)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--duration", type=float, default=0.0, help="wall-clock cap (s); 0=off")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--policy-seats", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--min-batch", type=int, default=0, help="assert batch_rows >= this (512 on rig)")
    args = p.parse_args()

    cfg = TrainConfig(
        num_actors=args.actors,
        num_updates=args.updates,
        rollout_len=args.rollout,
        duration_s=args.duration,
        device=args.device,
        policy_seats=tuple(args.policy_seats),
        min_infer_batch=args.min_batch,
    )
    stats = run_training(cfg)
    print(stats.summary())


if __name__ == "__main__":
    _main()


__all__ = [
    "VecLearner",
    "run_training",
    "TrainStats",
    "ppo_update",
    "sample_actions",
]
