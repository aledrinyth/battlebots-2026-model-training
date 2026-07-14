"""Vectorized actor process (P3.2, single-box) — numpy/CPU-only experience gen.

One OS process per entry in ``num_actors``, for the **single-machine** learner
(``learner.VecLearner``). Each actor hosts one :class:`~train.harness.game_runner.GameRunner`
(a sequence of 8-player games with fixed policy-seat rows) and exchanges obs /
actions with the central learner over a shared-memory slab, synchronizing on a
two-hit barrier. It never imports torch: the actor *is* the match-legal numpy
surface, so the learner can batch every actor's obs into one GPU forward while
the actors stay CPU-bound.

The per-game reward/stash/rollover logic lives in :class:`GameRunner` (shared
with the two-box ``rollout_worker.LocalCollector``); this module only marshals a
runner's rows into/out of the shared-memory slab and handles the barrier
protocol + clean abort-on-error (no deadlock).

Fixed-slot rows and the one-round-late reward are documented on
:class:`GameRunner`; the slab layout on :mod:`train.harness.shm`.
"""

from __future__ import annotations

from threading import BrokenBarrierError
from typing import Dict, Tuple

from train.harness.game_runner import GameRunner
from train.harness.shm import SlabNames, SlabSet


def actor_process(
    actor_index: int,
    policy_seats: Tuple[int, ...],
    scripted_by_seat: Dict[int, str],
    base_seed: int,
    mass_scale: float,
    reward_clip: float,
    slab_names: SlabNames,
    barrier,
    stop_event,
    error_event,
    timeout: float,
    food_shaping_coef: float = 0.0,
    dr_food_density: float = 0.0,
) -> None:
    """Run one actor until ``stop_event`` / a broken barrier. Spawn entry point."""
    slabs: SlabSet | None = None
    try:
        slabs = SlabSet.attach(slab_names)
        obs_slab = slabs.obs[actor_index]      # (seats, OBS_WIDTH)
        meta_slab = slabs.meta[actor_index]    # (seats, META_W)
        act_slab = slabs.act[actor_index]      # (seats, 2)

        runner = GameRunner(
            stream_index=actor_index,
            policy_seats=policy_seats,
            scripted_by_seat=scripted_by_seat,
            base_seed=base_seed,
            mass_scale=mass_scale,
            reward_clip=reward_clip,
            food_shaping_coef=food_shaping_coef,
            dr_food_density=dr_food_density,
        )

        while True:
            # ---- fill slab from the runner's fixed-slot rows ----
            frame = runner.observe()
            obs_slab[:] = frame.obs
            meta_slab[:, 0] = frame.active
            meta_slab[:, 1] = frame.is_reset
            meta_slab[:, 2] = frame.prev_reward
            meta_slab[:, 3] = frame.prev_done
            meta_slab[:, 4] = frame.prev_acted

            # ---- barrier 1: learner reads + infers ----
            try:
                barrier.wait(timeout)
            except BrokenBarrierError:
                break
            if stop_event.is_set():
                break

            # ---- barrier 2: learner has written actions ----
            try:
                barrier.wait(timeout)
            except BrokenBarrierError:
                break

            runner.step(act_slab[:, 0], act_slab[:, 1])

    except Exception:  # pragma: no cover - defensive; surfaced via error_event
        if error_event is not None:
            error_event.set()
        try:
            barrier.abort()
        except Exception:
            pass
        raise
    finally:
        if slabs is not None:
            slabs.close()


__all__ = ["actor_process"]
