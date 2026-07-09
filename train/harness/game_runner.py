"""GameRunner (P3.2) — one in-process game as fixed-slot rows, numpy/CPU only.

The reward/stash/rollover logic for a single 8-player game is subtle (Δmass
computed across the death gap, transitions reported one round late, seamless
game rollover), and getting it wrong silently corrupts the training signal. So it
lives here **once** and is shared by both experience-generation paths:

* the single-box remote actor (`vec_actor.actor_process`, which writes these rows
  into a shared-memory slab and syncs on barriers), and
* the two-box local collector (`rollout_worker.LocalCollector`, which writes them
  straight into a rollout buffer and runs inference in-process).

A runner owns one :class:`~train.harness.actor.Actor` (its fixed ``policy_seats``
occupy the row slots) and exposes exactly two calls per round:

* :meth:`observe` — featurize the alive policy seats into fixed-slot rows and
  surface the *previous* action's reward/done (known only now, post-step);
* :meth:`step` — decode the learner's per-slot actions, step the env, compute
  each acted seat's Δmass reward, and roll a finished game over to a fresh one.

This module never imports torch — it is the match-legal numpy surface. Actions
arrive as already-decided ``(heading_bin, split)`` values (the policy lives
elsewhere), so no policy forward happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

from bot.adapters.act_v1 import CanonicalAction
from train.harness.actor import Actor
from train.harness.flat_obs import OBS_WIDTH, flatten


def total_mass(actor: Actor, pid: int) -> float:
    """Total mass of player ``pid`` right now (``Σ blob.mass``; 0 if dead)."""
    return float(sum(b.mass for b in actor.env.state.players[pid].blobs.values()))


def nearest_food_dist(actor: Actor, pid: int) -> float:
    """Distance from player ``pid``'s mass centroid to the nearest food.

    Used only by the optional Stage-0 food shaping (``food_shaping_coef > 0``).
    Returns ``inf`` if the player is dead or no food exists (⇒ zero shaping).
    """
    player = actor.env.state.players[pid]
    if not player.blobs:
        return float("inf")
    px, py = player.x, player.y
    best = float("inf")
    for f in actor.env.state.map.foods:
        fx, fy = f.pos
        d2 = (px - fx) * (px - fx) + (py - fy) * (py - fy)
        if d2 < best:
            best = d2
    return best ** 0.5 if best != float("inf") else float("inf")


def _make_actor(seed: int, policy_seats: Tuple[int, ...], scripted_by_seat: Mapping[int, str]) -> Actor:
    from train.opponents.scripted import make_bot

    scripted = {int(pid): make_bot(name) for pid, name in scripted_by_seat.items()}
    a = Actor(seed=seed, policy_seats=policy_seats, scripted=scripted, emit_update=False)
    a.reset()
    return a


def game_seed(base_seed: int, stream_index: int, game_counter: int) -> int:
    """Deterministic, distinct seed per (stream, game) — reproducibility (P3.4)."""
    return base_seed + stream_index * 1_000_003 + game_counter * 7


@dataclass(frozen=True)
class ObsFrame:
    """One round of fixed-slot rows for a single game (``S = len(seats)``).

    ``obs`` rows are zero for inactive (dead-gap) seats. ``prev_*`` describe the
    transition each seat's *last* action produced (0 for a seat that didn't act
    last round), to be written into the buffer's previous step by the caller.
    """

    active: np.ndarray      # (S,) float32 {0,1}
    is_reset: np.ndarray    # (S,) float32 {0,1}
    obs: np.ndarray         # (S, OBS_WIDTH) float32
    prev_reward: np.ndarray  # (S,) float32
    prev_done: np.ndarray    # (S,) float32
    prev_acted: np.ndarray   # (S,) float32


class GameRunner:
    """Drives one game's fixed ``policy_seats`` rows across its lifetime.

    Args mirror the reward config: ``mass_scale`` divides Δmass, ``reward_clip``
    bounds it (guarding the death step, which loses the whole body).
    """

    def __init__(
        self,
        stream_index: int,
        policy_seats: Tuple[int, ...],
        scripted_by_seat: Mapping[int, str],
        base_seed: int,
        mass_scale: float,
        reward_clip: float,
        food_shaping_coef: float = 0.0,
    ) -> None:
        self.stream_index = stream_index
        self.seats = tuple(policy_seats)
        self.scripted_by_seat = dict(scripted_by_seat)
        self.base_seed = base_seed
        self.mass_scale = mass_scale
        self.reward_clip = reward_clip
        #: Optional dense food-approach shaping (see TrainConfig.food_shaping_coef).
        #: 0.0 ⇒ pure Δmass, and the nearest-food scan below is skipped entirely.
        self.food_shaping_coef = float(food_shaping_coef)

        self._game_counter = 0
        self.actor = _make_actor(
            game_seed(base_seed, stream_index, 0), self.seats, self.scripted_by_seat
        )
        self._stash: Dict[int, Tuple[float, float, float]] = {s: (0.0, 0.0, 0.0) for s in self.seats}
        self._mass_before: Dict[int, float] = {}
        self._dist_before: Dict[int, float] = {}

    # -- the round exchange --------------------------------------------------

    def observe(self) -> ObsFrame:
        """Featurize alive policy seats into fixed-slot rows; surface prev transition."""
        seat_obs = self.actor.observe()  # {seat: CanonicalObs} for alive policy seats
        n = len(self.seats)
        active = np.zeros(n, dtype=np.float32)
        is_reset = np.zeros(n, dtype=np.float32)
        obs = np.zeros((n, OBS_WIDTH), dtype=np.float32)
        pr = np.zeros(n, dtype=np.float32)
        pd = np.zeros(n, dtype=np.float32)
        pa = np.zeros(n, dtype=np.float32)

        self._mass_before = {}
        self._dist_before = {}
        for slot, seat in enumerate(self.seats):
            r, d, a = self._stash[seat]
            pr[slot], pd[slot], pa[slot] = r, d, a
            if seat in seat_obs:
                flatten(seat_obs[seat], out=obs[slot])
                active[slot] = 1.0
                is_reset[slot] = 1.0 if seat_obs[seat].is_reset else 0.0
                self._mass_before[seat] = total_mass(self.actor, seat)
                if self.food_shaping_coef:
                    self._dist_before[seat] = nearest_food_dist(self.actor, seat)
        return ObsFrame(active, is_reset, obs, pr, pd, pa)

    def step(self, heading: np.ndarray, split: np.ndarray) -> None:
        """Apply per-slot ``(heading_bin, split)``, step, compute Δmass rewards."""
        seat_obs = self.actor.observe()  # cached; the seats that will act this round
        actions: Dict[int, CanonicalAction] = {}
        for slot, seat in enumerate(self.seats):
            if seat in seat_obs:
                actions[seat] = CanonicalAction(
                    heading_bin=int(round(float(heading[slot]))),
                    split=bool(split[slot] > 0.5),
                )
        done = self.actor.step(actions)

        new_stash: Dict[int, Tuple[float, float, float]] = {s: (0.0, 0.0, 0.0) for s in self.seats}
        for seat in seat_obs:
            player = self.actor.env.state.players[seat]
            mass_after = total_mass(self.actor, seat)
            delta = (mass_after - self._mass_before[seat]) / self.mass_scale
            r = float(np.clip(delta, -self.reward_clip, self.reward_clip))
            if self.food_shaping_coef and player.alive and delta <= 0.0:
                # Dense food-approach shaping: reward *net* progress toward the
                # nearest food. SYMMETRIC (approaching is +, retreating is −) so a
                # blob that just jitters in place nets ~0 — a one-sided max(0,·)
                # bonus is farmable by oscillating (it harvests only the down-half
                # of the distance noise) and yields a stuck, non-approaching policy.
                # SKIPPED on any step that ate (``delta > 0``): eating moves the
                # nearest-food target to the next, farther food, which a symmetric
                # term would wrongly punish — the Δmass reward already covers eating.
                d_after = nearest_food_dist(self.actor, seat)
                d_before = self._dist_before.get(seat, d_after)
                if d_before != float("inf") and d_after != float("inf"):
                    r += self.food_shaping_coef * (d_before - d_after)
            seat_done = 1.0 if (not player.alive or done) else 0.0
            new_stash[seat] = (r, seat_done, 1.0)
        self._stash = new_stash

        if done:
            self._game_counter += 1
            self.actor = _make_actor(
                game_seed(self.base_seed, self.stream_index, self._game_counter),
                self.seats,
                self.scripted_by_seat,
            )
            # keep _stash — it carries the finished game's last transitions (done=1)
            # to be reported on the next (fresh-game) round.


__all__ = ["GameRunner", "ObsFrame", "total_mass", "nearest_food_dist", "game_seed"]
