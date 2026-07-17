"""Deploy-time defensive rails (Track 2 — a gated, shippable safety override).

The tournament's #1 mass leak for our policy is death: ~5.4 eliminations per
match to split-hunters (each resets us to spawn mass). This module is a
deterministic, torch-free override applied to the policy's chosen action right
before it is serialized — it vetoes the two clearly-suicidal move classes the
learned policy still walks into:

1. **Walking into a predator.** If the chosen heading points toward a nearby
   enemy that can eat us now (``can_eat_me``) or is within split-hunt reach
   (``split_threat`` high), steer to the opposite (flee) heading instead. And
   never *split* while such a predator is adjacent — a split halves us into
   easy prey.
2. **Popping ourselves on a virus.** If we are large enough to shatter
   (a visible virus flagged ``is_dangerous``) and the heading drives us into a
   nearby one, deflect around it.

It reads only the ``CanonicalObs`` threat features the featurizer already
computes (verified mechanics — ``can_eat_me`` / ``split_threat`` / virus
``is_dangerous``), so it needs no extra engine access and runs in microseconds.
Both the shipped deploy path (`bot.main`) and the gate harness apply the *same*
function, so what we measure is what we ship.

**This is opt-in and gated.** A rail that over-flees can starve the bot
(never approaching food near bigger blobs), so it ships only if it beats the
un-railed policy on the meta-clone field. Thresholds are module constants for
exactly that tuning.
"""

from __future__ import annotations

import math

from bot.adapters.act_v1 import HEADING_BINS, CanonicalAction, bin_to_vector
from bot.adapters.obs_v1 import CanonicalObs

__all__ = ["apply_rails", "RailConfig"]

_STEP = 2.0 * math.pi / HEADING_BINS

# enemy feature indices (obs_v1 ENEMY layout)
_E_DX, _E_DY, _E_CAN_EAT_ME, _E_SPLIT_THREAT = 0, 1, 6, 7
# virus feature indices (obs_v1 VIRUS layout)
_V_DX, _V_DY, _V_DIST, _V_DANGER = 0, 1, 2, 4


class RailConfig:
    """Tunable trigger thresholds (obs positions are normalized by half-vision)."""

    #: flee a predator whose split-hunt threat is at least this (0..1)
    split_threat_flee: float = 0.5
    #: only redirect when the heading points within this cosine of the threat
    #: (cos 60° = 0.5 → heading is "toward" the threat cluster)
    toward_cos: float = 0.5
    #: a dangerous virus nearer than this (normalized dist) can pop us
    virus_near: float = 0.45
    #: deflect from a virus only when the heading points this close to it
    virus_toward_cos: float = 0.70


def _vec_to_bin(x: float, y: float) -> int:
    return int(round(math.atan2(y, x) / _STEP)) % HEADING_BINS


def apply_rails(
    obs: CanonicalObs, action: CanonicalAction, cfg: RailConfig = RailConfig()
) -> CanonicalAction:
    """Return ``action`` unchanged, or a defensively-overridden version."""
    hx, hy = bin_to_vector(action.heading_bin)
    new_bin = action.heading_bin
    new_split = action.split

    # --- Rail 1: predators -------------------------------------------------
    tx = ty = 0.0          # resultant direction TOWARD threatening enemies
    predator_adjacent = False
    enemy, emask = obs.enemy, obs.enemy_mask
    for i in range(enemy.shape[0]):
        if emask[i] < 0.5:
            continue
        threat = enemy[i, _E_CAN_EAT_ME] > 0.5 or enemy[i, _E_SPLIT_THREAT] >= cfg.split_threat_flee
        if not threat:
            continue
        predator_adjacent = True
        dx, dy = float(enemy[i, _E_DX]), float(enemy[i, _E_DY])
        n = math.hypot(dx, dy)
        if n > 1e-9:
            tx += dx / n
            ty += dy / n

    if predator_adjacent:
        new_split = False  # never fragment next to something that can eat us
        tn = math.hypot(tx, ty)
        if tn > 1e-9 and (hx * tx + hy * ty) / tn > cfg.toward_cos:
            # heading drives into the threat cluster → flee the opposite way
            new_bin = _vec_to_bin(-tx, -ty)
            hx, hy = bin_to_vector(new_bin)

    # --- Rail 2: dangerous viruses ----------------------------------------
    virus, vmask = obs.virus, obs.virus_mask
    for i in range(virus.shape[0]):
        if vmask[i] < 0.5 or virus[i, _V_DANGER] < 0.5:
            continue
        if float(virus[i, _V_DIST]) > cfg.virus_near:
            continue
        vx, vy = float(virus[i, _V_DX]), float(virus[i, _V_DY])
        n = math.hypot(vx, vy)
        if n < 1e-9:
            continue
        if (hx * vx + hy * vy) / n > cfg.virus_toward_cos:
            # steer ~90° around the virus, on whichever side our heading leans
            # (a pure perpendicular never degenerates to zero, unlike projecting
            # out the toward-component when the heading is dead-on).
            px, py = -vy / n, vx / n
            if hx * px + hy * py < 0.0:
                px, py = -px, -py
            new_bin = _vec_to_bin(px, py)
            hx, hy = bin_to_vector(new_bin)

    if new_bin == action.heading_bin and new_split == action.split:
        return action
    return CanonicalAction(heading_bin=new_bin, split=new_split)
