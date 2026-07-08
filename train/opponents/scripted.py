"""Scripted baseline bots (P2.4).

All bots share the :class:`ScriptedBot` interface and act on the wrapper's
``QueryMovePlayer`` observations. They emit vector-heading ``MovePlayer`` moves;
the engine normalizes the heading, and a zero heading ``(0, 0)`` is a legal
"stay" (verified — ``DirectionModel`` accepts it, mechanics-spec §3). None of
them split except :class:`ChaserBot`, which splits only when a split still lands
the eat (so it never fragments itself into easy prey).

Roster (playbook P2.4):

* :class:`RandomBot`    — uniform random heading (the baseline everything beats).
* :class:`DontMoveBot`  — always stays (wrapper twin of the kit's ``dont_move``).
* :class:`GreedyFoodBot`— head to the nearest food, else drift to arena center.
* :class:`CautiousBot`  — flee bigger, else chase smaller, else food (wrapper twin
  of the kit's ``cautious`` example bot — same 1.1× thresholds).
* :class:`CowardBot`    — flee *any* bigger blob (inverse-distance weighted), else
  seek food.
* :class:`ChaserBot`    — pursue the nearest blob it can actually eat, splitting to
  catch when the split still eats.

Radii use the whole-player radius (``you.radius`` = √Σmass), matching the kit's
example bots; blobs are single early on, so this equals the acting blob's radius
in the common case.

**Own position = ``query.you.pos``, not ``query.view_center``.** As of engine
v2026.1.10, ``get_player_view_center`` clamps the viewport center to
``[half_vision, arena − half_vision]`` so the vision window never spills off the
map; it therefore diverges from the player's true centroid near the walls (by up
to ``half_vision``, which *grows* with the player). Navigating from that clamped
center pins a wall-hugging bot to a phantom target and starves it. ``you.pos`` is
the mass-weighted centroid (``player.x, player.y``) — exactly what
``view_center`` returned pre-v2026.1.10 — so all heading math uses it.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from bot import spec
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

_ARENA_CENTER = spec.ARENA_SIZE / 2.0


def _move(query: QueryMovePlayer, dx: float, dy: float, split: bool = False) -> MovePlayer:
    """Build a validated vector-heading move for the querying player.

    ``(0, 0)`` is passed through unchanged — a legal stay. The engine normalizes
    non-zero headings, so magnitude is irrelevant; only direction matters.
    """
    return MovePlayer(
        player_id=query.you.player_id,
        direction=DirectionModel(x=float(dx), y=float(dy)),
        split=bool(split),
    )


def _nearest(center: Tuple[float, float], items: Sequence, key_ok=None):
    """Nearest item (by squared distance to ``center``) whose ``key_ok(item)`` is
    truthy, as ``(dx, dy, dist2, item)``; ``None`` if none qualify."""
    cx, cy = center
    best = None
    best_d2 = math.inf
    for it in items:
        if key_ok is not None and not key_ok(it):
            continue
        dx = it.pos[0] - cx
        dy = it.pos[1] - cy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = (dx, dy, d2, it)
    return best


def _toward_center_or_stay(query: QueryMovePlayer) -> MovePlayer:
    """Drift toward arena center to find food; stay if already essentially there."""
    cx, cy = query.you.pos
    dx, dy = _ARENA_CENTER - cx, _ARENA_CENTER - cy
    if dx * dx + dy * dy < 1e-6:
        return _move(query, 0.0, 0.0)
    return _move(query, dx, dy)


class ScriptedBot:
    """Base class: stateless by default. Subclasses override :meth:`act`."""

    name = "scripted"

    def reset(self) -> None:  # noqa: D401 - hook for stateful bots
        """Reset per-episode state (stateless bots: nothing to do)."""

    def act(self, query: QueryMovePlayer) -> MovePlayer:  # pragma: no cover - abstract
        raise NotImplementedError


class RandomBot(ScriptedBot):
    """Uniform random heading each turn.

    Uses its **own** ``random.Random`` so it never perturbs the engine's global
    ``random`` stream (which drives spawns / turn order) — keeping games seeded by
    the wrapper reproducible regardless of how many random bots play.
    """

    name = "random"

    def __init__(self, seed: int = 0, split_prob: float = 0.0) -> None:
        self._seed = seed
        self._split_prob = split_prob
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def act(self, query: QueryMovePlayer) -> MovePlayer:
        ang = self._rng.uniform(0.0, 2.0 * math.pi)
        split = self._split_prob > 0.0 and self._rng.random() < self._split_prob
        return _move(query, math.cos(ang), math.sin(ang), split=split)


class DontMoveBot(ScriptedBot):
    """Always stays put — wrapper twin of the kit's ``dont_move`` example."""

    name = "dont_move"

    def act(self, query: QueryMovePlayer) -> MovePlayer:
        return _move(query, 0.0, 0.0)


class GreedyFoodBot(ScriptedBot):
    """Head straight for the nearest visible food; drift to center if none seen."""

    name = "greedy_food"

    def act(self, query: QueryMovePlayer) -> MovePlayer:
        near = _nearest(query.you.pos, query.visible_food)
        if near is None:
            return _toward_center_or_stay(query)
        dx, dy, _d2, _f = near
        return _move(query, dx, dy)


class CautiousBot(ScriptedBot):
    """Flee bigger, else chase smaller, else seek food — the kit's ``cautious``
    logic reimplemented against wrapper observations (same 1.1× thresholds)."""

    name = "cautious"
    THREAT_RATIO = 1.1
    PREY_RATIO = 1.1

    def act(self, query: QueryMovePlayer) -> MovePlayer:
        center = query.you.pos
        my_r = query.you.radius

        threat = _nearest(center, query.visible_blobs,
                          key_ok=lambda b: b.radius > my_r * self.THREAT_RATIO)
        if threat is not None:
            dx, dy, _d2, _b = threat
            return _move(query, -dx, -dy)  # flee

        prey = _nearest(center, query.visible_blobs,
                        key_ok=lambda b: my_r > b.radius * self.PREY_RATIO)
        if prey is not None:
            dx, dy, _d2, _b = prey
            return _move(query, dx, dy)

        food = _nearest(center, query.visible_food)
        if food is not None:
            dx, dy, _d2, _f = food
            return _move(query, dx, dy)
        return _move(query, 0.0, 0.0)


class CowardBot(ScriptedBot):
    """Flee *any* bigger blob, weighting nearer threats more (inverse distance);
    seek food when safe."""

    name = "coward"
    THREAT_RATIO = 1.0  # anything strictly bigger is a threat

    def act(self, query: QueryMovePlayer) -> MovePlayer:
        cx, cy = query.you.pos
        my_r = query.you.radius

        fx = fy = 0.0
        any_threat = False
        for b in query.visible_blobs:
            if b.radius <= my_r * self.THREAT_RATIO:
                continue
            any_threat = True
            dx = cx - b.pos[0]
            dy = cy - b.pos[1]
            d = math.hypot(dx, dy)
            if d < 1e-9:
                # Overlapping a bigger blob: bolt toward arena center as a tiebreak.
                fx += _ARENA_CENTER - cx
                fy += _ARENA_CENTER - cy
                continue
            w = 1.0 / d  # nearer threats dominate the resultant
            fx += (dx / d) * w
            fy += (dy / d) * w

        if any_threat and (fx * fx + fy * fy) > 1e-12:
            return _move(query, fx, fy)

        food = _nearest(query.you.pos, query.visible_food)
        if food is not None:
            dx, dy, _d2, _f = food
            return _move(query, dx, dy)
        return _toward_center_or_stay(query)


class ChaserBot(ScriptedBot):
    """Pursue the nearest blob it can eat; split to catch when the split still eats.

    Eat rule (verified §6c): eater **mass** ≥ ``EAT_SIZE_RATIO`` × target mass and
    the target center lies within the eater's disc. (Engine v2026.1.12 made this a
    *mass* ratio; mass = radius², so a mass ratio of 1.2 is a radius ratio √1.2 ≈
    1.095.) The chaser targets any blob it already out-masses by the ratio, and
    splits only when the target sits inside its **split reach** *and* a half-size
    child (mass ``m/2``) would still out-mass the target by the ratio — so a split
    always converts to a kill rather than scattering the chaser into prey. Falls
    back to food to grow when nothing is edible.
    """

    name = "chaser"

    def act(self, query: QueryMovePlayer) -> MovePlayer:
        center = query.you.pos
        my_r = query.you.radius
        my_mass = my_r * my_r

        prey = _nearest(
            center, query.visible_blobs,
            key_ok=lambda b: my_mass >= spec.EAT_SIZE_RATIO * b.radius * b.radius,
        )
        if prey is not None:
            dx, dy, d2, b = prey
            dist = math.sqrt(d2)
            # Split to catch: I can split (mass ≥ 2), the target is within reach,
            # and a half (mass my_mass/2) still out-masses it by the eat ratio.
            can_split = my_mass >= spec.SPLIT_MIN_MASS
            if (
                can_split
                and len(query.you.blobs) == 1
                and dist <= my_r + spec.SPLIT_REACH
                and dist > my_r  # not already covering it (would eat without split)
                and (my_mass * 0.5) >= spec.EAT_SIZE_RATIO * b.radius * b.radius
            ):
                return _move(query, dx, dy, split=True)
            return _move(query, dx, dy)

        food = _nearest(center, query.visible_food)
        if food is not None:
            dx, dy, _d2, _f = food
            return _move(query, dx, dy)
        return _toward_center_or_stay(query)


#: Name → zero-arg factory, for curriculum / league / ladder configs by string.
BOTS = {
    "random": RandomBot,
    "dont_move": DontMoveBot,
    "greedy_food": GreedyFoodBot,
    "cautious": CautiousBot,
    "coward": CowardBot,
    "chaser": ChaserBot,
}


def make_bot(name: str, **kwargs) -> ScriptedBot:
    """Instantiate a scripted bot by name (see :data:`BOTS`)."""
    try:
        cls = BOTS[name]
    except KeyError:
        raise KeyError(f"unknown scripted bot {name!r}; known: {sorted(BOTS)}") from None
    return cls(**kwargs)


__all__ = [
    "ScriptedBot",
    "RandomBot",
    "DontMoveBot",
    "GreedyFoodBot",
    "CautiousBot",
    "CowardBot",
    "ChaserBot",
    "BOTS",
    "make_bot",
]
