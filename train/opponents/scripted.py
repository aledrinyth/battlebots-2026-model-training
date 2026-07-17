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

import functools
import importlib.util
import math
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from bot import spec
from bot.heuristic import HeuristicController, HeuristicParams
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


class PotentialFieldBot(ScriptedBot):
    """Training-free two-phase controller (``bot.heuristic.HeuristicController``):
    deterministic greedy FEED while small/early, then latches to the
    potential-field HUNT (food/prey attraction, threat/virus/wall repulsion,
    conservative split-to-kill) once there is prey to eat. Every weight is a knob
    (:class:`~bot.heuristic.HeuristicParams`); carries no learned weights, so it
    ships as a tiny file. Stateful (the FEED→HUNT latch), so one instance per
    seat — the harness builds a fresh bot per game and calls :meth:`reset`."""

    name = "potential_field"

    def __init__(self, params: HeuristicParams | None = None) -> None:
        self._controller = HeuristicController(params)

    def reset(self) -> None:
        self._controller.reset()

    def act(self, query: QueryMovePlayer) -> MovePlayer:
        dx, dy, split = self._controller.act(query)
        return _move(query, dx, dy, split)


#: Cache: version tag -> HeuristicController class loaded from its dist file.
_DIST_CONTROLLER_CACHE: dict = {}


def _load_dist_controller(version: str):
    """Load ``HeuristicController`` from a *deployed* ``dist/potential_field_<v>.py``.

    Each shipped heuristic is a distinct code version (v7 added the death-zone
    veto + virus-pop gate over v6; v8 the gravity-well FEED, etc.), so pinning a
    training opponent to the exact submitted file — rather than re-deriving it —
    guarantees the league trains against the real bot. The dist file is a valid
    module defining ``HeuristicController``; we load it lazily (so importing this
    module never needs ``dist/``) and cache per version. The module must be
    registered in ``sys.modules`` *before* exec so its ``@dataclass`` params
    resolve their ``__module__``.
    """
    cls = _DIST_CONTROLLER_CACHE.get(version)
    if cls is None:
        path = _REPO_ROOT / "dist" / f"potential_field_{version}.py"
        if not path.exists():
            raise FileNotFoundError(
                f"deployed heuristic not found: {path} "
                f"(rsync dist/potential_field_{version}.py to run this opponent)"
            )
        mod_name = f"_pf_dist_{version}"
        spec_ = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec_)
        sys.modules[mod_name] = mod
        spec_.loader.exec_module(mod)
        cls = mod.HeuristicController
        _DIST_CONTROLLER_CACHE[version] = cls
    return cls


class _DistHeuristicBot(ScriptedBot):
    """A pinned, deployed potential-field version as a league opponent.

    Subclasses set ``version``; the controller is loaded from the matching
    dist file. Stateful (FEED→HUNT latch), so one instance per seat and the
    harness calls :meth:`reset` per game — same contract as
    :class:`PotentialFieldBot`."""

    version = ""

    def __init__(self) -> None:
        self._controller = _load_dist_controller(self.version)()

    def reset(self) -> None:
        self._controller.reset()

    def act(self, query: QueryMovePlayer) -> MovePlayer:
        dx, dy, split = self._controller.act(query)
        return _move(query, dx, dy, split)


class PotentialFieldV6Bot(_DistHeuristicBot):
    name = "potential_field_v6"
    version = "v6"


class PotentialFieldV7Bot(_DistHeuristicBot):
    name = "potential_field_v7"
    version = "v7"


class PotentialFieldV8Bot(_DistHeuristicBot):
    name = "potential_field_v8"
    version = "v8"


#: Name → zero-arg factory, for curriculum / league / ladder configs by string.
BOTS = {
    "random": RandomBot,
    "dont_move": DontMoveBot,
    "greedy_food": GreedyFoodBot,
    "cautious": CautiousBot,
    "coward": CowardBot,
    "chaser": ChaserBot,
    "potential_field": PotentialFieldBot,
    "potential_field_v6": PotentialFieldV6Bot,
    "potential_field_v7": PotentialFieldV7Bot,
    "potential_field_v8": PotentialFieldV8Bot,
}


#: Prefix that turns an opponent *name* into a frozen policy checkpoint. The
#: rest of the string is the exported actor-weights ``.npz`` (see
#: ``train/export.py``), resolved against the repo root when relative. This is
#: the single seam that lets a trained clone ride ``evaluate``/``promote`` as a
#: named opponent: both resolve every opponent through :func:`make_bot`, and
#: the name is a plain string, so it survives the spawn-worker boundary intact
#: (no shared state, no torch in the worker — ``NumpyPolicy`` is numpy-only).
FROZEN_PREFIX = "frozen:"
#: Like ``frozen:`` but applies the Track-2 deploy rails (``bot.rails``) — the
#: exact shipped override — so railed-vs-plain gates paired on the meta field.
RAILED_PREFIX = "railed:"

_REPO_ROOT = Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=64)
def _load_frozen_policy(path: str):
    """Load (and per-process cache) a frozen ``NumpyPolicy`` by resolved path.

    Mirrors ``train.harness.league_seats._frozen_bot``: the policy is
    immutable (``NumpyPolicy.act`` is pure — recurrent hidden lives in each
    :class:`~train.opponents.frozen.FrozenPolicyBot`), so one load can back
    every seat that draws this clone within a worker. Imported lazily to keep
    this module torch-free and dodge the frozen↔scripted import cycle.
    """
    from bot.inference import NumpyPolicy

    return NumpyPolicy.load(path)


def _resolve_frozen_path(spec_path: str) -> str:
    from pathlib import Path

    p = Path(spec_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"frozen opponent weights not found: {p} "
            f"(export a checkpoint with `python -m train.export`)"
        )
    return str(p)


def make_bot(name: str, **kwargs) -> ScriptedBot:
    """Instantiate an opponent by name.

    A plain name resolves through the scripted :data:`BOTS` registry. A
    ``frozen:<path>`` name loads that exported ``.npz`` as a
    :class:`~train.opponents.frozen.FrozenPolicyBot` — the mechanism behind
    R1 meta-clone opponents (impl doc §13.2 Option A).
    """
    if name.startswith(FROZEN_PREFIX):
        from train.opponents.frozen import FrozenPolicyBot

        path = _resolve_frozen_path(name[len(FROZEN_PREFIX):])
        return FrozenPolicyBot(path, policy=_load_frozen_policy(path))
    if name.startswith(RAILED_PREFIX):
        from train.opponents.frozen import RailedPolicyBot

        path = _resolve_frozen_path(name[len(RAILED_PREFIX):])
        return RailedPolicyBot(path, policy=_load_frozen_policy(path))
    try:
        cls = BOTS[name]
    except KeyError:
        raise KeyError(f"unknown scripted bot {name!r}; known: {sorted(BOTS)}") from None
    return cls(**kwargs)


__all__ = [
    "ScriptedBot",
    "PotentialFieldBot",
    "RandomBot",
    "DontMoveBot",
    "GreedyFoodBot",
    "CautiousBot",
    "CowardBot",
    "ChaserBot",
    "BOTS",
    "make_bot",
]
