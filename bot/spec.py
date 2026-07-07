# Progression annotation (reconstructed): Engine constants are the shared contract: changing them independently breaks training/deployment parity.
"""Canonical game constants — the single source of truth for the whole project.

**Boundary discipline (adaptation-pipeline requirement, P1.1):** no physics or
arena constant is ever hard-coded anywhere else in `bot/`, `train/`, or the
tooling. Everything is imported *here* straight from the vendored engine's
``lib.config`` and re-exported, plus a handful of *derived* quantities computed
from those primitives (never typed in by hand). When the engine pin changes, a
single ``lib.config`` edit flows through this module to every consumer, and the
spec-diff detector (P11.3) diffs exactly these inputs.

Only ``lib.config`` is imported (not ``engine.*``) so this module stays inside
the match-legal surface: the shipped bot uses ``lib`` models already, but never
needs the grader-side ``engine`` package. Timing / protocol constants
(2 s, 8 s, 4096 chars) live in ``engine.config.io_config`` and are documented in
``notes/protocol-audit.md`` + ``notes/mechanics-spec.md``; they are the
compliance harness's concern (P4), not the bot's, and are deliberately kept out
of this import path.
"""

from __future__ import annotations

# --- Arena / match shape (verbatim from lib.config.arena) -------------------
from lib.config.arena import (
    ARENA_SIZE,
    FOOD_COUNT,
    MAX_BLOB_COUNT,
    MAX_ROUNDS,
    NUM_PLAYERS,
    PLAYER_SPAWN_PADDING,
    RESPAWN_DELAY_ROUNDS,
    TURN_DURATION_SECONDS,
    VIRUS_COUNT,
    VIRUS_SIZE,
    VISION_REFERENCE_SUM_OF_RADII,
    VISION_SIZE,
)

# --- Player / physics constants (verbatim from lib.config.player) -----------
from lib.config.player import (
    BASE_PLAYER_SPEED,
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    MASS_DECAY_RATE,
    MERGE_ATTRACTION_SPEED,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
    SAME_PLAYER_OVERLAP_EPSILON,
    SPLIT_COOLDOWN_FRAMES,
    SPLIT_EJECT_DRAG,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
    STARTING_RADIUS,
)

# --- Derived quantities (computed from the primitives above, never typed) ---

#: Mass of a fresh / respawned blob and the floor mass decay can never cross
#: (``_apply_mass_decay`` clamps at ``STARTING_RADIUS**2``).
STARTING_MASS = STARTING_RADIUS * STARTING_RADIUS

#: Mass a single food pellet contributes: a blob eating food grows as
#: ``r' = sqrt(r**2 + FOOD_RADIUS**2)``, i.e. it gains ``FOOD_RADIUS**2`` mass.
FOOD_MASS = FOOD_RADIUS * FOOD_RADIUS

#: A blob may consume a virus only when ``blob.mass > virus.radius**2 * EAT_SIZE_RATIO``.
#: Engine v2026.1.11 corrected ``_can_consume_virus`` to compare *mass* against the
#: virus's *mass* (``virus.radius**2``), not its radius — so with VIRUS_SIZE=1.5 the
#: pop threshold is ``2.25 * 1.2 = 2.7`` (was the pre-v2026.1.11 "mass > 1.8").
VIRUS_POP_MASS_THRESHOLD = VIRUS_SIZE * VIRUS_SIZE * EAT_SIZE_RATIO

#: Mass gained from consuming a virus: ``total_mass = blob.mass + virus.radius**2``.
#: Engine v2026.1.11 corrected this to the virus's *mass* (``radius**2`` = 2.25),
#: not ``VIRUS_SIZE`` (1.5). The blob still shatters into ≤16 pieces on consumption.
VIRUS_MASS_GAIN = VIRUS_SIZE * VIRUS_SIZE

#: Total distance a freshly-launched split child coasts on eject momentum before
#: the 1e-4 velocity cutoff, i.e. the geometric-series sum
#: ``SPLIT_EJECT_SPEED * (1 + d + d**2 + ...) = SPLIT_EJECT_SPEED / (1 - SPLIT_EJECT_DRAG)``.
#: This ~8.9-unit reach (on a 60-unit arena) is the true danger radius of a
#: bigger enemy that can still split, not its body radius (see mechanics-spec).
SPLIT_REACH = SPLIT_EJECT_SPEED / (1.0 - SPLIT_EJECT_DRAG)

#: Half-side of the (square) vision window at the minimum vision size.
VISION_HALF_MIN = VISION_SIZE / 2.0

#: Per-round loop cadence in the live engine (seconds) — the wall-clock the
#: pipe runner sleeps to; irrelevant to the in-process wrapper (which never
#: sleeps) but part of the spec.
ROUND_SECONDS = TURN_DURATION_SECONDS


def movement_speed(radius: float) -> float:
    """Per-tick movement speed of a blob of the given radius.

    Mirrors ``StateMutator._movement_speed`` exactly:
    ``max(MIN_PLAYER_SPEED, BASE_PLAYER_SPEED / (1 + radius * PLAYER_SPEED_RADIUS_FACTOR))``.
    """
    return max(
        MIN_PLAYER_SPEED,
        BASE_PLAYER_SPEED / (1.0 + radius * PLAYER_SPEED_RADIUS_FACTOR),
    )


def vision_size_for(sum_of_radii: float) -> float:
    """Vision-window side length for a player whose blobs' radii sum to
    ``sum_of_radii``.

    Mirrors ``GameState.get_player_vision_size``:
    ``max(sum_of_radii / VISION_REFERENCE_SUM_OF_RADII, 1.0) ** 0.4 * VISION_SIZE``.
    """
    if sum_of_radii <= 0:
        return VISION_SIZE
    scale = max(sum_of_radii / VISION_REFERENCE_SUM_OF_RADII, 1.0) ** 0.4
    return scale * VISION_SIZE


__all__ = [
    # arena
    "ARENA_SIZE",
    "FOOD_COUNT",
    "MAX_BLOB_COUNT",
    "MAX_ROUNDS",
    "NUM_PLAYERS",
    "PLAYER_SPAWN_PADDING",
    "RESPAWN_DELAY_ROUNDS",
    "TURN_DURATION_SECONDS",
    "VIRUS_COUNT",
    "VIRUS_SIZE",
    "VISION_REFERENCE_SUM_OF_RADII",
    "VISION_SIZE",
    # player / physics
    "BASE_PLAYER_SPEED",
    "EAT_SIZE_RATIO",
    "FOOD_RADIUS",
    "MASS_DECAY_RATE",
    "MERGE_ATTRACTION_SPEED",
    "MIN_PLAYER_SPEED",
    "PLAYER_SPEED_RADIUS_FACTOR",
    "SAME_PLAYER_OVERLAP_EPSILON",
    "SPLIT_COOLDOWN_FRAMES",
    "SPLIT_EJECT_DRAG",
    "SPLIT_EJECT_SPEED",
    "SPLIT_MIN_MASS",
    "STARTING_RADIUS",
    # derived
    "STARTING_MASS",
    "FOOD_MASS",
    "VIRUS_POP_MASS_THRESHOLD",
    "VIRUS_MASS_GAIN",
    "SPLIT_REACH",
    "VISION_HALF_MIN",
    "ROUND_SECONDS",
    # helpers
    "movement_speed",
    "vision_size_for",
]
