"""P1.1 acceptance: `bot/spec.py` is the single source of truth and every value
in it is faithfully derived from the vendored engine's `lib.config` — nothing is
hard-coded, and the derived quantities are correct.

If the engine pin changes a constant, these assertions (which compare `bot.spec`
against `lib.config` directly) are the tripwire.
"""

from __future__ import annotations

import math

from lib.config import arena as _arena
from lib.config import player as _player

from bot import spec


def test_arena_constants_come_straight_from_lib_config() -> None:
    assert spec.ARENA_SIZE == _arena.ARENA_SIZE
    assert spec.NUM_PLAYERS == _arena.NUM_PLAYERS
    assert spec.MAX_ROUNDS == _arena.MAX_ROUNDS
    assert spec.VISION_SIZE == _arena.VISION_SIZE
    assert spec.VISION_REFERENCE_SUM_OF_RADII == _arena.VISION_REFERENCE_SUM_OF_RADII
    assert spec.FOOD_COUNT == _arena.FOOD_COUNT
    assert spec.VIRUS_COUNT == _arena.VIRUS_COUNT
    assert spec.VIRUS_SIZE == _arena.VIRUS_SIZE
    assert spec.MAX_BLOB_COUNT == _arena.MAX_BLOB_COUNT
    assert spec.PLAYER_SPAWN_PADDING == _arena.PLAYER_SPAWN_PADDING
    assert spec.RESPAWN_DELAY_ROUNDS == _arena.RESPAWN_DELAY_ROUNDS
    assert spec.TURN_DURATION_SECONDS == _arena.TURN_DURATION_SECONDS


def test_player_constants_come_straight_from_lib_config() -> None:
    assert spec.STARTING_RADIUS == _player.STARTING_RADIUS
    assert spec.BASE_PLAYER_SPEED == _player.BASE_PLAYER_SPEED
    assert spec.PLAYER_SPEED_RADIUS_FACTOR == _player.PLAYER_SPEED_RADIUS_FACTOR
    assert spec.MIN_PLAYER_SPEED == _player.MIN_PLAYER_SPEED
    assert spec.EAT_SIZE_RATIO == _player.EAT_SIZE_RATIO
    assert spec.FOOD_RADIUS == _player.FOOD_RADIUS
    assert spec.SPLIT_MIN_MASS == _player.SPLIT_MIN_MASS
    assert spec.SPLIT_COOLDOWN_FRAMES == _player.SPLIT_COOLDOWN_FRAMES
    assert spec.SPLIT_EJECT_SPEED == _player.SPLIT_EJECT_SPEED
    assert spec.SPLIT_EJECT_DRAG == _player.SPLIT_EJECT_DRAG
    assert spec.MERGE_ATTRACTION_SPEED == _player.MERGE_ATTRACTION_SPEED
    assert spec.SAME_PLAYER_OVERLAP_EPSILON == _player.SAME_PLAYER_OVERLAP_EPSILON
    assert spec.MASS_DECAY_RATE == _player.MASS_DECAY_RATE


def test_derived_quantities_are_correct() -> None:
    assert spec.STARTING_MASS == _player.STARTING_RADIUS ** 2
    assert spec.FOOD_MASS == _player.FOOD_RADIUS ** 2
    # Engine v2026.1.11: virus consumption/gain use the virus's *mass* (radius**2),
    # not its radius (was VIRUS_SIZE * ratio = 1.8 and VIRUS_SIZE = 1.5 pre-v2026.1.11).
    assert spec.VIRUS_POP_MASS_THRESHOLD == _arena.VIRUS_SIZE ** 2 * _player.EAT_SIZE_RATIO
    assert spec.VIRUS_MASS_GAIN == _arena.VIRUS_SIZE ** 2
    assert spec.SPLIT_REACH == _player.SPLIT_EJECT_SPEED / (1.0 - _player.SPLIT_EJECT_DRAG)
    # sanity on the headline numbers the spec doc quotes
    assert math.isclose(spec.STARTING_MASS, 0.81)
    assert math.isclose(spec.VIRUS_POP_MASS_THRESHOLD, 2.7)
    assert math.isclose(spec.VIRUS_MASS_GAIN, 2.25)
    assert math.isclose(spec.SPLIT_REACH, 8.8889, rel_tol=1e-3)


def test_helpers_match_engine_formulas() -> None:
    from engine.state.state_mutator import StateMutator

    # movement_speed mirrors StateMutator._movement_speed for a range of radii
    # (bound-method call on an uninitialised instance — the method reads no state).
    dummy = StateMutator.__new__(StateMutator)
    for r in (0.9, 2.0, 5.0, 12.0, 50.0):
        assert math.isclose(spec.movement_speed(r), StateMutator._movement_speed(dummy, r))

    # vision_size_for mirrors GameState.get_player_vision_size's math exactly.
    for s in (0.0, 6.0, 12.0, 24.0, 100.0):
        expected = spec.VISION_SIZE if s <= 0 else max(s / 12.0, 1.0) ** 0.4 * spec.VISION_SIZE
        assert math.isclose(spec.vision_size_for(s), expected)

    # VIRUS_POP_MASS_THRESHOLD tracks the engine's _can_consume_virus boundary
    # (blob.mass > virus.radius**2 * EAT_SIZE_RATIO) — guards against re-drift.
    from engine.state.blob_state import BlobState

    vr = _arena.VIRUS_SIZE
    below = BlobState.__new__(BlobState)
    below.radius = math.sqrt(spec.VIRUS_POP_MASS_THRESHOLD) * (1 - 1e-9)
    above = BlobState.__new__(BlobState)
    above.radius = math.sqrt(spec.VIRUS_POP_MASS_THRESHOLD) * (1 + 1e-9)
    assert not dummy._can_consume_virus(below, vr)
    assert dummy._can_consume_virus(above, vr)
