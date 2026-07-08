"""P1.5 acceptance: the NumPy-vectorized vision override is **bit-identical** to
the pristine vendored engine.

P1.5 optimizes only the observation/vision path (`get_visible_food /
get_visible_blobs / get_visible_viruses`), not `commit_round` physics — so P1.3
parity (`tests/test_wrapper_parity.py`) already guarantees the state trajectory is
untouched. What *this* test guarantees is the other half: that the faster vision
returns the exact same visible-entity lists (same membership, same order, same
model fields) as the engine it replaces, on real game states including
post-death/respawn frames. Together they certify the override is a pure speedup.

Run the full-length comparison with `--runslow`; the CI form checks a 150-round
prefix of two games.
"""

from __future__ import annotations

import math
import random

import pytest

from engine.state.game_state import GameState
from train.env import fast_vision
from train.env.wrapper import WrapperEnv, make_move


# --- comparison helpers -----------------------------------------------------

def _food_key(models):
    return [(m.food_id, m.pos) for m in models]


def _virus_key(models):
    return [(m.virus_id, m.pos, m.radius) for m in models]


def _blob_key(models):
    return [
        (m.player_id, m.team_id, m.blob_id, m.pos, m.radius, m.merge_cooldown)
        for m in models
    ]


def _assert_vision_matches(
    state: GameState, base_food, base_vir, base_blob, base_public_blob
) -> None:
    """For every alive player, the installed (fast) getters must equal the pristine
    getters bit-for-bit on this exact state — including v2026.1.14's public
    (re-indexed) blobs, which is what the live query actually sends."""
    for pid, player in state.players.items():
        if not player.alive:
            # dead ⇒ both return [] (guarded identically)
            assert state.get_visible_food(pid) == []
            assert base_food(state, pid) == []
            assert state.get_public_visible_blobs(pid) == []
            assert base_public_blob(state, pid) == []
            continue

        assert _food_key(state.get_visible_food(pid)) == _food_key(base_food(state, pid)), (
            f"visible_food diverged for player {pid} at round {state.round}"
        )
        assert _virus_key(state.get_visible_viruses(pid)) == _virus_key(base_vir(state, pid)), (
            f"visible_viruses diverged for player {pid} at round {state.round}"
        )
        assert _blob_key(state.get_visible_blobs(pid)) == _blob_key(base_blob(state, pid)), (
            f"visible_blobs diverged for player {pid} at round {state.round}"
        )
        # v2026.1.14: the public re-indexed blobs (what PlayerConnection now sends)
        # must be bit-identical to the pristine public method AND re-indexed per
        # target player (blob_id = 0,1,2… among that player's visible blobs).
        fast_public = state.get_public_visible_blobs(pid)
        assert _blob_key(fast_public) == _blob_key(base_public_blob(state, pid)), (
            f"public visible_blobs diverged for player {pid} at round {state.round}"
        )
        per_player: dict = {}
        for m in fast_public:
            expected = per_player.get(m.player_id, 0)
            assert m.blob_id == expected, (
                f"public blob_id not a per-player ordinal for player {pid}"
            )
            per_player[m.player_id] = expected + 1


def _random_moves(obs, rng: random.Random):
    moves = {}
    for pid in obs:
        ang = rng.uniform(0.0, 2.0 * math.pi)
        moves[pid] = make_move(pid, math.cos(ang), math.sin(ang), split=rng.random() < 0.05)
    return moves


def _compare_game(seed: int, rounds: int) -> None:
    # Capture the pristine getters before the wrapper installs the fast override.
    fast_vision.uninstall()
    base_food = GameState.get_visible_food
    base_vir = GameState.get_visible_viruses
    base_blob = GameState.get_visible_blobs
    base_public_blob = GameState.get_public_visible_blobs

    env = WrapperEnv(seed=seed, emit_update=True)  # reset() installs fast_vision
    obs = env.reset()
    assert fast_vision.is_installed(), "wrapper.reset() should install the fast vision"

    rng = random.Random(seed + 100)
    for _ in range(rounds):
        _assert_vision_matches(env.state, base_food, base_vir, base_blob, base_public_blob)
        obs, done = env.step(_random_moves(obs, rng))
        if done:
            break


# --- tests ------------------------------------------------------------------

def test_boundary_inclusive_is_bit_exact() -> None:
    """A target sitting *exactly* on the vision box edge (|Δ| == half_vision) must
    be reported visible by both paths — the one place float equality could flip a
    `<=` between Python and NumPy. It does not: both evaluate the same float64
    comparison."""
    fast_vision.uninstall()
    base_food = GameState.get_visible_food

    env = WrapperEnv(seed=3, emit_update=False)
    env.reset()
    state = env.state
    pid = 0
    cx, cy = state.get_player_view_center(pid)
    half = state.get_player_vision_size(pid) / 2

    # Place one food exactly on the right edge, one just outside, one just inside.
    foods = state.map.foods
    foods[0].pos = (cx + half, cy)          # exactly on edge → visible
    foods[1].pos = (cx + half + 1e-9, cy)   # just outside → not visible
    foods[2].pos = (cx, cy - half)          # exactly on edge (y) → visible

    fast_vision.install()
    fast_ids = [m.food_id for m in state.get_visible_food(pid)]
    base_ids = [m.food_id for m in base_food(state, pid)]
    assert fast_ids == base_ids
    assert foods[0].food_id in fast_ids
    assert foods[2].food_id in fast_ids
    assert foods[1].food_id not in fast_ids


def test_circle_visibility_from_outside_box() -> None:
    """Engine v2026.1.7–.9: viruses and player blobs use the *circle* test
    (`is_circle_in_vision`) — visible when the circle overlaps the box, i.e. from
    up to its radius *outside* the vision window. Food keeps the point test. This
    pins that split so the fast override can't regress to point-in-box, and checks
    it stays bit-identical to the pristine engine."""
    fast_vision.uninstall()
    base_vir = GameState.get_visible_viruses

    env = WrapperEnv(seed=5, emit_update=False)
    env.reset()
    state = env.state
    pid = 0
    cx, cy = state.get_player_view_center(pid)
    half = state.get_player_vision_size(pid) / 2

    viruses = state.map.viruses
    r = viruses[0].radius
    # center outside the box by 0.5·r (< r) ⇒ circle overlaps ⇒ visible
    viruses[0].pos = (cx + half + 0.5 * r, cy)
    # center outside by 1.5·r (> r) ⇒ circle clears the box ⇒ not visible
    viruses[1].pos = (cx + half + 1.5 * r, cy)
    # keep the rest well outside so they don't clutter the assertion
    for v in viruses[2:]:
        v.pos = (cx + 10 * half, cy + 10 * half)

    fast_vision.install()
    fast_ids = [m.virus_id for m in state.get_visible_viruses(pid)]
    base_ids = [m.virus_id for m in base_vir(state, pid)]
    assert fast_ids == base_ids, "fast circle-visibility diverged from the engine"
    assert viruses[0].virus_id in fast_ids, "circle overlapping the box must be visible"
    assert viruses[1].virus_id not in fast_ids, "circle clear of the box must be hidden"


def test_empty_vision_and_dead_match() -> None:
    """Degenerate frames (no entities in box, dead player) agree."""
    fast_vision.uninstall()
    base_food = GameState.get_visible_food
    base_vir = GameState.get_visible_viruses
    base_blob = GameState.get_visible_blobs
    base_public_blob = GameState.get_public_visible_blobs

    env = WrapperEnv(seed=7, emit_update=False)
    env.reset()
    state = env.state

    # Force a player dead and confirm both paths return [] for it.
    victim = next(iter(state.players.values()))
    victim.blobs.clear()
    fast_vision.install()
    _assert_vision_matches(state, base_food, base_vir, base_blob, base_public_blob)


def test_vision_equivalence_prefix() -> None:
    """CI form: two seeded games, 150-round prefix, fast vision == pristine."""
    _compare_game(seed=0, rounds=150)
    _compare_game(seed=1, rounds=150)


@pytest.mark.slow
def test_vision_equivalence_full_games() -> None:
    """Opt-in: full 1400-round games (seed 0 has real deaths/respawns) stay
    bit-identical end to end."""
    for seed in range(3):
        _compare_game(seed=seed, rounds=1400)
