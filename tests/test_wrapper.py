"""P1.2 acceptance: the in-process wrapper runs full games headless, holds every
state invariant each round, produces observations matching `QueryMovePlayer`
semantics exactly (dead ⇒ no observation), is deterministically seeded, and
conserves mass through player-eating.

The short invariant test runs in CI; the full 1400-round run is marked `slow`
(the literal acceptance criterion — run with `--runslow` / `RUN_SLOW=1`).
"""

from __future__ import annotations

import math
import random

import pytest

from bot import spec
from lib.interface.queries.query_move import QueryMovePlayer

from train.env.wrapper import WrapperEnv, make_move


# --- helpers ----------------------------------------------------------------

def _random_moves(obs, rng: random.Random):
    """A random legal move for every player in `obs` (its own separate RNG so it
    never perturbs the engine's global-`random` stream)."""
    moves = {}
    for pid in obs:
        ang = rng.uniform(0.0, 2.0 * math.pi)
        moves[pid] = make_move(pid, math.cos(ang), math.sin(ang), split=rng.random() < 0.03)
    return moves


def _assert_invariants(env: WrapperEnv) -> None:
    state = env.state
    size = state.map.size

    total_alive_blobs = 0
    for player in state.players.values():
        assert len(player.blobs) <= spec.MAX_BLOB_COUNT, "blob count exceeded 16"
        for blob in player.blobs.values():
            total_alive_blobs += 1
            assert math.isfinite(blob.x) and math.isfinite(blob.y), "non-finite position"
            assert math.isfinite(blob.radius) and blob.radius > 0.0, "non-positive/NaN radius"
            # clamp keeps centers within [0, size]
            assert -1e-6 <= blob.x <= size + 1e-6, f"x out of arena: {blob.x}"
            assert -1e-6 <= blob.y <= size + 1e-6, f"y out of arena: {blob.y}"
            # decayed mass never crosses the starting-mass floor (minus float slack)
            assert blob.mass >= spec.STARTING_MASS - 1e-6, "mass fell below the decay floor"
        assert player.alive == bool(player.blobs)

    # per-round regen restores the larder exactly
    assert len(state.map.foods) == spec.FOOD_COUNT, "food not topped to 160"
    assert len(state.map.viruses) == spec.VIRUS_COUNT, "viruses not topped to 6"
    assert total_alive_blobs > 0


def _run(seed: int, rounds: int, emit_update: bool, check=None, policy_seed: int = 0):
    env = WrapperEnv(seed=seed, emit_update=emit_update)
    obs = env.reset()
    if check is not None:
        check(env)
    rng = random.Random(policy_seed)
    n = 0
    done = False
    while not done and n < rounds:
        obs, done = env.step(_random_moves(obs, rng))
        n += 1
        if check is not None and not done:
            check(env)
    return env, n, done


# --- invariants -------------------------------------------------------------

def test_short_game_invariants_ci() -> None:
    """Invariants hold every round over a short game (CI-fast)."""
    env, n, done = _run(seed=1, rounds=180, emit_update=False, check=_assert_invariants)
    assert n == 180 and not done


@pytest.mark.slow
def test_full_1400_round_game_invariants() -> None:
    """P1.2 literal acceptance: a full 1400-round, 8-player game with random
    policies runs headless and every invariant holds every round."""
    env, n, done = _run(seed=7, rounds=spec.MAX_ROUNDS, emit_update=False, check=_assert_invariants)
    assert done, "game did not end"
    assert n == spec.MAX_ROUNDS, f"expected {spec.MAX_ROUNDS} rounds, ran {n}"
    # terminal signals are well-formed
    assert sorted(env.rankings()) == list(range(spec.NUM_PLAYERS))
    assert sorted(env.final_masses()) == list(range(spec.NUM_PLAYERS))
    with pytest.raises(RuntimeError):
        env.step({})  # no stepping past game end


# --- QueryMovePlayer semantics ---------------------------------------------

def test_observations_match_query_move_player_semantics() -> None:
    env = WrapperEnv(seed=3, emit_update=True)
    obs = env.reset()
    gs = env.state

    assert set(obs) == set(p for p in gs.players if gs.players[p].alive)
    for pid, q in obs.items():
        assert isinstance(q, QueryMovePlayer)
        assert q.round == gs.round
        assert q.you.player_id == pid
        # own blobs come from `you`, never from visible_blobs (enemies only)
        assert len(q.you.blobs) == len(gs.players[pid].blobs)
        assert all(vb.player_id != pid for vb in q.visible_blobs)
        # every observation field equals the engine's own accessor output.
        # v2026.1.14: the live query sends the PUBLIC (re-indexed) views, so the
        # wrapper must too — compare against get_public_visible_* accordingly.
        assert q.view_center == gs.get_player_view_center(pid)
        assert q.vision_size == gs.get_player_vision_size(pid)
        assert [f.food_id for f in q.visible_food] == [
            f.food_id for f in gs.get_public_visible_food(pid)
        ]
        assert [(b.player_id, b.blob_id) for b in q.visible_blobs] == [
            (b.player_id, b.blob_id) for b in gs.get_public_visible_blobs(pid)
        ]
        assert [v.virus_id for v in q.visible_viruses] == [
            v.virus_id for v in gs.get_public_visible_viruses(pid)
        ]
        # public ids are per-frame ordinals: food/virus 0..N-1 in visible order
        assert [f.food_id for f in q.visible_food] == list(range(len(q.visible_food)))
        assert [v.virus_id for v in q.visible_viruses] == list(range(len(q.visible_viruses)))
        # update stream is present and keyed by a sequential public index
        assert isinstance(q.update, dict)
        assert list(q.update.keys()) == list(range(len(q.update)))


def test_emit_update_flag_controls_stream() -> None:
    on = WrapperEnv(seed=3, emit_update=True).reset()
    off = WrapperEnv(seed=3, emit_update=False).reset()
    assert any(len(q.update) > 0 for q in on.values()), "update stream should be populated"
    assert all(len(q.update) == 0 for q in off.values()), "fast-path update must be empty"


# --- dead => no observation, reappears on respawn --------------------------

def test_dead_player_gets_no_observation_until_respawn() -> None:
    """Force a kill and assert the eaten player vanishes from observations, then
    reappears exactly at its respawn round."""
    env = WrapperEnv(seed=11, emit_update=False)
    obs = env.reset()

    # Hand-place a lopsided duel: player 0 huge over player 1, all others frozen.
    state = env.state
    for pid, p in state.players.items():
        p.blobs.clear()
        p._next_blob_id = 0
    from engine.state.blob_state import BlobState  # engine-side, test-only

    def put(pid, x, y, r):
        p = state.players[pid]
        p.blobs[0] = BlobState(blob_id=0, x=x, y=y, radius=r)
        p._next_blob_id = 1

    put(0, 30.0, 30.0, 4.0)   # eater
    put(1, 30.5, 30.0, 1.0)   # victim, well inside player 0's disc, ratio 4 >= 1.2
    for pid in range(2, spec.NUM_PLAYERS):
        put(pid, 5.0 + pid, 5.0, 0.9)

    obs = env._observations()
    assert 1 in obs, "victim should be observable before the kill"

    # Player 0 stays put and eats player 1 this round.
    moves = {pid: make_move(pid, 0.0, 0.0) for pid in obs}
    obs, done = env.step(moves)

    assert 1 not in obs, "eaten player must receive no observation"
    victim = env.state.players[1]
    assert not victim.alive
    assert victim.respawn_at_round is not None and victim.respawn_at_round > env.state.round

    # Step forward: player 1 is absent from every observation while dead, and
    # reappears once the respawn round's commit runs (~RESPAWN_DELAY_ROUNDS later).
    respawned_at = None
    for _ in range(spec.RESPAWN_DELAY_ROUNDS + 5):
        if env.state.players[1].alive:
            respawned_at = env.state.round
            assert 1 in obs, "respawned player must be observable again"
            break
        assert 1 not in obs, "victim must stay unobserved while dead"
        obs, done = env.step({pid: make_move(pid, 0.0, 0.0) for pid in obs})
        if done:
            break
    assert respawned_at is not None, "victim never respawned"


# --- determinism ------------------------------------------------------------

def _trajectory(seed: int, rounds: int):
    env = WrapperEnv(seed=seed, emit_update=False)
    obs = env.reset()
    rng = random.Random(1234)
    for _ in range(rounds):
        obs, done = env.step(_random_moves(obs, rng))
        if done:
            break
    return [
        (pid, b.blob_id, b.x, b.y, b.radius)
        for pid, p in env.state.players.items()
        for b in p.blobs.values()
    ]


def test_deterministic_seeding() -> None:
    a = _trajectory(seed=5, rounds=60)
    b = _trajectory(seed=5, rounds=60)
    c = _trajectory(seed=6, rounds=60)
    assert a == b, "same seed + same moves must give an identical trajectory"
    assert a != c, "different seed should diverge"


# --- mass conservation through eating ---------------------------------------

def test_player_eating_conserves_mass() -> None:
    """Wrap `_resolve_player_eating` and assert total blob mass is unchanged
    across the call every round of a game — eating only redistributes mass."""
    env = WrapperEnv(seed=21, emit_update=False)
    obs = env.reset()

    original = env.mutator._resolve_player_eating
    violations: list[float] = []

    def total_mass() -> float:
        return sum(
            b.mass for p in env.state.players.values() for b in p.blobs.values()
        )

    def wrapped() -> None:
        before = total_mass()
        original()
        after = total_mass()
        if abs(after - before) > 1e-9 * max(1.0, before):
            violations.append(after - before)

    env.mutator._resolve_player_eating = wrapped  # type: ignore[method-assign]

    rng = random.Random(2)
    done = False
    n = 0
    while not done and n < 300:
        obs, done = env.step(_random_moves(obs, rng))
        n += 1
    assert not violations, f"player-eating changed total mass: {violations[:5]}"
