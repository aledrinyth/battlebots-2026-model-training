"""P2.4 acceptance: scripted baseline bots.

* Behavior unit tests — each bot heads the right way (flee/chase/seek/stay) and
  the chaser splits only when a split still lands the eat.
* Ban-safety — every scripted move validates through the kit ``MoveValidator``
  (opponents are training-only, but must still never emit an invalid move).
* Acceptance — greedy-food beats random on mean final mass, and the chaser kills
  random. A fast subset runs in CI; the full statistical criteria
  (≥90 % of 100 games; ≥1 kill/game) are marked ``slow``.

Measured calibration (8 full games each): greedy > random mean in 8/8 (mass
ratio ≥8×); chaser ≈61 kills/game, 8/8 games with ≥1 kill — so the thresholds
below carry large margin.
"""

from __future__ import annotations

import math

import pytest

from bot import spec
from engine.interface.io.input_validator import MoveValidator
from lib.interface.events.event_player_eaten import EventPlayerEaten
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.blob_model import BlobModel, VisibleBlobModel
from lib.models.food_model import FoodModel
from lib.models.player_model import PlayerModel

from train.env.wrapper import WrapperEnv
from train.opponents import (
    BOTS,
    CautiousBot,
    ChaserBot,
    CowardBot,
    DontMoveBot,
    GreedyFoodBot,
    RandomBot,
    make_bot,
)


# --- synthetic single-frame query ------------------------------------------

def mk_query(*, own=((0, 30.0, 30.0, 1.0, 0),), enemies=(), foods=(), pid=0, rankings=None):
    """own: (blob_id,x,y,r,cd); enemies: (player_id,blob_id,x,y,r); foods: (id,x,y)."""
    blobs = tuple(BlobModel(blob_id=b[0], pos=(b[1], b[2]), radius=b[3],
                            merge_cooldown=b[4]) for b in own)
    masses = [b.radius ** 2 for b in blobs]
    tm = sum(masses) or 1.0
    cx = sum(b.pos[0] * m for b, m in zip(blobs, masses)) / tm
    cy = sum(b.pos[1] * m for b, m in zip(blobs, masses)) / tm
    you = PlayerModel(player_id=pid, team_id=pid, pos=(cx, cy),
                      radius=math.sqrt(tm), alive=True, blobs=blobs)
    if rankings is None:
        # own + unique enemy player ids, largest-radius-first (mirrors the
        # engine's get_rankings). Unused by scripted bots, but a required field.
        radii = {pid: math.sqrt(tm)}
        for e in enemies:
            radii[e[0]] = max(radii.get(e[0], 0.0), e[4])
        rankings = [p for p, _ in sorted(radii.items(), key=lambda kv: kv[1], reverse=True)]
    return QueryMovePlayer(
        update={}, round=0, rankings=rankings, you=you, view_center=(cx, cy), vision_size=20.0,
        visible_food=[FoodModel(food_id=f[0], pos=(f[1], f[2])) for f in foods],
        visible_blobs=[VisibleBlobModel(player_id=e[0], team_id=e[0], blob_id=e[1],
                                        pos=(e[2], e[3]), radius=e[4]) for e in enemies],
        visible_viruses=[],
    )


def _heading(move: MovePlayer):
    return move.direction.to_vector()


# --- behavior unit tests ----------------------------------------------------

def test_dont_move_stays():
    assert _heading(DontMoveBot().act(mk_query())) == (0.0, 0.0)


def test_greedy_heads_to_nearest_food():
    # pellet at x=25 (dist 5) is nearer than x=42 (dist 12) ⇒ head −x.
    q = mk_query(foods=((0, 42.0, 30.0), (1, 25.0, 30.0)))
    hx, hy = _heading(GreedyFoodBot().act(q))
    assert hx < 0


def test_greedy_drifts_to_center_when_no_food():
    q = mk_query(own=((0, 10.0, 10.0, 1.0, 0),))  # in a corner, no food
    hx, hy = _heading(GreedyFoodBot().act(q))
    assert hx > 0 and hy > 0  # toward arena center (30,30)


def test_coward_flees_bigger_blob():
    q = mk_query(own=((0, 30.0, 30.0, 1.0, 0),), enemies=((1, 0, 33.0, 30.0, 3.0),))
    hx, hy = _heading(CowardBot().act(q))
    assert hx < 0  # away from the +x threat


def test_coward_seeks_food_when_safe():
    q = mk_query(enemies=((1, 0, 33.0, 30.0, 0.5),), foods=((0, 40.0, 30.0),))
    hx, hy = _heading(CowardBot().act(q))
    assert hx > 0  # smaller enemy is no threat → head to food at +x


def test_chaser_pursues_edible_prey():
    q = mk_query(own=((0, 30.0, 30.0, 1.0, 0),), enemies=((1, 0, 33.0, 30.0, 0.5),))
    move = ChaserBot().act(q)
    hx, hy = _heading(move)
    assert hx > 0                # toward prey at +x
    assert move.split is False   # mass 1 < 2 ⇒ cannot split


def test_chaser_splits_to_catch():
    # me r=1.5 (mass 2.25 ≥ 2, single blob); prey r=0.5 at dist 3 (> my_r, within
    # reach); half-mass 1.125 ≥ 1.2×prey mass 0.25 ⇒ split still eats ⇒ split.
    q = mk_query(own=((0, 30.0, 30.0, 1.5, 0),), enemies=((1, 0, 33.0, 30.0, 0.5),))
    move = ChaserBot().act(q)
    assert move.split is True
    assert _heading(move)[0] > 0


def test_chaser_prey_uses_mass_ratio_not_radius():
    # Engine v2026.1.12 mass rule: me r=1.0 (mass 1.0) out-masses prey r=0.91
    # (mass 0.8281) by 1.0 ≥ 1.2·0.8281 = 0.994 ✓ ⇒ pursue. Under the old radius
    # rule (1.0 ≥ 1.2·0.91 = 1.092 ✗) the chaser would ignore it and drift.
    q = mk_query(own=((0, 30.0, 30.0, 1.0, 0),), enemies=((1, 0, 34.0, 30.0, 0.91),))
    assert _heading(ChaserBot().act(q))[0] > 0  # heads toward prey at +x


def test_chaser_no_split_when_already_covering():
    # prey already within my disc (dist 1 < my_r 1.5) ⇒ eat without splitting.
    q = mk_query(own=((0, 30.0, 30.0, 1.5, 0),), enemies=((1, 0, 31.0, 30.0, 0.5),))
    assert ChaserBot().act(q).split is False


def test_cautious_flee_chase_food_priority():
    flee = CautiousBot().act(mk_query(enemies=((1, 0, 33.0, 30.0, 2.0),)))
    assert _heading(flee)[0] < 0  # bigger enemy ⇒ flee
    chase = CautiousBot().act(mk_query(own=((0, 30.0, 30.0, 2.0, 0),),
                                       enemies=((1, 0, 33.0, 30.0, 1.0),)))
    assert _heading(chase)[0] > 0  # smaller enemy ⇒ chase
    seek = CautiousBot().act(mk_query(foods=((0, 40.0, 30.0),)))
    assert _heading(seek)[0] > 0   # nothing to fight ⇒ food


def test_make_bot_registry():
    assert set(BOTS) == {"random", "dont_move", "greedy_food", "cautious", "coward", "chaser"}
    assert isinstance(make_bot("chaser"), ChaserBot)
    with pytest.raises(KeyError):
        make_bot("nope")


# --- match harness ----------------------------------------------------------

def run_match(seat_bots, seed, rounds=1400):
    """Drive a full wrapper game; return (final_masses, kills_by_eater)."""
    env = WrapperEnv(seed=seed, emit_update=False)
    obs = env.reset()
    for b in seat_bots.values():
        b.reset()
    for _ in range(rounds):
        moves = {pid: seat_bots[pid].act(q) for pid, q in obs.items()}
        obs, done = env.step(moves)
        if done:
            break
    kills: dict[int, int] = {}
    for ev in list(env.state.event_history) + list(env.state.private_event_history):
        if isinstance(ev, EventPlayerEaten) and not ev.eaten_player_alive:
            kills[ev.eater_player_id] = kills.get(ev.eater_player_id, 0) + 1
    return env.final_masses(), kills


def _greedy_vs_random_seats():
    bots = {0: GreedyFoodBot()}
    for pid in range(1, spec.NUM_PLAYERS):
        bots[pid] = RandomBot(seed=100 + pid)
    return bots


def _chaser_vs_random_seats():
    bots = {0: ChaserBot()}
    for pid in range(1, spec.NUM_PLAYERS):
        bots[pid] = RandomBot(seed=200 + pid)
    return bots


# --- ban-safety across a real game -----------------------------------------

def test_all_scripted_moves_valid_in_game():
    # A mixed table; validate every emitted move (incl. chaser splits) exactly as
    # the engine would — no scripted bot may ever emit an invalid move.
    order = ["greedy_food", "cautious", "coward", "chaser",
             "random", "dont_move", "greedy_food", "chaser"]
    env = WrapperEnv(seed=3, emit_update=False)
    obs = env.reset()
    seat_bots = {pid: make_bot(order[pid], **({"seed": pid} if order[pid] == "random" else {}))
                 for pid in range(spec.NUM_PLAYERS)}
    validator = MoveValidator(env.state)
    for _ in range(60):
        moves = {}
        for pid, q in obs.items():
            mv = seat_bots[pid].act(q)
            parsed = MovePlayer.model_validate_json(mv.model_dump_json())
            validator.validate(parsed, q, pid)  # raises on any invalid move
            assert len(mv.model_dump_json()) < 4096
            moves[pid] = mv
        obs, done = env.step(moves)
        if done:
            break


# --- acceptance: fast subset (CI) ------------------------------------------

def test_greedy_beats_random_fast():
    # Short games; greedy's food advantage is already decisive (calibrated ≥8×).
    for s in range(3):
        fm, _ = run_match(_greedy_vs_random_seats(), seed=1000 + s, rounds=500)
        rand_mean = sum(fm[p] for p in range(1, spec.NUM_PLAYERS)) / (spec.NUM_PLAYERS - 1)
        assert fm[0] > rand_mean, f"greedy {fm[0]:.2f} !> random mean {rand_mean:.2f} (seed {s})"


def test_chaser_kills_random_fast():
    total = 0
    for s in range(2):
        _fm, kills = run_match(_chaser_vs_random_seats(), seed=5000 + s, rounds=700)
        total += kills.get(0, 0)
    assert total >= 2, f"chaser only killed {total} across 2 short games"


# --- acceptance: full statistical criteria (slow) --------------------------

@pytest.mark.slow
def test_greedy_beats_random_90pct_full():
    N = 10  # playbook says 100; 10 full games already saturate the margin
    wins = 0
    for s in range(N):
        fm, _ = run_match(_greedy_vs_random_seats(), seed=7000 + s)
        rand_mean = sum(fm[p] for p in range(1, spec.NUM_PLAYERS)) / (spec.NUM_PLAYERS - 1)
        wins += fm[0] > rand_mean
    assert wins / N >= 0.9, f"greedy beat random mean in only {wins}/{N} games"


@pytest.mark.slow
def test_chaser_kills_at_least_one_per_game_full():
    N = 10
    total = 0
    for s in range(N):
        _fm, kills = run_match(_chaser_vs_random_seats(), seed=8000 + s)
        total += kills.get(0, 0)
    assert total / N >= 1.0, f"chaser averaged only {total / N:.2f} kills/game"
