"""R0 reconstruction + conversion acceptance (impl doc §12).

Synthetic-stream tests pin the walker's structural rules (block
segmentation, death/respawn transitions, food/virus accounting, the
v2026.1.14 public-ordinal re-indexing) against hand-computed expectations;
the checked-in engine replay ``tests/fixtures/game.json`` then reproduces
the §7.1 validation block wholesale (the impl-doc "done-when", ported off
the deleted match-13472 onto a checked-in replay of the same format).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from train.replay.convert import check_report, convert_replay
from train.replay.reconstruct import ReplayError, Walker, build_query

REPO = Path(__file__).resolve().parent.parent
FIXTURE_GAME = REPO / "tests" / "fixtures" / "game.json"


# --- synthetic replay ---------------------------------------------------------
#
# Two players on the real 60×60 arena. Storyline:
#   round 0: both move; p0 eats food 1 (the *middle* of three foods — its
#            removal must shift the later public ordinals);
#   round 1: p0 eats p1 (death);
#   round 2: p0 alone (p1 dead ⇒ no move event);
#   round 3: p1 respawns in the snapshot;
#   round 4: both move again — p1's query round jumps 1→4 ⇒ is_reset.
# p0 finishes bigger ⇒ ranks (p0=1, p1=2); event_player_won says p0.


def blob(bid, x, y, r, cd=0):
    return {"blob_id": bid, "pos": [x, y], "radius": r, "merge_cooldown": cd}


def snap(pid, alive, blobs):
    radius = sum(b["radius"] ** 2 for b in blobs) ** 0.5
    return {"event_type": "event_player_moved", "player_id": pid,
            "pos": [0.0, 0.0], "radius": radius, "alive": alive, "blobs": blobs}


def move(pid, x=1.0, y=0.0, split=False):
    return {"event_type": "move_player", "player_id": pid,
            "direction": {"x": x, "y": y, "degrees": None}, "split": split}


def synthetic_events():
    return [
        {
            "event_type": "event_game_started",
            "turn_order": [1, 0],
            "arena_size": 60.0,
            "vision_size": 20.0,
            "turn_duration_seconds": 0.1,
            "max_rounds": 1400,
            "engine_version": "2026.1.15",
            "players": [
                {"player_id": 0, "team_id": 10, "pos": [10.0, 10.0],
                 "radius": 1.0, "alive": True, "blobs": [blob(0, 10.0, 10.0, 1.0)]},
                {"player_id": 1, "team_id": 20, "pos": [50.0, 50.0],
                 "radius": 0.9, "alive": True, "blobs": [blob(0, 50.0, 50.0, 0.9)]},
            ],
        },
        {"event_type": "event_food_spawned", "foods": [
            {"food_id": 0, "pos": [12.0, 10.0]},
            {"food_id": 1, "pos": [11.0, 11.0]},
            {"food_id": 2, "pos": [14.0, 10.0]},
        ]},
        {"event_type": "event_virus_spawned", "viruses": [
            {"virus_id": 0, "pos": [30.0, 30.0], "radius": 1.5},
        ]},
        # round 0
        move(1, 0.0, 1.0),
        move(0, 1.0, 0.0),
        {"event_type": "event_food_eaten", "player_id": 0, "blob_id": 0,
         "food_ids": [1], "new_radius": 1.011},
        snap(0, True, [blob(0, 11.0, 10.0, 1.011)]),
        snap(1, True, [blob(0, 50.0, 50.5, 0.9)]),
        # round 1: p0 teleports next to p1 in the fiction and eats it
        move(1, 1.0, 0.0),
        move(0, 0.0, 1.0),
        {"event_type": "event_player_eaten",
         "eater_player_id": 0, "eater_blob_id": 0, "eater_pos": [50.0, 50.0],
         "eaten_player_id": 1, "eaten_blob_id": 0, "eaten_pos": [50.0, 50.5],
         "eater_radius": 1.35, "eaten_radius": 0.9, "eaten_player_alive": False},
        snap(0, True, [blob(0, 50.0, 50.0, 1.35)]),
        snap(1, False, []),
        # round 2: p1 dead — only p0 moves
        move(0, -1.0, 0.0),
        snap(0, True, [blob(0, 49.0, 50.0, 1.35)]),
        snap(1, False, []),
        # round 3: p1 respawns during the round (no move from it yet)
        move(0, -1.0, 0.0),
        snap(0, True, [blob(0, 48.0, 50.0, 1.35)]),
        snap(1, True, [blob(1, 20.0, 20.0, 0.9)]),
        # round 4: both move; p1's round gap 1 -> 4 must flag is_reset
        move(1, 0.0, 0.0),          # a "stay" — no heading label
        move(0, 1.0, 0.0, split=True),
        snap(0, True, [blob(0, 49.0, 50.0, 1.35)]),
        snap(1, True, [blob(1, 20.0, 20.0, 0.9)]),
        {"event_type": "event_player_won", "player_id": 0},
    ]


def test_walker_segmentation_and_terminal_state():
    w = Walker(synthetic_events(), source="synthetic")
    decisions = [(rnd, mv["player_id"]) for rnd, mv in w.decisions()]
    assert decisions == [
        (0, 1), (0, 0), (1, 1), (1, 0), (2, 0), (3, 0), (4, 1), (4, 0),
    ]
    assert w.rounds == 5
    assert w.winner_pid == 0
    assert w.final_ranks() == {0: 1, 1: 2}
    assert w.eaten_total == 1 and w.eaten_in_vision == 1
    assert w.eaten_in_reach == 1


def test_eating_untracked_food_is_fatal():
    """The strict stream-consistency signal: an off-by-one segmentation
    would eat pellets that are not on the reconstructed map."""
    events = synthetic_events()
    idx = next(i for i, e in enumerate(events)
               if e["event_type"] == "event_food_eaten")
    events[idx]["food_ids"] = [77]  # never spawned
    w = Walker(events, source="s")
    with pytest.raises(ReplayError, match="not on the reconstructed map"):
        list(w.decisions())


def test_public_food_ordinals_shift_after_eat():
    """v2026.1.14 semantics: public ids are per-frame ordinals in map order,
    so eating the middle food renumbers what follows."""
    w = Walker(synthetic_events(), source="synthetic")
    it = w.decisions()

    next(it)  # round 0, p1 (far corner — sees nothing)
    _, mv = next(it)  # round 0, p0
    q0 = build_query(w.gs, 0, 0)
    # all three foods visible from (10,10); ordinals 0,1,2 in spawn order
    assert [f.food_id for f in q0.visible_food] == [0, 1, 2]
    assert [tuple(f.pos) for f in q0.visible_food] == [
        (12.0, 10.0), (11.0, 11.0), (14.0, 10.0)]

    next(it)  # round 1, p1
    _, mv = next(it)  # round 1, p0 — food 1 has been eaten
    q1 = build_query(w.gs, 0, 1)
    # p0 now sits at (50,50); no food is near — check the *map* order instead
    assert [f.food_id for f in w.gs.map.foods] == [0, 2]
    assert [tuple(f.pos) for f in w.gs.map.foods] == [
        (12.0, 10.0), (14.0, 10.0)]
    # and the public view of a player at the old spot would renumber them —
    # the ordinals come from enumerate() over visible-in-map-order.
    assert q1.round == 1


def test_convert_labels_resets_and_stays():
    result = convert_replay(synthetic_events(), source="synthetic")
    a = result.arrays
    assert result.n_rows == 8
    assert a["obs"].shape == (8, 736)

    # per-row (seat, round, is_reset) in stream order
    rows = list(zip(a["seat"].tolist(), a["round"].tolist(),
                    a["is_reset"].tolist()))
    assert rows == [
        (1, 0, 1), (0, 0, 1), (1, 1, 0), (0, 1, 0),
        (0, 2, 0), (0, 3, 0), (1, 4, 1), (0, 4, 0),
    ]
    # p1's final move was a stay: split of the heading loss
    assert a["has_heading"].tolist() == [1, 1, 1, 1, 1, 1, 0, 1]
    assert a["heading_bin"][6] == -1
    # p0's final move carried split=True
    assert a["split"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    # heading bins: p1 moved +y (bin 6), p0 +x (bin 0)
    assert a["heading_bin"][0] == 6 and a["heading_bin"][1] == 0

    assert result.seats[0].final_rank == 1
    assert result.seats[1].final_rank == 2
    assert result.seats[0].n_rows == 5 and result.seats[1].n_rows == 3
    assert check_report(result) == []


def test_convert_is_deterministic():
    r1 = convert_replay(synthetic_events(), source="a")
    r2 = convert_replay(synthetic_events(), source="b")
    for k in r1.arrays:
        assert np.array_equal(r1.arrays[k], r2.arrays[k]), k


# --- structural rejection ----------------------------------------------------


def _events_with(mutator):
    events = synthetic_events()
    mutator(events)
    return events


def _consume(events):
    w = Walker(events, source="synthetic")
    list(w.decisions())
    return w


def test_rejects_malformed_streams():
    with pytest.raises(ReplayError, match="expected event_game_started"):
        Walker(synthetic_events()[1:], source="s")

    with pytest.raises(ReplayError, match="unknown event type"):
        _consume(_events_with(lambda e: e.insert(3, {"event_type": "event_new"})))

    with pytest.raises(ReplayError, match="aborted game"):
        _consume(_events_with(
            lambda e: e.insert(3, {"event_type": "event_player_banned"})))

    with pytest.raises(ReplayError, match="no event_player_won"):
        _consume(synthetic_events()[:-1])

    # a move from a player the state says is dead: inject a p1 move right
    # after round 2's p0 move (p1 died in round 1)
    def dead_move(e):
        idx = [i for i, ev in enumerate(e)
               if ev["event_type"] == "move_player"][4]
        e.insert(idx + 1, move(1))
    with pytest.raises(ReplayError, match="dead player"):
        _consume(_events_with(dead_move))

    # snapshot bookkeeping: drop one snapshot -> caught either as an
    # interleaved block (next round's moves arrive mid-block) or as a
    # snapshot-count mismatch, depending on where the drop lands
    def drop_snap(e):
        idx = next(i for i, ev in enumerate(e)
                   if ev["event_type"] == "event_player_moved")
        del e[idx]
    with pytest.raises(ReplayError, match="snapshot"):
        _consume(_events_with(drop_snap))

    # and dropping a whole trailing block leaves a partial-block remainder
    def drop_last_snaps(e):
        idxs = [i for i, ev in enumerate(e)
                if ev["event_type"] == "event_player_moved"]
        del e[idxs[-1]]
    with pytest.raises(ReplayError, match="partial snapshot block"):
        _consume(_events_with(drop_last_snaps))


def test_rank_vs_won_mismatch_is_fatal():
    def wrong_winner(e):
        e[-1] = {"event_type": "event_player_won", "player_id": 1}
    w = Walker(_events_with(wrong_winner), source="s")
    list(w.decisions())
    with pytest.raises(ReplayError, match="disagrees"):
        w.final_ranks()


# --- the §7.1 acceptance block on a real engine replay -----------------------


def test_fixture_game_reproduces_impl_doc_validation():
    """The impl-doc §14.1 "done-when", on the checked-in v2026.1.13 replay:
    every alive move becomes exactly one validated row; resets decompose into
    first-frames + respawns; every eaten pellet lay inside the eater's
    reconstructed pre-move window."""
    events = json.loads(FIXTURE_GAME.read_text())
    result = convert_replay(events, source=str(FIXTURE_GAME))
    r = result.report

    assert r["engine_version"] == "2026.1.13"
    assert r["rounds"] == 1400
    n_moves = sum(1 for e in events if e["event_type"] == "move_player")
    assert result.n_rows == r["move_events"] == n_moves == 10390
    assert result.arrays["obs"].shape == (10390, 736)
    assert r["heading_bins_covered"] == 24
    assert r["reset_rows"] == 35 and r["first_frames"] == 8 and r["respawns"] == 27
    assert r["eaten_total"] == r["eaten_in_vision"] == 539  # 100 % plain-window
    assert r["eaten_in_reach"] == 539
    assert check_report(result) == []
