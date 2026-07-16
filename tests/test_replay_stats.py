"""Tourney step 1 — tools/replay_stats.py acceptance.

Three layers:

* a **synthetic replay** built event-by-event with hand-computed expected
  stats — exercises every counter (splits cmd vs effective, pellets vs
  events, deaths/respawn, capture ratios incl. the pre-2026.1.14 missing
  ``eaten_radius``, ranking ties, timeline) and every structural validator
  (interleave, partial block, unknown event, version quarantine);
* the checked-in engine replay ``tests/fixtures/game.json`` (v2026.1.13,
  22 557 events) — the parser must agree with the engine's own
  ``event_player_won`` and global event counts;
* the real tournament replay ``match_replays/match-37245-replay.json``
  (gitignored data, so skipped when absent) — must reproduce the figures
  frozen in notes/HANDOFF-tourney-diagnosis-session.md §2/§7.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from replay_stats import (  # noqa: E402
    COMPATIBLE_ENGINE_VERSIONS,
    MatchStats,
    ReplayFormatError,
    aggregate_teams,
    clone_candidates,
    main,
    parse_replay,
)

FIXTURE_GAME = REPO / "tests" / "fixtures" / "game.json"
REAL_REPLAY = REPO / "match_replays" / "match-37245-replay.json"


# --- synthetic replay ---------------------------------------------------------
#
# 2 players (pid 0 = team 10, pid 1 = team 20), 3 rounds. Storyline:
#   round 0: p0 splits (ineffective: mass 0.81 < SPLIT_MIN_MASS), both eat food
#            (p0 one event with TWO pellets, p1 one event with one pellet);
#   round 1: p0 (radius 2.0) eats p1's only blob (radius 1.0) -> p1 dies
#            (death round 1); p0's snapshot shows two blobs (it split-grew),
#            p1's snapshot is dead/empty;
#   round 2: only p0 moves (p1 dead => no move_player), p0 splits again —
#            now EFFECTIVE (prev snapshot has a blob with mass 4.0 >= 2.0).
# Final: p0 radius sqrt(5), p1 dead radius 0 -> ranks 1, 2; p0 wins.


def blob(bid: int, x: float, y: float, r: float) -> dict:
    return {"blob_id": bid, "pos": [x, y], "radius": r, "merge_cooldown": 0}


def snap(pid: int, alive: bool, blobs: list[dict]) -> dict:
    total = sum(b["radius"] ** 2 for b in blobs)
    return {
        "event_type": "event_player_moved",
        "player_id": pid,
        "pos": [1.0, 1.0],
        "radius": total**0.5,
        "alive": alive,
        "blobs": blobs,
    }


def move(pid: int, split: bool = False) -> dict:
    return {
        "event_type": "move_player",
        "player_id": pid,
        "direction": {"x": 1.0, "y": 0.0, "degrees": None},
        "split": split,
    }


def synthetic_events(engine_version: str = "2026.1.15",
                     with_eaten_radius: bool = True) -> list[dict]:
    started = {
        "event_type": "event_game_started",
        "turn_order": [1, 0],
        "arena_size": 60.0,
        "vision_size": 20.0,
        "turn_duration_seconds": 0.1,
        "max_rounds": 3,
        "engine_version": engine_version,
        "players": [
            {
                "player_id": 0,
                "team_id": 10,
                "pos": [1.0, 1.0],
                "radius": 0.9,
                "alive": True,
                "blobs": [blob(0, 1.0, 1.0, 0.9)],
            },
            {
                "player_id": 1,
                "team_id": 20,
                "pos": [5.0, 5.0],
                "radius": 0.9,
                "alive": True,
                "blobs": [blob(0, 5.0, 5.0, 0.9)],
            },
        ],
    }
    eaten = {
        "event_type": "event_player_eaten",
        "eater_player_id": 0,
        "eater_blob_id": 0,
        "eater_pos": [5.0, 5.0],
        "eaten_player_id": 1,
        "eaten_blob_id": 0,
        "eaten_pos": [5.0, 5.0],
        "eater_radius": 2.0,  # post-absorption
        "eaten_player_alive": False,
    }
    if with_eaten_radius:
        eaten["eaten_radius"] = 1.0
    return [
        started,
        # round 0
        move(1),
        move(0, split=True),  # ineffective: mass 0.81 < 2.0
        {"event_type": "event_food_eaten", "player_id": 0, "blob_id": 0,
         "food_ids": [7, 8], "new_radius": 0.92},
        {"event_type": "event_food_eaten", "player_id": 1, "blob_id": 0,
         "food_ids": [9], "new_radius": 0.91},
        snap(0, True, [blob(0, 1.0, 1.0, 0.92)]),
        snap(1, True, [blob(0, 5.0, 5.0, 0.91)]),
        {"event_type": "event_food_spawned", "foods": [{"food_id": 30, "pos": [2.0, 2.0]}]},
        # round 1
        move(1),
        move(0),
        eaten,
        snap(0, True, [blob(0, 5.0, 5.0, 2.0), blob(1, 6.0, 6.0, 0.0)]),
        snap(1, False, []),
        # round 2: p1 dead -> no move from it
        move(0, split=True),  # EFFECTIVE: prev snapshot blob mass 4.0 >= 2.0
        snap(0, True, [blob(0, 5.0, 5.0, 2.0), blob(2, 7.0, 7.0, 1.0)]),
        snap(1, False, []),
        {"event_type": "event_player_won", "player_id": 0},
    ]


@pytest.fixture()
def parsed() -> MatchStats:
    return parse_replay(synthetic_events(), source="synthetic", timeline_step=1)


def test_synthetic_counters(parsed: MatchStats) -> None:
    p0, p1 = parsed.players[0], parsed.players[1]

    assert parsed.rounds == 3
    assert parsed.winner_pid == 0
    assert parsed.warnings == []

    # splits: commands vs effective (round-0 split was below SPLIT_MIN_MASS)
    assert p0.splits_cmd == 2
    assert p0.splits_eff == 1
    assert p1.splits_cmd == 0

    # foods in pellets, not events
    assert p0.foods == 2 and p0.food_events == 1
    assert p1.foods == 1 and p1.food_events == 1

    # capture / death bookkeeping
    assert p0.captures == 1 and p1.blobs_lost == 1
    assert p1.deaths == 1 and p1.death_rounds == [1]
    assert p0.deaths == 0

    # ratios: post = 2.0/1.0; pre = sqrt(4-1)/1 = sqrt(3)
    assert p0.capture_ratios_post == [2.0]
    assert p0.capture_ratios_pre == pytest.approx([3**0.5])

    # moves: p1 issues none in round 2 (dead players are not queried)
    assert p0.moves == 3 and p1.moves == 2
    assert p0.alive_rounds == 3 and p1.alive_rounds == 1

    # masses: final = last snapshot, peak over all snapshots
    assert p0.final_mass == pytest.approx(5.0)
    assert p0.peak_mass == pytest.approx(5.0)
    assert p1.final_mass == 0.0
    assert p1.peak_mass == pytest.approx(0.91**2)
    assert p0.max_blobs == 2

    # ranking: engine rule (final radius desc, tie -> lower pid)
    assert p0.rank == 1 and p1.rank == 2


def test_synthetic_timeline_every_round(parsed: MatchStats) -> None:
    assert sorted(parsed.timeline) == [0, 1, 2]
    assert parsed.timeline[0][0] == pytest.approx(0.92**2)
    assert parsed.timeline[1][1] == 0.0
    assert parsed.timeline[2][0] == pytest.approx(5.0)


def test_final_round_always_in_timeline() -> None:
    m = parse_replay(synthetic_events(), source="s", timeline_step=100)
    # step 100 would only record round 0; the final round is forced in
    assert sorted(m.timeline) == [0, 2]
    assert m.timeline[2][0] == pytest.approx(5.0)


def test_missing_eaten_radius_degrades_gracefully() -> None:
    # engine < 2026.1.14 replays carry no eaten_radius
    m = parse_replay(synthetic_events(with_eaten_radius=False), source="s")
    p0 = m.players[0]
    assert p0.captures == 1
    assert p0.capture_ratios_post == [] and p0.capture_ratios_pre == []


def test_aggregation_and_clone_candidates(parsed: MatchStats) -> None:
    teams = aggregate_teams([parsed, parsed])  # same match twice
    t10, t20 = teams[10], teams[20]
    assert t10.matches == 2 and t10.wins == 2 and t10.mean_rank == 1.0
    assert t10.splits_cmd == 4 and t10.splits_eff == 2
    assert t20.deaths == 2 and t20.mean_rank == 2.0
    assert t10.moves == 6  # BC row volume: 3 moves x 2 matches

    cands = clone_candidates(teams, exclude_team=20, min_matches=2)
    assert [t.team_id for t in cands] == [10]
    assert clone_candidates(teams, exclude_team=None, min_matches=3) == []


# --- structural validation ------------------------------------------------


def test_rejects_interleaved_snapshot_block() -> None:
    events = synthetic_events()
    # inject a move inside round 0's snapshot block
    idx = next(i for i, e in enumerate(events)
               if e["event_type"] == "event_player_moved")
    events.insert(idx + 1, move(1))
    with pytest.raises(ReplayFormatError, match="interleaved"):
        parse_replay(events, source="s")


def test_rejects_trailing_partial_block() -> None:
    events = synthetic_events()
    events.append(snap(0, True, [blob(0, 1.0, 1.0, 1.0)]))
    with pytest.raises(ReplayFormatError, match="partial"):
        parse_replay(events, source="s")

    # a non-snapshot event arriving mid-block is interleaving
    events = synthetic_events()
    events.insert(len(events) - 1, snap(0, True, [blob(0, 1.0, 1.0, 1.0)]))
    with pytest.raises(ReplayFormatError, match="interleaved"):
        parse_replay(events, source="s")


def test_rejects_duplicate_snapshot_in_block() -> None:
    events = synthetic_events()
    idx = next(i for i, e in enumerate(events)
               if e["event_type"] == "event_player_moved")
    events[idx + 1] = snap(0, True, [])  # second p0 snapshot, p1 missing
    with pytest.raises(ReplayFormatError, match="covers"):
        parse_replay(events, source="s")


def test_rejects_unknown_event_and_bad_header() -> None:
    events = synthetic_events()
    events.insert(1, {"event_type": "event_totally_new"})
    with pytest.raises(ReplayFormatError, match="unknown event"):
        parse_replay(events, source="s")
    with pytest.raises(ReplayFormatError, match="event_game_started"):
        parse_replay(synthetic_events()[1:], source="s")
    with pytest.raises(ReplayFormatError, match="no event_player_won"):
        parse_replay(synthetic_events()[:-1], source="s")


def test_rank_vs_won_mismatch_is_warned() -> None:
    events = synthetic_events()
    events[-1] = {"event_type": "event_player_won", "player_id": 1}
    m = parse_replay(events, source="s")
    assert any("disagrees" in w for w in m.warnings)


def test_version_quarantine_via_cli(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    bad = tmp_path / "match-1-replay.json"
    bad.write_text(json.dumps(synthetic_events(engine_version="2099.1.1")))
    ok = tmp_path / "match-2-replay.json"
    ok.write_text(json.dumps(synthetic_events()))

    assert main([str(tmp_path), "--team", "10"]) == 0
    out = capsys.readouterr().out
    assert "1 parsed, 1 quarantined" in out
    assert "2099.1.1" in out

    # --allow-version rescues it
    assert main([str(tmp_path), "--team", "10",
                 "--allow-version", "2099.1.1"]) == 0
    assert "2 parsed, 0 quarantined" in capsys.readouterr().out

    # nothing valid -> exit 1
    assert main([str(bad), "--team", "10"]) == 1


def test_json_export_roundtrip(tmp_path: Path) -> None:
    replay = tmp_path / "match-9-replay.json"
    replay.write_text(json.dumps(synthetic_events()))
    out = tmp_path / "stats.json"
    assert main([str(replay), "--team", "20", "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["our_team"] == 20
    (m,) = data["matches"]
    assert m["winner_team"] == 10 and m["rounds"] == 3
    by_pid = {p["player_id"]: p for p in m["players"]}
    assert by_pid[0]["splits_eff"] == 1
    assert by_pid[1]["death_rounds"] == [1]
    assert m["timeline"]["2"]["0"] == pytest.approx(5.0)


# --- checked-in engine replay (v2026.1.13) ---------------------------------


def test_fixture_game_json_parses_and_matches_engine_verdict() -> None:
    with open(FIXTURE_GAME) as f:
        events = json.load(f)
    m = parse_replay(events, source=str(FIXTURE_GAME))

    assert m.engine_version == "2026.1.13"
    assert m.engine_version in COMPATIBLE_ENGINE_VERSIONS
    assert m.rounds == 1400
    assert m.warnings == []  # rank reconstruction agrees with the engine

    # global counts against the raw stream
    assert sum(p.moves for p in m.players.values()) == sum(
        1 for e in events if e["event_type"] == "move_player"
    )
    assert sum(p.captures for p in m.players.values()) == sum(
        1 for e in events if e["event_type"] == "event_player_eaten"
    )
    assert sum(p.foods for p in m.players.values()) == sum(
        len(e["food_ids"]) for e in events
        if e["event_type"] == "event_food_eaten"
    )
    assert sum(p.shatters for p in m.players.values()) == sum(
        1 for e in events if e["event_type"] == "event_virus_consumed"
    )
    # 2026.1.13: no eaten_radius anywhere -> no ratios
    assert all(not p.capture_ratios_post for p in m.players.values())
    assert m.players[m.winner_pid].rank == 1


# --- the real tournament replay (gitignored; skip when absent) ----------------


@pytest.mark.skipif(not REAL_REPLAY.exists(),
                    reason="tournament replays are local-only data")
def test_match_37245_reproduces_handoff_figures() -> None:
    """Frozen in HANDOFF-tourney-diagnosis-session.md §2/§7 (2026-07-17)."""
    with open(REAL_REPLAY) as f:
        events = json.load(f)
    m = parse_replay(events, source=str(REAL_REPLAY))

    assert m.engine_version == "2026.1.15"
    assert m.rounds == 1400
    assert m.warnings == []

    us = m.players[m.pid_of_team(53)]
    win = m.players[m.winner_pid]

    assert m.team_of(m.winner_pid) == 49
    assert us.rank == 2
    assert us.final_mass == pytest.approx(10.9, abs=0.05)
    assert us.splits_cmd == 0
    assert us.shatters == 3
    assert us.death_rounds == [743]

    assert win.final_mass == pytest.approx(28.4, abs=0.05)
    assert win.peak_mass == pytest.approx(45.9, abs=0.05)
    assert win.splits_cmd == 12
    assert win.captures == 26
    # handoff counted food_eaten EVENTS (279/389); pellets are the true count
    assert us.food_events == 279 and us.foods == 293
    assert win.food_events == 389 and win.foods == 479
    # "captures at radius ratios 1.5-3.5" (post-absorption) was the handoff's
    # eyeball of this distribution; the exact range is 1.54-4.42 with the
    # median in the split-hunt band.
    rs = sorted(win.capture_ratios_post)
    assert rs[0] > 1.5 and rs[-1] < 4.5
    assert 2.0 < rs[len(rs) // 2] < 3.0
