"""P4.1 acceptance: the compliance harness flags what the engine would ban.

Three subprocess tests run short, round-capped games through the REAL pipe
runner (engine in-process, bots as subprocesses — the bench_env §(b) rig):

* a bot that sleeps 3 s on one turn is **banned** by the engine's own SIGALRM
  and flagged (single-move breach);
* a bot that sleeps 4 ms every turn survives a capped game but is flagged by
  the worst-case **projection** (cumulative breach at full length);
* a clean bot passes with no flags and sane stats.

Threshold logic is additionally unit-tested on synthetic captures (no
subprocesses), so the flag semantics stay covered even where the pipe rig
can't run. The full-length "< 6 s for the real bot" clause is slow-marked
(P4.4 full re-runs it before any submission).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("engine")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import compliance  # noqa: E402  (path set above)
from compliance import PlayerCapture, TurnRecord, analyze  # noqa: E402

BOTS = "tests/fixtures/bots"
# Bots need a beat to import the kit before the engine's 3 s open-pipe alarm
# starts; the sleeper tests don't measure init, so trade honesty for CI calm.
START_DELAY = 1.0


# --- subprocess acceptance (real pipe runner) --------------------------------

def test_single_move_breach_is_flagged() -> None:
    report = compliance.run_compliance(
        [f"1:{BOTS}/sleeper_single.py", f"7:{BOTS}/dont_move.py"],
        rounds=10, seed=0, start_delay=START_DELAY,
    )
    assert not report["ok"]
    game = report["games"][0]
    # Ground truth: the ENGINE banned seat 0 (sleep 3 s > 2 s SIGALRM).
    assert game["engine_result"]["result_type"] == "PLAYER_BANNED"
    assert game["engine_result"]["player"] == 0
    assert "BANNED" in game["players"][0]["flags"]
    # The innocent seats must not be flagged.
    for pid in range(1, 8):
        assert game["players"][pid]["flags"] == []


def test_cumulative_drip_is_flagged_by_projection() -> None:
    report = compliance.run_compliance(
        [f"1:{BOTS}/sleeper_cumulative.py", f"7:{BOTS}/dont_move.py"],
        rounds=40, seed=0, start_delay=START_DELAY,
    )
    assert not report["ok"]
    game = report["games"][0]
    # 40 capped rounds don't reach 8 s — the game itself succeeds ...
    assert game["engine_result"]["result_type"] == "SUCCESS"
    # ... but 1400 × (4 ms + engine tax) projects past the 6 s margin line.
    p0 = game["players"][0]
    assert "CUM_PROJECTED" in p0["flags"]
    assert p0["projected_full_game_s"] > game["limits"]["usable_cumulative_s"]
    for pid in range(1, 8):
        assert game["players"][pid]["flags"] == []


def test_clean_bot_passes_with_sane_stats() -> None:
    rounds = 30
    report = compliance.run_compliance(
        [f"8:{BOTS}/dont_move.py"], rounds=rounds, seed=0, start_delay=START_DELAY,
    )
    assert report["ok"], report["flags"]
    game = report["games"][0]
    assert game["engine_result"]["result_type"] == "SUCCESS"
    usable = game["limits"]["usable_cumulative_s"]
    for pid, p in game["players"].items():
        assert p["flags"] == []
        assert p["turns"] >= rounds - 1
        assert 0.0 < p["cumulative_s"] < usable
        assert p["projected_full_game_s"] < usable  # instant bot ≈ engine tax only
        assert p["clock_ms"]["p50"] > 0.0
        assert p["clock_ms"]["max"] >= p["clock_ms"]["p50"]
        # bot-window is a sub-interval of the ban-clock window
        assert p["bot_window_ms"]["p50"] <= p["clock_ms"]["max"]
        assert p["max_move_len"] and p["max_move_len"] < game["limits"]["max_characters_read"]
        assert p["max_query_len"] and p["max_query_len"] > p["max_move_len"]
        assert "start" in p["rss_kb"] and "end" in p["rss_kb"]
        assert p["pipe_open_wait_s"] >= 0.0


@pytest.mark.slow
def test_full_game_stays_under_cumulative_margin() -> None:
    """P4.1 acceptance clause 3 on the current best stand-in bot.

    Until P5.4 lands `bot/main.py`, the stand-in is the scripted `greedy`
    fixture (real compute per turn, worst realistic per-turn cost among the
    fixtures). Re-point at the packaged bot when it exists.
    """
    report = compliance.run_compliance(
        [f"8:{BOTS}/dont_move.py"], rounds=None, seed=0, start_delay=START_DELAY,
    )
    assert report["ok"], report["flags"]
    game = report["games"][0]
    for p in game["players"].values():
        assert p["cumulative_s"] < game["limits"]["usable_cumulative_s"]


# --- threshold semantics on synthetic captures (subprocess-free) -------------

def _raw(players: dict[int, PlayerCapture], result: dict | None = None) -> dict:
    return {
        "seed": 0,
        "rounds_cap": 50,
        "engine_wall_s": 1.0,
        "result": result or {"result_type": "SUCCESS"},
        "players": players,
    }


def _turns(n: int, delta_s: float, move_len: int = 100) -> list[TurnRecord]:
    return [
        TurnRecord(round=i + 1, clock_delta_s=delta_s,
                   bot_window_s=delta_s * 0.5, query_len=1000, move_len=move_len)
        for i in range(n)
    ]


def test_analyze_flags_cumulative_breach() -> None:
    cap = PlayerCapture(script="x.py", turns=_turns(50, 0.001))
    cap.final_cumulative_s = 6.5  # > 0.75 × 8
    rep = analyze(_raw({0: cap}))
    assert "CUM_BREACH" in rep["players"][0]["flags"]
    assert not rep["ok"]


def test_analyze_flags_projection() -> None:
    # 5 ms/turn × 1400 = 7 s > 6 s, even though 50 turns only spent 0.25 s
    cap = PlayerCapture(script="x.py", turns=_turns(50, 0.005))
    cap.final_cumulative_s = 0.25
    rep = analyze(_raw({0: cap}))
    assert "CUM_PROJECTED" in rep["players"][0]["flags"]


def test_analyze_flags_single_move_near_miss() -> None:
    turns = _turns(49, 0.001) + [TurnRecord(round=50, clock_delta_s=1.9,
                                            bot_window_s=1.9, query_len=1000,
                                            move_len=100)]
    cap = PlayerCapture(script="x.py", turns=turns)
    cap.final_cumulative_s = 2.0
    rep = analyze(_raw({0: cap}))  # 1.9 s > 0.75 × 2 s, but no ban fired
    assert "SINGLE_MOVE_NEAR" in rep["players"][0]["flags"]


def test_analyze_flags_rss_leak_and_msg_len() -> None:
    cap = PlayerCapture(script="x.py", turns=_turns(50, 0.001, move_len=4000))
    cap.final_cumulative_s = 0.1
    cap.rss_kb = {"start": 100_000, "end": 100_000 + 60 * 1024}  # +60 MB
    rep = analyze(_raw({0: cap}))
    flags = rep["players"][0]["flags"]
    assert "RSS_LEAK" in flags
    assert "MSG_NEAR" in flags  # 4000 > 0.75 × 4096


def test_analyze_clean_capture_passes() -> None:
    cap = PlayerCapture(script="x.py", turns=_turns(50, 0.002))
    cap.final_cumulative_s = 0.1
    cap.rss_kb = {"start": 100_000, "mid": 100_500, "end": 101_000}
    rep = analyze(_raw({0: cap}))
    assert rep["ok"]
    assert rep["players"][0]["flags"] == []
    # limits came from the engine's own constants, not hard-coding
    assert rep["limits"]["cumulative_timeout_s"] == 8
    assert rep["limits"]["single_move_timeout_s"] == 2
    assert rep["limits"]["max_characters_read"] == 4096
    assert rep["limits"]["max_rounds"] == 1400


def test_analyze_attributes_ban_to_the_banned_seat_only() -> None:
    caps = {0: PlayerCapture(script="x.py", turns=_turns(3, 0.001)),
            1: PlayerCapture(script="y.py", turns=_turns(3, 0.001))}
    rep = analyze(_raw(caps, result={
        "result_type": "PLAYER_BANNED", "ban_type": "TIMEOUT",
        "player": 1, "reason": "You took too long to respond.",
    }))
    assert rep["players"][0]["flags"] == []
    assert rep["players"][1]["flags"] == ["BANNED"]
    assert not rep["ok"]
