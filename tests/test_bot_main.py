"""P5.4 acceptance: the real bot survives the real runner, death included.

Three subprocess tests drive ``bot/main.py`` (the shipping entry point, dev
weights fallback = the P4.2 random-init fixture) through the real pipe
runner via the P4.1 compliance rig:

* a capped game: engine SUCCESS, zero flags, ledger + init lines on stderr;
* a **forced death/respawn cycle** — a predator blob is injected mid-game
  through the engine's own ``commit_round`` (the P4.2 surgery pattern), the
  engine's eat/respawn machinery runs for real, and the bot must (a) skip
  the dead rounds (the engine never queries dead players), (b) detect the
  round gap and log the hidden reset, (c) finish the game unflagged;
* (slow) a full-length game — the playbook's "zero exceptions, zero invalid
  moves" clause end-to-end; ``make compliance-full`` re-runs this at 10
  games before any submission.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("engine")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import compliance  # noqa: E402  (path set above)

BOT = "bot/main.py"
FILLER = "tests/fixtures/bots/dont_move.py"
START_DELAY = 1.0


def test_bot_survives_a_capped_real_runner_game() -> None:
    report = compliance.run_compliance(
        [f"1:{BOT}", f"7:{FILLER}"], rounds=60, seed=0, start_delay=START_DELAY,
    )
    assert report["ok"], report["flags"]
    game = report["games"][0]
    assert game["engine_result"]["result_type"] == "SUCCESS"
    p0 = game["players"][0]
    assert p0["turns"] == 60
    assert p0["flags"] == []


def test_bot_stderr_carries_init_and_reset_lines() -> None:
    raw = compliance.run_game(
        [f"1:{BOT}", f"7:{FILLER}"], seed=0, rounds=60, start_delay=START_DELAY,
    )
    err = raw["players"][0].stderr_tail
    assert "[bot] init:" in err                # import+init ledger line
    assert "[bot] reset detected" in err       # game-start reset (round gap from None)
    assert "fatal" not in err


def test_forced_death_respawn_cycle_through_the_real_runner() -> None:
    """Inject a predator via the engine's own commit path; the bot must ride
    through the death gap and come back with a fresh hidden state."""
    import engine.state.state_mutator as _sm
    from engine.state.blob_state import BlobState

    KILL_AT_COMMIT = 20
    ROUNDS = 80  # death ~20, respawn ~+30, then ~30 alive rounds

    orig = _sm.StateMutator.commit_round
    state = {"commits": 0, "injected": False}

    def predatory_commit(self, moves):
        state["commits"] += 1
        if state["commits"] == KILL_AT_COMMIT and not state["injected"]:
            state["injected"] = True
            me = self.state.players[0]
            if me.alive:  # drop a 3.0-radius blob (≫ 1.2 × ours) on our centroid
                self.state.players[1].blobs = {
                    0: BlobState(blob_id=0, x=me.x, y=me.y, radius=3.0,
                                 merge_cooldown=500)
                }
        return orig(self, moves)

    _sm.StateMutator.commit_round = predatory_commit
    try:
        raw = compliance.run_game(
            [f"1:{BOT}", f"7:{FILLER}"], seed=0, rounds=ROUNDS,
            start_delay=START_DELAY,
        )
    finally:
        _sm.StateMutator.commit_round = orig

    assert state["injected"], "the predator never got injected"
    assert raw["result"]["result_type"] == "SUCCESS", raw["result"]

    p0 = raw["players"][0]
    # The engine never queries dead players: the death gap must show.
    from lib.config.arena import RESPAWN_DELAY_ROUNDS
    assert len(p0.turns) <= ROUNDS - RESPAWN_DELAY_ROUNDS + 2
    rounds_seen = [t.round for t in p0.turns]
    gaps = [b - a for a, b in zip(rounds_seen, rounds_seen[1:])]
    assert max(gaps) >= RESPAWN_DELAY_ROUNDS, f"no respawn gap in {rounds_seen}"

    # The bot saw it too: one reset at game start + one at respawn, no fatal.
    err = p0.stderr_tail
    assert err.count("[bot] reset detected") >= 2, err
    assert "fatal" not in err

    # And the verdict machinery still passes it.
    rep = compliance.analyze(raw)
    assert rep["players"][0]["flags"] == [], rep["players"][0]


@pytest.mark.slow
def test_full_game_zero_exceptions_zero_invalid_moves() -> None:
    report = compliance.run_compliance(
        [f"1:{BOT}", f"7:{FILLER}"], rounds=None, seed=0, start_delay=START_DELAY,
    )
    assert report["ok"], report["flags"]
    game = report["games"][0]
    assert game["engine_result"]["result_type"] == "SUCCESS"
    assert game["players"][0]["cumulative_s"] < game["limits"]["usable_cumulative_s"]
