"""P0.2 acceptance: the recorded reference match parses and is complete.

These fixtures (``game.json`` + ``results.json``) are the ground-truth output of
a full match run through the real pipe runner (``simulation``) with the vendored
engine pinned in ``vendor/agario-public``. They are the reference the in-process
wrapper is later proven equivalent to (P1.3).

Constants come from the engine itself (``lib.config.arena``) — nothing about the
match shape is hard-coded here (adaptation-pipeline boundary discipline).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.config.arena import MAX_ROUNDS, NUM_PLAYERS

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads((FIXTURES / "results.json").read_text())


@pytest.fixture(scope="module")
def recording() -> list[dict]:
    return json.loads((FIXTURES / "game.json").read_text())


def test_results_parse_and_are_a_success(results: dict) -> None:
    assert results["result_type"] == "SUCCESS"


def test_ranking_is_a_full_permutation(results: dict) -> None:
    assert sorted(results["ranking"]) == list(range(NUM_PLAYERS))


def test_final_masses_cover_every_player(results: dict) -> None:
    masses = results["final_masses"]
    assert sorted(int(k) for k in masses) == list(range(NUM_PLAYERS))
    assert all(m >= 0.0 for m in masses.values())


def test_recording_brackets_a_full_match(recording: list[dict]) -> None:
    assert recording[0]["event_type"] == "event_game_started"
    assert recording[0]["max_rounds"] == MAX_ROUNDS
    # A winner is declared only after the final round commits.
    assert recording[-1]["event_type"] == "event_player_won"


def test_recording_is_non_trivial(recording: list[dict]) -> None:
    # A real 8-player, 1400-round match produces tens of thousands of events;
    # a stub or crashed match would produce a handful.
    assert len(recording) > MAX_ROUNDS, "recording is too short to be a full match"
