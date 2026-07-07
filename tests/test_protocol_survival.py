"""P0.4 acceptance: a hardcoded-move toy bot survives a full game.

The reference fixture was recorded with 4× cautious + 4× ``dont_move`` (a
hardcoded bot that always emits a zero heading). If the toy bot ever mishandled
the protocol — slow pipe open, bad framing, oversized/invalid move, closed pipe
— the engine would ban it and emit ``event_player_bannned``. Zero ban events
across a full 1400-round match ⇒ every bot, including the hardcoded one,
survived the wire protocol end-to-end.

We also require at least one eat event, so the recording genuinely exercised the
death/respawn path the audit describes (dead players receive no queries).
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# Note the engine's spelling: three n's.
BAN_EVENT = "event_player_bannned"
EAT_EVENT = "event_player_eaten"


@pytest.fixture(scope="module")
def event_counts() -> collections.Counter:
    recording = json.loads((FIXTURES / "game.json").read_text())
    return collections.Counter(e["event_type"] for e in recording)


def test_no_bot_was_banned(event_counts: collections.Counter) -> None:
    assert event_counts[BAN_EVENT] == 0, "a bot was banned during the reference match"


def test_death_respawn_cycle_exercised(event_counts: collections.Counter) -> None:
    assert event_counts[EAT_EVENT] > 0, "no eat events — death/respawn path untested"


def test_match_reached_a_winner(event_counts: collections.Counter) -> None:
    assert event_counts["event_player_won"] == 1
