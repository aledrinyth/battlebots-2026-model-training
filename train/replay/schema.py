"""Shared constants + row schema for the replay dataset (impl doc §8).

The dataset row arrays, their dtypes, and the engine-version compatibility
set live here so ``convert``/``dataset``/``ingest`` can never drift apart.
"""

from __future__ import annotations

import numpy as np

#: Engine versions whose replays are reconstruction-valid against the current
#: vendored engine. Audited 2026-07-17 (notes/tourney-replay-stats.md §3):
#: .13→.14 changed live-query censoring only (the replay is the *private*
#: history — unaffected — and physics/config are byte-identical); .14→.15
#: changed ban-event logging only. Anything else must be quarantined, never
#: silently mixed (impl doc §9).
COMPATIBLE_ENGINE_VERSIONS = frozenset({"2026.1.13", "2026.1.14", "2026.1.15"})

#: Replay event types the reconstruction understands. A replay containing
#: anything else is structurally unknown and must be rejected.
KNOWN_EVENT_TYPES = frozenset(
    {
        "event_game_started",
        "move_player",
        "event_player_moved",
        "event_food_eaten",
        "event_food_spawned",
        "event_virus_spawned",
        "event_virus_consumed",
        "event_player_eaten",
        "event_player_won",
    }
)

#: A banned/cancelled game is not a clean demonstration source — reject the
#: whole file rather than train on a partial match.
ABORT_EVENT_TYPES = frozenset(
    {"event_player_banned", "event_game_ended_cancelled"}
)

#: Dataset arrays per shard: name -> dtype (impl doc §8). ``obs`` is
#: ``(N, OBS_WIDTH)`` float32; everything else is ``(N,)``.
ROW_DTYPES = {
    "obs": np.float32,
    "heading_bin": np.int8,     # 0..23, or -1 when has_heading == 0
    "has_heading": np.uint8,    # 0 for "stay" rows (mask the heading loss)
    "split": np.uint8,
    "is_reset": np.uint8,       # GRU reset flag for TBPTT batching
    "rank_weight": np.float32,  # terminal-rank quality weight
    "seat": np.int8,            # player_id within the match
    "bot_id": np.int32,         # interned identity; sidecar maps id -> name
    "episode": np.int32,        # (replay, seat) segment id
    "round": np.int16,
}
