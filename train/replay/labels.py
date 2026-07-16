"""Action labels + terminal-rank quality weight (impl doc §6).

``heading_to_bin`` is the exact inverse of ``act_v1.bin_to_vector`` (the
24-bin roundtrip is test-enforced). ``(0, 0)`` — "stay" — has no heading:
the row keeps its split label but masks the heading CE term via
``has_heading = 0``.
"""

from __future__ import annotations

import math
from typing import Optional

from bot.adapters import act_v1


def heading_to_bin(x: float, y: float) -> Optional[int]:
    """Nearest heading bin for a direction vector; ``None`` for "stay"."""
    if x == 0.0 and y == 0.0:
        return None
    step = 2.0 * math.pi / act_v1.HEADING_BINS
    return int(round(math.atan2(y, x) / step)) % act_v1.HEADING_BINS


def rank_weight(final_rank: int, n_players: int) -> float:
    """Linear terminal-rank quality weight: rank 1 → 1.0 … last → 0.0.

    Enforced, not optional (impl doc §6): BC must not imitate a bot that
    finished last. Consumers select harder cuts (e.g. winners-only ⇒
    ``weight == 1.0``, top-3 of 8 ⇒ ``weight >= 5/7``) by thresholding.
    """
    if n_players < 2:
        raise ValueError("rank_weight needs >= 2 players")
    if not 1 <= final_rank <= n_players:
        raise ValueError(f"rank {final_rank} out of range 1..{n_players}")
    return (n_players - final_rank) / (n_players - 1)
