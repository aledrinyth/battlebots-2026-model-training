"""Fail-safe wrapper around the per-turn pipeline (P4.3, consumed by P5.4).

A match bot may never let an exception escape to the pipe: an unanswered
query is a ban. Every turn therefore runs through :func:`safe_move` — if the
real pipeline (parse → featurize → forward → serialize) raises *anything*,
the bot still answers with a legal fallback:

1. **flee-to-centre** when we have a parsed query in hand (walking toward the
   arena centre beats sitting still against the wall while poisoned frames
   persist), else
2. **zero move** (the kit's own example fallback — always legal).

Match-legal surface only: ``lib`` models + :mod:`bot.spec` (never ``engine``).
"""

from __future__ import annotations

from typing import Callable, Optional

from lib.interface.events.moves.move_player import MovePlayer
from lib.models.penguin_model import DirectionModel

from bot.spec import ARENA_SIZE

__all__ = ["safe_move", "zero_move", "flee_to_center_move"]


def zero_move(player_id: int) -> MovePlayer:
    """A zero heading — always a legal move (the kit example's fallback)."""
    return MovePlayer(
        player_id=player_id,
        direction=DirectionModel(x=0.0, y=0.0),
        split=False,
    )


def flee_to_center_move(query, player_id: int) -> MovePlayer:
    """Head for the arena centre — the safest heading with no other signal."""
    x, y = query.you.pos
    dx = ARENA_SIZE / 2.0 - float(x)
    dy = ARENA_SIZE / 2.0 - float(y)
    if not (abs(dx) > 1e-9 or abs(dy) > 1e-9):  # already there (or NaN-free zero)
        return zero_move(player_id)
    return MovePlayer(
        player_id=player_id,
        direction=DirectionModel(x=dx, y=dy),
        split=False,
    )


def safe_move(
    compute: Callable[[], MovePlayer],
    player_id: int,
    query: Optional[object] = None,
) -> MovePlayer:
    """Run ``compute()``; on *any* failure return a legal fallback move.

    Also refuses a successful-but-invalid result (wrong type / wrong
    ``player_id`` — the engine would ban either), converting it to the
    fallback as well.
    """
    try:
        move = compute()
        if not isinstance(move, MovePlayer) or move.player_id != player_id:
            raise ValueError(f"pipeline produced an invalid move: {move!r}")
        return move
    except Exception:
        pass
    if query is not None:
        try:
            return flee_to_center_move(query, player_id)
        except Exception:
            pass
    return zero_move(player_id)
