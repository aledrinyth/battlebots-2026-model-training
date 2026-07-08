"""Scripted baseline bots (P2.4) — ladder anchors, curriculum opponents, league seed.

Every bot here implements the same tiny interface against the wrapper's
observation dicts (``QueryMovePlayer``), so the trainer, the curriculum
(P6.1), and the evaluation ladder (P7.1) can all drive them uniformly:

    bot.reset()                      # start of an episode (stateless bots: no-op)
    move = bot.act(query)            # QueryMovePlayer -> MovePlayer

They read the raw query (never the featurizer) and emit moves through the
match-legal geometry, so they are cheap and deterministic given their own RNG.
"""

from train.opponents.scripted import (
    BOTS,
    CautiousBot,
    ChaserBot,
    CowardBot,
    DontMoveBot,
    GreedyFoodBot,
    RandomBot,
    ScriptedBot,
    make_bot,
)

__all__ = [
    "ScriptedBot",
    "RandomBot",
    "DontMoveBot",
    "GreedyFoodBot",
    "CautiousBot",
    "CowardBot",
    "ChaserBot",
    "BOTS",
    "make_bot",
]
