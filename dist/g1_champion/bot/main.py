"""P5.4 — the match-time entry point: pipe loop, fail-safe, time ledger.

Per-turn path (the exact one P4.2 gates): kit ``Connection`` wire framing →
``QueryMovePlayer.model_validate_json`` → :class:`bot.featurizer.Featurizer`
→ :class:`bot.inference.NumpyPolicy` (greedy decode) → ``ActionAdapter`` →
serialize. Every turn runs under :func:`bot.safety.safe_move` — nothing may
escape to the pipe unanswered.

Design decisions (recorded in notes/P4-compliance.md):

* **Kit protocol layer, not the kit ``Game`` mirror.** ``helper.Game``
  replays every update event into a client-side state copy the featurizer
  never reads. We keep the kit's ``Connection`` (the risky wire-framing
  bit) and parse the query directly — the path the P4.2 scene bank times.
* **player_id from the submission directory** (``submission{N}`` is the
  engine's own layout), confirmed against the first parsed query — available
  before the first byte arrives, so even a first-query parse failure still
  answers with a *valid* player id.
* **GRU hidden hygiene**: the featurizer raises ``is_reset`` on any round
  gap (game start, respawn) and the policy zeroes the hidden on it; the loop
  additionally guards ``np.isfinite`` so one poisoned frame can't corrupt
  the rest of the game (P4.3 fuzz finding).
* **GC parked**: refcounting frees our per-turn garbage (arrays, models);
  a gen-0 collect runs every 32 turns *after* the move is flushed — outside
  the engine's timed window.
* **Time ledger to stderr**: per-turn window = query-bytes-in-hand →
  move-flushed (the part we control), logged every 100 turns and on every
  reset; import+init time logged before the first turn.

Weights resolution: ``$BOT_WEIGHTS`` → ``weights.npz`` next to this file →
the repo's random-init fixture (dev fallback, so the unpackaged tree runs
under the compliance harness).
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from helper.interface import Connection
from lib.interface.queries.query_move import QueryMovePlayer

# Run as a script (the engine launches `python …/main.py` with cwd=submissionN),
# the `bot` package's parent isn't on sys.path — bootstrap it. Holds for both
# layouts: repo tree (…/repo/bot/main.py → repo/) and P5.5 bundle
# (…/bundle/bot/main.py → bundle/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.adapters.act_v1 import ActionAdapter  # noqa: E402
from bot.featurizer import Featurizer  # noqa: E402
from bot.inference import NumpyPolicy  # noqa: E402
from bot.safety import safe_move  # noqa: E402

_LEDGER_EVERY = 100  # turns between stderr ledger lines
_GC_EVERY = 32       # turns between gen-0 collects (post-flush)


def _log(msg: str) -> None:
    print(f"[bot] {msg}", file=sys.stderr, flush=True)


def _resolve_weights() -> Path:
    env = os.environ.get("BOT_WEIGHTS")
    if env:
        return Path(env)
    local = Path(__file__).resolve().parent / "weights.npz"
    if local.exists():
        return local
    # Dev fallback: the unpackaged repo tree (P5.5 bundles the real weights).
    return (Path(__file__).resolve().parent.parent
            / "tests" / "fixtures" / "adversarial" / "weights_random.npz")


def _player_id_from_cwd() -> int | None:
    """The engine's own layout: we are launched with cwd=…/submission{N}."""
    name = Path.cwd().name
    if name.startswith("submission") and name[len("submission"):].isdigit():
        return int(name[len("submission"):])
    return None


def main() -> None:
    t0 = time.perf_counter()
    player_id = _player_id_from_cwd()

    # Open the pipes FIRST — the engine's 3 s open-pipe alarm is running and
    # its wait accrues to our 8 s ban clock.
    conn = Connection()
    t_pipes = time.perf_counter()

    policy = NumpyPolicy.load(str(_resolve_weights()))
    featurizer = Featurizer()
    hidden = policy.initial_hidden()

    gc.collect()
    gc.disable()
    _log(f"init: pipes {t_pipes - t0:.3f}s, ready {time.perf_counter() - t0:.3f}s, "
         f"player_id(cwd)={player_id}")

    turns = 0
    cum = 0.0
    worst = 0.0

    while True:
        try:
            raw = conn._receive()  # kit framing; blocks until the next query
        except EOFError:
            _log(f"engine closed the pipe after {turns} turns "
                 f"(cum {cum:.3f}s) — game over.")
            return
        t_in = time.perf_counter()

        query = None
        try:
            query = QueryMovePlayer.model_validate_json(raw)
            qpid = query.you.player_id
            if player_id is None:
                player_id = qpid
            elif qpid != player_id:  # engine-impossible; trust the engine
                _log(f"player_id mismatch: cwd={player_id} query={qpid} — using query's")
                player_id = qpid
        except Exception:
            if player_id is None:  # first-ever query AND cwd was unhelpful
                try:
                    player_id = int(json.loads(raw)["you"]["player_id"])
                except Exception:
                    player_id = 0  # last resort — answer *something*

        def compute():
            obs = featurizer.featurize(query)
            if obs.is_reset:
                _log(f"reset detected at round {query.round} (turn {turns}) — hidden zeroed")
            action, new_hidden = policy.act(obs, hidden)
            if not np.isfinite(new_hidden).all():  # P4.3: never carry a poisoned state
                _log(f"non-finite hidden at round {query.round} — reinitialized")
                new_hidden = policy.initial_hidden()
            return action, new_hidden

        # safe_move expects a move-producing callable; carry the hidden out-of-band.
        result: dict = {}

        def guarded_move():
            action, new_hidden = compute()
            result["hidden"] = new_hidden
            return ActionAdapter.to_move(player_id, action)

        move = safe_move(guarded_move, player_id, query=query)
        conn.send_move(move)
        t_out = time.perf_counter()

        hidden = result.get("hidden", hidden)
        turns += 1
        spent = t_out - t_in
        cum += spent
        worst = max(worst, spent)
        if turns % _LEDGER_EVERY == 0:
            _log(f"ledger: turn {turns} round {query.round if query else '?'} "
                 f"cum {cum:.3f}s avg {cum / turns * 1e3:.3f}ms worst {worst * 1e3:.3f}ms")
        if turns % _GC_EVERY == 0:
            gc.collect(0)  # post-flush: outside the engine's timed window


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # outer loop / pipe failures — nothing left to answer
        _log(f"fatal: {type(e).__name__}: {e}")
        raise
