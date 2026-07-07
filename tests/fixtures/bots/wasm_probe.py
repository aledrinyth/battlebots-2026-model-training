"""Gate G0 probe bot (P0.3).

Proves wasmtime works end-to-end inside a real match submission:

1. Instantiate ``hello.wasm`` **before** opening the pipe loop (this is the cost
   that must fit inside the engine's 3 s pipe-open alarm alongside imports).
2. Log the instantiation time to stderr (captured in ``submission.err``) so the
   match harness can assert it stayed under the 500 ms budget.
3. Call the wasm ``add`` export **every turn** to choose the heading, so the
   module isn't just loaded but actually drives a legal move each round.

If wasmtime were unavailable or instantiation blew the pipe-open window, the bot
would never open its pipes and the engine would ban/skip it — so "match
completes with this bot alive" is itself part of the gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The `simulation` runner launches this file in place (it does not copy it), so
# __file__ is stable: repo/tests/fixtures/bots/wasm_probe.py -> repo/v2/wasm.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "v2" / "wasm"))
from wasm_core import load_add  # noqa: E402

# ---- Gate G0: instantiate BEFORE any pipe I/O ----------------------------
_add, _instantiation_ms = load_add()
_MAGIC = _add(2, 3)
assert _MAGIC == 5, f"wasm add broken: add(2,3)={_MAGIC}"
print(
    f"[wasm-gate] instantiation_ms={_instantiation_ms:.3f} add(2,3)={_MAGIC}",
    file=sys.stderr,
    flush=True,
)

# ---- Now it is safe to open the pipes -------------------------------------
from helper.game import Game  # noqa: E402
from lib.interface.events.moves.move_player import MovePlayer  # noqa: E402
from lib.interface.events.moves.typing import MoveType  # noqa: E402
from lib.interface.queries.query_move import QueryMovePlayer  # noqa: E402
from lib.interface.queries.typing import QueryType  # noqa: E402
from lib.models.penguin_model import DirectionModel  # noqa: E402

# Four cardinal headings; the wasm result selects which one each turn.
_HEADINGS = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]


def main() -> None:
    game = Game()
    turn = 0

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            nonlocal turn
            match query:
                case QueryMovePlayer():
                    # Seek nearest food if any is visible; otherwise let the
                    # wasm module pick a cardinal heading (result USED in the
                    # legal move, every turn).
                    me = game.state.me
                    if game.state.visible_food:
                        food = min(
                            game.state.visible_food,
                            key=lambda f: (f.pos[0] - me.x) ** 2 + (f.pos[1] - me.y) ** 2,
                        )
                        dx, dy = food.pos[0] - me.x, food.pos[1] - me.y
                    else:
                        idx = _add(turn, 0) % len(_HEADINGS)
                        dx, dy = _HEADINGS[idx]
                    turn += 1
                    return MovePlayer(
                        player_id=me.player_id,
                        direction=DirectionModel(x=dx, y=dy),
                    )
            raise RuntimeError(f"Unsupported query type: {type(query)}")

        game.send_move(choose_move(query))


if __name__ == "__main__":
    main()
