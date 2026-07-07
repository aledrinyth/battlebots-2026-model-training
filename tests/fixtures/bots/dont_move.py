"""Stationary reference bot (fixture): always emits a zero heading.

A zero direction is a legal move (the kit's own example uses it as its
fallback). Serves as a passive ladder anchor and keeps the reference match
deterministic on this player's side.
"""

from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.events.moves.typing import MoveType
from lib.interface.queries.query_move import QueryMovePlayer
from lib.interface.queries.typing import QueryType
from lib.models.penguin_model import DirectionModel


def main() -> None:
    game = Game()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryMovePlayer():
                    return MovePlayer(
                        player_id=game.state.me.player_id,
                        direction=DirectionModel(x=0.0, y=0.0),
                    )
            raise RuntimeError(f"Unsupported query type: {type(query)}")

        game.send_move(choose_move(query))


if __name__ == "__main__":
    main()
