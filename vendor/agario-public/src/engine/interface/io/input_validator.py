from copy import deepcopy
from typing import TYPE_CHECKING

from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.events.moves.typing import MoveType
from lib.interface.queries.base_query import BaseQuery

import string

# VALID_DIRECTIONS = UP, DOWN, LEFT, RIGHT

if TYPE_CHECKING:
    from engine.state.game_state import GameState


class MoveValidator:
    def __init__(self, state: "GameState"):
        self.state = state

    def validate(self, event: MoveType, query: BaseQuery, player_id: int) -> None:
        self._validate_move(event, query, player_id)

        match event:
            case MovePlayer() as e:
                self._validate_move_player_direction(e, query, player_id)

    def _validate_move(self, e: MoveType, query: BaseQuery, player_id: int) -> None:
        if not e.player_id == player_id:
            raise ValueError(
                "You set the move 'player_id' to a player_id other than your own."
            )

    def _validate_move_player_direction(
        self, e: MovePlayer, query: BaseQuery, player_id: int
    ) -> None:
        if not self.state.players[player_id].alive:
            raise ValueError("Dead players cannot submit moves.")
