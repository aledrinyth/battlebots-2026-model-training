from typing import Literal

from lib.interface.events.moves.base_move import BaseMove
from lib.models.penguin_model import DirectionModel


class MovePlayer(BaseMove):
    event_type: Literal["move_player"] = "move_player"
    player_id: int
    direction: DirectionModel
    split: bool = False
