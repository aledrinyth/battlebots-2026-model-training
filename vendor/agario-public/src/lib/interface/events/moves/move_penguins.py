from lib.interface.events.moves.base_move import BaseMove
from lib.models.penguin_model import DirectionModel

from typing import Literal

    
class PushPenguins(BaseMove):
    event_type: Literal["push_penguins"] = "push_penguins"
    player_id: int
    directions: dict[int, DirectionModel] # maps penguin_id -> direction