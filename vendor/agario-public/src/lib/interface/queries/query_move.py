from typing import Literal, Sequence

from lib.interface.queries.base_query import BaseQuery
from lib.models.blob_model import VisibleBlobModel
from lib.models.food_model import FoodModel
from lib.models.player_model import PlayerModel
from lib.models.virus_model import VirusModel


class QueryMovePlayer(BaseQuery):
    """Authoritative per-turn payload sent to a bot player."""

    query_type: Literal["move_player"] = "move_player"
    round: int
    rankings: Sequence[int]
    you: PlayerModel
    view_center: tuple[float, float]
    vision_size: float
    visible_food: Sequence[FoodModel]
    visible_blobs: Sequence[VisibleBlobModel]
    visible_viruses: Sequence[VirusModel]
