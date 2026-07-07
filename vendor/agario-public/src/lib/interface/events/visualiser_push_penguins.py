from lib.interface.events.base_event import BaseEvent
from lib.models.penguin_model import DirectionModel

from typing import Literal, Mapping

class VisualiserPushPenguins(BaseEvent):
    event_type: Literal["visualiser_push_penguins"] = "visualiser_push_penguins"
    directions: Mapping[tuple[int, int], DirectionModel]