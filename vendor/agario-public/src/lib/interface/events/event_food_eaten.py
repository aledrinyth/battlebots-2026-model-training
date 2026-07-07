from typing import Literal, Sequence

from lib.interface.events.base_event import BaseEvent


class EventFoodEaten(BaseEvent):
    event_type: Literal["event_food_eaten"] = "event_food_eaten"
    player_id: int
    blob_id: int
    food_ids: Sequence[int]
    new_radius: float
