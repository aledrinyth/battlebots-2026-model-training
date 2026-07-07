from typing import Literal, Sequence

from lib.interface.events.base_event import BaseEvent
from lib.models.food_model import FoodModel


class EventFoodSpawned(BaseEvent):
    event_type: Literal["event_food_spawned"] = "event_food_spawned"
    foods: Sequence[FoodModel]
