from lib.interface.events.base_event import BaseEvent

from typing import Literal, Mapping


class EventPenguinsMoved(BaseEvent):
    event_type: Literal["event_penguins_moved"] = "event_penguins_moved"
    player_id: int
    final_destinations: Mapping[int, tuple[float, float]] # maps penguin_id -> new destination