from lib.interface.events.base_event import BaseEvent

from typing import Literal, Mapping


class EventPrivatePhysicsTick(BaseEvent):
    event_type: Literal["event_private_physics_tick"] = "event_private_physics_tick"
    final_destinations: Mapping[tuple[int, int], tuple[float, float]] # maps (playe_id, penguin_id) -> new destination