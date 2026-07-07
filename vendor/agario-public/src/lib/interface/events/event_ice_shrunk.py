from typing import Literal

from lib.interface.events.base_event import BaseEvent


class EventIceShrunk(BaseEvent):
    event_type: Literal["event_ice_shrunk"] = "event_ice_shrunk"
    new_size: float
