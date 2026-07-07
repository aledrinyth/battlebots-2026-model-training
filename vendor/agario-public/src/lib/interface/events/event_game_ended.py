from lib.interface.events.base_event import BaseEvent

from typing import Literal

class EventGameEndedCancelled(BaseEvent):
    event_type: Literal["event_game_ended_cancelled"] = "event_game_ended_cancelled"
    reason: str
