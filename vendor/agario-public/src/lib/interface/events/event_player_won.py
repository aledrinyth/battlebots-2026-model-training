from lib.interface.events.base_event import BaseEvent

from typing import Literal


class EventPlayerWon(BaseEvent):
    event_type: Literal["event_player_won"] = "event_player_won"
    player_id: int
