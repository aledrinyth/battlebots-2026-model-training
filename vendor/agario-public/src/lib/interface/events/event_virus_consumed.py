from typing import Literal

from lib.interface.events.base_event import BaseEvent


class EventVirusConsumed(BaseEvent):
    event_type: Literal["event_virus_consumed"] = "event_virus_consumed"
    player_id: int
    blob_id: int
    virus_id: int
    virus_pos: tuple[float, float]
    pieces_created: int
