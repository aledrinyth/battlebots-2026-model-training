from typing import Literal

from lib.interface.events.base_event import BaseEvent
from lib.models.blob_model import BlobModel


class EventPlayerMoved(BaseEvent):
    event_type: Literal["event_player_moved"] = "event_player_moved"
    player_id: int
    pos: tuple[float, float]
    radius: float
    alive: bool
    blobs: tuple[BlobModel, ...] = ()
