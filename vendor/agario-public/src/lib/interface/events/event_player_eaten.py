from typing import Literal

from lib.interface.events.base_event import BaseEvent


class EventPlayerEaten(BaseEvent):
    event_type: Literal["event_player_eaten"] = "event_player_eaten"
    eater_player_id: int
    eater_blob_id: int
    eater_pos: tuple[float, float]
    eaten_player_id: int
    eaten_blob_id: int
    eaten_pos: tuple[float, float]
    eater_radius: float
    eaten_radius: float
    eaten_player_alive: bool


class PublicEventPlayerEaten(BaseEvent):
    event_type: Literal["public_event_player_eaten"] = "public_event_player_eaten"
    eater_player_id: int | None = None
    eater_pos: tuple[float, float] | None = None
    eater_radius: float | None = None
    eaten_player_id: int | None = None
    eaten_pos: tuple[float, float] | None = None
    eaten_radius: float | None = None
