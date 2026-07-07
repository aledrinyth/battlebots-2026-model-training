from lib.interface.events.base_event import BaseEvent

from typing import Literal

from lib.interface.events.moves.typing import MoveType


class EventPenguinFallenOff(BaseEvent):
    event_type: Literal["event_penguin_fallen_off"] = "event_penguin_fallen_off"
    player_id: int
    penguin_id: int
    coordinates: tuple[float, float]