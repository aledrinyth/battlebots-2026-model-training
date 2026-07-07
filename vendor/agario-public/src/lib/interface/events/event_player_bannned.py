from lib.interface.events.base_event import BaseEvent

from typing import Any, Literal, Optional, Union

from lib.interface.events.moves.typing import MoveType
from lib.interface.io.ban_type import BanType


class EventPlayerBanned(BaseEvent):
    event_type: Literal["event_player_banned"] = "event_player_banned"
    player_id: int
    ban_type: BanType
    reason: str
    details: Optional[Union[list[Any], MoveType, Any]]
