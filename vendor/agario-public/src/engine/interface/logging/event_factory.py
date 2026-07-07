from engine.interface.io.exceptions import (
    BrokenPipeException,
    CumulativeTimeoutException,
    InvalidMessageException,
    InvalidMoveException,
    PlayerException,
    TimeoutException,
)

from lib.interface.events.event_player_bannned import EventPlayerBanned
from lib.interface.io.ban_type import BanType


def event_banned_factory(e: PlayerException) -> "EventPlayerBanned":
    ban_type: BanType
    details = e.details
    match e:
        case TimeoutException() as e:
            ban_type = "TIMEOUT"
        case CumulativeTimeoutException() as e:
            ban_type = "CUMULATIVE_TIMEOUT"
        case BrokenPipeException():
            ban_type = "BROKEN_PIPE"
        case InvalidMessageException() as e:
            ban_type = "INVALID_MESSAGE"
        case InvalidMoveException() as e:
            ban_type = "INVALID_MOVE"
        case _:
            raise RuntimeError("An unspecified PlayerException was raised.")

    return EventPlayerBanned(
        player_id=e.player_id,
        reason=e.error_message,
        ban_type=ban_type,
        details=details,
    )
