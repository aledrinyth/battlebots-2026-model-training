from pydantic import Field
from lib.interface.events.event_game_ended import (
    EventGameEndedCancelled,
)
from lib.interface.events.event_food_eaten import EventFoodEaten
from lib.interface.events.event_food_spawned import EventFoodSpawned
from lib.interface.events.event_game_started import (
    EventGameStarted,
    PublicEventGameStarted,
)
from lib.interface.events.event_player_eaten import (
    EventPlayerEaten,
    PublicEventPlayerEaten,
)
from lib.interface.events.event_player_bannned import EventPlayerBanned
from lib.interface.events.event_player_moved import EventPlayerMoved
from lib.interface.events.event_player_won import EventPlayerWon
from lib.interface.events.event_virus_spawned import EventVirusSpawned
from lib.interface.events.event_virus_consumed import EventVirusConsumed
from lib.interface.events.moves.typing import MoveType

from typing import Annotated, TypeAlias, Union


EventType: TypeAlias = Annotated[
    Union[
        EventPlayerMoved,
        EventPlayerEaten,
        PublicEventPlayerEaten,
        EventFoodSpawned,
        EventFoodEaten,
        EventVirusSpawned,
        EventVirusConsumed,
        EventGameStarted,
        EventPlayerBanned,
        EventPlayerWon,
        EventGameEndedCancelled,
        PublicEventGameStarted,
        MoveType,
    ],
    Field(discriminator="event_type"),
]

# Export these things
__all__ = [
    "EventType",
    "EventPlayerWon",
]
