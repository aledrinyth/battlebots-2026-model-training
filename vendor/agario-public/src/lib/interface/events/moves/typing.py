from pydantic import Field
from lib.interface.events.moves.move_player import MovePlayer

from typing import Annotated, Union, TypeAlias

MoveType: TypeAlias = Annotated[
    Union[
        MovePlayer,
    ],
    Field(discriminator="event_type"),
]
