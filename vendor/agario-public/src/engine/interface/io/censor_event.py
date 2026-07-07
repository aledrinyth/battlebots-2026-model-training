from typing import TYPE_CHECKING, Optional

from lib.interface.events.event_game_started import (
    EventGameStarted,
    PublicEventGameStarted,
)
from lib.interface.events.event_player_eaten import (
    EventPlayerEaten,
    PublicEventPlayerEaten,
)
from lib.interface.events.event_player_moved import EventPlayerMoved
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.events.typing import EventType

if TYPE_CHECKING:
    from engine.state.game_state import GameState


class CensorEvent:
    def __init__(self, state: "GameState") -> None:
        self.state = state

    def censor(self, event: EventType, player_id: int) -> Optional[EventType]:
        """Return the bot-visible form of an engine event, or None if it stays hidden."""

        match event:
            case EventGameStarted() as e:
                return PublicEventGameStarted(
                    turn_order=e.turn_order,
                    arena_size=e.arena_size,
                    vision_size=self.state.get_player_vision_size(player_id),
                    turn_duration_seconds=e.turn_duration_seconds,
                    max_rounds=e.max_rounds,
                    players=[player.get_public() for player in e.players],
                    you=filter(
                        lambda x: x.player_id == player_id, e.players
                    ).__next__(),
                    engine_version=e.engine_version,
                )
            case EventPlayerMoved() as e:
                if e.player_id == player_id:
                    return e

                visible_blobs = tuple(
                    self.state.get_public_visible_player_blobs(e.player_id, player_id)
                )
                if visible_blobs:
                    total_mass = sum(blob.radius * blob.radius for blob in visible_blobs)
                    pos = (
                        sum(blob.pos[0] * blob.radius * blob.radius for blob in visible_blobs)
                        / total_mass,
                        sum(blob.pos[1] * blob.radius * blob.radius for blob in visible_blobs)
                        / total_mass,
                    )
                    return EventPlayerMoved(
                        player_id=e.player_id,
                        pos=pos,
                        radius=total_mass ** 0.5,
                        alive=e.alive,
                        blobs=visible_blobs,
                    )
                return None
            case EventPlayerEaten() as e:
                eater_visible = player_id == e.eater_player_id or self.state.player_can_see_circle(
                    player_id,
                    e.eater_pos[0],
                    e.eater_pos[1],
                    e.eater_radius,
                )
                eaten_visible = player_id == e.eaten_player_id or self.state.player_can_see_circle(
                    player_id,
                    e.eaten_pos[0],
                    e.eaten_pos[1],
                    e.eaten_radius,
                )
                if eater_visible or eaten_visible:
                    return PublicEventPlayerEaten(
                        eater_player_id=e.eater_player_id if eater_visible else None,
                        eater_pos=e.eater_pos if eater_visible else None,
                        eater_radius=e.eater_radius if eater_visible else None,
                        eaten_player_id=e.eaten_player_id if eaten_visible else None,
                        eaten_pos=e.eaten_pos if eaten_visible else None,
                        eaten_radius=e.eaten_radius if eaten_visible else None,
                    )
                return None
            case MovePlayer() as e:
                if e.player_id == player_id:
                    return e
                return None

        return event
