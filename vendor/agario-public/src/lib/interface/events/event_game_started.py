from lib.interface.events.base_event import BaseEvent

from typing import Literal, Sequence

from lib.models.player_model import PlayerModel, PublicPlayerModel


class EventGameStarted(BaseEvent):
    event_type: Literal["event_game_started"] = "event_game_started"
    turn_order: list[int]
    arena_size: float
    vision_size: float
    turn_duration_seconds: float
    max_rounds: int
    players: Sequence[PlayerModel]
    engine_version: str | None = None


class PublicEventGameStarted(BaseEvent):
    """Bot-safe startup snapshot derived from the engine's full game-start event."""

    event_type: Literal["public_event_game_started"] = "public_event_game_started"
    turn_order: list[int]
    arena_size: float
    vision_size: float
    turn_duration_seconds: float
    max_rounds: int
    you: PlayerModel
    players: Sequence[PublicPlayerModel]
    engine_version: str | None = None
