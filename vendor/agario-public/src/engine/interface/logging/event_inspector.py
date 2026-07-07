from typing import Union
from engine.interface.io.game_result import (
    GameBanResult,
    GameCancelledResult,
    GameCrashedResult,
    GameSuccessResult,
)
from lib.interface.events.event_game_ended import (
    EventGameEndedCancelled,
)
from lib.interface.events.event_food_eaten import EventFoodEaten
from lib.interface.events.event_food_spawned import EventFoodSpawned
from lib.interface.events.event_virus_spawned import EventVirusSpawned
from lib.interface.events.event_virus_consumed import EventVirusConsumed
from lib.interface.events.event_player_bannned import EventPlayerBanned
from lib.interface.events.event_player_eaten import EventPlayerEaten
from lib.interface.events.event_player_won import EventPlayerWon
from lib.interface.events.typing import EventType
from lib.interface.events.event_game_started import EventGameStarted
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.events.event_player_moved import EventPlayerMoved

from pydantic import RootModel


class EventInspector:
    def __init__(
        self,
        history: list[EventType],
        ranking: list[int],
        final_masses: dict[int, float],
    ) -> None:
        self.history = history
        self.ranking = ranking
        self.final_masses = final_masses

    def get_result(
        self,
    ) -> Union[
        GameBanResult, GameSuccessResult, GameCancelledResult, GameCrashedResult
    ]:
        match self.history[-1]:
            case EventGameEndedCancelled() as e:
                return GameCancelledResult(reason=e.reason)
            case EventPlayerBanned() as e:
                return GameBanResult(
                    ban_type=e.ban_type, player=e.player_id, reason=e.reason
                )
            case EventPlayerWon():
                return GameSuccessResult(
                    ranking=self.ranking,
                    final_masses=self.final_masses,
                )
            case _:
                return GameCrashedResult(reason="Game engine crashed.")

    def get_recording_json(self) -> str:
        return RootModel(self.history).model_dump_json()

    def get_visualiser_json(self) -> str:
        visualiser_json: list[EventType] = []
        for event in self.history:
            match event:
                case EventGameStarted() as e:
                    visualiser_json.append(e)

                case MovePlayer() as e:
                    visualiser_json.append(e)

                case EventPlayerMoved() as e:
                    visualiser_json.append(e)

                case EventPlayerEaten() as e:
                    visualiser_json.append(e)

                case EventFoodSpawned() as e:
                    visualiser_json.append(e)

                case EventFoodEaten() as e:
                    visualiser_json.append(e)

                case EventVirusSpawned() as e:
                    visualiser_json.append(e)

                case EventVirusConsumed() as e:
                    visualiser_json.append(e)

                case EventPlayerWon() as e:
                    visualiser_json.append(e)

        return RootModel(visualiser_json).model_dump_json(indent=4)
