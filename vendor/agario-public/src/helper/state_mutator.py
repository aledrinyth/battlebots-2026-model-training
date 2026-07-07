from helper.client_state import ClientSate
from helper.state.client_player_state import ClientPlayer

from lib.interface.events.event_player_bannned import EventPlayerBanned
from lib.interface.events.event_player_eaten import (
    EventPlayerEaten,
    PublicEventPlayerEaten,
)
from lib.interface.events.event_player_moved import EventPlayerMoved
from lib.interface.events.event_player_won import EventPlayerWon
from lib.interface.events.event_game_ended import (
    EventGameEndedCancelled,
)
from lib.interface.events.event_game_started import (
    PublicEventGameStarted,
)
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.events.typing import EventType


import logging

# Configure logging to display INFO messages and write to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    filename='app1.log',
    filemode='w' # Use 'w' to overwrite each time
)


class StateMutator:
    def __init__(self, state: ClientSate) -> None:
        self.state = state

    def commit(self, i: int, event: EventType) -> None:
        self.state.event_history.append(event)

        match event:
            case PublicEventGameStarted() as e:
                self.state.turn_order = e.turn_order
                self.state.map.size = e.arena_size
                self.state.vision_size = e.vision_size
                self.state.view_center = e.you.pos
                self.state.turn_duration_seconds = e.turn_duration_seconds
                self.state.max_rounds = e.max_rounds
                self.state.total_players = len(e.players)
                self.state.game_over = False
                self.state.winner_player_id = None
                self.state.me = ClientPlayer(e.you)

            case EventGameEndedCancelled() as e:
                self.state.game_over = True

            case EventPlayerBanned() as e:
                self.state.game_over = True

            case EventPlayerWon() as e:
                self.state.game_over = True
                self.state.winner_player_id = e.player_id

            case EventPlayerMoved() | EventPlayerEaten() | PublicEventPlayerEaten():
                pass

            case MovePlayer() as e:
                pass

            case _:
                raise RuntimeError(f"Unrecognised event: {event}")
