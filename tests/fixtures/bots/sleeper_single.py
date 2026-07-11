"""Compliance fixture (P4.1): breaches the SINGLE-MOVE timeout once.

Behaves like ``dont_move`` except on its 3rd query, where it sleeps past the
engine's 2 s SIGALRM (``TIMEOUT_SECONDS``) before answering. The engine must
ban it (``TimeoutException``) and the compliance harness must flag it.
"""

import time

from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.models.penguin_model import DirectionModel

SLEEP_ON_TURN = 3
SLEEP_SECONDS = 3.0  # > TIMEOUT_SECONDS (2 s)


def main() -> None:
    game = Game()
    turn = 0
    while True:
        game.get_next_query()
        turn += 1
        if turn == SLEEP_ON_TURN:
            time.sleep(SLEEP_SECONDS)
        game.send_move(
            MovePlayer(
                player_id=game.state.me.player_id,
                direction=DirectionModel(x=0.0, y=0.0),
            )
        )


if __name__ == "__main__":
    main()
