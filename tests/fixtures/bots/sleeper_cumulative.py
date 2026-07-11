"""Compliance fixture (P4.1): breaches the CUMULATIVE budget by slow drip.

Behaves like ``dont_move`` but sleeps 4 ms on every query. Over a full game
that is 1400 × (4 ms + engine tax ≈ 1.8 ms) ≈ 8.1 s — past the 8 s
``CUMULATIVE_TIMEOUT_SECONDS`` and far past the harness's margin line (6 s).
In round-capped compliance runs the ``CUM_PROJECTED`` flag must catch it.
"""

import time

from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.models.penguin_model import DirectionModel

SLEEP_PER_TURN = 0.004  # the playbook P4.1 acceptance number


def main() -> None:
    game = Game()
    while True:
        game.get_next_query()
        time.sleep(SLEEP_PER_TURN)
        game.send_move(
            MovePlayer(
                player_id=game.state.me.player_id,
                direction=DirectionModel(x=0.0, y=0.0),
            )
        )


if __name__ == "__main__":
    main()
