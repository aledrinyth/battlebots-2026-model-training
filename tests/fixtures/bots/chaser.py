"""Aggressive chaser reference bot (fixture).

Relentlessly pursues the nearest blob it can eat and *splits* to catch when a
half-mass child would still land the kill. Falls back to food, then to the arena
centre. Unlike ``cautious`` it presses attacks and drives prey toward walls — so
it exercises a defender's flee / virus-bait / wall-escape logic, which passive
opponents never trigger. Eat rule follows the engine (mass = radius**2, eater
mass >= EAT_SIZE_RATIO * target mass).
"""

import math

from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.events.moves.typing import MoveType
from lib.interface.queries.query_move import QueryMovePlayer
from lib.interface.queries.typing import QueryType
from lib.models.penguin_model import DirectionModel

from lib.config.player import (
    EAT_SIZE_RATIO,
    SPLIT_EJECT_DRAG,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
)

SPLIT_REACH = SPLIT_EJECT_SPEED / (1.0 - SPLIT_EJECT_DRAG)


def choose_move(game: Game) -> tuple[float, float, bool]:
    me = game.state.me
    ox, oy = me.x, me.y
    my_mass = me.radius * me.radius

    # Nearest blob we out-mass by the eat ratio.
    best = None
    best_d2 = math.inf
    for b in game.state.visible_blobs:
        if my_mass < EAT_SIZE_RATIO * b.radius * b.radius:
            continue
        dx, dy = b.pos[0] - ox, b.pos[1] - oy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = (dx, dy, math.sqrt(d2), b)
    if best is not None:
        dx, dy, dist, b = best
        split = (
            my_mass >= SPLIT_MIN_MASS
            and len(me.blobs) == 1
            and me.radius < dist <= me.radius + SPLIT_REACH
            and (my_mass * 0.5) >= EAT_SIZE_RATIO * b.radius * b.radius
        )
        return (dx, dy, split)

    if game.state.visible_food:
        f = min(
            game.state.visible_food,
            key=lambda f: (f.pos[0] - ox) ** 2 + (f.pos[1] - oy) ** 2,
        )
        return (f.pos[0] - ox, f.pos[1] - oy, False)

    return (30.0 - ox, 30.0 - oy, False)


def main() -> None:
    game = Game()

    while True:
        query = game.get_next_query()

        def build_move(query: QueryType) -> MoveType:
            match query:
                case QueryMovePlayer():
                    dx, dy, split = choose_move(game)
                    return MovePlayer(
                        player_id=game.state.me.player_id,
                        direction=DirectionModel(x=dx, y=dy),
                        split=split,
                    )
            raise RuntimeError(f"Unsupported query type: {type(query)}")

        game.send_move(build_move(query))


if __name__ == "__main__":
    main()
