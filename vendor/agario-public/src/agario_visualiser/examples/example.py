from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.events.moves.typing import MoveType
from lib.interface.queries.query_move import QueryMovePlayer
from lib.interface.queries.typing import QueryType
from lib.models.penguin_model import DirectionModel


def choose_direction(game: Game) -> tuple[float, float]:
    center_x = game.state.me.x
    center_y = game.state.me.y
    my_radius = game.state.me.radius

    threats = [
        (blob.pos[0] - center_x, blob.pos[1] - center_y)
        for blob in game.state.visible_blobs
        if blob.radius > my_radius * 1.1
    ]
    if threats:
        nearest = min(threats, key=lambda pos: pos[0] * pos[0] + pos[1] * pos[1])
        return (-nearest[0], -nearest[1])

    prey = [
        (blob.pos[0] - center_x, blob.pos[1] - center_y)
        for blob in game.state.visible_blobs
        if my_radius > blob.radius * 1.1
    ]
    if prey:
        return min(prey, key=lambda pos: pos[0] * pos[0] + pos[1] * pos[1])

    if game.state.visible_food:
        return min(
            [
                (food.pos[0] - center_x, food.pos[1] - center_y)
                for food in game.state.visible_food
            ],
            key=lambda pos: pos[0] * pos[0] + pos[1] * pos[1],
        )

    return (0.0, 0.0)


def main() -> None:
    game = Game()

    while True:
        query = game.get_next_query()

        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryMovePlayer():
                    dx, dy = choose_direction(game)
                    return MovePlayer(
                        player_id=game.state.me.player_id,
                        direction=DirectionModel(x=dx, y=dy),
                    )
            raise RuntimeError(f"Unsupported query type: {type(query)}")

        game.send_move(choose_move(query))


if __name__ == "__main__":
    main()
