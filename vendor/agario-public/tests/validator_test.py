import math

import pytest
from pydantic import ValidationError

from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.blob_model import BlobModel
from lib.models.food_model import FoodModel
from lib.models.penguin_model import DirectionModel
from lib.models.player_model import PlayerModel
from lib.models.virus_model import VirusModel


@pytest.mark.parametrize(
    "payload",
    [
        '{"event_type":"move_player","player_id":0,"direction":{"x":NaN,"y":0.0},"split":false}',
        '{"event_type":"move_player","player_id":0,"direction":{"x":Infinity,"y":0.0},"split":false}',
        '{"event_type":"move_player","player_id":0,"direction":{"degrees":-Infinity},"split":false}',
    ],
)
def test_move_player_json_rejects_non_finite_direction(payload: str) -> None:
    with pytest.raises(ValidationError):
        MovePlayer.model_validate_json(payload)


@pytest.mark.parametrize(
    ("kwargs",),
    [
        ({"x": math.nan, "y": 0.0},),
        ({"x": math.inf, "y": 0.0},),
        ({"degrees": math.nan},),
    ],
)
def test_direction_model_rejects_non_finite_python_values(
    kwargs: dict[str, float]
) -> None:
    with pytest.raises(ValidationError):
        DirectionModel(**kwargs)


def test_query_move_player_rejects_non_finite_state_values() -> None:
    with pytest.raises(ValidationError):
        QueryMovePlayer(
            update={},
            round=7,
            rankings=[0, 1, 2, 3],
            you=PlayerModel(
                player_id=0,
                team_id=0,
                pos=(12.0, 18.0),
                radius=4.0,
                alive=True,
                blobs=(BlobModel(blob_id=7, pos=(12.0, 18.0), radius=4.0),),
            ),
            view_center=(math.nan, 18.0),
            vision_size=20.0,
            visible_food=(FoodModel(food_id=1, pos=(10.0, 10.0)),),
            visible_blobs=(),
            visible_viruses=(VirusModel(virus_id=3, pos=(16.0, 16.0), radius=2.0),),
        )
