import json

from engine.state.blob_state import BlobState
from engine.state.game_state import GameState
from lib.config.arena import NUM_PLAYERS
from lib.models.food_model import FoodModel


def _make_state(tmp_path, monkeypatch) -> GameState:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    with open(tmp_path / "input" / "catalog.json", "w") as file:
        json.dump([{"team_id": index} for index in range(NUM_PLAYERS)], file)
    return GameState()


def test_public_visible_entities_use_redacted_ids(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    viewer = state.players[0]
    other = state.players[1]
    for player_id, player in state.players.items():
        if player_id not in (viewer.id, other.id):
            player.blobs = {}
    viewer.blobs = {0: BlobState(blob_id=0, x=20.0, y=20.0, radius=1.0)}
    other.blobs = {
        7: BlobState(blob_id=7, x=29.0, y=20.0, radius=1.0),
        11: BlobState(blob_id=11, x=29.5, y=20.0, radius=1.0),
    }
    state.map.foods = [
        FoodModel(food_id=41, pos=(19.0, 20.0)),
        FoodModel(food_id=43, pos=(21.0, 20.0)),
    ]
    state.map.viruses = []
    state.map.spawn_virus(pos=(30.5, 20.0), radius=1.0)

    assert [food.food_id for food in state.get_public_visible_food(viewer.id)] == [0, 1]
    assert [blob.blob_id for blob in state.get_public_visible_blobs(viewer.id)] == [0, 1]
    assert [virus.virus_id for virus in state.get_public_visible_viruses(viewer.id)] == [0]
