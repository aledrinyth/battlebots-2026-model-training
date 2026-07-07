import json

from engine.interface.io.censor_event import CensorEvent
from engine.interface.io.player_connection import PlayerConnection
from engine.state.game_state import GameState
from lib.config.arena import NUM_PLAYERS
from lib.interface.events.moves.move_player import MovePlayer
from lib.models.penguin_model import DirectionModel


def _make_state(tmp_path, monkeypatch) -> GameState:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    with open(tmp_path / "input" / "catalog.json", "w") as file:
        json.dump([{"team_id": index} for index in range(NUM_PLAYERS)], file)
    return GameState()


def test_record_updates_use_dense_public_indices(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PlayerConnection, "_open_pipes", lambda self: None)
    state = _make_state(tmp_path, monkeypatch)
    state.event_history = [
        MovePlayer(
            player_id=1,
            direction=DirectionModel(x=1.0, y=0.0),
            split=False,
        ),
        MovePlayer(
            player_id=0,
            direction=DirectionModel(x=0.0, y=1.0),
            split=False,
        ),
    ]
    connection = PlayerConnection(player_id=0)

    update = connection._get_record_update_dict(state, CensorEvent(state))

    assert list(update) == [0]
    assert update[0] == state.event_history[1]
