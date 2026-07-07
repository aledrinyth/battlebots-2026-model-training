import json

from engine.interface.io.censor_event import CensorEvent
from engine.state.blob_state import BlobState
from engine.state.game_state import GameState
from lib.config.arena import NUM_PLAYERS
from lib.interface.events.event_player_eaten import (
    EventPlayerEaten,
    PublicEventPlayerEaten,
)
from lib.interface.events.event_player_moved import EventPlayerMoved
from lib.interface.events.moves.move_player import MovePlayer
from lib.models.penguin_model import DirectionModel


def _make_state(tmp_path, monkeypatch) -> GameState:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    with open(tmp_path / "input" / "catalog.json", "w") as file:
        json.dump([{"team_id": index} for index in range(NUM_PLAYERS)], file)
    return GameState()


def test_move_player_is_only_visible_to_the_issuing_player(
    tmp_path, monkeypatch
) -> None:
    state = _make_state(tmp_path, monkeypatch)
    censor = CensorEvent(state)
    move = MovePlayer(
        player_id=1,
        direction=DirectionModel(x=1.0, y=0.0),
        split=True,
    )

    assert censor.censor(move, player_id=1) == move
    assert censor.censor(move, player_id=0) is None


def test_player_moved_redacts_other_players_blob_ids(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    state.players[0].blobs = {0: BlobState(blob_id=0, x=20.0, y=20.0, radius=1.0)}
    state.players[1].blobs = {
        7: BlobState(blob_id=7, x=29.0, y=20.0, radius=1.0),
        11: BlobState(blob_id=11, x=29.5, y=20.0, radius=1.0),
    }
    censor = CensorEvent(state)
    moved = EventPlayerMoved(
        player_id=1,
        pos=(29.25, 20.0),
        radius=2**0.5,
        alive=True,
        blobs=tuple(blob._to_model() for blob in state.players[1].sorted_blobs()),
    )

    censored = censor.censor(moved, player_id=0)

    assert isinstance(censored, EventPlayerMoved)
    assert [blob.blob_id for blob in censored.blobs] == [0, 1]


def test_player_eaten_hides_non_visible_side_details(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    state.players[0].blobs = {0: BlobState(blob_id=0, x=20.0, y=20.0, radius=1.0)}
    censor = CensorEvent(state)
    eaten = EventPlayerEaten(
        eater_player_id=1,
        eater_blob_id=9,
        eater_pos=(31.2, 20.0),
        eaten_player_id=2,
        eaten_blob_id=4,
        eaten_pos=(29.8, 20.0),
        eater_radius=0.5,
        eaten_radius=0.5,
        eaten_player_alive=True,
    )

    censored = censor.censor(eaten, player_id=0)

    assert censored == PublicEventPlayerEaten(
        eaten_player_id=2,
        eaten_pos=(29.8, 20.0),
        eaten_radius=0.5,
    )


def test_player_eaten_is_visible_when_circles_clip_the_vision_edge(
    tmp_path, monkeypatch
) -> None:
    state = _make_state(tmp_path, monkeypatch)
    state.players[0].blobs = {0: BlobState(blob_id=0, x=20.0, y=20.0, radius=1.0)}
    censor = CensorEvent(state)
    eaten = EventPlayerEaten(
        eater_player_id=1,
        eater_blob_id=0,
        eater_pos=(30.8, 20.0),
        eaten_player_id=2,
        eaten_blob_id=0,
        eaten_pos=(30.4, 20.0),
        eater_radius=1.0,
        eaten_radius=1.0,
        eaten_player_alive=False,
    )

    censored = censor.censor(eaten, player_id=0)

    assert censored == PublicEventPlayerEaten(
        eater_player_id=1,
        eater_pos=(30.8, 20.0),
        eater_radius=1.0,
        eaten_player_id=2,
        eaten_pos=(30.4, 20.0),
        eaten_radius=1.0,
    )
