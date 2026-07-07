import helper.game as helper_game
from helper.client_state import ClientSate
from helper.game import Game
from helper.state.client_player_state import ClientPlayer
from helper.state_mutator import StateMutator
from lib.interface.events.event_game_started import PublicEventGameStarted
from lib.interface.events.event_player_won import EventPlayerWon
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.blob_model import BlobModel
from lib.models.food_model import FoodModel
from lib.models.player_model import PlayerModel, PublicPlayerModel
from lib.models.virus_model import VirusModel


def test_client_state_exposes_me_but_not_players() -> None:
    state = ClientSate()

    assert not hasattr(state, "players")
    assert state.map_size == 0.0
    assert not hasattr(state.map, "foods")
    assert not hasattr(state.map, "viruses")


def test_public_game_started_populates_me_and_total_players() -> None:
    state = ClientSate()
    mutator = StateMutator(state)
    event = PublicEventGameStarted(
        turn_order=[0, 1, 2, 3],
        arena_size=60.0,
        vision_size=20.0,
        turn_duration_seconds=0.1,
        max_rounds=250,
        players=[
            PublicPlayerModel(player_id=0, alive=True),
            PublicPlayerModel(player_id=1, alive=True),
            PublicPlayerModel(player_id=2, alive=True),
            PublicPlayerModel(player_id=3, alive=True),
        ],
        you=PlayerModel(
            player_id=2,
            team_id=2,
            pos=(12.0, 18.0),
            radius=3.5,
            alive=True,
            blobs=(BlobModel(blob_id=7, pos=(12.0, 18.0), radius=3.5),),
        ),
    )

    mutator.commit(0, event)

    assert state.total_players == 4
    assert state.me.player_id == 2
    assert state.me.x == 12.0
    assert state.me.y == 18.0
    assert state.me.radius == 3.5
    assert state.map_size == 60.0
    assert not hasattr(state, "players")


def test_player_won_sets_public_winner_id() -> None:
    state = ClientSate()
    mutator = StateMutator(state)

    mutator.commit(0, EventPlayerWon(player_id=1))

    assert state.game_over
    assert state.winner_player_id == 1


def test_game_syncs_rankings_from_query(monkeypatch) -> None:
    query = QueryMovePlayer(
        update={},
        round=7,
        rankings=[2, 0, 3, 1],
        you=PlayerModel(
            player_id=2,
            team_id=2,
            pos=(12.0, 18.0),
            radius=4.0,
            alive=True,
            blobs=(BlobModel(blob_id=7, pos=(12.0, 18.0), radius=4.0),),
        ),
        view_center=(12.0, 18.0),
        vision_size=20.0,
        visible_food=(FoodModel(food_id=1, pos=(10.0, 10.0)),),
        visible_blobs=(),
        visible_viruses=(VirusModel(virus_id=3, pos=(16.0, 16.0), radius=2.0),),
    )

    class FakeConnection:
        def get_next_query(self) -> QueryMovePlayer:
            return query

        def send_move(self, move: object) -> None:
            return None

    monkeypatch.setattr(helper_game, "Connection", FakeConnection)

    game = Game()
    game.state.me = ClientPlayer(query.you)
    game.get_next_query()

    assert game.state.round == 7
    assert game.state.rankings == [2, 0, 3, 1]
    assert game.state.me.player_id == 2
