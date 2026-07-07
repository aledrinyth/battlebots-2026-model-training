from lib.game.game_logic import GameLogic
from lib.interface.events.typing import EventType
from lib.models.blob_model import VisibleBlobModel
from lib.models.food_model import FoodModel
from lib.models.virus_model import VirusModel
from helper.state.client_player_state import ClientPlayer


class ClientMapView:
    def __init__(self) -> None:
        self.size = 0.0


class ClientSate(GameLogic):
    def __init__(self) -> None:
        self.round = -1
        self.max_rounds = 0
        self.map = ClientMapView()
        self.vision_size = 0.0
        self.view_center = (0.0, 0.0)
        self.turn_duration_seconds = 0.0
        self.total_players = 0
        self.visible_food: list[FoodModel] = []
        self.visible_blobs: list[VisibleBlobModel] = []
        self.visible_viruses: list[VirusModel] = []

        self.game_over = False
        self.winner_player_id: int | None = None

        self.event_history: list["EventType"] = []
        self.new_events: int = 0
        self.turn_order: list[int] = []
        self.rankings: list[int] = []

        self.me: ClientPlayer

    @property
    def map_size(self) -> float:
        return self.map.size
