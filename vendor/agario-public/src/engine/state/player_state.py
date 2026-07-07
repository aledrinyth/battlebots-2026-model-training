from engine.interface.io.player_connection import PlayerConnection

from engine.state.blob_state import BlobState
from lib.models.player_model import PlayerModel
from lib.config.player import STARTING_RADIUS


class PlayerState:
    def __init__(self, player_id: int, team_id: int) -> None:
        self.id = player_id
        self.team_id = team_id
        self.blobs: dict[int, BlobState] = {
            0: BlobState(
                blob_id=0,
                x=0.0,
                y=0.0,
                radius=STARTING_RADIUS,
            )
        }
        self._next_blob_id = 1
        self.round_died: int = -1
        self.respawn_at_round: int | None = None
        self.connection: PlayerConnection

    @property
    def alive(self) -> bool:
        return bool(self.blobs)

    @property
    def radius(self) -> float:
        return sum(blob.mass for blob in self.blobs.values()) ** 0.5

    @property
    def x(self) -> float:
        if not self.blobs:
            return 0.0
        total_mass = sum(blob.mass for blob in self.blobs.values())
        if total_mass == 0:
            return 0.0
        return sum(blob.x * blob.mass for blob in self.blobs.values()) / total_mass

    @property
    def y(self) -> float:
        if not self.blobs:
            return 0.0
        total_mass = sum(blob.mass for blob in self.blobs.values())
        if total_mass == 0:
            return 0.0
        return sum(blob.y * blob.mass for blob in self.blobs.values()) / total_mass

    def next_blob_id(self) -> int:
        blob_id = self._next_blob_id
        while blob_id in self.blobs:
            blob_id += 1
        self._next_blob_id = blob_id + 1
        return blob_id

    def sorted_blobs(self) -> list[BlobState]:
        return sorted(self.blobs.values(), key=lambda blob: blob.blob_id)

    def connect(self) -> None:
        self.connection = PlayerConnection(self.id)

    def _to_model(self) -> PlayerModel:
        return PlayerModel(
            player_id=self.id,
            team_id=self.team_id,
            pos=(self.x, self.y),
            radius=self.radius,
            alive=self.alive,
            blobs=tuple(blob._to_model() for blob in self.sorted_blobs()),
        )
