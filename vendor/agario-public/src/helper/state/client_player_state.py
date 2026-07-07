from lib.models.blob_model import BlobModel
from lib.models.player_model import PublicPlayerModel, PlayerModel


class ClientPlayer:
    def __init__(self, player_model: PublicPlayerModel | PlayerModel) -> None:
        self.player_id = player_model.player_id
        self.x = 0.0
        self.y = 0.0
        self.radius = 0.0
        self.alive = player_model.alive
        self.blobs: dict[int, BlobModel] = {}
        self.round_died: int = -1

        if isinstance(player_model, PlayerModel):
            self.sync_from_model(player_model)

    def sync_from_model(self, player_model: PlayerModel) -> None:
        self.x = player_model.pos[0]
        self.y = player_model.pos[1]
        self.radius = player_model.radius
        self.alive = player_model.alive
        self.blobs = {blob.blob_id: blob for blob in player_model.blobs}

    def sync_snapshot(
        self,
        *,
        pos: tuple[float, float],
        radius: float,
        alive: bool,
        blobs: tuple[BlobModel, ...],
    ) -> None:
        self.x = pos[0]
        self.y = pos[1]
        self.radius = radius
        self.alive = alive
        self.blobs = {blob.blob_id: blob for blob in blobs}
