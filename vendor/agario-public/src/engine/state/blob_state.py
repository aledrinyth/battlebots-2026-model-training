from lib.models.blob_model import BlobModel, VisibleBlobModel


class BlobState:
    def __init__(
        self,
        blob_id: int,
        x: float,
        y: float,
        radius: float,
        merge_cooldown: int = 0,
        eject_vx: float = 0.0,
        eject_vy: float = 0.0,
    ) -> None:
        self.blob_id = blob_id
        self.x = x
        self.y = y
        self.radius = radius
        self.merge_cooldown = merge_cooldown
        self.eject_vx = eject_vx
        self.eject_vy = eject_vy

    @property
    def mass(self) -> float:
        return self.radius * self.radius

    def _to_model(self) -> BlobModel:
        return BlobModel(
            blob_id=self.blob_id,
            pos=(self.x, self.y),
            radius=self.radius,
            merge_cooldown=self.merge_cooldown,
        )

    def _to_visible_model(self, player_id: int, team_id: int) -> VisibleBlobModel:
        return VisibleBlobModel(
            player_id=player_id,
            team_id=team_id,
            blob_id=self.blob_id,
            pos=(self.x, self.y),
            radius=self.radius,
            merge_cooldown=self.merge_cooldown,
        )
