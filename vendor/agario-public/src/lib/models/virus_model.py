from lib.base_model import FiniteBaseModel


class VirusModel(FiniteBaseModel):
    virus_id: int
    pos: tuple[float, float]
    radius: float
