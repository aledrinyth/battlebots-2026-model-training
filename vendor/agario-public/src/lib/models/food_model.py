from lib.base_model import FiniteBaseModel


class FoodModel(FiniteBaseModel):
    food_id: int
    pos: tuple[float, float]
