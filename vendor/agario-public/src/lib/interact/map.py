import random

from lib.config.arena import ARENA_SIZE
from lib.models.food_model import FoodModel
from lib.models.virus_model import VirusModel


class Map:
    def __init__(self) -> None:
        self.size = ARENA_SIZE
        self.foods: list[FoodModel] = []
        self.viruses: list[VirusModel] = []
        self._next_food_id = 0
        self._next_virus_id = 0

    def spawn_food(self) -> FoodModel:
        food = FoodModel(
            food_id=self._next_food_id,
            pos=(
                random.uniform(0.0, self.size),
                random.uniform(0.0, self.size),
            ),
        )
        self._next_food_id += 1
        self.foods.append(food)
        return food

    def spawn_virus(self, pos: tuple[float, float], radius: float) -> VirusModel:
        virus = VirusModel(
            virus_id=self._next_virus_id,
            pos=pos,
            radius=radius,
        )
        self._next_virus_id += 1
        self.viruses.append(virus)
        return virus

    def remove_food(self, food_id: int) -> None:
        self.foods = [food for food in self.foods if food.food_id != food_id]
