from typing import final
import math

from lib.models.penguin_model import PenguinModel
from lib.config.player import MAX_MAGNITUDE

class Direction:
    def __init__(self, x: float, y: float, magnitude: float):
        self.dx = x
        self.dy = y
        self.magnitude = magnitude

    def get_normalised(self) -> tuple[float, float]:
        self.magnitude = min(self.magnitude, MAX_MAGNITUDE)
        hypotenuse = math.sqrt(self.dx * self.dx + self.dy * self.dy)
        if hypotenuse == 0:
            return (0, 0)
        return (self.dx / hypotenuse * self.magnitude, self.dy / hypotenuse * self.magnitude)

class Penguin:
    def __init__(self, player_id: int, penguin_id: int, starting_position: tuple[float, float]) -> None:
        self.player_id = player_id
        self.penguin_id = penguin_id
        self.alive = True
        self.x: float = starting_position[0]
        self.y: float = starting_position[1]
        self.vx: float = 0
        self.vy: float = 0


    def __str__(self) -> str:
        return(f"Penguin: id=({self.player_id}, {self.penguin_id}) pos=({self.x}, {self.y}) v=({self.vx}, {self.vy})")
    
    def stop(self) -> None:
        self.vx = 0
        self.vy = 0
    
    @classmethod
    def from_model(cls, penguin_model: "PenguinModel") -> "Penguin":
        return cls(
            penguin_model.player_id,
            penguin_model.penguin_id,
            penguin_model.pos,
        )
    
    @final
    def _to_model(self) -> PenguinModel:
        return PenguinModel(
            player_id=self.player_id,
            penguin_id=self.penguin_id,
            pos=(self.x, self.y)
        )