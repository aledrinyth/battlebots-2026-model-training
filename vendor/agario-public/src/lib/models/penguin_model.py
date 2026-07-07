from pydantic import model_validator

from lib.base_model import FiniteBaseModel


class DirectionModel(FiniteBaseModel):
    x: float | None = None
    y: float | None = None
    degrees: float | None = None

    @model_validator(mode="after")
    def validate_representation(self) -> "DirectionModel":
        has_vector = self.x is not None or self.y is not None
        if has_vector and (self.x is None or self.y is None):
            raise ValueError("You must provide both 'x' and 'y' when using a vector.")
        if has_vector == (self.degrees is not None):
            raise ValueError(
                "Provide exactly one direction representation: either ('x', 'y') or 'degrees'."
            )
        return self

    def to_vector(self) -> tuple[float, float]:
        if self.degrees is not None:
            from math import cos, radians, sin

            angle = radians(self.degrees)
            return (cos(angle), sin(angle))
        assert self.x is not None
        assert self.y is not None
        return (self.x, self.y)


# Kept for backwards compatibility with older internal modules that may still import it.
class PenguinModel(FiniteBaseModel):
    player_id: int
    penguin_id: int
    pos: tuple[float, float]
