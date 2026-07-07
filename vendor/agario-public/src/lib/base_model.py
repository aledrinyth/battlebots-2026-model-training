from pydantic import BaseModel, ConfigDict


class FiniteBaseModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
