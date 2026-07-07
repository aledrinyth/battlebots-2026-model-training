from lib.base_model import FiniteBaseModel


class BaseEvent(FiniteBaseModel):
    event_type: str
