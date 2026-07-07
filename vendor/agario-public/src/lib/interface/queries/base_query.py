from lib.interface.events.typing import EventType
from lib.base_model import FiniteBaseModel

from typing import Mapping


class BaseQuery(FiniteBaseModel):
    query_type: str
    update: Mapping[int, EventType]
