from typing import Literal, Sequence

from lib.interface.events.base_event import BaseEvent
from lib.models.virus_model import VirusModel


class EventVirusSpawned(BaseEvent):
    event_type: Literal["event_virus_spawned"] = "event_virus_spawned"
    viruses: Sequence[VirusModel]
