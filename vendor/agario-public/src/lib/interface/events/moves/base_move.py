from lib.interface.events.base_event import BaseEvent


class BaseMove(BaseEvent):
    player_id: int
