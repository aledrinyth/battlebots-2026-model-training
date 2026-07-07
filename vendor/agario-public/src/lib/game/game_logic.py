from collections import defaultdict, deque
from typing import Callable, Iterator, Protocol


class ArenaView(Protocol):
    size: float


class SharedGameState(Protocol):
    map: ArenaView


class GameLogic(SharedGameState):
    pass
