# from lib.interact.tile import Tile
# from lib.config.scoring import MONASTARY_POINTS
# from lib.config.map_config import MAX_MAP_LENGTH

# from abc import ABC, abstractmethod
# from typing import Iterator, final

# MONASTARY_COUNT = 9


# class TileSubsciber(ABC):
#     """
#     TileSubsciber
#     _For special meeple claim rules. This does impy that the tile must have a meeple for a subsciber to be created_
#     """

#     @abstractmethod
#     def on_tile_changed(self, tile: "Tile") -> bool:
#         pass

#     def register_to(self, publisher: "TilePublisherBus") -> None:
#         for pos in self._watching():
#             publisher.register(pos, self)

#     @abstractmethod
#     def _watching(self) -> list[tuple[int, int]]:
#         pass

#     @abstractmethod
#     def _reward(self) -> list[tuple[int, int, Tile, str]]:
#         pass


# class MonastaryNeighbourSubsciber(TileSubsciber):
#     def __init__(
#         self, center: tuple[int, int], player_id: int, tile: "Tile", claim: str
#     ) -> None:
#         self.center = center
#         self.filled: set[Tile] = set()
#         self.registered = False
#         self.player_id = player_id
#         self.tile = tile
#         self.claim = claim
#         super().__init__()

#     @final
#     def on_tile_changed(self, tile: "Tile") -> bool:
#         if tile not in self.filled:
#             self.filled.add(tile)

#         if len(self.filled) >= MONASTARY_COUNT:
#             return True

#         return False

#     def register_to(
#         self, publisher: "TilePublisherBus", grid: list[list["Tile | None"]] = list()
#     ) -> None:
#         for x, y in self._watching():
#             assert len(grid) == MAX_MAP_LENGTH

#             tile = grid[y][x]

#             if tile is not None:
#                 self.filled.add(tile)
#         return super().register_to(publisher)

#     @final
#     def _watching(self) -> list[tuple[int, int]]:
#         x, y = self.center

#         # May be out of range, but this is okay as Publisher will never notify
#         return [
#             (x + i, y + j)
#             for i in range(-1, 2)
#             for j in range(-1, 2)
#             # if not (i == j == 0)
#         ]

#     @final
#     def _reward(self) -> list[tuple[int, int, Tile, str]]:
#         assert self.tile.placed_pos
#         return [(self.player_id, MONASTARY_POINTS, self.tile, self.claim)]


# class TilePublisherBus:
#     _singleton: "TilePublisherBus | None" = None

#     def __init__(self) -> None:
#         self.watchers: dict[tuple[int, int], list[TileSubsciber]] = {}

#     def __new__(cls) -> "TilePublisherBus":
#         if cls._singleton is None:
#             cls._singleton = super().__new__(cls)

#         return cls._singleton

#     def register(self, position: tuple[int, int], watcher: TileSubsciber) -> None:
#         self.watchers.setdefault(position, []).append(watcher)

#     def check_notify(self, tile: "Tile") -> Iterator[TileSubsciber]:
#         assert tile.placed_pos

#         current_subscribers = list(self.watchers.get(tile.placed_pos, []))

#         for subsciber in current_subscribers:
#             if subsciber.on_tile_changed(tile):
#                 for pos_watched in subsciber._watching():
#                     self.watchers[pos_watched].remove(subsciber)
#                 yield subsciber
