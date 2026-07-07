import json
import os
from pathlib import Path
import tempfile
import unittest

from engine.state.game_state import GameState
# from engine.game.tile_subscriber import MonastaryNeighbourSubsciber, TilePublisherBus

from lib.config.arena import NUM_PLAYERS
from lib.interact.map import Map


class TestMoveDown(unittest.TestCase):
    def setUp(self) -> None:
        self._cwd = Path.cwd()
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmpdir.name)
        os.chdir(tmp_path)
        (tmp_path / "input").mkdir()
        (tmp_path / "output").mkdir()
        with open(tmp_path / "input" / "catalog.json", "w") as file:
            json.dump([{"team_id": index} for index in range(NUM_PLAYERS)], file)
        self.state = GameState()
        self.state.map = Map()

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._tmpdir.cleanup()
    
    # def test_connect_to_player(self):
    #     pass # To do later

    def test_move_down(self):
        pass # to implement after first test passes
        
# class TestPlaceTile(unittest.TestCase):
#     def setUp(self) -> None:
#         self.state = GameState()
#         self.state.map = Map()

#         # Helper for moving in grid
#         self.DIRECTION_OFFSET = {
#             "top_edge": (0, -1),
#             "bottom_edge": (0, 1),
#             "left_edge": (-1, 0),
#             "right_edge": (1, 0),
#         }

#         # Helper for finding opposite edge
#         self.OPPOSITE_EDGE = {
#             "top_edge": "bottom_edge",
#             "bottom_edge": "top_edge",
#             "left_edge": "right_edge",
#             "right_edge": "left_edge",
#         }

#     def tearDown(self) -> None:
#         return super().tearDown()

#     def test_basic_road_traversal(self) -> None:
#         self.state.start_base_phase()
#         U1 = self.state.map.get_tile_by_type("U", pop=True)
#         U2 = self.state.map.get_tile_by_type("U", pop=True)
#         U3 = self.state.map.get_tile_by_type("U", pop=True)

#         pos1 = (85, 85)
#         pos2 = (85, 84)
#         pos3 = (85, 83)

#         U1.placed_pos = pos1
#         U2.placed_pos = pos2
#         U3.placed_pos = pos3

#         self.state.map._grid[pos1[1]][pos1[0]] = U1
#         self.state.map._grid[pos2[1]][pos2[0]] = U2
#         self.state.map._grid[pos3[1]][pos3[0]] = U3

#         # We want all edges of all tiles
#         assert set(self.state._traverse_connected_component(U1, "top_edge")) == set(
#             [
#                 (U1, "top_edge"),
#                 (U2, "top_edge"),
#                 (U3, "top_edge"),
#                 (U1, "bottom_edge"),
#                 (U2, "bottom_edge"),
#                 (U3, "bottom_edge"),
#             ]
#         )

#     def test_basic_city_traversal(self) -> None:
#         self.state.start_base_phase()
#         E1 = self.state.map.get_tile_by_type("E", pop=True)
#         e1_pos = (85, 85)
#         E1.placed_pos = e1_pos
#         E1.rotate_clockwise(1)
#         self.state.map._grid[e1_pos[1]][e1_pos[0]] = E1
#         assert not self.state.get_completed_components(E1)

#         E2 = self.state.map.get_tile_by_type("E", pop=True)
#         e2_pos = (87, 85)
#         E2.placed_pos = e2_pos
#         E2.rotate_clockwise(3)
#         self.state.map._grid[e2_pos[1]][e2_pos[0]] = E2
#         assert not self.state.get_completed_components(E1)

#         F1 = self.state.map.get_tile_by_type("F", pop=True)
#         f1_pos = (86, 85)
#         F1.placed_pos = f1_pos
#         self.state.map._grid[f1_pos[1]][f1_pos[0]] = F1
#         assert self.state.get_completed_components(E1)

#         assert set(self.state._traverse_connected_component(E1, "right_edge")) == set(
#             [
#                 (F1, "right_edge"),
#                 (F1, "left_edge"),
#                 (E1, "right_edge"),
#                 (E2, "left_edge"),
#             ]
#         )

#         assert self.state._get_reward(E1, "right_edge") == 8

#     def test_basic_monastary(self) -> None:
#         self.state.start_base_phase()
#         self.publisher = TilePublisherBus()

#         E1 = self.state.map.get_tile_by_type("E", pop=True)
#         e1_pos = (85, 84)
#         E1.placed_pos = e1_pos
#         self.state.map._grid[e1_pos[1]][e1_pos[0]] = E1

#         E2 = self.state.map.get_tile_by_type("E", pop=True)
#         e2_pos = (86, 85)
#         E2.placed_pos = e2_pos
#         E2.rotate_clockwise(1)
#         self.state.map._grid[e2_pos[1]][e2_pos[0]] = E2

#         E3 = self.state.map.get_tile_by_type("E", pop=True)
#         e3_pos = (85, 86)
#         E3.placed_pos = e3_pos
#         E3.rotate_clockwise(2)
#         self.state.map._grid[e3_pos[1]][e3_pos[0]] = E3

#         E4 = self.state.map.get_tile_by_type("E", pop=True)
#         e4_pos = (84, 85)
#         E4.placed_pos = e4_pos
#         E4.rotate_clockwise(3)
#         self.state.map._grid[e4_pos[1]][e4_pos[0]] = E4

#         V1 = self.state.map.get_tile_by_type("V", pop=True)
#         v1_pos = (84, 84)
#         V1.placed_pos = v1_pos
#         V1.rotate_clockwise(1)
#         self.state.map._grid[v1_pos[1]][v1_pos[0]] = V1

#         V2 = self.state.map.get_tile_by_type("V", pop=True)
#         v2_pos = (86, 84)
#         V2.placed_pos = v2_pos
#         V2.rotate_clockwise(2)
#         self.state.map._grid[v2_pos[1]][v2_pos[0]] = V2

#         V3 = self.state.map.get_tile_by_type("V", pop=True)
#         v3_pos = (86, 86)
#         V3.placed_pos = v3_pos
#         V3.rotate_clockwise(3)
#         self.state.map._grid[v3_pos[1]][v3_pos[0]] = V3

#         V4 = self.state.map.get_tile_by_type("V", pop=True)
#         v4_pos = (84, 86)
#         V4.placed_pos = v4_pos
#         self.state.map._grid[v4_pos[1]][v4_pos[0]] = V4

#         # Mimic MoveMeeplePlace
#         B = self.state.map.get_tile_by_type("B", pop=True)
#         b_pos = (85, 85)
#         B.placed_pos = b_pos
#         self.state.map._grid[b_pos[1]][b_pos[0]] = B

#         s = MonastaryNeighbourSubsciber(b_pos, 0, B, MONASTARY_IDENTIFIER)
#         s.register_to(self.publisher, grid=self.state.map._grid)

#         subscribers = list(self.publisher.check_notify(B))
#         assert len(subscribers) == 1

#         assert subscribers[0]._reward() == [(0, 9, B, MONASTARY_IDENTIFIER)]

#         subscribers = list(self.publisher.check_notify(B))
#         assert len(subscribers) == 0

#     def test_complex_monastary(self) -> None:
#         self.state.start_base_phase()
#         self.publisher = TilePublisherBus()

#         E1 = self.state.map.get_tile_by_type("E", pop=True)
#         e1_pos = (85, 84)
#         E1.placed_pos = e1_pos
#         self.state.map._grid[e1_pos[1]][e1_pos[0]] = E1

#         E2 = self.state.map.get_tile_by_type("E", pop=True)
#         e2_pos = (86, 85)
#         E2.placed_pos = e2_pos
#         E2.rotate_clockwise(1)
#         self.state.map._grid[e2_pos[1]][e2_pos[0]] = E2

#         E3 = self.state.map.get_tile_by_type("E", pop=True)
#         e3_pos = (85, 86)
#         E3.placed_pos = e3_pos
#         E3.rotate_clockwise(2)
#         self.state.map._grid[e3_pos[1]][e3_pos[0]] = E3

#         # Mimic MoveMeeplePlace
#         B1 = self.state.map.get_tile_by_type("B", pop=True)
#         b1_pos = (84, 85)
#         B1.placed_pos = b1_pos
#         self.state.map._grid[b1_pos[1]][b1_pos[0]] = B1

#         s = MonastaryNeighbourSubsciber(b1_pos, 1, B1, MONASTARY_IDENTIFIER)
#         s.register_to(self.publisher, grid=self.state.map._grid)

#         V1 = self.state.map.get_tile_by_type("V", pop=True)
#         v1_pos = (84, 84)
#         V1.placed_pos = v1_pos
#         V1.rotate_clockwise(1)
#         self.state.map._grid[v1_pos[1]][v1_pos[0]] = V1
#         subscribers = list(self.publisher.check_notify(V1))
#         assert len(subscribers) == 0

#         V2 = self.state.map.get_tile_by_type("V", pop=True)
#         v2_pos = (86, 84)
#         V2.placed_pos = v2_pos
#         V2.rotate_clockwise(2)
#         self.state.map._grid[v2_pos[1]][v2_pos[0]] = V2
#         subscribers = list(self.publisher.check_notify(V2))
#         assert len(subscribers) == 0

#         V3 = self.state.map.get_tile_by_type("V", pop=True)
#         v3_pos = (86, 86)
#         V3.placed_pos = v3_pos
#         V3.rotate_clockwise(3)
#         self.state.map._grid[v3_pos[1]][v3_pos[0]] = V3
#         subscribers = list(self.publisher.check_notify(V3))
#         assert len(subscribers) == 0

#         V4 = self.state.map.get_tile_by_type("V", pop=True)
#         v4_pos = (84, 86)
#         V4.placed_pos = v4_pos
#         self.state.map._grid[v4_pos[1]][v4_pos[0]] = V4
#         subscribers = list(self.publisher.check_notify(V4))
#         assert len(subscribers) == 0

#         V5 = self.state.map.get_tile_by_type("V", pop=True)
#         v5_pos = (83, 86)
#         V5.placed_pos = v5_pos
#         V5.rotate_clockwise(2)
#         self.state.map._grid[v5_pos[1]][v5_pos[0]] = V5
#         subscribers = list(self.publisher.check_notify(V5))
#         assert len(subscribers) == 0

#         U1 = self.state.map.get_tile_by_type("U", pop=True)
#         u1_pos = (83, 85)
#         U1.placed_pos = u1_pos
#         self.state.map._grid[u1_pos[1]][u1_pos[0]] = U1
#         subscribers = list(self.publisher.check_notify(U1))
#         assert len(subscribers) == 0

#         P1 = self.state.map.get_tile_by_type("U", pop=True)
#         p1_pos = (83, 84)
#         P1.placed_pos = p1_pos
#         self.state.map._grid[p1_pos[1]][p1_pos[0]] = P1
#         subscribers = list(self.publisher.check_notify(P1))
#         assert len(subscribers) == 0

#         # Mimic MoveMeeplePlace
#         B2 = self.state.map.get_tile_by_type("B", pop=True)
#         b2_pos = (85, 85)
#         B2.placed_pos = b2_pos
#         self.state.map._grid[b2_pos[1]][b2_pos[0]] = B2
#         subscribers = list(self.publisher.check_notify(B2))
#         assert len(subscribers) == 1
#         assert subscribers[0]._reward() == [(1, 9, B1, MONASTARY_IDENTIFIER)]

#         s = MonastaryNeighbourSubsciber(b2_pos, 0, B2, MONASTARY_IDENTIFIER)
#         s.register_to(self.publisher, grid=self.state.map._grid)

#         subscribers = list(self.publisher.check_notify(B2))
#         assert len(subscribers) == 1

#         assert subscribers[0]._reward() == [(0, 9, B2, MONASTARY_IDENTIFIER)]

#         subscribers = list(self.publisher.check_notify(B2))
#         assert len(subscribers) == 0

#     def test_complex_road_traversal(self) -> None:
#         self.state.start_base_phase()
#         start = self.state.map.get_tile_by_type("A", pop=True)
#         start.rotate_clockwise(2)
#         start.placed_pos = (85, 85)
#         self.state.map._grid[85][85] = start

#         expected_set: set[tuple["Tile", str]] = set()
#         expected_set.add((start, "top_edge"))

#         # K1
#         K1 = self.state.map.get_tile_by_type("K", pop=True)
#         k1_pos = (85, 84)
#         K1.rotate_clockwise(2)
#         K1.placed_pos = k1_pos
#         self.state.map._grid[k1_pos[1]][k1_pos[0]] = K1
#         expected_set.add((K1, "bottom_edge"))
#         expected_set.add((K1, "right_edge"))
#         assert not self.state.get_completed_components(start)

#         # K2
#         K2 = self.state.map.get_tile_by_type("K", pop=True)
#         k2_pos = (86, 84)
#         K2.placed_pos = k2_pos
#         self.state.map._grid[k2_pos[1]][k2_pos[0]] = K2
#         expected_set.add((K2, "left_edge"))
#         expected_set.add((K2, "top_edge"))
#         assert not self.state.get_completed_components(start)

#         # U1
#         U1 = self.state.map.get_tile_by_type("U", pop=True)
#         u1_pos = (86, 83)
#         U1.placed_pos = u1_pos
#         self.state.map._grid[u1_pos[1]][u1_pos[0]] = U1
#         expected_set.add((U1, "bottom_edge"))
#         expected_set.add((U1, "top_edge"))
#         assert not self.state.get_completed_components(start)

#         # P1
#         P1 = self.state.map.get_tile_by_type("P", pop=True)
#         p1_pos = (86, 82)
#         P1.placed_pos = p1_pos
#         self.state.map._grid[p1_pos[1]][p1_pos[0]] = P1
#         expected_set.add((P1, "bottom_edge"))
#         expected_set.add((P1, "right_edge"))
#         assert not self.state.get_completed_components(start)

#         # U2
#         U2 = self.state.map.get_tile_by_type("U", pop=True)
#         u2_pos = (87, 82)
#         U2.rotate_clockwise(1)
#         U2.placed_pos = u2_pos
#         self.state.map._grid[u2_pos[1]][u2_pos[0]] = U2
#         expected_set.add((U2, "right_edge"))
#         expected_set.add((U2, "left_edge"))
#         assert not self.state.get_completed_components(start)

#         # J1
#         J1 = self.state.map.get_tile_by_type("J", pop=True)
#         j1_pos = (88, 82)
#         J1.rotate_clockwise(2)
#         J1.placed_pos = j1_pos
#         self.state.map._grid[j1_pos[1]][j1_pos[0]] = J1
#         expected_set.add((J1, "left_edge"))
#         expected_set.add((J1, "top_edge"))
#         assert not self.state.get_completed_components(start)

#         # U3
#         U3 = self.state.map.get_tile_by_type("U", pop=True)
#         u3_pos = (88, 81)
#         U3.placed_pos = u3_pos
#         self.state.map._grid[u3_pos[1]][u3_pos[0]] = U3
#         expected_set.add((U3, "bottom_edge"))
#         expected_set.add((U3, "top_edge"))
#         assert not self.state.get_completed_components(start)

#         # J2
#         J2 = self.state.map.get_tile_by_type("J", pop=True)
#         j2_pos = (88, 80)
#         J2.placed_pos = j2_pos
#         self.state.map._grid[j2_pos[1]][j2_pos[0]] = J2
#         expected_set.add((J2, "bottom_edge"))
#         expected_set.add((J2, "right_edge"))
#         assert not self.state.get_completed_components(start)

#         # U4
#         U4 = self.state.map.get_tile_by_type("U", pop=True)
#         u4_pos = (89, 80)
#         U4.rotate_clockwise(1)
#         U4.placed_pos = u4_pos
#         self.state.map._grid[u4_pos[1]][u4_pos[0]] = U4
#         expected_set.add((U4, "right_edge"))
#         expected_set.add((U4, "left_edge"))
#         assert not self.state.get_completed_components(start)

#         # End tile
#         end = self.state.map.get_tile_by_type("A", pop=True)
#         end_pos = (90, 80)
#         end.rotate_clockwise(1)
#         end.placed_pos = end_pos
#         self.state.map._grid[end_pos[1]][end_pos[0]] = end
#         expected_set.add((end, "left_edge"))
#         assert self.state.get_completed_components(start)

#         result = self.state._traverse_connected_component(start, "top_edge")

#         self.assertEqual(set(result), expected_set)

#     def test_complex_city_traversal(self) -> None:
#         self.state.start_base_phase()

#         expected_set: set[tuple["Tile", str]] = set()

#         C1 = self.state.map.get_tile_by_type("C", pop=True)
#         c1_pos = (85, 85)
#         C1.placed_pos = c1_pos
#         self.state.map._grid[c1_pos[1]][c1_pos[0]] = C1
#         expected_set.add((C1, "left_edge"))
#         expected_set.add((C1, "right_edge"))
#         expected_set.add((C1, "bottom_edge"))
#         expected_set.add((C1, "top_edge"))
#         assert not self.state.get_completed_components(C1)

#         M1 = self.state.map.get_tile_by_type("M", pop=True)
#         m1_pos = (86, 85)
#         M1.placed_pos = m1_pos
#         self.state.map._grid[m1_pos[1]][m1_pos[0]] = M1
#         expected_set.add((M1, "top_edge"))
#         expected_set.add((M1, "left_edge"))
#         assert not self.state.get_completed_components(C1)

#         N1 = self.state.map.get_tile_by_type("N", pop=True)
#         n1_pos = (86, 84)
#         N1.placed_pos = n1_pos
#         N1.rotate_clockwise(2)
#         self.state.map._grid[n1_pos[1]][n1_pos[0]] = N1
#         expected_set.add((N1, "bottom_edge"))
#         expected_set.add((N1, "right_edge"))
#         assert not self.state.get_completed_components(C1)

#         E1 = self.state.map.get_tile_by_type("E", pop=True)
#         e1_pos = (87, 84)
#         E1.placed_pos = e1_pos
#         E1.rotate_clockwise(3)
#         self.state.map._grid[e1_pos[1]][e1_pos[0]] = E1
#         expected_set.add((E1, "left_edge"))
#         assert not self.state.get_completed_components(C1)

#         E2 = self.state.map.get_tile_by_type("E", pop=True)
#         e2_pos = (85, 84)
#         E2.placed_pos = e2_pos
#         E2.rotate_clockwise(2)
#         self.state.map._grid[e2_pos[1]][e2_pos[0]] = E2
#         expected_set.add((E2, "bottom_edge"))
#         assert not self.state.get_completed_components(C1)

#         E3 = self.state.map.get_tile_by_type("E", pop=True)
#         e3_pos = (84, 85)
#         E3.placed_pos = e3_pos
#         E3.rotate_clockwise(1)
#         self.state.map._grid[e3_pos[1]][e3_pos[0]] = E3
#         expected_set.add((E3, "right_edge"))
#         assert not self.state.get_completed_components(C1)

#         I1 = self.state.map.get_tile_by_type("I", pop=True)
#         i1_pos = (85, 86)
#         I1.placed_pos = i1_pos
#         I1.rotate_clockwise(2)
#         self.state.map._grid[i1_pos[1]][i1_pos[0]] = I1
#         expected_set.add((I1, "top_edge"))
#         # print_map(self.state.map._grid, range(75, 96))
#         assert self.state.get_completed_components(C1)

#         result = set(self.state._traverse_connected_component(C1, "right_edge"))
#         self.assertEqual(result, expected_set)

#         result = self.state.get_completed_components(C1)
#         assert len(result) == 1

#         EXPECTED_REWARD = 18
#         EXPECTED_REWARD_NO_MODS = 14

#         for connected_component in result.values():
#             assert len(set([t for t, _ in connected_component])) == 7

#             reward = StructureType.get_points(C1.internal_edges["right_edge"]) * len(
#                 set([t for t, _ in connected_component])
#             )

#             assert reward == EXPECTED_REWARD_NO_MODS

#             for t in set([t for t, _ in connected_component]):
#                 reward = TileModifier.apply_point_modifiers(
#                     t, StructureType.CITY, reward
#                 )

#             assert reward == EXPECTED_REWARD

#         reward = self.state._get_reward(C1, "right_edge")
#         self.assertEqual(reward, EXPECTED_REWARD)
