import math
import json

from engine.state.blob_state import BlobState
from engine.state.game_state import GameState
from engine.state.state_mutator import StateMutator
from lib.config.arena import (
    MAX_BLOB_COUNT,
    NUM_PLAYERS,
    VISION_REFERENCE_SUM_OF_RADII,
    VISION_SIZE,
    VIRUS_COUNT,
)
from lib.config.player import SAME_PLAYER_OVERLAP_EPSILON, SPLIT_MIN_MASS
from lib.interface.events.moves.move_player import MovePlayer
from lib.models.food_model import FoodModel
from lib.models.penguin_model import DirectionModel
import engine.state.state_mutator as state_mutator_module


def _make_state(tmp_path, monkeypatch) -> GameState:
    monkeypatch.chdir(tmp_path)
    # Disable mass decay for tests that check mass conservation
    monkeypatch.setattr(state_mutator_module, "MASS_DECAY_RATE", 0.0)
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    with open(tmp_path / "input" / "catalog.json", "w") as file:
        json.dump([{"team_id": index} for index in range(NUM_PLAYERS)], file)
    return GameState()


def _is_overlapping(blob_a: BlobState, blob_b: BlobState) -> bool:
    distance = math.hypot(blob_a.x - blob_b.x, blob_a.y - blob_b.y)
    return distance + SAME_PLAYER_OVERLAP_EPSILON < blob_a.radius + blob_b.radius


def test_split_hits_each_eligible_blob_and_conserves_mass(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    mutator = StateMutator(state)
    player = state.players[0]
    state.map.foods = []
    state.map.viruses = []

    for other_id, other in state.players.items():
        if other_id != player.id:
            other.blobs.clear()

    eligible_mass = SPLIT_MIN_MASS + 1.0
    player.blobs = {
        0: BlobState(blob_id=0, x=10.0, y=10.0, radius=math.sqrt(eligible_mass)),
        1: BlobState(blob_id=1, x=14.0, y=10.0, radius=math.sqrt(eligible_mass)),
        2: BlobState(
            blob_id=2,
            x=18.0,
            y=10.0,
            radius=math.sqrt(SPLIT_MIN_MASS) - 0.05,
        ),
    }

    total_mass_before = sum(blob.mass for blob in player.blobs.values())
    mutator.commit_round(
        [
            MovePlayer(
                player_id=player.id,
                direction=DirectionModel(x=1.0, y=0.0),
                split=True,
            )
        ]
    )

    assert len(player.blobs) == 5
    total_mass_after = sum(blob.mass for blob in player.blobs.values())
    assert math.isclose(total_mass_before, total_mass_after, rel_tol=1e-9)

    split_children = [blob for blob in player.blobs.values() if blob.blob_id >= 3]
    assert len(split_children) == 2
    for child in split_children:
        assert child.merge_cooldown > 0


def test_same_player_cooldown_blocks_merge_until_ready(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    mutator = StateMutator(state)
    player = state.players[0]
    state.map.foods = []
    state.map.viruses = []

    for other_id, other in state.players.items():
        if other_id != player.id:
            other.blobs.clear()

    player.blobs = {
        0: BlobState(
            blob_id=0,
            x=20.0,
            y=20.0,
            radius=1.0,
            merge_cooldown=3,
        ),
        1: BlobState(
            blob_id=1,
            x=21.5,
            y=20.0,
            radius=1.0,
            merge_cooldown=3,
        ),
    }

    mutator.commit_round(
        [
            MovePlayer(
                player_id=player.id,
                direction=DirectionModel(x=0.0, y=0.0),
            )
        ]
    )

    assert len(player.blobs) == 2
    blob_a, blob_b = player.sorted_blobs()
    assert not _is_overlapping(blob_a, blob_b)

    for blob in player.blobs.values():
        blob.merge_cooldown = 0
    state.map.foods = []
    player.blobs[0].x = 20.0
    player.blobs[0].y = 20.0
    player.blobs[1].x = 21.5
    player.blobs[1].y = 20.0

    mutator.commit_round(
        [
            MovePlayer(
                player_id=player.id,
                direction=DirectionModel(x=0.0, y=0.0),
            )
        ]
    )

    assert len(player.blobs) == 1
    merged_blob = next(iter(player.blobs.values()))
    assert math.isclose(merged_blob.radius, math.sqrt(2.0), rel_tol=1e-9)


def test_subnormal_direction_moves_at_normal_speed(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    mutator = StateMutator(state)
    player = state.players[0]
    state.map.foods = []
    state.map.viruses = []

    for other_id, other in state.players.items():
        if other_id != player.id:
            other.blobs.clear()

    start_x = 20.0
    start_y = 20.0
    player.blobs = {
        0: BlobState(blob_id=0, x=start_x, y=start_y, radius=2.0),
    }

    speed = mutator._movement_speed(player.blobs[0].radius)
    mutator.commit_round(
        [
            MovePlayer(
                player_id=player.id,
                direction=DirectionModel(x=5e-324, y=5e-324),
            )
        ]
    )

    moved_blob = player.blobs[0]
    delta_x = moved_blob.x - start_x
    delta_y = moved_blob.y - start_y
    expected_component = speed / math.sqrt(2.0)

    assert math.isclose(math.hypot(delta_x, delta_y), speed, rel_tol=1e-9)
    assert math.isclose(delta_x, expected_component, rel_tol=1e-9)
    assert math.isclose(delta_y, expected_component, rel_tol=1e-9)


def test_view_center_and_visibility_use_player_centroid(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    player = state.players[0]
    state.map.viruses = []

    player.blobs = {
        0: BlobState(blob_id=0, x=10.0, y=10.0, radius=1.0),
        1: BlobState(blob_id=1, x=30.0, y=10.0, radius=1.0),
    }

    assert state.get_player_view_center(player.id) == (20.0, 10.0)
    assert math.isclose(state.get_player_vision_size(player.id), VISION_SIZE, rel_tol=1e-9)

    assert state.player_can_see_point(player.id, 23.5, 10.0)
    assert not state.player_can_see_point(player.id, 9.9, 10.0)


def test_view_center_is_clipped_to_keep_vision_inside_arena(
    tmp_path, monkeypatch
) -> None:
    state = _make_state(tmp_path, monkeypatch)
    player = state.players[0]
    state.map.viruses = []

    player.blobs = {
        0: BlobState(blob_id=0, x=1.0, y=1.0, radius=1.0),
    }
    assert state.get_player_view_center(player.id) == (10.0, 10.0)

    player.blobs = {
        0: BlobState(blob_id=0, x=59.0, y=59.0, radius=1.0),
    }
    assert state.get_player_view_center(player.id) == (50.0, 50.0)


def test_blob_and_virus_become_visible_when_edges_enter_vision(
    tmp_path, monkeypatch
) -> None:
    state = _make_state(tmp_path, monkeypatch)
    viewer = state.players[0]
    other = state.players[1]
    state.map.foods = []
    state.map.viruses = []

    viewer.blobs = {
        0: BlobState(blob_id=0, x=20.0, y=20.0, radius=1.0),
    }
    other.blobs = {
        0: BlobState(blob_id=0, x=30.5, y=20.0, radius=1.0),
    }
    virus = state.map.spawn_virus(pos=(30.75, 20.0), radius=1.0)

    visible_blob_ids = {
        (blob.player_id, blob.blob_id) for blob in state.get_visible_blobs(viewer.id)
    }
    visible_virus_ids = {virus.virus_id for virus in state.get_visible_viruses(viewer.id)}

    assert (other.id, 0) in visible_blob_ids
    assert visible_virus_ids == {virus.virus_id}


def test_circle_visibility_respects_diagonal_corner_intersection(
    tmp_path, monkeypatch
) -> None:
    state = _make_state(tmp_path, monkeypatch)
    viewer = state.players[0]
    state.map.viruses = []

    viewer.blobs = {
        0: BlobState(blob_id=0, x=20.0, y=20.0, radius=1.0),
    }

    assert not state.player_can_see_circle(viewer.id, 30.9, 30.9, 1.0)
    assert state.player_can_see_circle(viewer.id, 30.6, 30.6, 1.0)


def test_vision_size_scales_from_sum_of_blob_radii(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    player = state.players[0]
    state.map.viruses = []

    player.blobs = {
        0: BlobState(blob_id=0, x=10.0, y=10.0, radius=64.0),
        1: BlobState(blob_id=1, x=20.0, y=20.0, radius=64.0),
    }

    expected = (
        math.pow(max(128.0 / VISION_REFERENCE_SUM_OF_RADII, 1.0), 0.4) * VISION_SIZE
    )
    assert math.isclose(state.get_player_vision_size(player.id), expected, rel_tol=1e-9)


def test_split_respects_max_blob_count(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    mutator = StateMutator(state)
    player = state.players[0]
    state.map.foods = []
    state.map.viruses = []

    for other_id, other in state.players.items():
        if other_id != player.id:
            other.blobs.clear()

    eligible_mass = SPLIT_MIN_MASS + 1.0
    player.blobs = {
        blob_id: BlobState(
            blob_id=blob_id,
            x=10.0 + blob_id,
            y=10.0,
            radius=math.sqrt(eligible_mass),
            merge_cooldown=5,
        )
        for blob_id in range(MAX_BLOB_COUNT)
    }

    total_mass_before = sum(blob.mass for blob in player.blobs.values())
    mutator.commit_round(
        [
            MovePlayer(
                player_id=player.id,
                direction=DirectionModel(x=1.0, y=0.0),
                split=True,
            )
        ]
    )

    assert len(player.blobs) == MAX_BLOB_COUNT
    total_mass_after = sum(blob.mass for blob in player.blobs.values())
    assert math.isclose(total_mass_before, total_mass_after, rel_tol=1e-9)


def test_virus_pop_splits_blob_up_to_cap(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    mutator = StateMutator(state)
    player = state.players[0]
    state.map.foods = []
    state.map.viruses = []

    for other_id, other in state.players.items():
        if other_id != player.id:
            other.blobs.clear()

    # Blob must be large enough to consume virus: mass > virus_mass * EAT_SIZE_RATIO
    # With virus_radius=2.0, virus_mass=4.0, need blob.mass > 4.8, so radius > 2.19
    player.blobs = {
        0: BlobState(blob_id=0, x=20.0, y=20.0, radius=2.5),
    }
    state.map.spawn_virus(pos=(20.0, 20.0), radius=2.0)

    mutator.commit_round(
        [
            MovePlayer(
                player_id=player.id,
                direction=DirectionModel(x=0.0, y=0.0),
            )
        ]
    )

    assert len(player.blobs) == MAX_BLOB_COUNT
    assert len(state.map.viruses) == VIRUS_COUNT
    expected_total_mass = 2.5 * 2.5 + 2.0 * 2.0  # 10.25
    total_mass_after = sum(blob.mass for blob in player.blobs.values())
    assert math.isclose(total_mass_after, expected_total_mass, rel_tol=1e-9)

    expected_piece_radius = math.sqrt(expected_total_mass / MAX_BLOB_COUNT)
    for blob in player.blobs.values():
        assert math.isclose(blob.radius, expected_piece_radius, rel_tol=1e-9)
        assert blob.merge_cooldown > 0


def test_virus_does_not_pop_when_blob_is_not_large_enough(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    mutator = StateMutator(state)
    player = state.players[0]
    state.map.foods = []
    state.map.viruses = []

    for other_id, other in state.players.items():
        if other_id != player.id:
            other.blobs.clear()

    player.blobs = {
        0: BlobState(blob_id=0, x=20.0, y=20.0, radius=1.4),
    }
    state.map.spawn_virus(pos=(20.0, 20.0), radius=2.0)

    total_mass_before = sum(blob.mass for blob in player.blobs.values())
    mutator.commit_round(
        [
            MovePlayer(
                player_id=player.id,
                direction=DirectionModel(x=0.0, y=0.0),
            )
        ]
    )

    assert len(player.blobs) == 1
    assert len(state.map.viruses) == VIRUS_COUNT
    total_mass_after = sum(blob.mass for blob in player.blobs.values())
    assert math.isclose(total_mass_after, total_mass_before, rel_tol=1e-9)


def test_virus_does_not_pop_on_partial_overlap_without_center_containment(
    tmp_path, monkeypatch
) -> None:
    state = _make_state(tmp_path, monkeypatch)
    mutator = StateMutator(state)
    player = state.players[0]
    state.map.foods = []
    state.map.viruses = []

    for other_id, other in state.players.items():
        if other_id != player.id:
            other.blobs.clear()

    player.blobs = {
        0: BlobState(blob_id=0, x=20.0, y=20.0, radius=2.5),
    }
    original_virus = state.map.spawn_virus(pos=(24.0, 20.0), radius=2.0)

    total_mass_before = sum(blob.mass for blob in player.blobs.values())
    mutator.commit_round(
        [
            MovePlayer(
                player_id=player.id,
                direction=DirectionModel(x=0.0, y=0.0),
            )
        ]
    )

    assert len(player.blobs) == 1
    assert original_virus.virus_id in {virus.virus_id for virus in state.map.viruses}
    total_mass_after = sum(blob.mass for blob in player.blobs.values())
    assert math.isclose(total_mass_after, total_mass_before, rel_tol=1e-9)


def test_food_growth_is_clamped_to_arena_same_round(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    mutator = StateMutator(state)
    player = state.players[0]
    state.map.foods = []
    state.map.viruses = []

    for other_id, other in state.players.items():
        if other_id != player.id:
            other.blobs.clear()

    player.blobs = {
        0: BlobState(blob_id=0, x=0.9, y=10.0, radius=0.9),
    }
    state.map.foods = [FoodModel(food_id=0, pos=(0.9, 10.0))]

    mutator.commit_round(
        [
            MovePlayer(
                player_id=player.id,
                direction=DirectionModel(x=0.0, y=0.0),
            )
        ]
    )

    blob = player.blobs[0]
    assert math.isclose(blob.x, blob.radius, rel_tol=1e-9)


def test_player_growth_is_clamped_to_arena_same_round(tmp_path, monkeypatch) -> None:
    state = _make_state(tmp_path, monkeypatch)
    mutator = StateMutator(state)
    eater = state.players[0]
    target = state.players[1]
    state.map.foods = []
    state.map.viruses = []

    for other_id, other in state.players.items():
        if other_id not in {eater.id, target.id}:
            other.blobs.clear()

    eater.blobs = {
        0: BlobState(blob_id=0, x=1.0, y=10.0, radius=1.0),
    }
    target.blobs = {
        0: BlobState(blob_id=0, x=1.0, y=10.0, radius=0.5),
    }

    mutator.commit_round(
        [
            MovePlayer(
                player_id=eater.id,
                direction=DirectionModel(x=0.0, y=0.0),
            ),
            MovePlayer(
                player_id=target.id,
                direction=DirectionModel(x=0.0, y=0.0),
            ),
        ]
    )

    blob = eater.blobs[0]
    assert math.isclose(blob.x, blob.radius, rel_tol=1e-9)
    assert not target.alive
