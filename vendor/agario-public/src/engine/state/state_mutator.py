import math
import random

from engine.state.blob_state import BlobState
from engine.state.game_state import GameState

from lib.config.arena import MAX_BLOB_COUNT, PLAYER_SPAWN_PADDING, RESPAWN_DELAY_ROUNDS
from lib.interface.events.event_game_ended import EventGameEndedCancelled
from lib.interface.events.event_food_eaten import EventFoodEaten
from lib.interface.events.event_food_spawned import EventFoodSpawned
from lib.interface.events.event_game_started import EventGameStarted
from lib.interface.events.event_player_bannned import EventPlayerBanned
from lib.interface.events.event_player_eaten import EventPlayerEaten
from lib.interface.events.event_player_moved import EventPlayerMoved
from lib.interface.events.event_player_won import EventPlayerWon
from lib.interface.events.event_virus_spawned import EventVirusSpawned
from lib.interface.events.event_virus_consumed import EventVirusConsumed
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.events.typing import EventType

from lib.config.player import (
    BASE_PLAYER_SPEED,
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    MASS_DECAY_RATE,
    MERGE_ATTRACTION_SPEED,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
    SAME_PLAYER_OVERLAP_EPSILON,
    SPLIT_COOLDOWN_FRAMES,
    SPLIT_EJECT_DRAG,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
    STARTING_RADIUS,
)


class StateMutator:
    def __init__(self, state: GameState) -> None:
        self.state = state

    def commitPrivate(self, event: EventType) -> None:
        self.state.private_event_history.append(event)

    def commit(self, event: EventType) -> None:
        self.state.event_history.append(event)
        self.state.private_event_history.append(event)

        match event:
            case (
                EventGameStarted()
                | MovePlayer()
                | EventGameEndedCancelled()
                | EventPlayerBanned()
                | EventPlayerWon()
            ):
                pass

    def _combine_radii(self, radius_a: float, radius_b: float) -> float:
        return math.sqrt(radius_a * radius_a + radius_b * radius_b)

    def _movement_speed(self, radius: float) -> float:
        return max(
            MIN_PLAYER_SPEED,
            BASE_PLAYER_SPEED / (1.0 + radius * PLAYER_SPEED_RADIUS_FACTOR),
        )

    def _normalise_vector(self, dx: float, dy: float) -> tuple[float, float]:
        max_component = max(abs(dx), abs(dy))
        if not math.isfinite(max_component) or max_component == 0.0:
            return (0.0, 0.0)
        # Scale first so subnormal inputs do not collapse into a non-unit vector.
        scaled_x = dx / max_component
        scaled_y = dy / max_component
        magnitude = math.hypot(scaled_x, scaled_y)
        return (scaled_x / magnitude, scaled_y / magnitude)

    def _blob_direction_for_split(self, dx: float, dy: float) -> tuple[float, float]:
        unit_x, unit_y = self._normalise_vector(dx, dy)
        if unit_x == 0.0 and unit_y == 0.0:
            return (1.0, 0.0)
        return (unit_x, unit_y)

    def _clamp_blob_to_arena(self, blob: BlobState) -> None:
        blob.x = min(max(blob.radius, blob.x), self.state.map.size - blob.radius)
        blob.y = min(max(blob.radius, blob.y), self.state.map.size - blob.radius)

    def _move_blob(self, blob: BlobState, direction_x: float, direction_y: float) -> None:
        if direction_x != 0.0 or direction_y != 0.0:
            speed = self._movement_speed(blob.radius)
            blob.x += direction_x * speed
            blob.y += direction_y * speed

        blob.x += blob.eject_vx
        blob.y += blob.eject_vy
        blob.eject_vx *= SPLIT_EJECT_DRAG
        blob.eject_vy *= SPLIT_EJECT_DRAG
        if abs(blob.eject_vx) < 1e-4:
            blob.eject_vx = 0.0
        if abs(blob.eject_vy) < 1e-4:
            blob.eject_vy = 0.0
        blob.merge_cooldown = max(0, blob.merge_cooldown - 1)
        self._clamp_blob_to_arena(blob)

    def _apply_split(self, event: MovePlayer) -> None:
        player = self.state.players[event.player_id]
        if not player.alive or not event.split:
            return

        direction_x, direction_y = event.direction.to_vector()
        split_x, split_y = self._blob_direction_for_split(direction_x, direction_y)
        launch_scale = 1.0 if direction_x != 0.0 or direction_y != 0.0 else 0.0

        starting_blob_ids = [blob.blob_id for blob in player.sorted_blobs()]
        for blob_id in starting_blob_ids:
            if len(player.blobs) >= MAX_BLOB_COUNT:
                break

            blob = player.blobs.get(blob_id)
            if blob is None or blob.mass < SPLIT_MIN_MASS:
                continue

            child_radius = math.sqrt(blob.mass / 2.0)
            blob.radius = child_radius
            blob.merge_cooldown = SPLIT_COOLDOWN_FRAMES

            child = BlobState(
                blob_id=player.next_blob_id(),
                x=blob.x + split_x * (blob.radius + child_radius + SAME_PLAYER_OVERLAP_EPSILON),
                y=blob.y + split_y * (blob.radius + child_radius + SAME_PLAYER_OVERLAP_EPSILON),
                radius=child_radius,
                merge_cooldown=SPLIT_COOLDOWN_FRAMES,
                eject_vx=split_x * SPLIT_EJECT_SPEED * launch_scale,
                eject_vy=split_y * SPLIT_EJECT_SPEED * launch_scale,
            )
            self._clamp_blob_to_arena(child)
            player.blobs[child.blob_id] = child

    def _replacement_positions(
        self,
        center_x: float,
        center_y: float,
        piece_radius: float,
        piece_count: int,
    ) -> list[tuple[float, float]]:
        cols = math.ceil(math.sqrt(piece_count))
        rows = math.ceil(piece_count / cols)
        spacing = piece_radius * 2.0 + SAME_PLAYER_OVERLAP_EPSILON
        x_offset = (cols - 1) * spacing / 2.0
        y_offset = (rows - 1) * spacing / 2.0

        positions: list[tuple[float, float]] = []
        for index in range(piece_count):
            row = index // cols
            col = index % cols
            positions.append(
                (
                    center_x + col * spacing - x_offset,
                    center_y + row * spacing - y_offset,
                )
            )
        return positions

    def _split_blob_evenly(
        self,
        player_id: int,
        blob_id: int,
        total_mass: float,
        piece_count: int,
    ) -> None:
        player = self.state.players[player_id]
        blob = player.blobs.get(blob_id)
        if blob is None:
            return

        if piece_count <= 1:
            blob.radius = math.sqrt(total_mass)
            self._clamp_blob_to_arena(blob)
            return

        piece_radius = math.sqrt(total_mass / piece_count)
        positions = self._replacement_positions(blob.x, blob.y, piece_radius, piece_count)

        for index, (x, y) in enumerate(positions):
            if index == 0:
                target_blob = blob
            else:
                target_blob = BlobState(
                    blob_id=player.next_blob_id(),
                    x=x,
                    y=y,
                    radius=piece_radius,
                    merge_cooldown=SPLIT_COOLDOWN_FRAMES,
                )
                player.blobs[target_blob.blob_id] = target_blob

            target_blob.x = x
            target_blob.y = y
            target_blob.radius = piece_radius
            target_blob.eject_vx = 0.0
            target_blob.eject_vy = 0.0
            target_blob.merge_cooldown = SPLIT_COOLDOWN_FRAMES
            self._clamp_blob_to_arena(target_blob)

    def _can_hit_virus(
        self,
        blob: BlobState,
        virus_x: float,
        virus_y: float,
        virus_radius: float,
    ) -> bool:
        return (
            (blob.x - virus_x) ** 2 + (blob.y - virus_y) ** 2
            <= blob.radius * blob.radius
        )

    def _can_consume_virus(self, blob: BlobState, virus_radius: float) -> bool:
        return blob.mass > (virus_radius ** 2) * EAT_SIZE_RATIO

    def _resolve_viruses(self) -> None:
        remaining_viruses = []
        for virus in self.state.map.viruses:
            living_blobs = [
                (player.id, blob)
                for player in self.state.players.values()
                if player.alive
                for blob in player.blobs.values()
            ]
            living_blobs.sort(key=lambda item: (-item[1].radius, item[0], item[1].blob_id))

            candidates = [
                (player_id, blob)
                for player_id, blob in living_blobs
                if self._can_hit_virus(blob, virus.pos[0], virus.pos[1], virus.radius)
                and self._can_consume_virus(blob, virus.radius)
            ]
            if not candidates:
                remaining_viruses.append(virus)
                continue

            player_id, blob = min(
                candidates,
                key=lambda item: (-item[1].radius, item[0], item[1].blob_id),
            )
            player = self.state.players[player_id]
            total_mass_after_consumption = blob.mass + virus.radius * virus.radius
            max_piece_count = MAX_BLOB_COUNT - len(player.blobs) + 1
            piece_count = max(1, max_piece_count)
            self._split_blob_evenly(
                player_id=player_id,
                blob_id=blob.blob_id,
                total_mass=total_mass_after_consumption,
                piece_count=piece_count,
            )
            self.state.private_event_history.append(
                EventVirusConsumed(
                    player_id=player_id,
                    blob_id=blob.blob_id,
                    virus_id=virus.virus_id,
                    virus_pos=virus.pos,
                    pieces_created=piece_count,
                )
            )

        self.state.map.viruses = remaining_viruses

    def _apply_same_player_attraction(self) -> None:
        for player in self.state.players.values():
            if len(player.blobs) <= 1:
                continue

            total_mass = sum(blob.mass for blob in player.blobs.values())
            if total_mass == 0:
                continue

            centroid_x = (
                sum(blob.x * blob.mass for blob in player.blobs.values()) / total_mass
            )
            centroid_y = (
                sum(blob.y * blob.mass for blob in player.blobs.values()) / total_mass
            )
            for blob in player.blobs.values():
                dx = centroid_x - blob.x
                dy = centroid_y - blob.y
                distance = math.hypot(dx, dy)
                if distance == 0:
                    continue
                step = min(MERGE_ATTRACTION_SPEED, distance)
                blob.x += dx / distance * step
                blob.y += dy / distance * step
                self._clamp_blob_to_arena(blob)

    def _merge_blobs(self, player_id: int, survivor_id: int, consumed_id: int) -> None:
        player = self.state.players[player_id]
        survivor = player.blobs[survivor_id]
        consumed = player.blobs[consumed_id]
        combined_mass = survivor.mass + consumed.mass
        survivor.x = (
            survivor.x * survivor.mass + consumed.x * consumed.mass
        ) / combined_mass
        survivor.y = (
            survivor.y * survivor.mass + consumed.y * consumed.mass
        ) / combined_mass
        survivor.eject_vx = (
            survivor.eject_vx * survivor.mass + consumed.eject_vx * consumed.mass
        ) / combined_mass
        survivor.eject_vy = (
            survivor.eject_vy * survivor.mass + consumed.eject_vy * consumed.mass
        ) / combined_mass
        survivor.radius = math.sqrt(combined_mass)
        survivor.merge_cooldown = 0
        del player.blobs[consumed_id]
        self._clamp_blob_to_arena(survivor)

    def _merge_touching_same_player_blobs(self) -> bool:
        for player in self.state.players.values():
            blobs = player.sorted_blobs()
            for index, blob_a in enumerate(blobs):
                for blob_b in blobs[index + 1 :]:
                    if blob_a.merge_cooldown > 0 or blob_b.merge_cooldown > 0:
                        continue

                    dx = blob_b.x - blob_a.x
                    dy = blob_b.y - blob_a.y
                    distance = math.hypot(dx, dy)
                    if distance > blob_a.radius + blob_b.radius + SAME_PLAYER_OVERLAP_EPSILON:
                        continue

                    survivor, consumed = sorted(
                        (blob_a, blob_b),
                        key=lambda blob: (-blob.mass, blob.blob_id),
                    )
                    self._merge_blobs(player.id, survivor.blob_id, consumed.blob_id)
                    return True
        return False

    def _separate_same_player_blobs(self, iterations: int = 4) -> None:
        for _ in range(iterations):
            changed = False
            for player in self.state.players.values():
                blobs = player.sorted_blobs()
                for index, blob_a in enumerate(blobs):
                    for blob_b in blobs[index + 1 :]:
                        dx = blob_b.x - blob_a.x
                        dy = blob_b.y - blob_a.y
                        distance = math.hypot(dx, dy)
                        min_distance = (
                            blob_a.radius + blob_b.radius + SAME_PLAYER_OVERLAP_EPSILON
                        )
                        if distance >= min_distance:
                            continue

                        if distance == 0:
                            if blob_a.blob_id < blob_b.blob_id:
                                nx, ny = (1.0, 0.0)
                            else:
                                nx, ny = (0.0, 1.0)
                        else:
                            nx = dx / distance
                            ny = dy / distance

                        overlap = min_distance - distance
                        total_mass = blob_a.mass + blob_b.mass
                        move_a = overlap * (blob_b.mass / total_mass)
                        move_b = overlap * (blob_a.mass / total_mass)
                        blob_a.x -= nx * move_a
                        blob_a.y -= ny * move_a
                        blob_b.x += nx * move_b
                        blob_b.y += ny * move_b
                        self._clamp_blob_to_arena(blob_a)
                        self._clamp_blob_to_arena(blob_b)
                        changed = True
            if not changed:
                break

    def _stabilise_same_player_blobs(self) -> None:
        self._apply_same_player_attraction()
        while self._merge_touching_same_player_blobs():
            pass
        self._separate_same_player_blobs()
        while self._merge_touching_same_player_blobs():
            pass
        self._separate_same_player_blobs()

    def _resolve_food(self) -> None:
        living_blobs = [
            (player.id, blob)
            for player in self.state.players.values()
            if player.alive
            for blob in player.blobs.values()
        ]
        living_blobs.sort(key=lambda item: (-item[1].radius, item[0], item[1].blob_id))

        remaining_foods = []
        eaten_by_blob: dict[tuple[int, int], list[int]] = {}
        for food in self.state.map.foods:
            candidates = [
                (player_id, blob)
                for player_id, blob in living_blobs
                if (blob.x - food.pos[0]) ** 2 + (blob.y - food.pos[1]) ** 2
                <= blob.radius * blob.radius
            ]
            if not candidates:
                remaining_foods.append(food)
                continue

            eater_id, eater_blob = min(
                candidates,
                key=lambda item: (-item[1].radius, item[0], item[1].blob_id),
            )
            eater_blob.radius = self._combine_radii(eater_blob.radius, FOOD_RADIUS)
            self._clamp_blob_to_arena(eater_blob)
            eaten_by_blob.setdefault((eater_id, eater_blob.blob_id), []).append(food.food_id)

        self.state.map.foods = remaining_foods
        for (player_id, blob_id), food_ids in eaten_by_blob.items():
            eater_blob = self.state.players[player_id].blobs[blob_id]
            self.commitPrivate(
                EventFoodEaten(
                    player_id=player_id,
                    blob_id=blob_id,
                    food_ids=food_ids,
                    new_radius=eater_blob.radius,
                )
            )

    def _can_eat_blob(self, eater: BlobState, target: BlobState) -> bool:
        if eater.mass < target.mass * EAT_SIZE_RATIO:
            return False
        return (
            (eater.x - target.x) ** 2 + (eater.y - target.y) ** 2
            <= eater.radius * eater.radius
        )

    def _resolve_player_eating(self) -> None:
        changed = True
        while changed:
            changed = False
            living_blobs = sorted(
                [
                    (player.id, blob.blob_id)
                    for player in self.state.players.values()
                    if player.alive
                    for blob in player.blobs.values()
                ],
                key=lambda item: (
                    -self.state.players[item[0]].blobs[item[1]].radius,
                    item[0],
                    item[1],
                ),
            )
            for eater_player_id, eater_blob_id in living_blobs:
                eater_player = self.state.players[eater_player_id]
                eater_blob = eater_player.blobs.get(eater_blob_id)
                if eater_blob is None:
                    continue

                for target_player_id, target_blob_id in living_blobs:
                    if eater_player_id == target_player_id:
                        continue

                    target_player = self.state.players[target_player_id]
                    target_blob = target_player.blobs.get(target_blob_id)
                    if target_blob is None:
                        continue

                    if not self._can_eat_blob(eater_blob, target_blob):
                        continue

                    eaten_pos = (target_blob.x, target_blob.y)
                    eater_blob.radius = self._combine_radii(
                        eater_blob.radius,
                        target_blob.radius,
                    )
                    self._clamp_blob_to_arena(eater_blob)
                    del target_player.blobs[target_blob_id]
                    if not target_player.alive:
                        target_player.round_died = self.state.round
                        target_player.respawn_at_round = self.state.round + RESPAWN_DELAY_ROUNDS

                    self.commit(
                        EventPlayerEaten(
                            eater_player_id=eater_player_id,
                            eater_blob_id=eater_blob_id,
                            eater_pos=(eater_blob.x, eater_blob.y),
                            eaten_player_id=target_player_id,
                            eaten_blob_id=target_blob_id,
                            eaten_pos=eaten_pos,
                            eater_radius=eater_blob.radius,
                            eaten_radius=target_blob.radius,
                            eaten_player_alive=target_player.alive,
                        )
                    )
                    changed = True
                    break
                if changed:
                    break

    def _emit_player_snapshots(self) -> None:
        for player in self.state.players.values():
            self.commit(
                EventPlayerMoved(
                    player_id=player.id,
                    pos=(player.x, player.y),
                    radius=player.radius,
                    alive=player.alive,
                    blobs=tuple(blob._to_model() for blob in player.sorted_blobs()),
                )
            )

    def _respawn_dead_players(self) -> None:
        padding = max(PLAYER_SPAWN_PADDING, STARTING_RADIUS * 2)
        lo = padding
        hi = self.state.map.size - padding
        for player in self.state.players.values():
            if player.alive or player.respawn_at_round is None:
                continue
            if self.state.round < player.respawn_at_round:
                continue
            for _ in range(10000):
                x = random.uniform(lo, hi)
                y = random.uniform(lo, hi)
                clear = all(
                    math.hypot(x - blob.x, y - blob.y) >= STARTING_RADIUS * 4
                    for other in self.state.players.values()
                    if other.alive
                    for blob in other.blobs.values()
                )
                if clear:
                    break
            blob_id = player.next_blob_id()
            player.blobs[blob_id] = BlobState(blob_id=blob_id, x=x, y=y, radius=STARTING_RADIUS)
            player.respawn_at_round = None

    def _apply_mass_decay(self) -> None:
        min_mass = STARTING_RADIUS * STARTING_RADIUS
        for player in self.state.players.values():
            if not player.alive:
                continue
            for blob in player.blobs.values():
                current_mass = blob.mass
                if current_mass <= min_mass:
                    continue  # Don't decay or increase if at/below minimum
                new_mass = current_mass * (1.0 - MASS_DECAY_RATE)
                if new_mass < min_mass:
                    new_mass = min_mass
                blob.radius = math.sqrt(new_mass)

    def commit_round(self, events: list[MovePlayer]) -> None:
        for event in events:
            self.commit(event)

        direction_by_player: dict[int, tuple[float, float]] = {}
        for event in events:
            self._apply_split(event)
            dx, dy = event.direction.to_vector()
            direction_by_player[event.player_id] = self._normalise_vector(dx, dy)

        for event in events:
            player = self.state.players[event.player_id]
            if not player.alive:
                continue
            direction_x, direction_y = direction_by_player[event.player_id]
            for blob in player.blobs.values():
                self._move_blob(blob, direction_x, direction_y)

        self._apply_mass_decay()
        self._stabilise_same_player_blobs()
        self._resolve_viruses()
        self._stabilise_same_player_blobs()
        self._resolve_food()
        self._resolve_player_eating()
        self._stabilise_same_player_blobs()
        self._respawn_dead_players()
        self._emit_player_snapshots()

        spawned_food = self.state._ensure_food_count()
        if spawned_food:
            self.commitPrivate(EventFoodSpawned(foods=spawned_food))
        spawned_viruses = self.state._ensure_virus_count()
        if spawned_viruses:
            self.commitPrivate(EventVirusSpawned(viruses=spawned_viruses))
