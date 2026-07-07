#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
from importlib.resources import files
import json
import math
import queue
import runpy
import sys
import threading
import traceback
from time import perf_counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import helper.game as helper_game
from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.events.moves.typing import MoveType
from lib.models.penguin_model import DirectionModel


DEFAULT_SUBMISSION = Path(
    str(files("agario_visualiser.examples").joinpath("example.py"))
).resolve()


@dataclass
class Snapshot:
    round_number: int = -1
    max_rounds: int = 0
    arena_size: float = 0.0
    vision_size: float = 1.0
    turn_duration_seconds: float = 0.1
    player_id: int = -1
    position: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    alive: bool = True
    your_blobs: list[tuple[int, float, float, float, int]] = field(default_factory=list)
    visible_food: list[tuple[float, float]] = field(default_factory=list)
    visible_viruses: list[tuple[int, float, float, float]] = field(default_factory=list)
    visible_blobs: list[tuple[int, int, float, float, float, int]] = field(
        default_factory=list
    )
    status: str = "Waiting for engine..."
    game_over: bool = False
    placement: Optional[int] = None
    total_players: int = 0
    rankings: list[int] = field(default_factory=list)


@dataclass
class BlobState:
    player_id: int
    blob_id: int
    x: float
    y: float
    radius: float
    alive: bool
    is_self: bool = False


@dataclass
class CameraFrame:
    left: float
    top: float
    right: float
    bottom: float
    scale_x: float
    scale_y: float


@dataclass
class RenderViewState:
    x: float
    y: float
    vision_size: float


class InputState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: set[str] = set()
        self._split_requested = False

    def press(self, key: str) -> None:
        with self._lock:
            self._pressed.add(key)
            if key == "space":
                self._split_requested = True

    def release(self, key: str) -> None:
        with self._lock:
            self._pressed.discard(key)

    def vector(self) -> tuple[float, float]:
        with self._lock:
            pressed = set(self._pressed)

        x = 0.0
        y = 0.0
        if "left" in pressed or "a" in pressed:
            x -= 1.0
        if "right" in pressed or "d" in pressed:
            x += 1.0
        if "up" in pressed or "w" in pressed:
            y -= 1.0
        if "down" in pressed or "s" in pressed:
            y += 1.0
        return (x, y)

    def consume_split(self) -> bool:
        with self._lock:
            split_requested = self._split_requested
            self._split_requested = False
        return split_requested


class VisualiserApp:
    def __init__(
        self,
        masquerade: bool,
        window_size: int,
        countdown_duration_seconds: float,
    ) -> None:
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Agar.io Visualiser")
        self.window_size = window_size
        self.masquerade = masquerade
        self.input_state = InputState()
        self.snapshot = Snapshot()
        self.snapshot_lock = threading.Lock()
        self.animation_lock = threading.Lock()
        self.pending_snapshots: queue.Queue[Snapshot] = queue.Queue()
        self.animation_started_at = perf_counter()
        self.animation_duration = 0.1
        self.start_blobs: dict[tuple[int, int], BlobState] = {}
        self.target_blobs: dict[tuple[int, int], BlobState] = {}
        self.start_view = RenderViewState(x=0.0, y=0.0, vision_size=1.0)
        self.target_view = RenderViewState(x=0.0, y=0.0, vision_size=1.0)
        self.view_mode = "relative"
        self.closed = False
        self.countdown_duration_seconds = max(0.0, countdown_duration_seconds)
        self.countdown_started_at: float | None = (
            perf_counter() if self.countdown_duration_seconds > 0.0 else None
        )

        self.status_var = tk.StringVar(value="Waiting for engine...")
        self.info_var = tk.StringVar(value="")

        frame = tk.Frame(self.root, bg="#08131a")
        frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            frame,
            width=window_size,
            height=window_size,
            bg="#0b1720",
            highlightthickness=0,
        )
        self.canvas.pack(padx=12, pady=(12, 8))

        status = tk.Label(
            frame,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            bg="#08131a",
            fg="#d9eef6",
            font=("Menlo", 12),
        )
        status.pack(fill="x", padx=12)

        info = tk.Label(
            frame,
            textvariable=self.info_var,
            anchor="w",
            justify="left",
            bg="#08131a",
            fg="#89a7b5",
            font=("Menlo", 10),
        )
        info.pack(fill="x", padx=12, pady=(4, 12))

        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.report_callback_exception = self._report_callback_exception
        self.root.after(33, self._tick)

    def _on_close(self) -> None:
        self.closed = True
        self.root.destroy()

    def _on_key_press(self, event: object) -> None:
        key = getattr(event, "keysym", "").lower()
        if key:
            if key == "v" and key not in self.input_state._pressed:
                self.view_mode = "relative" if self.view_mode == "arena" else "arena"
            self.input_state.press(key)

    def _on_key_release(self, event: object) -> None:
        key = getattr(event, "keysym", "").lower()
        if key:
            self.input_state.release(key)

    def current_direction(self) -> tuple[float, float]:
        return self.input_state.vector()

    def consume_split(self) -> bool:
        return self.input_state.consume_split()

    def update_snapshot(self, snapshot: Snapshot) -> None:
        self.pending_snapshots.put(snapshot)

    def _apply_snapshot(self, snapshot: Snapshot) -> None:
        new_target_blobs = self._snapshot_to_blobs(snapshot)
        new_target_view = RenderViewState(
            x=snapshot.position[0],
            y=snapshot.position[1],
            vision_size=max(snapshot.vision_size, 1e-6),
        )
        now = perf_counter()

        with self.snapshot_lock:
            self.snapshot = snapshot
        with self.animation_lock:
            current_blobs = self._interpolated_blobs_locked(now)
            current_view = self._interpolated_view_locked(now)
            self.start_blobs = {
                blob_id: current_blobs.get(blob_id, target_blob)
                for blob_id, target_blob in new_target_blobs.items()
            }
            self.target_blobs = new_target_blobs
            self.start_view = current_view
            self.target_view = new_target_view
            self.animation_started_at = now
            self.animation_duration = max(snapshot.turn_duration_seconds, 1e-6)

    def _report_callback_exception(
        self,
        exc: type[BaseException],
        val: BaseException,
        tb: object,
    ) -> None:
        message = "".join(traceback.format_exception(exc, val, tb))
        self.status_var.set("Visualiser UI error")
        self.info_var.set(message.splitlines()[-1] if message else str(val))
        print(message, file=sys.stderr, flush=True)

    def _snapshot_to_blobs(self, snapshot: Snapshot) -> dict[tuple[int, int], BlobState]:
        blobs: dict[tuple[int, int], BlobState] = {}
        for blob_id, x, y, radius, _cooldown in snapshot.your_blobs:
            blobs[(snapshot.player_id, blob_id)] = BlobState(
                player_id=snapshot.player_id,
                blob_id=blob_id,
                x=x,
                y=y,
                radius=radius,
                alive=snapshot.alive,
                is_self=True,
            )
        for player_id, blob_id, x, y, radius, _cooldown in snapshot.visible_blobs:
            blobs[(player_id, blob_id)] = BlobState(
                player_id=player_id,
                blob_id=blob_id,
                x=x,
                y=y,
                radius=radius,
                alive=True,
                is_self=False,
            )
        return blobs

    def _interpolated_blobs_locked(self, now: float) -> dict[tuple[int, int], BlobState]:
        if not self.target_blobs:
            return {}
        progress = min(
            1.0,
            max(0.0, (now - self.animation_started_at) / self.animation_duration),
        )
        interpolated: dict[tuple[int, int], BlobState] = {}
        for blob_id, target_blob in self.target_blobs.items():
            start_blob = self.start_blobs.get(blob_id, target_blob)
            interpolated[blob_id] = BlobState(
                player_id=target_blob.player_id,
                blob_id=target_blob.blob_id,
                x=start_blob.x + (target_blob.x - start_blob.x) * progress,
                y=start_blob.y + (target_blob.y - start_blob.y) * progress,
                radius=start_blob.radius
                + (target_blob.radius - start_blob.radius) * progress,
                alive=target_blob.alive,
                is_self=target_blob.is_self,
            )
        return interpolated

    def _interpolated_view_locked(self, now: float) -> RenderViewState:
        progress = min(
            1.0,
            max(0.0, (now - self.animation_started_at) / self.animation_duration),
        )
        return RenderViewState(
            x=self.start_view.x + (self.target_view.x - self.start_view.x) * progress,
            y=self.start_view.y + (self.target_view.y - self.start_view.y) * progress,
            vision_size=self.start_view.vision_size
            + (self.target_view.vision_size - self.start_view.vision_size) * progress,
        )

    def _tick(self) -> None:
        try:
            while True:
                self._apply_snapshot(self.pending_snapshots.get_nowait())
        except queue.Empty:
            pass

        with self.snapshot_lock:
            snapshot = self.snapshot
        now = perf_counter()
        with self.animation_lock:
            blobs = self._interpolated_blobs_locked(now)
            view = self._interpolated_view_locked(now)

        self._draw_snapshot(snapshot, blobs, view)
        if not self.closed:
            self.root.after(33, self._tick)

    def _countdown_remaining(self) -> float:
        if self.countdown_started_at is None:
            return 0.0
        return max(
            0.0,
            self.countdown_duration_seconds - (perf_counter() - self.countdown_started_at),
        )

    def _ordinal(self, value: int) -> str:
        if 10 <= value % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"

    def _draw_snapshot(
        self,
        snapshot: Snapshot,
        blobs: dict[tuple[int, int], BlobState],
        view: RenderViewState,
    ) -> None:
        canvas = self.canvas
        canvas.delete("all")

        pad = 20
        arena_size = max(snapshot.arena_size, 1e-6)
        canvas.create_rectangle(
            pad,
            pad,
            self.window_size - pad,
            self.window_size - pad,
            outline="#1f3340",
            width=2,
        )

        mode = "Masquerade" if self.masquerade else "Submission"
        controls_text = (
            "Controls: WASD/arrows, Space splits, V toggles view"
            if self.masquerade
            else "View-only mode. Press V to toggle view."
        )
        view_label = "Arena" if self.view_mode == "arena" else "Relative"
        self.status_var.set(
            f"Round {snapshot.round_number + 1}/{snapshot.max_rounds or '?'}  "
            f"Player {snapshot.player_id if snapshot.player_id >= 0 else '?'}  "
            f"Radius {snapshot.radius:.2f}  "
            f"Pos ({view.x:.2f}, {view.y:.2f})  "
            f"Mode {mode}  View {view_label}"
        )
        self.info_var.set(
            f"{snapshot.status}  Food {len(snapshot.visible_food)}  "
            f"Viruses {len(snapshot.visible_viruses)}  "
            f"Blobs {len(snapshot.visible_blobs)}  "
            f"Turn {snapshot.turn_duration_seconds:.2f}s  "
            f"{controls_text}"
        )
        countdown_remaining = self._countdown_remaining()

        self_blobs = [blob for blob in blobs.values() if blob.is_self]
        if not self_blobs and not snapshot.game_over and countdown_remaining <= 0:
            canvas.create_text(
                self.window_size / 2,
                self.window_size / 2,
                text=snapshot.status,
                fill="#89a7b5",
                font=("Menlo", 14),
            )
            return
        current_self = max(self_blobs, key=lambda blob: blob.radius) if self_blobs else None

        camera = self._camera_for_view(view, arena_size, pad)
        self._draw_grid(canvas, camera, arena_size, pad)

        for x_world, y_world in snapshot.visible_food:
            if not self._is_in_camera(camera, x_world, y_world):
                continue
            x = pad + (x_world - camera.left) * camera.scale_x
            y = pad + (y_world - camera.top) * camera.scale_y
            radius = max(2.0, 0.15 * min(camera.scale_x, camera.scale_y))
            canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#ffd166",
                outline="",
            )

        if self.view_mode == "arena":
            self._draw_vision_box(
                canvas,
                view,
                snapshot.arena_size,
                camera,
                pad,
            )

        for blob in blobs.values():
            if not blob.alive:
                continue
            if not self._is_circle_in_camera(camera, blob.x, blob.y, blob.radius):
                continue
            x = pad + (blob.x - camera.left) * camera.scale_x
            y = pad + (blob.y - camera.top) * camera.scale_y
            screen_radius = max(4.0, blob.radius * min(camera.scale_x, camera.scale_y))
            if blob.is_self:
                fill = "#6ef3a5"
                outline = "#f5fffa"
                label = (
                    "YOU"
                    if current_self is not None and blob.blob_id == current_self.blob_id
                    else ""
                )
            else:
                fill = "#ff6b6b" if blob.radius > snapshot.radius else "#4ecdc4"
                outline = "#d9eef6"
                label = str(blob.player_id)
            canvas.create_oval(
                x - screen_radius,
                y - screen_radius,
                x + screen_radius,
                y + screen_radius,
                fill=fill,
                outline=outline,
                width=2 if blob.is_self else 1,
            )
            if label:
                canvas.create_text(
                    x,
                    y,
                    text=label,
                    fill="#08131a",
                    font=("Menlo", 10, "bold"),
                )

        for _virus_id, x_world, y_world, virus_radius in snapshot.visible_viruses:
            if not self._is_circle_in_camera(camera, x_world, y_world, virus_radius):
                continue
            x = pad + (x_world - camera.left) * camera.scale_x
            y = pad + (y_world - camera.top) * camera.scale_y
            screen_radius = max(
                5.0,
                virus_radius * min(camera.scale_x, camera.scale_y),
            )
            self._draw_virus(canvas, x, y, screen_radius)

        self._draw_coordinates(canvas, camera, arena_size, pad)
        self._draw_leaderboard(canvas, snapshot, pad)

        self.status_var.set(
            f"Round {snapshot.round_number + 1}/{snapshot.max_rounds or '?'}  "
            f"Player {snapshot.player_id}  Radius {snapshot.radius:.2f}  "
            f"Pos ({view.x:.2f}, {view.y:.2f})  "
            f"Mode {mode}  View {view_label}"
        )
        self.info_var.set(
            f"{snapshot.status}  Food {len(snapshot.visible_food)}  "
            f"Viruses {len(snapshot.visible_viruses)}  "
            f"Blobs {len(snapshot.visible_blobs)}  "
            f"Turn {snapshot.turn_duration_seconds:.2f}s  "
            f"{controls_text}"
        )
        if snapshot.game_over:
            self._draw_end_screen(canvas, snapshot)
            return

        if countdown_remaining > 0:
            self._draw_countdown(canvas, countdown_remaining)

    def _draw_leaderboard(
        self,
        canvas: object,
        snapshot: Snapshot,
        pad: int,
    ) -> None:
        if not snapshot.rankings:
            return

        visible_rankings = snapshot.rankings[:8]
        width = 156
        line_height = 18
        header_height = 24
        height = header_height + 10 + line_height * len(visible_rankings)
        left = self.window_size - pad - width - 8
        top = pad + 8

        canvas.create_rectangle(
            left,
            top,
            left + width,
            top + height,
            fill="#08131a",
            outline="#1f3340",
            width=2,
            stipple="gray50",
        )
        canvas.create_text(
            left + 10,
            top + 12,
            text="Leaderboard",
            fill="#f5fffa",
            font=("Menlo", 11, "bold"),
            anchor="w",
        )

        for index, player_id in enumerate(visible_rankings, start=1):
            is_self = player_id == snapshot.player_id
            line_y = top + header_height + 4 + (index - 1) * line_height
            canvas.create_text(
                left + 10,
                line_y,
                text=f"#{index}",
                fill="#89a7b5",
                font=("Menlo", 10),
                anchor="w",
            )
            canvas.create_text(
                left + width - 10,
                line_y,
                text=f"P{player_id}{'  YOU' if is_self else ''}",
                fill="#6ef3a5" if is_self else "#d9eef6",
                font=("Menlo", 10, "bold" if is_self else "normal"),
                anchor="e",
            )

    def _draw_virus(
        self,
        canvas: object,
        x: float,
        y: float,
        radius: float,
    ) -> None:
        points: list[float] = []
        spike_count = 12
        inner_radius = radius * 0.92
        outer_radius = radius * 0.98
        for index in range(spike_count * 2):
            angle = (math.pi * index / spike_count) - (math.pi / 2)
            point_radius = outer_radius if index % 2 == 0 else inner_radius
            points.extend(
                [
                    x + math.cos(angle) * point_radius,
                    y + math.sin(angle) * point_radius,
                ]
            )

        canvas.create_polygon(
            points,
            fill="#39b54a",
            outline="#d4ffd9",
            width=2,
            smooth=False,
        )
        core_radius = radius * 0.84
        canvas.create_oval(
            x - core_radius,
            y - core_radius,
            x + core_radius,
            y + core_radius,
            fill="#58cf5f",
            outline="",
        )

    def _draw_countdown(self, canvas: object, countdown_remaining: float) -> None:
        count = max(1, math.ceil(countdown_remaining))
        canvas.create_rectangle(
            0,
            0,
            self.window_size,
            self.window_size,
            fill="#08131a",
            stipple="gray50",
            outline="",
        )
        canvas.create_text(
            self.window_size / 2,
            self.window_size / 2 - 24,
            text="Match starts in",
            fill="#d9eef6",
            font=("Menlo", 18),
        )
        canvas.create_text(
            self.window_size / 2,
            self.window_size / 2 + 28,
            text=str(count),
            fill="#f5fffa",
            font=("Menlo", 64, "bold"),
        )

    def _draw_end_screen(self, canvas: object, snapshot: Snapshot) -> None:
        canvas.create_rectangle(
            0,
            0,
            self.window_size,
            self.window_size,
            fill="#08131a",
            stipple="gray50",
            outline="",
        )
        title = "Match Over"
        subtitle = snapshot.status
        if snapshot.placement is not None and snapshot.total_players > 0:
            placement_text = f"Your rank: {snapshot.placement}/{snapshot.total_players}"
        elif snapshot.total_players > 0:
            placement_text = f"Your rank: ?/{snapshot.total_players}"
        else:
            placement_text = "Your rank: unavailable"

        canvas.create_text(
            self.window_size / 2,
            self.window_size / 2 - 52,
            text=title,
            fill="#f5fffa",
            font=("Menlo", 26, "bold"),
        )
        canvas.create_text(
            self.window_size / 2,
            self.window_size / 2,
            text=placement_text,
            fill="#6ef3a5",
            font=("Menlo", 28, "bold"),
        )
        canvas.create_text(
            self.window_size / 2,
            self.window_size / 2 + 42,
            text=subtitle,
            fill="#d9eef6",
            font=("Menlo", 14),
        )
        canvas.create_text(
            self.window_size / 2,
            self.window_size / 2 + 78,
            text="Close the window to exit.",
            fill="#89a7b5",
            font=("Menlo", 12),
        )

    def _camera_for_view(
        self,
        view: RenderViewState,
        arena_size: float,
        pad: int,
    ) -> CameraFrame:
        drawable = self.window_size - pad * 2
        if self.view_mode == "arena":
            return CameraFrame(
                left=0.0,
                top=0.0,
                right=arena_size,
                bottom=arena_size,
                scale_x=drawable / arena_size,
                scale_y=drawable / arena_size,
            )

        field_size = min(view.vision_size, arena_size)
        half = field_size / 2
        left = max(0.0, min(view.x - half, arena_size - field_size))
        top = max(0.0, min(view.y - half, arena_size - field_size))
        return CameraFrame(
            left=left,
            top=top,
            right=left + field_size,
            bottom=top + field_size,
            scale_x=drawable / field_size,
            scale_y=drawable / field_size,
        )

    def _is_in_camera(self, camera: CameraFrame, x: float, y: float) -> bool:
        return camera.left <= x <= camera.right and camera.top <= y <= camera.bottom

    def _is_circle_in_camera(
        self,
        camera: CameraFrame,
        x: float,
        y: float,
        radius: float,
    ) -> bool:
        dx_outside = max(camera.left - x, 0.0, x - camera.right)
        dy_outside = max(camera.top - y, 0.0, y - camera.bottom)
        return dx_outside * dx_outside + dy_outside * dy_outside <= radius * radius

    def _draw_grid(
        self,
        canvas: object,
        camera: CameraFrame,
        arena_size: float,
        pad: int,
    ) -> None:
        grid_step = 2.0
        major_step = 10.0
        x = math.floor(camera.left / grid_step) * grid_step
        while x <= camera.right + 1e-6:
            screen_x = pad + (x - camera.left) * camera.scale_x
            color = "#18313b" if abs((x / major_step) - round(x / major_step)) < 1e-6 else "#10232c"
            canvas.create_line(
                screen_x,
                pad,
                screen_x,
                self.window_size - pad,
                fill=color,
            )
            x += grid_step

        y = math.floor(camera.top / grid_step) * grid_step
        while y <= camera.bottom + 1e-6:
            screen_y = pad + (y - camera.top) * camera.scale_y
            color = "#18313b" if abs((y / major_step) - round(y / major_step)) < 1e-6 else "#10232c"
            canvas.create_line(
                pad,
                screen_y,
                self.window_size - pad,
                screen_y,
                fill=color,
            )
            y += grid_step

    def _draw_vision_box(
        self,
        canvas: object,
        view: RenderViewState,
        arena_size: float,
        camera: CameraFrame,
        pad: int,
    ) -> None:
        half = view.vision_size / 2
        left = max(0.0, view.x - half)
        top = max(0.0, view.y - half)
        right = min(arena_size, view.x + half)
        bottom = min(arena_size, view.y + half)
        canvas.create_rectangle(
            pad + (left - camera.left) * camera.scale_x,
            pad + (top - camera.top) * camera.scale_y,
            pad + (right - camera.left) * camera.scale_x,
            pad + (bottom - camera.top) * camera.scale_y,
            outline="#89a7b5",
            dash=(6, 4),
            width=1,
        )

    def _draw_coordinates(
        self,
        canvas: object,
        camera: CameraFrame,
        arena_size: float,
        pad: int,
    ) -> None:
        start = int(math.floor(camera.left / 10.0) * 10)
        end = int(math.ceil(camera.right / 10.0) * 10)
        for value in range(start, min(end, int(arena_size)) + 1, 10):
            screen_pos = pad + (value - camera.left) * camera.scale_x
            canvas.create_text(
                screen_pos,
                self.window_size - pad + 10,
                text=str(value),
                fill="#507180",
                font=("Menlo", 9),
            )
        start_y = int(math.floor(camera.top / 10.0) * 10)
        end_y = int(math.ceil(camera.bottom / 10.0) * 10)
        for value in range(start_y, min(end_y, int(arena_size)) + 1, 10):
            screen_pos = pad + (value - camera.top) * camera.scale_y
            canvas.create_text(
                pad - 10,
                screen_pos,
                text=str(value),
                fill="#507180",
                font=("Menlo", 9),
            )

    def run(self) -> None:
        self.root.mainloop()


def estimate_placement(game: Game) -> tuple[Optional[int], int]:
    state = game.state
    if state.total_players <= 0:
        return (None, 0)
    if state.winner_player_id == state.me.player_id:
        return (1, state.total_players)
    return (None, state.total_players)


def load_match_result(player_id: int) -> tuple[Optional[int], int]:
    results_path = Path.cwd().resolve().parent / "output" / "results.json"
    if not results_path.exists():
        return (None, 0)

    try:
        with open(results_path, "r") as file:
            result = json.load(file)
    except (OSError, json.JSONDecodeError):
        return (None, 0)

    ranking = result.get("ranking")
    if result.get("result_type") != "SUCCESS" or not isinstance(ranking, list):
        return (None, 0)

    if player_id in ranking:
        return (ranking.index(player_id) + 1, len(ranking))
    return (None, len(ranking))


def build_snapshot(
    game: Game,
    *,
    status: str = "Connected to engine.",
    game_over: bool | None = None,
    placement: Optional[int] = None,
    total_players: Optional[int] = None,
) -> Snapshot:
    state = game.state
    return Snapshot(
        round_number=state.round,
        max_rounds=state.max_rounds,
        arena_size=state.map.size,
        vision_size=state.vision_size,
        turn_duration_seconds=state.turn_duration_seconds,
        player_id=state.me.player_id,
        position=state.view_center,
        radius=state.me.radius,
        alive=state.me.alive,
        your_blobs=[
            (
                blob.blob_id,
                blob.pos[0],
                blob.pos[1],
                blob.radius,
                blob.merge_cooldown,
            )
            for blob in state.me.blobs.values()
        ],
        visible_food=[food.pos for food in state.visible_food],
        visible_viruses=[
            (
                virus.virus_id,
                virus.pos[0],
                virus.pos[1],
                virus.radius,
            )
            for virus in state.visible_viruses
        ],
        visible_blobs=[
            (
                blob.player_id,
                blob.blob_id,
                blob.pos[0],
                blob.pos[1],
                blob.radius,
                blob.merge_cooldown,
            )
            for blob in state.visible_blobs
        ],
        status=status,
        game_over=state.game_over if game_over is None else game_over,
        placement=placement,
        total_players=state.total_players if total_players is None else total_players,
        rankings=list(state.rankings),
    )


def clone_snapshot(snapshot: Snapshot, *, status: str, game_over: bool = True) -> Snapshot:
    return Snapshot(
        round_number=snapshot.round_number,
        max_rounds=snapshot.max_rounds,
        arena_size=snapshot.arena_size,
        vision_size=snapshot.vision_size,
        turn_duration_seconds=snapshot.turn_duration_seconds,
        player_id=snapshot.player_id,
        position=snapshot.position,
        radius=snapshot.radius,
        alive=snapshot.alive,
        your_blobs=list(snapshot.your_blobs),
        visible_food=list(snapshot.visible_food),
        visible_viruses=list(snapshot.visible_viruses),
        visible_blobs=list(snapshot.visible_blobs),
        status=status,
        game_over=game_over,
        placement=snapshot.placement,
        total_players=snapshot.total_players,
        rankings=list(snapshot.rankings),
    )


def build_terminal_snapshot(
    game: Game | None,
    latest_snapshot: Snapshot | None,
    *,
    status: str,
) -> Snapshot:
    if latest_snapshot is not None:
        return clone_snapshot(latest_snapshot, status=status)
    if game is not None and hasattr(game.state, "me"):
        placement, total_players = estimate_placement(game)
        return build_snapshot(
            game,
            status=status,
            game_over=True,
            placement=placement,
            total_players=total_players,
        )
    return Snapshot(status=status, game_over=True)


def finalise_snapshot(final_snapshot: Snapshot, game: Game | None) -> Snapshot:
    if game is None or not hasattr(game.state, "me"):
        return final_snapshot

    placement, total_players = load_match_result(game.state.me.player_id)
    if placement is None:
        placement, total_players = estimate_placement(game)

    final_snapshot.placement = placement
    if total_players > 0:
        final_snapshot.total_players = total_players
    if final_snapshot.placement == 1:
        final_snapshot.status = "You win!"
    return final_snapshot


@contextlib.contextmanager
def patched_submission_environment(
    script_path: Path,
    replacement_game: type[Game],
):
    original_game = helper_game.Game
    original_argv = list(sys.argv)
    script_dir = str(script_path.parent)
    inserted_path = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        inserted_path = True

    helper_game.Game = replacement_game
    sys.argv = [str(script_path)]
    try:
        yield
    finally:
        helper_game.Game = original_game
        sys.argv = original_argv
        if inserted_path:
            with contextlib.suppress(ValueError):
                sys.path.remove(script_dir)


def run_masquerade_worker(
    app: VisualiserApp | None,
    stop_event: threading.Event,
) -> None:
    game: Game | None = None
    latest_snapshot: Snapshot | None = None
    final_snapshot: Snapshot | None = None

    try:
        game = Game()
        if app is not None:
            app.update_snapshot(
                Snapshot(
                    status="Connected to engine. Waiting for first update...",
                    turn_duration_seconds=0.1,
                )
            )
        while True:
            game.get_next_query()
            snapshot = build_snapshot(game, status="Keyboard control active.")
            latest_snapshot = snapshot
            if app is not None:
                app.update_snapshot(snapshot)

            move_vector = app.current_direction() if app is not None else (0.0, 0.0)
            game.send_move(
                MovePlayer(
                    player_id=game.state.me.player_id,
                    direction=DirectionModel(x=move_vector[0], y=move_vector[1]),
                    split=app.consume_split() if app is not None else False,
                )
            )

            if snapshot.game_over:
                final_snapshot = build_snapshot(
                    game,
                    status="Match complete.",
                    game_over=True,
                )
                break

            if not snapshot.alive:
                final_snapshot = build_snapshot(
                    game,
                    status="You were eaten.",
                    game_over=True,
                )
                break
    except Exception as exc:
        if not isinstance(exc, EOFError):
            traceback.print_exc(file=sys.stderr)
        final_snapshot = build_terminal_snapshot(
            game,
            latest_snapshot,
            status="Match complete." if isinstance(exc, EOFError) else "Game state frozen.",
        )
    finally:
        if final_snapshot is not None and app is not None:
            app.update_snapshot(finalise_snapshot(final_snapshot, game))
        stop_event.set()


def run_delegated_worker(
    delegate_script: Path,
    app: VisualiserApp | None,
    stop_event: threading.Event,
) -> None:
    script_path = delegate_script.resolve()
    game: Game | None = None
    latest_snapshot: Snapshot | None = None
    final_snapshot: Snapshot | None = None

    class InstrumentedGame(Game):
        def __init__(self) -> None:
            nonlocal game
            super().__init__()
            game = self
            if app is not None:
                app.update_snapshot(
                    Snapshot(
                        status=f"Connected to engine. Running {script_path.name}...",
                        turn_duration_seconds=0.1,
                    )
                )

        def get_next_query(self):
            nonlocal latest_snapshot
            query = super().get_next_query()
            snapshot = build_snapshot(self, status=f"Running {script_path.name}.")
            latest_snapshot = snapshot
            if app is not None:
                app.update_snapshot(snapshot)
            return query

    try:
        with patched_submission_environment(script_path, InstrumentedGame):
            runpy.run_path(str(script_path), run_name="__main__")

        exit_status = f"{script_path.name} exited."
        if latest_snapshot is not None:
            if latest_snapshot.game_over:
                exit_status = "Match complete."
            elif not latest_snapshot.alive:
                exit_status = "You were eaten."
        final_snapshot = build_terminal_snapshot(
            game,
            latest_snapshot,
            status=exit_status,
        )
    except Exception as exc:
        if not isinstance(exc, EOFError):
            traceback.print_exc(file=sys.stderr)
        status = "Match complete." if isinstance(exc, EOFError) else "Game state frozen."
        if latest_snapshot is not None:
            if latest_snapshot.game_over:
                status = "Match complete."
            elif not latest_snapshot.alive:
                status = "You were eaten."
        final_snapshot = build_terminal_snapshot(
            game,
            latest_snapshot,
            status=status,
        )
    finally:
        if final_snapshot is not None and app is not None:
            app.update_snapshot(finalise_snapshot(final_snapshot, game))
        stop_event.set()


def run_submission(
    masquerade: bool,
    headless: bool,
    window_size: int,
    countdown_seconds: float,
    delegate_script: Path | None,
) -> None:
    app: VisualiserApp | None = None
    if not headless:
        try:
            app = VisualiserApp(
                masquerade=masquerade,
                window_size=window_size,
                countdown_duration_seconds=countdown_seconds,
            )
        except Exception as exc:
            if masquerade:
                raise SystemExit(
                    f"[visualiser] failed to create GUI ({exc}); --masquerade requires a window"
                ) from exc
            print(
                f"[visualiser] failed to create GUI ({exc}); falling back to headless mode",
                file=sys.stderr,
            )
            app = None
            headless = True

    stop_event = threading.Event()

    if app is not None:
        app.update_snapshot(
            Snapshot(
                status=(
                    "Connecting to engine..."
                    if masquerade
                    else f"Connecting to engine for {(delegate_script or DEFAULT_SUBMISSION).name}..."
                ),
                turn_duration_seconds=0.1,
            )
        )

    if masquerade:
        thread_target = lambda: run_masquerade_worker(app, stop_event)
    else:
        thread_target = lambda: run_delegated_worker(
            delegate_script or DEFAULT_SUBMISSION,
            app,
            stop_event,
        )
    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()

    if app is None:
        stop_event.wait()
        return
    app.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agar.io visualiser submission client."
    )
    parser.add_argument(
        "--masquerade",
        action="store_true",
        help="Use keyboard input so the visualiser itself acts as the submission.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the GUI while still running the submission logic.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=720,
        help="Window size in pixels for the square arena canvas.",
    )
    parser.add_argument(
        "--countdown-seconds",
        type=float,
        default=0.0,
        help="Display a startup countdown for this many seconds before the engine begins.",
    )
    parser.add_argument(
        "--delegate-script",
        type=Path,
        default=None,
        help="Run this submission inside the visualiser process and show its first-person view.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.headless and args.masquerade:
        raise SystemExit("--headless cannot be combined with --masquerade.")
    if args.masquerade and args.delegate_script is not None:
        raise SystemExit("--delegate-script cannot be combined with --masquerade.")
    run_submission(
        masquerade=args.masquerade,
        headless=args.headless,
        window_size=args.window_size,
        countdown_seconds=args.countdown_seconds,
        delegate_script=args.delegate_script,
    )


if __name__ == "__main__":
    main()
