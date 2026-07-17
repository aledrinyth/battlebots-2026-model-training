"""spar_aggro — AGGRESSIVE SPARRING BOT (hand-written; NOT a ship candidate).

Purpose: reproduce the early player-hunting pressure that the v8/v10/v11/v12
self-play pool lacks — the thing that actually beats us in real tournaments
(see memory pf-v11-tournament-losses: winners get ~145 player-kills to our 17).

Design centres on the ONE mechanic that makes player-kills possible in this
engine: bigger blobs are SLOWER than smaller ones (speed = max(0.25, 1.1/(1+
0.08 r)) decreases with radius), so a hunter can never simply chase down prey —
it must SPLIT-LUNGE (a split child travels 1.6/(1-0.82)=8.9 units of free
travel). So the hunter:
  1. SURVIVE — flee anything that can eat one of our blobs next turn (hard floor).
  2. KILL — when a smaller enemy sits inside split reach and a half of our blob
     still out-masses it (and the split can't be punished), split toward it.
  3. otherwise FORAGE food (gravity well) to reach split mass fast.

Kept deliberately simple and independent of the v-series code so it is a real
(non-shared-blind-spot) stressor. Standalone submission: kit + stdlib only.
"""
from __future__ import annotations

import gc
import json
import math
from typing import Tuple

from helper.interface import Connection
from lib.config.arena import ARENA_SIZE, MAX_BLOB_COUNT, VIRUS_SIZE
from lib.config.player import (
    BASE_PLAYER_SPEED,
    EAT_SIZE_RATIO,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
    SPLIT_EJECT_DRAG,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
)
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

SPLIT_REACH = SPLIT_EJECT_SPEED / (1.0 - SPLIT_EJECT_DRAG)  # ~8.89
RATIO = EAT_SIZE_RATIO


def _speed(r: float) -> float:
    return max(MIN_PLAYER_SPEED, BASE_PLAYER_SPEED / (1.0 + r * PLAYER_SPEED_RADIUS_FACTOR))


def _decide(query) -> Tuple[float, float, bool]:
    you = query.you
    blobs = you.blobs
    if not blobs:
        return 0.0, 0.0, False
    cx, cy = you.pos
    my_max_r = max(b.radius for b in blobs)
    my_max_mass = my_max_r * my_max_r
    my_min_mass = min(b.radius * b.radius for b in blobs)
    n_blobs = len(blobs)

    threats = []   # (dist, blob, mass) — can eat one of our blobs
    prey = []      # (dist, blob, mass) — our biggest blob can eat it
    for b in query.visible_blobs:
        if b.player_id == you.player_id:
            continue
        bm = b.radius * b.radius
        d = math.hypot(b.pos[0] - cx, b.pos[1] - cy)
        if bm >= RATIO * my_min_mass:
            threats.append((d, b, bm))
        if my_max_mass >= RATIO * bm:
            prey.append((d, b, bm))

    # 1. SURVIVE
    flee_x = flee_y = 0.0
    danger = False
    for d, b, bm in threats:
        reach = b.radius + _speed(b.radius)
        if bm >= SPLIT_MIN_MASS and bm * 0.5 >= RATIO * my_max_mass:
            reach += SPLIT_REACH
        gap = d - reach
        if gap <= 2.5:
            danger = True
            if d > 1e-6:
                w = 1.0 / (max(gap, 0.05) ** 2)
                flee_x += (cx - b.pos[0]) / d * w
                flee_y += (cy - b.pos[1]) / d * w
    if danger and (abs(flee_x) > 1e-9 or abs(flee_y) > 1e-9):
        m = ARENA_SIZE * 0.5
        flee_x += (m - cx) * 0.02
        flee_y += (m - cy) * 0.02
        return flee_x, flee_y, False

    # 2. KILL: safe split-lunge
    half = my_max_mass * 0.5
    if my_max_mass >= 2.0 * SPLIT_MIN_MASS and n_blobs < MAX_BLOB_COUNT:
        best = None
        for d, b, bm in prey:
            if half < RATIO * bm * 1.05:
                continue
            if not (my_max_r * 0.5 < d <= my_max_r + SPLIT_REACH):
                continue
            punished = False
            for d2, tb, tbm in threats:
                if tb is b:
                    continue
                if tbm >= RATIO * half and d2 <= tb.radius + _speed(tb.radius) + SPLIT_REACH + 2.0:
                    punished = True
                    break
            if punished:
                continue
            if best is None or d < best[0]:
                best = (d, b)
        if best is not None:
            _, b = best
            return b.pos[0] - cx, b.pos[1] - cy, True

    # 2b. STALK
    target = None
    if prey and my_max_mass >= SPLIT_MIN_MASS:
        target = min(prey, key=lambda t: t[0])[1]

    # 3. FORAGE (gravity well)
    fx = fy = 0.0
    for f in query.visible_food:
        dx = f.pos[0] - cx
        dy = f.pos[1] - cy
        d2 = dx * dx + dy * dy
        if d2 < 1e-6:
            continue
        w = 1.0 / d2
        inv = 1.0 / math.sqrt(d2)
        fx += dx * inv * w
        fy += dy * inv * w
    if target is not None:
        tx, ty = target.pos[0] - cx, target.pos[1] - cy
        tn = math.hypot(tx, ty)
        if tn > 1e-6:
            fx += tx / tn * 0.6
            fy += ty / tn * 0.6
    if abs(fx) < 1e-9 and abs(fy) < 1e-9:
        m = ARENA_SIZE * 0.5
        fx, fy = (m - cx), (m - cy)
    margin = 5.0
    if cx < margin:
        fx += (margin - cx) * 0.3
    elif cx > ARENA_SIZE - margin:
        fx -= (cx - (ARENA_SIZE - margin)) * 0.3
    if cy < margin:
        fy += (margin - cy) * 0.3
    elif cy > ARENA_SIZE - margin:
        fy -= (cy - (ARENA_SIZE - margin)) * 0.3
    return fx, fy, False


def _zero(pid):
    return MovePlayer(player_id=pid, direction=DirectionModel(x=0.0, y=0.0), split=False)


def main() -> None:
    conn = Connection()
    gc.collect()
    gc.disable()
    player_id = None
    while True:
        try:
            raw = conn._receive()
        except EOFError:
            return
        query = None
        try:
            query = QueryMovePlayer.model_validate_json(raw)
            player_id = query.you.player_id
        except Exception:
            if player_id is None:
                try:
                    player_id = int(json.loads(raw)["you"]["player_id"])
                except Exception:
                    player_id = 0
        try:
            dx, dy, split = _decide(query)
            move = MovePlayer(player_id=player_id,
                              direction=DirectionModel(x=float(dx), y=float(dy)),
                              split=bool(split))
        except Exception:
            move = _zero(player_id)
        conn.send_move(move)


if __name__ == "__main__":
    main()
