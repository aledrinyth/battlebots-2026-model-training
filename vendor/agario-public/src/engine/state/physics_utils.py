import math

from lib.config.player import PENGUIN_DRAG, PENGUIN_RADIUS
from lib.config.arena import DT
from lib.interact.penguin import Penguin


def apply_drag_2d(vx: float, vy: float) -> tuple[float, float]:
    speed: float = math.sqrt(vx * vx + vy * vy)
    if speed == 0:
        return 0.0, 0.0

    new_speed: float = max(0.0, speed - PENGUIN_DRAG * DT)
    scale: float = new_speed / speed
    return vx * scale, vy * scale

def is_colliding(p1: Penguin, p2: Penguin) -> bool:
    dx: float = p1.x - p2.x
    dy: float = p1.y - p2.y
    return dx*dx + dy*dy <= (2 * PENGUIN_RADIUS) ** 2

def resolve_collision(p1: Penguin, p2: Penguin) -> None:
    dx: float = p2.x - p1.x
    dy: float = p2.y - p1.y
    dist: float = math.sqrt(dx*dx + dy*dy)

    # Normal vector
    if dist == 0:
        rvx: float = p2.vx - p1.vx
        rvy: float = p2.vy - p1.vy
        rv_mag: float = math.sqrt(rvx * rvx + rvy * rvy)
        if rv_mag > 0:
            nx = rvx / rv_mag
            ny = rvy / rv_mag
        else:
            # Deterministic fallback normal when both overlap and have identical velocity.
            nx = 1.0
            ny = 0.0
        dist = 0.0
    else:
        nx = dx / dist
        ny = dy / dist

    # Overlap correction
    overlap: float = 2 * PENGUIN_RADIUS - dist
    if overlap > 0:
        p1.x -= nx * overlap / 2
        p1.y -= ny * overlap / 2
        p2.x += nx * overlap / 2
        p2.y += ny * overlap / 2

    # Relative velocity
    dvx: float = p2.vx - p1.vx
    dvy: float = p2.vy - p1.vy

    # Velocity along normal
    rel_vel: float = dvx * nx + dvy * ny

    # If moving away, ignore
    if rel_vel >= 0:
        return

    # Elastic collision impulse (equal mass)
    # With rel_vel = (v2 - v1) dot n, equal-mass elastic response uses rel_vel directly.
    impulse: float = rel_vel

    p1.vx += impulse * nx
    p1.vy += impulse * ny
    p2.vx -= impulse * nx
    p2.vy -= impulse * ny

def is_moving(p: Penguin) -> bool:
    return p.vx * p.vx + p.vy * p.vy > 0.1 * 0.1
