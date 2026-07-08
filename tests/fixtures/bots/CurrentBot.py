from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.events.moves.typing import MoveType
from lib.interface.queries.query_move import QueryMovePlayer
from lib.interface.queries.typing import QueryType
from lib.models.penguin_model import DirectionModel

from lib.config.arena import ARENA_SIZE
from lib.config.player import (
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    MASS_DECAY_RATE,
    SPLIT_EJECT_DRAG,
    SPLIT_EJECT_SPEED,
    SPLIT_MIN_MASS,
)

# --- Derived constants (computed from engine primitives, never hard-coded) ---

#: Total distance a freshly-launched split child coasts on eject momentum before
#: the velocity cutoff: geometric sum ``SPLIT_EJECT_SPEED / (1 - SPLIT_EJECT_DRAG)``.
#: This — not an enemy's body radius — is the true reach of a splitting attacker.
SPLIT_REACH = SPLIT_EJECT_SPEED / (1.0 - SPLIT_EJECT_DRAG)

#: A split child keeps half the parent's mass.
_SPLIT_CHILD_MASS = 0.5

#: Mass one food pellet adds (``r' = sqrt(r**2 + FOOD_RADIUS**2)`` → +FOOD_RADIUS**2).
FOOD_MASS = FOOD_RADIUS * FOOD_RADIUS

#: Break-even mass: at this size one pellet/tick exactly offsets ``MASS_DECAY_RATE``
#: mass lost/tick, so beyond it grazing cannot sustain the blob → hunt players.
HUNT_MODE_MASS = FOOD_MASS / MASS_DECAY_RATE

_ARENA_CENTER = ARENA_SIZE / 2.0

#: Threats within this radius are considered
_PREDATOR_INFLUENCE = 2 * SPLIT_REACH + 3             # big blobs tend to split once or twice(v27)
_GROW_VIRUS_INFLUENCE = 10.0
_VIRUS_INFLUENCE = 2.0

#: Multipliers to determine importance of objects
_ATTRACTIVE_MULTIPLIER = 0.05
_REPULSIVE_MULTIPLIER = 1.0
_PASSIVE_FOOD_MULTIPLIER = 1.25
_HUNT_FOOD_MULTPLIER = 0.3
_PASSIVE_PREY_MULTIPLIER = 5.0
_HUNT_PREY_MULTIPLIER = 8.0
_PREDATOR_MULTIPLIER = 5.0
_GROW_VIRUS_MULTIPLIER = 4.0
_VIRUS_MULTIPLIER = 0.25

def _my_masses(game: Game) -> Tuple[float, float]:
    """(min, max) blob mass across our blobs; falls back to whole-player radius."""
    masses = [b.radius * b.radius for b in game.state.me.blobs.values()]
    if not masses:
        m = game.state.me.radius * game.state.me.radius
        return m, m, m
    return min(masses), max(masses), sum(masses)

def _enemy_player_mass(blobs: Sequence, ox: float, oy: float) -> Dict[int, float]:
    """Total *visible* mass per enemy player_id — their post-merge potential."""
    totals: Dict[int, float] = defaultdict(float)
    for b in blobs:
        dx = b.pos[0] - ox
        dy = b.pos[1] - oy
        d2 = dx * dx + dy * dy
        if d2 > 250:      # in case players try double-split us when split by virus(v26)
            continue
        totals[b.player_id] += b.radius * b.radius
    return totals

def _virus_pop_mass(virus_radius: float) -> float:
    """Mass a blob must exceed to consume (and thus be popped by) this virus.

    Mirrors ``StateMutator._can_consume_virus``: ``blob.mass > virus_r**2 * ratio``.
    """
    return (virus_radius ** 2) * EAT_SIZE_RATIO

def _food_reachable(fx: float, fy: float, r: float) -> bool:
    """Can a blob of radius ``r`` ever bring pellet ``(fx, fy)`` inside its disc?

    The centre is wall-clamped to ``[r, S-r]`` and food is eaten within the blob
    radius, so the pellet is reachable iff the closest legal centre is within
    ``r`` of it. Corner/deep-edge pellets fail this and are skipped, which stops a
    big blob jamming itself in a corner it can never eat out of.
    """
    s = ARENA_SIZE
    lo, hi = r, s - r
    if lo > hi:  # blob wider than the arena half — degenerate, treat as reachable
        return True
    cx = min(max(lo, fx), hi)
    cy = min(max(lo, fy), hi)
    return (cx - fx) ** 2 + (cy - fy) ** 2 <= r * r

def _nearest(ox: float, oy: float, items: Sequence, key_ok=None):
    """Nearest item to ``(ox, oy)`` passing ``key_ok`` → (dx, dy, dist, item)."""
    best = None
    best_d2 = math.inf
    for it in items:
        if key_ok is not None and not key_ok(it):
            continue
        dx = it.pos[0] - ox
        dy = it.pos[1] - oy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = (dx, dy, math.sqrt(d2), it)
    return best

def choose_move(game: Game) -> Tuple[float, float, bool]:
    """Return the ``(dx, dy, split)`` for this turn.
       Uses a greedy algorithm to maximise food eaten below 2 mass.
       Eats viruses to quickly grow after that.
       When big enough, eat blobs & viruses to sustain mass."""
    me = game.state.me
    ox, oy = me.x, me.y
    my_min_mass, my_max_mass, my_total_mass = _my_masses(game)
    my_max_r = math.sqrt(my_max_mass)
    blobs = game.state.visible_blobs
    viruses = game.state.visible_viruses
    food = game.state.visible_food
    num_blobs = max(1, len(blobs))          # number of blobs we have(v27)

    if my_total_mass < 1.5:           # we can start killing quickly(v27)
        for b in blobs:
            mass = b.radius * b.radius
            dx = b.pos[0] - ox
            dy = b.pos[1] - oy
            d2 = dx * dx + dy * dy
            if d2 > 90:
                continue
            if mass > EAT_SIZE_RATIO * my_total_mass:
                dx = b.pos[0] - ox
                dy = b.pos[1] - oy
                return (-dx, -dy, False)
        food = _nearest(ox, oy, food, key_ok=lambda f: _food_reachable(f.pos[0], f.pos[1], my_max_r))
        if food is not None:
            return (food[0], food[1], False)

    direction = [0.0,0.0,False]
    predators = 0
    virus_predators = 0
    prey = []
    if my_total_mass > HUNT_MODE_MASS and food is not None:
        for f in food:
            if not _food_reachable(f.pos[0], f.pos[1], my_max_r):
                continue
            dx = f.pos[0] - ox
            dy = f.pos[1] - oy
            d2 = dx * dx + dy * dy
            multiplier = _ATTRACTIVE_MULTIPLIER * _HUNT_FOOD_MULTPLIER / d2
            direction[0] += multiplier * dx
            direction[1] += multiplier * dy

    elif food is not None:
        for f in food:
            if not _food_reachable(f.pos[0], f.pos[1], my_max_r):
                continue
            dx = f.pos[0] - ox
            dy = f.pos[1] - oy
            d2 = dx * dx + dy * dy
            multiplier = _ATTRACTIVE_MULTIPLIER * _PASSIVE_FOOD_MULTIPLIER / d2
            direction[0] += multiplier * dx
            direction[1] += multiplier * dy

    for b in blobs:
        mass = b.radius * b.radius
        split_blobs = 17 - num_blobs
        if split_blobs > 1 and my_max_mass / split_blobs <= EAT_SIZE_RATIO * mass:  # if we have 16 blobs eat viruses(v28)
            virus_predators += 1       # new counter to keep track of blobs that can eat us if we pop(v27)
        if mass * EAT_SIZE_RATIO >= my_min_mass:         # if they are predator
            dx = b.pos[0] - ox
            dy = b.pos[1] - oy
            d2 = dx * dx + dy * dy
            if d2 > _PREDATOR_INFLUENCE * _PREDATOR_INFLUENCE:
                continue
            predators += 1
            multiplier = _REPULSIVE_MULTIPLIER * _PREDATOR_MULTIPLIER * mass / d2
            direction[0] -= multiplier * dx
            direction[1] -= multiplier * dy

        elif my_max_mass >= mass * EAT_SIZE_RATIO:       # if they are prey
            dx = b.pos[0] - ox
            dy = b.pos[1] - oy
            d2 = dx * dx + dy * dy
            if d2 < SPLIT_REACH * SPLIT_REACH:           # split-kill immediately(v26)
                return (dx,dy,True)
            if mass >= HUNT_MODE_MASS:
                multiplier = _HUNT_PREY_MULTIPLIER
            else:
                multiplier = _PASSIVE_PREY_MULTIPLIER
            multiplier *= _ATTRACTIVE_MULTIPLIER / d2
            direction[0] += multiplier * dx
            direction[1] += multiplier * dy

    for v in viruses:
        dx = v.pos[0] - ox
        dy = v.pos[1] - oy
        d2 = dx * dx + dy * dy
        pop_mass = _virus_pop_mass(v.radius)
        if my_max_mass < pop_mass:
            break
        elif virus_predators==0 and d2 <= _GROW_VIRUS_INFLUENCE * _GROW_VIRUS_INFLUENCE:      # bigger range for eating viruses
            multiplier = -_GROW_VIRUS_MULTIPLIER               # eat viruses to grow quickly(v<25)
        elif d2 <= _VIRUS_INFLUENCE * _VIRUS_INFLUENCE:
            multiplier = _VIRUS_MULTIPLIER
        multiplier *= _REPULSIVE_MULTIPLIER / d2
        direction[0] -= multiplier * dx
        direction[1] -= multiplier * dy

    if direction[0] * direction[0] + direction[1] * direction[1] > 1e-6:       # just move towards centre if no best choice(v<20)
        if predators == 0 and len(prey)==1 and  _SPLIT_CHILD_MASS * my_total_mass >= EAT_SIZE_RATIO * prey[0] and len(blobs) < 8: # doesn't repeated split(v25)
            direction[2] = True
        elif predators > 0:                   # avoids fleeing into edge(v25)
            xval = abs(direction[0])
            yval = abs(direction[1])
            if xval < yval:
                if ox < my_max_r + 3 and direction[0] < 0:
                    direction[0] *= -1
                elif ox > ARENA_SIZE - (my_max_r + 3) and direction[0] > 0:
                    direction[0] *= -1
                elif oy < my_max_r + 3 and direction[1] < 0:
                    direction[1] *= -1
                elif oy > ARENA_SIZE - (my_max_r + 3) and direction[1] > 0:
                    direction[1] *= -1
            else:
                if oy < my_max_r + 3 and direction[1] < 0:
                    direction[1] *= -1
                elif oy > ARENA_SIZE - (my_max_r + 3) and direction[1] > 0:
                    direction[1] *= -1
                elif ox < my_max_r + 3 and direction[0] < 0:
                    direction[0] *= -1
                elif ox > ARENA_SIZE - (my_max_r + 3) and direction[0] > 0:
                    direction[0] *= -1
        return tuple(direction)

    dx, dy = _ARENA_CENTER - ox, _ARENA_CENTER - oy
    if dx * dx + dy * dy < 1e-4:
        return (0.0, 0.0, False)
    return (dx, dy, False)

def main() -> None:
    game = Game()

    while True:
        query = game.get_next_query()

        def build_move(query: QueryType) -> MoveType:
            match query:
                case QueryMovePlayer():
                    dx, dy, split = choose_move(game)
                    return MovePlayer(
                        player_id=game.state.me.player_id,
                        direction=DirectionModel(x=dx, y=dy),
                        split=split,
                    )
            raise RuntimeError(f"Unsupported query type: {type(query)}")

        game.send_move(build_move(query))

if __name__ == '__main__':