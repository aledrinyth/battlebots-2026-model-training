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

#: Extra margin (world units) added to a virus's pop-contact distance before we
#: bother steering around it; farther viruses do not constrain a greedy heading.
_VIRUS_AVOID_BUFFER = 2.0

# --- TEAM-15 counter-play tuning ---------------------------------------------
#: Below this mass a blob has a "fragile childhood" (TEAM-15 bust their games at
#: mass 2–9): we never self-fragment and we weight crowd-avoidance harder.
CHILDHOOD_MASS = 15.0
#: Enemies within this range bend our grazing/exploring heading away (soft crowd
#: avoidance — "paths bend away from traffic before it's dangerous").
CROWD_RANGE = 12.0
#: Weight of the crowd-avoidance bias vs. the desired heading, adult vs. child.
CROWD_WEIGHT_ADULT = 0.6
CROWD_WEIGHT_CHILD = 1.6
#: Radians/round the idle sweep rotates so perpetual motion also covers ground.
SWEEP_RATE = 0.15

# --- Value-based targeting ----------------------------------------------------
#: We chase the prey with the highest *value* = mass / (dist + PREY_DIST_BIAS),
#: not merely the nearest — a big boost a little farther beats a tiny one close.
#: The bias (world units) softens the divide so an adjacent speck can't outrank a
#: much larger blob just out of reach; ~ a few blob radii.
PREY_DIST_BIAS = 4.0
#: Multiplier on a prey's value when its owner is *fragmented* (>=2 visible blobs
#: = freshly split / virus-popped) — preserves the TEAM-15 punish-the-weak
#: priority as a weighting rather than an absolute override.
FRAG_PREY_BONUS = 1.6
#: Only prefer a bigger-but-farther target within this range (world units, ~vision
#: size). Beyond it we don't over-commit — chasing far prey across contested space
#: is how a value-seeker gets picked off by aggressive opponents. A closer target
#: outside no candidate-within-range is still taken via the un-capped fallback.
CHASE_RANGE = 16.0

# --- Virus baiting ------------------------------------------------------------
#: When a heavy enemy is chasing us and we're too small to be popped ourselves,
#: we flee *toward* a virus that would shatter the pursuer (our hunt then mops up
#: the debris). Only bait with a virus whose bearing is at least this aligned
#: (cosine) with the raw flee heading, so leading the chaser onto it does not send
#: us back into its jaws.
BAIT_ALIGN_MIN = 0.2

# --- Wall / corner escape -----------------------------------------------------
#: Within this distance of a wall, a flee heading's into-the-wall component is
#: cancelled so we slide *along* the wall instead of jamming into it or a corner
#: and getting pinned there while an enemy closes in. Kept small so it engages
#: only when genuinely about to pin — the bot lives on edges, so a wide margin
#: would perturb ordinary direct-away fleeing and cost more than it saves.
WALL_MARGIN = 2.5


def _can_eat(eater_mass: float, target_mass: float) -> bool:
    """Engine eat rule (v2026.1.12+): eater mass >= target mass * EAT_SIZE_RATIO."""
    return eater_mass >= target_mass * EAT_SIZE_RATIO


def _unit(dx: float, dy: float) -> Tuple[float, float]:
    m = math.hypot(dx, dy)
    if m < 1e-12:
        return (0.0, 0.0)
    return (dx / m, dy / m)


def _my_masses(game: Game) -> Tuple[float, float]:
    """(min, max) blob mass across our blobs; falls back to whole-player radius."""
    masses = [b.radius * b.radius for b in game.state.me.blobs.values()]
    if not masses:
        m = game.state.me.radius * game.state.me.radius
        return m, m
    return min(masses), max(masses)


def _enemy_player_mass(blobs: Sequence) -> Dict[int, float]:
    """Total *visible* mass per enemy player_id — their post-merge potential."""
    totals: Dict[int, float] = defaultdict(float)
    for b in blobs:
        totals[b.player_id] += b.radius * b.radius
    return totals


def _enemy_blob_counts(blobs: Sequence) -> Dict[int, int]:
    """Visible blob count per enemy player_id — ≥2 means fragmented (just split)."""
    counts: Dict[int, int] = defaultdict(int)
    for b in blobs:
        counts[b.player_id] += 1
    return counts


def _virus_pop_mass(virus_radius: float) -> float:
    """Mass a blob must exceed to consume (and thus be popped by) this virus.

    Mirrors ``StateMutator._can_consume_virus``: ``blob.mass > virus_r**2 * ratio``.
    """
    return (virus_radius ** 2) * EAT_SIZE_RATIO


def _is_threat(enemy_mass: float, enemy_r: float, my_min_mass: float, dist: float) -> bool:
    """True if ``enemy`` can eat our weakest blob now, or by splitting onto it."""
    if _can_eat(enemy_mass, my_min_mass):
        return True  # out-masses us: contact is death
    if (
        enemy_mass >= SPLIT_MIN_MASS
        and _can_eat(enemy_mass * _SPLIT_CHILD_MASS, my_min_mass)
        and dist <= enemy_r + SPLIT_REACH
    ):
        return True
    return False


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


def _near_poppable_virus(ox: float, oy: float, my_max_r: float, viruses: Sequence) -> bool:
    """True if a virus we could pop sits within our split reach — splitting risky."""
    my_max_mass = my_max_r * my_max_r
    for v in viruses:
        if my_max_mass <= _virus_pop_mass(v.radius):
            continue
        if math.hypot(ox - v.pos[0], oy - v.pos[1]) <= my_max_r + SPLIT_REACH:
            return True
    return False


def _crowd_bias(ox: float, oy: float, blobs: Sequence) -> Tuple[float, float]:
    """Unit vector away from nearby enemy *mass* (0,0 if the coast is clear).

    Mass- and closeness-weighted repulsion from every visible enemy blob within
    ``CROWD_RANGE`` (linear falloff to zero at the edge). Blended into the
    grazing/exploring heading it makes our path bend away from traffic before it
    is dangerous — the TEAM-15 "edge-living halves the directions trouble comes
    from" effect, but dynamic (no static wall-hugging → no corner trap).
    """
    fx = fy = 0.0
    for b in blobs:
        dx = ox - b.pos[0]
        dy = oy - b.pos[1]
        d = math.hypot(dx, dy)
        if d >= CROWD_RANGE or d < 1e-9:
            continue
        w = (b.radius * b.radius) * (CROWD_RANGE - d) / CROWD_RANGE
        fx += (dx / d) * w
        fy += (dy / d) * w
    return _unit(fx, fy)


def _avoidance_heading(
    ox: float,
    oy: float,
    my_min_mass: float,
    my_max_r: float,
    blobs: Sequence,
    viruses: Sequence,
) -> Optional[Tuple[float, float]]:
    """Inverse-distance repulsion from eat-threats + poppable nearby viruses, or
    ``None`` if nothing is dangerous this turn."""
    fx = fy = 0.0
    danger = False

    for b in blobs:
        dx = ox - b.pos[0]
        dy = oy - b.pos[1]
        dist = math.hypot(dx, dy)
        if not _is_threat(b.radius * b.radius, b.radius, my_min_mass, dist):
            continue
        danger = True
        if dist < 1e-9:
            fx += _ARENA_CENTER - ox  # overlapping: bolt toward centre
            fy += _ARENA_CENTER - oy
            continue
        w = 1.0 / dist  # nearer threats dominate the resultant
        fx += (dx / dist) * w
        fy += (dy / dist) * w

    # Viruses matter only once our biggest blob could actually be popped, and the
    # engine pops on centre-overlap (within our own radius), so the danger
    # distance is ~our radius plus a small margin.
    my_max_mass = my_max_r * my_max_r
    for v in viruses:
        if my_max_mass <= _virus_pop_mass(v.radius):
            continue
        dx = ox - v.pos[0]
        dy = oy - v.pos[1]
        dist = math.hypot(dx, dy)
        if dist > my_max_r + _VIRUS_AVOID_BUFFER:
            continue
        danger = True
        if dist < 1e-9:
            fx += _ARENA_CENTER - ox
            fy += _ARENA_CENTER - oy
            continue
        w = 1.0 / dist
        fx += (dx / dist) * w
        fy += (dy / dist) * w

    if not danger or (fx * fx + fy * fy) < 1e-12:
        return None
    return (fx, fy)


def _wall_escape(fx: float, fy: float, ox: float, oy: float, r: float) -> Tuple[float, float]:
    """Redirect a flee heading so we slide *along* walls instead of pinning.

    Any component pushing us into a wall within ``WALL_MARGIN`` is cancelled,
    turning "flee straight into the wall" into "flee along it" — which rounds a
    corner rather than jamming into it. If that cancels the whole heading (a
    corner with the enemy in the only open direction), we fall back to the
    wall-tangent escape that heads *least* toward the enemy: the interior-pointing
    axis direction with the largest dot against the original repulsion.
    """
    s = ARENA_SIZE
    fx0, fy0 = fx, fy
    near_lo_x = ox - r < WALL_MARGIN
    near_hi_x = (s - r) - ox < WALL_MARGIN
    near_lo_y = oy - r < WALL_MARGIN
    near_hi_y = (s - r) - oy < WALL_MARGIN
    if near_lo_x and fx < 0.0:
        fx = 0.0
    if near_hi_x and fx > 0.0:
        fx = 0.0
    if near_lo_y and fy < 0.0:
        fy = 0.0
    if near_hi_y and fy > 0.0:
        fy = 0.0
    if fx * fx + fy * fy > 1e-9:
        return (fx, fy)
    # Fully blocked: slide along a wall. Build the interior-pointing tangents on
    # each free-enough axis and take the one that moves least toward the enemy.
    cands = []
    if not (near_lo_x and near_hi_x):
        sx = 1.0 if near_lo_x else (-1.0 if near_hi_x else (1.0 if fx0 >= 0.0 else -1.0))
        cands.append((sx, 0.0))
    if not (near_lo_y and near_hi_y):
        sy = 1.0 if near_lo_y else (-1.0 if near_hi_y else (1.0 if fy0 >= 0.0 else -1.0))
        cands.append((0.0, sy))
    if not cands:  # boxed on all four sides (degenerate) — head for the centre
        return (_ARENA_CENTER - ox, _ARENA_CENTER - oy)
    return max(cands, key=lambda c: c[0] * fx0 + c[1] * fy0)


def _bait_virus_heading(
    ox: float,
    oy: float,
    my_max_mass: float,
    flee_dir: Tuple[float, float],
    threat_masses: Sequence,
    viruses: Sequence,
) -> Optional[Tuple[float, float]]:
    """Heading toward a virus we can lure the pursuer into, or ``None``.

    A blob big enough to *consume* a virus is shattered by it. So when a heavy
    enemy is chasing us we can flee toward a virus it would pop, then our hunt
    mops up the debris. We only bait when:

    * we are too small to be popped by that virus ourselves (``my_max_mass <=
      pop_mass``) — pathing onto it is safe for us, and
    * the heaviest pursuer *can* be popped (its mass > ``pop_mass``), and
    * the virus lies roughly in our flee direction (cosine >= ``BAIT_ALIGN_MIN``)
      so leading the chaser onto it doesn't double us back into danger.

    Among qualifying viruses we pick the best-aligned, nearest one.
    """
    if not threat_masses:
        return None
    max_threat_mass = max(threat_masses)
    fdx, fdy = flee_dir
    if fdx == 0.0 and fdy == 0.0:
        return None
    best = None
    best_score = 0.0
    for v in viruses:
        pop = _virus_pop_mass(v.radius)
        if my_max_mass > pop:        # a virus that would pop us too — unsafe
            continue
        if max_threat_mass <= pop:   # pursuer can't be popped — no bait value
            continue
        dx, dy = v.pos[0] - ox, v.pos[1] - oy
        d = math.hypot(dx, dy)
        if d < 1e-6:
            continue
        align = (dx / d) * fdx + (dy / d) * fdy
        if align < BAIT_ALIGN_MIN:
            continue
        score = align / (d + 1.0)    # aligned + near = a shield we reach quickly
        if score > best_score:
            best_score = score
            best = (dx, dy)
    return best


def _prey_value(target_mass: float, dist: float, fragmented: bool) -> float:
    """Chase value of a prey blob: mass reward, distance-discounted, with a bonus
    for fragmented (freshly-split / just-popped) owners."""
    value = target_mass / (dist + PREY_DIST_BIAS)
    if fragmented:
        value *= FRAG_PREY_BONUS
    return value


def _best_by_value(ox: float, oy: float, items: Sequence, key_ok, frag_of=None,
                   max_dist: float = math.inf):
    """Highest-``_prey_value`` item passing ``key_ok`` within ``max_dist`` →
    (dx, dy, dist, item). Falls back to the nearest passing item if every
    candidate is beyond ``max_dist`` (so we still take a lone far kill rather than
    starve), but never *prefers* a far big target over a near one."""
    best = None
    best_val = 0.0
    nearest = None
    nearest_d = math.inf
    for it in items:
        if not key_ok(it):
            continue
        dx = it.pos[0] - ox
        dy = it.pos[1] - oy
        dist = math.hypot(dx, dy)
        if dist < nearest_d:
            nearest_d = dist
            nearest = (dx, dy, dist, it)
        if dist > max_dist:
            continue
        frag = frag_of(it) if frag_of is not None else False
        val = _prey_value(it.radius * it.radius, dist, frag)
        if val > best_val:
            best_val = val
            best = (dx, dy, dist, it)
    return best if best is not None else nearest


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


def _steer(target: Tuple[float, float], ox: float, oy: float, blobs: Sequence, childhood: bool) -> Tuple[float, float]:
    """Blend a desired heading with crowd-avoidance (stronger while small)."""
    tx, ty = _unit(*target)
    bx, by = _crowd_bias(ox, oy, blobs)
    if bx == 0.0 and by == 0.0:
        return (tx, ty) if (tx or ty) else target
    weight = CROWD_WEIGHT_CHILD if childhood else CROWD_WEIGHT_ADULT
    return (tx + bx * weight, ty + by * weight)


def choose_move(game: Game) -> Tuple[float, float, bool]:
    """Return the greedy ``(dx, dy, split)`` for this turn."""
    me = game.state.me
    ox, oy = me.x, me.y
    my_min_mass, my_max_mass = _my_masses(game)
    my_max_r = math.sqrt(my_max_mass)
    childhood = my_max_mass < CHILDHOOD_MASS
    blobs = game.state.visible_blobs
    viruses = game.state.visible_viruses
    enemy_mass = _enemy_player_mass(blobs)
    enemy_count = _enemy_blob_counts(blobs)

    # 1 + 2. Survive: flee eat-threats and poppable viruses. When a heavy pursuer
    #    can be lured into a virus (and we're too small to be popped ourselves),
    #    bend the flee toward that virus to shatter it; then wall-slide the heading
    #    so we escape along walls instead of pinning in a corner.
    flee = _avoidance_heading(ox, oy, my_min_mass, my_max_r, blobs, viruses)
    if flee is not None:
        heading = flee
        threat_masses = [
            b.radius * b.radius
            for b in blobs
            if _is_threat(
                b.radius * b.radius,
                b.radius,
                my_min_mass,
                math.hypot(ox - b.pos[0], oy - b.pos[1]),
            )
        ]
        bait = _bait_virus_heading(
            ox, oy, my_max_mass, _unit(*flee), threat_masses, viruses
        )
        if bait is not None:
            heading = bait
        heading = _wall_escape(heading[0], heading[1], ox, oy, my_max_r)
        return (heading[0], heading[1], False)

    # 3. Hunt prey — a blob we out-mass whose *owner* we also out-mass whole, so a
    #    virus-split enemy can't bait us into a re-merge. Counter-play: prefer
    #    *fragmented* owners (freshly split / virus-popped) so we punish TEAM-15
    #    right when they pop a virus to spread, denying the compounding.
    def prey_ok(b) -> bool:
        return (
            _can_eat(my_max_mass, b.radius * b.radius)
            and enemy_mass[b.player_id] < my_max_mass
        )

    #    Pick by *value* (mass, distance-discounted) rather than nearest, so a big
    #    boost a little farther beats a tiny one underfoot; fragmented owners carry
    #    a value bonus so they still get punished first.
    prey = _best_by_value(
        ox, oy, blobs, key_ok=prey_ok,
        frag_of=lambda b: enemy_count[b.player_id] >= 2, max_dist=CHASE_RANGE,
    )
    if prey is not None:
        dx, dy, dist, target = prey
        child_mass = my_max_mass * _SPLIT_CHILD_MASS
        can_split = (
            not childhood  # never self-fragment during the fragile childhood
            and my_max_mass >= SPLIT_MIN_MASS
            and len(me.blobs) <= 1
            and my_max_r < dist <= my_max_r + SPLIT_REACH
            and _can_eat(child_mass, target.radius * target.radius)
            and enemy_mass[target.player_id] < child_mass
            and not _near_poppable_virus(ox, oy, my_max_r, viruses)
        )
        return (dx, dy, can_split)

    # 4. Size-dependent: big blobs can't live on food, so bear down on the nearest
    #    player we safely out-mass (whole) instead of grazing.
    if my_max_mass >= HUNT_MODE_MASS:
        hunt = _nearest(ox, oy, blobs, key_ok=lambda b: enemy_mass[b.player_id] < my_max_mass)
        if hunt is not None:
            return (hunt[0], hunt[1], False)

    # Feed: nearest *reachable* pellet (unreachable corner/edge food is skipped),
    #    steered away from nearby crowds so we graze in open space.
    food = _nearest(ox, oy, game.state.visible_food,
                    key_ok=lambda f: _food_reachable(f.pos[0], f.pos[1], my_max_r))
    if food is not None:
        return _steer((food[0], food[1]), ox, oy, blobs, childhood) + (False,)

    # 5. Explore — perpetual motion, never idle. Head to open space toward centre;
    #    if already essentially at centre, sweep a slowly-rotating heading so we
    #    keep covering ground. Both are bent away from nearby crowds.
    to_center = (_ARENA_CENTER - ox, _ARENA_CENTER - oy)
    if to_center[0] * to_center[0] + to_center[1] * to_center[1] < 1.0:
        ang = game.state.round * SWEEP_RATE
        to_center = (math.cos(ang), math.sin(ang))
    return _steer(to_center, ox, oy, blobs, childhood) + (False,)


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


if __name__ == "__main__":
    main()
