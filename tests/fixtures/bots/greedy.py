"""Greedy reference bot — the traditional baseline the NN policy is trained against.

A single self-contained submission (same shape as the kit's ``example.py`` and the
``cautious`` / ``dont_move`` fixtures): it drives ``helper.game.Game`` and emits one
vector-heading ``MovePlayer`` per turn. No project-internal imports — only the
match-legal ``helper`` + ``lib`` surface — so this file can be submitted as-is and
also used as a scripted expert / opponent when generating training trajectories.

Decision order each turn (pure greedy, no lookahead):

1. **Survive.** Flee every enemy blob that could eat one of ours — directly, or by
   splitting onto us — as an inverse-distance-weighted repulsion.
2. **Avoid viruses that would pop us.** A blob big enough to *consume* a virus is
   shattered into pieces on contact, so we steer around viruses once our biggest
   blob can be popped and one is close.
3. **Hunt prey — punish the fragmented first.** Chase an enemy blob we out-mass
   whose *owner* we also out-mass whole (so a virus-split enemy can't lure us in
   and re-merge to eat us), preferring owners that are *fragmented* (≥2 visible
   blobs = freshly split / virus-popped) and small. Split to close only when a
   half-mass child still lands the eat and stays bigger than the owner's whole
   mass — and never during our fragile childhood.
4. **Raid a virus-split cluster.** Even when an enemy's *whole* mass out-sizes us,
   a fresh virus pop shatters it into up-to-16 pieces that are individually edible
   and can't re-merge for 18 ticks. If a forward simulation of the dive
   (:func:`_simulate_raid`) proves that, eating greedily during that window, we
   out-mass the leftover before it recombines — so we then devour the remnant too
   — we dive in. The theoretical ceiling is ≈13.3× our mass
   (:func:`max_raidable_mass_ceiling`); real reach is set by distance + the window.
5. **Feed / hunt (size-dependent), crowd-avoidant.** Above the break-even mass
   ``FOOD_MASS / MASS_DECAY_RATE`` grazing can't offset decay, so we bear down on
   the nearest safely-smaller player; below it we take the nearest **reachable**
   pellet. Either way the grazing/exploring heading is bent away from nearby enemy
   mass so we live in emptier space (edges emerge naturally) — more strongly while
   small.
6. **Explore — never idle.** Perpetual motion: sweep toward centre / open space
   every tick, never stop.

Counter-play vs. the observed TEAM-15 archetype (perpetual-motion, edge-hugging,
virus-pop-to-spread, compounding — but a 45%-bust *fragile childhood*, killed
young at mass 2–9): we adopt their robust winners (never idle, bend away from
traffic) but refuse their fragility (we do **not** eat viruses to spread). Instead
we *exploit* it — hunt fragmented/just-popped and sub-15-mass enemies on sight,
and keep our own childhood disciplined (no self-fragmentation, extra caution).

Eat criterion follows the engine (v2026.1.12+, ``_can_eat_blob``): eater **mass**
>= target **mass** * ``EAT_SIZE_RATIO`` (mass == radius**2), not a radius ratio.
"""

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

from lib.config.arena import ARENA_SIZE, MAX_BLOB_COUNT, VIRUS_SIZE
from lib.config.player import (
    BASE_PLAYER_SPEED,
    EAT_SIZE_RATIO,
    FOOD_RADIUS,
    MASS_DECAY_RATE,
    MERGE_ATTRACTION_SPEED,
    MIN_PLAYER_SPEED,
    PLAYER_SPEED_RADIUS_FACTOR,
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

# --- Split-cluster raid tuning -----------------------------------------------
#: Flat mass a blob gains by consuming a standard virus (radius**2); a popped
#: cluster's total mass is the pre-pop blob mass plus this.
VIRUS_MASS_GAIN = VIRUS_SIZE * VIRUS_SIZE
#: Secure the win (out-mass the remnant by the eat ratio) at least this many ticks
#: before the pieces can re-merge — slack against imperfect convergence / a fleeing
#: cluster. We also re-decide every tick, and Survive still pre-empts a real threat.
RAID_TIME_MARGIN = 2


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


def _speed(radius: float) -> float:
    """Per-tick blob speed (mirrors ``StateMutator._movement_speed``)."""
    return max(MIN_PLAYER_SPEED, BASE_PLAYER_SPEED / (1.0 + radius * PLAYER_SPEED_RADIUS_FACTOR))


def _blobs_by_player(blobs: Sequence) -> Dict[int, list]:
    grouped: Dict[int, list] = defaultdict(list)
    for b in blobs:
        grouped[b.player_id].append(b)
    return grouped


def max_raidable_mass_ceiling(my_mass: float, n_pieces: int = MAX_BLOB_COUNT) -> float:
    """Theoretical ceiling on a pre-pop enemy blob we could consume via a dive.

    A virus pop shatters mass into ``n_pieces`` equal pieces (a lone blob → 16) and
    adds ``VIRUS_MASS_GAIN``. A piece is edible only while our mass ≥
    ``EAT_SIZE_RATIO`` × piece mass, so the whole cluster is *in principle*
    consumable when its post-pop total ``T`` satisfies ``T / n ≤ my_mass /
    EAT_SIZE_RATIO``. Hence the biggest pre-pop blob, **ignoring travel time and
    geometry**, is ``n · my_mass / EAT_SIZE_RATIO − VIRUS_MASS_GAIN`` (≈ 13.3× our
    mass for n=16). Real reach is far lower — travel distance, piece spread, and
    the 18-tick merge window bind it — which is why the live decision *simulates*
    the dive (:func:`_simulate_raid`) on the actual cluster instead.
    """
    return n_pieces * my_mass / EAT_SIZE_RATIO - VIRUS_MASS_GAIN


def _simulate_raid(ox: float, oy: float, my_mass: float, pieces: Sequence, deadline: int) -> bool:
    """Forward-simulate diving a fresh split cluster and eating greedily.

    Each tick until ``deadline - RAID_TIME_MARGIN``: we step toward the (uneaten)
    cluster centroid at our size-dependent speed; the pieces drift to that centroid
    at ``MERGE_ATTRACTION_SPEED``; then we eat — cascading, because each eat grows
    us and may pull more piece centres inside our disc (mirrors the engine's
    ``while changed`` eat loop) — every piece we out-mass by the eat ratio and
    cover. Returns ``True`` as soon as we out-mass the uneaten *remnant* by the eat
    ratio (so once it re-merges we devour it too), having eaten ≥1 piece.
    """
    state = [[b.pos[0], b.pos[1], b.radius * b.radius, False] for b in pieces]  # x, y, mass, eaten
    mx, my_, mmass = ox, oy, my_mass
    horizon = max(1, deadline - RAID_TIME_MARGIN)
    for _t in range(horizon):
        rem = [p for p in state if not p[3]]
        if not rem:
            break
        tot = sum(p[2] for p in rem)
        cx = sum(p[0] * p[2] for p in rem) / tot
        cy = sum(p[1] * p[2] for p in rem) / tot

        mr = math.sqrt(mmass)
        v = _speed(mr)
        dx, dy = cx - mx, cy - my_
        d = math.hypot(dx, dy)
        if d > 1e-9:
            step = min(v, d)
            mx += dx / d * step
            my_ += dy / d * step

        for p in rem:  # pieces converge toward their common centroid
            dxp, dyp = cx - p[0], cy - p[1]
            dp = math.hypot(dxp, dyp)
            if dp > 1e-9:
                s = min(MERGE_ATTRACTION_SPEED, dp)
                p[0] += dxp / dp * s
                p[1] += dyp / dp * s

        mr = math.sqrt(mmass)
        changed = True
        while changed:  # cascade: growing as we eat sweeps in more pieces this tick
            changed = False
            for p in state:
                if p[3]:
                    continue
                if mmass >= p[2] * EAT_SIZE_RATIO and (mx - p[0]) ** 2 + (my_ - p[1]) ** 2 <= mr * mr:
                    mmass += p[2]
                    p[3] = True
                    mr = math.sqrt(mmass)
                    changed = True

        remnant = sum(p[2] for p in state if not p[3])
        if any(p[3] for p in state) and mmass >= EAT_SIZE_RATIO * remnant:
            return True
    return False


def _raid_target(ox: float, oy: float, my_mass: float, blobs: Sequence) -> Optional[Tuple[float, float]]:
    """Heading to dive the nearest *winnable* fresh split cluster, or ``None``.

    A cluster qualifies only if every visible blob is still on merge cooldown (so
    it cannot re-merge yet) and :func:`_simulate_raid` proves that, diving now, we
    out-mass the remnant by the eat ratio before it re-merges.
    """
    best = None
    best_d2 = math.inf
    for pieces in _blobs_by_player(blobs).values():
        if len(pieces) < 2:
            continue  # a lone blob isn't a split cluster
        if any(b.merge_cooldown <= 0 for b in pieces):
            continue  # something can already merge → not a clean window
        deadline = min(b.merge_cooldown for b in pieces)
        if deadline <= RAID_TIME_MARGIN:
            continue
        nd2 = min((b.pos[0] - ox) ** 2 + (b.pos[1] - oy) ** 2 for b in pieces)
        if nd2 >= best_d2:
            continue
        if _simulate_raid(ox, oy, my_mass, pieces, deadline):
            tot = sum(b.radius * b.radius for b in pieces)
            cx = sum(b.pos[0] * b.radius * b.radius for b in pieces) / tot
            cy = sum(b.pos[1] * b.radius * b.radius for b in pieces) / tot
            best_d2 = nd2
            best = (cx - ox, cy - oy)
    return best


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

    # 1 + 2. Survive: flee eat-threats and poppable viruses.
    flee = _avoidance_heading(ox, oy, my_min_mass, my_max_r, blobs, viruses)
    if flee is not None:
        return (flee[0], flee[1], False)

    # 3. Hunt prey — a blob we out-mass whose *owner* we also out-mass whole, so a
    #    virus-split enemy can't bait us into a re-merge. Counter-play: prefer
    #    *fragmented* owners (freshly split / virus-popped) so we punish TEAM-15
    #    right when they pop a virus to spread, denying the compounding.
    def prey_ok(b) -> bool:
        return (
            _can_eat(my_max_mass, b.radius * b.radius)
            and enemy_mass[b.player_id] < my_max_mass
        )

    prey = _nearest(ox, oy, blobs, key_ok=lambda b: prey_ok(b) and enemy_count[b.player_id] >= 2)
    if prey is None:
        prey = _nearest(ox, oy, blobs, key_ok=prey_ok)
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

    # 4. Raid a fresh virus-split cluster we don't out-mass *whole* but can still
    #    win by diving: eat pieces during their merge cooldown until we out-mass the
    #    remnant, then devour the recombined leftover too. Only commits when a
    #    forward simulation proves the win; Survive above still pre-empts any real
    #    threat every tick, and we re-decide as the cluster shrinks.
    raid = _raid_target(ox, oy, my_max_mass, blobs)
    if raid is not None:
        return (raid[0], raid[1], False)

    # 5. Size-dependent: big blobs can't live on food, so bear down on the nearest
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

    # 6. Explore — perpetual motion, never idle. Head to open space toward centre;
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
