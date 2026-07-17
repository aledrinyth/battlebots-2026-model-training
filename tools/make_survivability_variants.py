#!/usr/bin/env python3
"""Generate early-game-survivability variants of potential_field_v11.

Each variant is a standalone copy of dist/potential_field_v11.py with a small,
localized patch to the SHARED early-game core (FEED forager, virus detour,
_safe_heading death-zone veto — byte-identical across v8/v10/v11). Techniques:

  v11a  mass-adaptive death-zone buffer  — widen the no-go disk while tiny
  v11b  mass-adaptive FEED flee margin   — flee from further away while tiny
  v11c  crowd avoidance in FEED          — steer away from dense predator regions
  v11d  strict virus-detour gate         — 18-tick lockout window, not 1-tick
  v11e  min-mass HUNT gate               — stay in the safe forager until bigger
  v11f  wall avoidance in FEED           — don't forage into corners while tiny
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = (REPO / "dist/potential_field_v11.py").read_text()

FIELD_ANCHOR = "                                         # shatter tiny).\n"


def patch(src: str, subs: list[tuple[str, str]]) -> str:
    for old, new in subs:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"anchor found {n}x (want 1):\n{old[:120]!r}")
        src = src.replace(old, new)
    return src


def add_fields(src: str, fields: str) -> str:
    return patch(src, [(FIELD_ANCHOR, FIELD_ANCHOR + fields)])


VARIANTS: dict[str, callable] = {}


def variant(name):
    def deco(fn):
        VARIANTS[name] = fn
        return fn
    return deco


# --- v11a: mass-adaptive death-zone buffer -----------------------------------
@variant("v11a")
def _a(src):
    src = add_fields(src,
        "\n    # --- v11a: mass-adaptive death-zone buffer (early caution) ---\n"
        "    dz_small_bonus: float = 1.5   # extra death-zone buffer while tiny,\n"
        "                                  # ramping to 0 by dz_small_ref mass\n"
        "    dz_small_ref: float = 6.0\n")
    return patch(src, [(
        "    eaters = []\n"
        "    for b in query.visible_blobs:\n"
        "        ebm = b.radius * b.radius\n"
        "        if ebm < ratio * min_my_mass:\n"
        "            continue  # cannot eat even our smallest blob — not a death threat\n"
        "        disk = b.radius + _speed(b.radius) + params.death_zone_buffer\n"
        "        eaters.append((b.pos[0], b.pos[1], ebm, disk))\n",
        "    # v11a: widen the no-go disk while our smallest blob is tiny — small\n"
        "    # blobs die most, so a fatter early buffer trades a little foraging\n"
        "    # reach for fewer movement kills; ramps to baseline by dz_small_ref.\n"
        "    dz_buf = params.death_zone_buffer + params.dz_small_bonus * max(\n"
        "        0.0, 1.0 - min_my_mass / params.dz_small_ref)\n"
        "    eaters = []\n"
        "    for b in query.visible_blobs:\n"
        "        ebm = b.radius * b.radius\n"
        "        if ebm < ratio * min_my_mass:\n"
        "            continue  # cannot eat even our smallest blob — not a death threat\n"
        "        disk = b.radius + _speed(b.radius) + dz_buf\n"
        "        eaters.append((b.pos[0], b.pos[1], ebm, disk))\n")])


# --- v11b: mass-adaptive FEED flee margin ------------------------------------
@variant("v11b")
def _b(src):
    src = add_fields(src,
        "\n    # --- v11b: mass-adaptive FEED flee margin (flee sooner tiny) ---\n"
        "    feed_flee_small_bonus: float = 4.0  # extra flee margin at spawn mass,\n"
        "                                        # ramping to 0 by feed_flee_small_ref\n"
        "    feed_flee_small_ref: float = 6.0\n")
    src = patch(src, [(
        "    ratio = spec.EAT_SIZE_RATIO\n"
        "    fx = fy = 0.0\n"
        "    fleeing = False\n",
        "    ratio = spec.EAT_SIZE_RATIO\n"
        "    # v11b: flee from further away while tiny (bigger margin at low mass).\n"
        "    flee_margin = params.feed_flee_margin + params.feed_flee_small_bonus * max(\n"
        "        0.0, 1.0 - my_min_mass / params.feed_flee_small_ref)\n"
        "    fx = fy = 0.0\n"
        "    fleeing = False\n")])
    return patch(src, [(
        "        if d <= reach + params.feed_flee_margin:\n",
        "        if d <= reach + flee_margin:\n")])


# --- v11c: crowd avoidance in FEED -------------------------------------------
@variant("v11c")
def _c(src):
    src = add_fields(src,
        "\n    # --- v11c: crowd avoidance in FEED (avoid dense predator regions) ---\n"
        "    crowd_radius: float = 16.0   # awareness radius for counting eaters\n"
        "    crowd_min_count: int = 2     # >= this many eaters near -> open space\n"
        "    crowd_weight: float = 3.0    # strength of the away-from-crowd bias\n")
    return patch(src, [(
        "    # gravity well over all visible food: densest cluster wins\n"
        "    fx = fy = 0.0\n"
        "    for f in query.visible_food:\n",
        "    # gravity well over all visible food: densest cluster wins\n"
        "    fx = fy = 0.0\n"
        "    # v11c: when several eaters cluster nearby (even if none is striking\n"
        "    # yet), bias away from their centroid toward open space so we don't\n"
        "    # forage into a trap. Blends with the food well below.\n"
        "    ecx = ecy = 0.0\n"
        "    ecount = 0\n"
        "    for b in query.visible_blobs:\n"
        "        bm = b.radius * b.radius\n"
        "        if bm < spec.EAT_SIZE_RATIO * my_min_mass:\n"
        "            continue\n"
        "        if math.hypot(b.pos[0] - cx, b.pos[1] - cy) <= params.crowd_radius:\n"
        "            ecx += b.pos[0]; ecy += b.pos[1]; ecount += 1\n"
        "    if ecount >= params.crowd_min_count:\n"
        "        ax, ay = cx - ecx / ecount, cy - ecy / ecount\n"
        "        am = math.hypot(ax, ay) or 1e-9\n"
        "        fx += ax / am * params.crowd_weight\n"
        "        fy += ay / am * params.crowd_weight\n"
        "    for f in query.visible_food:\n")])


# --- v11d: strict virus-detour gate ------------------------------------------
@variant("v11d")
def _d(src):
    return patch(src, [(
        "        virus = _nearest_consumable_virus(query)\n"
        "        safe = virus is not None and _virus_pop_safe(query, virus, p)\n",
        "        virus = _nearest_consumable_virus(query)\n"
        "        # v11d: gate the FEED-phase virus detour on the STRICT 18-tick\n"
        "        # lockout window (not the 1-tick _virus_pop_safe). A pop scatters\n"
        "        # us into ~16 tiny pieces merge-locked for SPLIT_COOLDOWN_FRAMES;\n"
        "        # only detour when no enemy can reach the scatter across it all.\n"
        "        safe = virus is not None and _virus_pop_safe_window(\n"
        "            query, virus, p, spec.SPLIT_COOLDOWN_FRAMES)\n")])


# --- v11e: min-mass HUNT gate ------------------------------------------------
@variant("v11e")
def _e(src):
    src = add_fields(src,
        "\n    # --- v11e: minimum mass before latching HUNT (safer FEED longer) ---\n"
        "    hunt_min_mass: float = 3.0   # don't HUNT until our biggest blob is\n"
        "                                 # at least this — hunting tiny over-commits\n")
    return patch(src, [(
        "        # normal FEED -> HUNT latch\n"
        "        if query.round >= p.feed_until_round and _has_eatable_enemy(query):\n"
        "            self._hunting = True\n",
        "        # normal FEED -> HUNT latch\n"
        "        # v11e: also require a minimum mass — hunting while tiny (mass\n"
        "        # ~1-2) over-commits, so forage in the safer FEED policy first.\n"
        "        big_enough = bool(blobs) and max(\n"
        "            b.radius * b.radius for b in blobs) >= p.hunt_min_mass\n"
        "        if (query.round >= p.feed_until_round and _has_eatable_enemy(query)\n"
        "                and big_enough):\n"
        "            self._hunting = True\n")])


# --- v11f: wall avoidance in FEED --------------------------------------------
@variant("v11f")
def _f(src):
    src = add_fields(src,
        "\n    # --- v11f: wall avoidance while foraging (avoid corners tiny) ---\n"
        "    feed_wall_margin: float = 8.0   # push inward within this of a wall\n"
        "    feed_wall_weight: float = 2.5   # inward push strength (scaled small)\n"
        "    feed_wall_ref: float = 8.0      # mass by which the push has faded\n")
    return patch(src, [(
        "        w = 1.0 / (d ** params.feed_food_falloff)\n"
        "        fx += dx / d * w\n"
        "        fy += dy / d * w\n"
        "    if fx * fx + fy * fy > 1e-18:\n"
        "        return fx, fy, False\n",
        "        w = 1.0 / (d ** params.feed_food_falloff)\n"
        "        fx += dx / d * w\n"
        "        fy += dy / d * w\n\n"
        "    # v11f: steer away from walls while foraging so we don't get cornered.\n"
        "    # The push fades as we grow (a big blob's own field handles corners);\n"
        "    # strongest while tiny, when a wall pin is fatal.\n"
        "    wscale = params.feed_wall_weight * max(\n"
        "        0.0, 1.0 - my_max_mass / params.feed_wall_ref)\n"
        "    if wscale > 0.0:\n"
        "        arena = spec.ARENA_SIZE\n"
        "        m = params.feed_wall_margin\n"
        "        if cx < m:\n"
        "            fx += wscale * (m - cx) / m\n"
        "        if arena - cx < m:\n"
        "            fx -= wscale * (m - (arena - cx)) / m\n"
        "        if cy < m:\n"
        "            fy += wscale * (m - cy) / m\n"
        "        if arena - cy < m:\n"
        "            fy -= wscale * (m - (arena - cy)) / m\n"
        "    if fx * fx + fy * fy > 1e-18:\n"
        "        return fx, fy, False\n")])


# --- v11g: split-reach-aware death zone while tiny (anti split-kill) ---------
@variant("v11g")
def _g(src):
    src = add_fields(src,
        "\n    # --- v11g: credit enemy split-reach in the veto while tiny ---\n"
        "    dz_split_frac: float = 0.5   # fraction of the 8.9u split reach added\n"
        "                                 # to a split-capable eater's disk at spawn\n"
        "    dz_split_ref: float = 6.0    # ...ramping to 0 by this mass\n")
    return patch(src, [(
        "    eaters = []\n"
        "    for b in query.visible_blobs:\n"
        "        ebm = b.radius * b.radius\n"
        "        if ebm < ratio * min_my_mass:\n"
        "            continue  # cannot eat even our smallest blob — not a death threat\n"
        "        disk = b.radius + _speed(b.radius) + params.death_zone_buffer\n"
        "        eaters.append((b.pos[0], b.pos[1], ebm, disk))\n",
        "    # v11g: while tiny, credit an enemy that could SPLIT-kill one of our\n"
        "    # blobs part of its 8.9-unit split reach in the no-go disk (normally\n"
        "    # excluded to avoid paralysis). Small blobs die mostly to split-lunges;\n"
        "    # the credit ramps to 0 by dz_split_ref mass so a grown bot is free.\n"
        "    split_small = params.dz_split_frac * max(\n"
        "        0.0, 1.0 - min_my_mass / params.dz_split_ref)\n"
        "    eaters = []\n"
        "    for b in query.visible_blobs:\n"
        "        ebm = b.radius * b.radius\n"
        "        if ebm < ratio * min_my_mass:\n"
        "            continue  # cannot eat even our smallest blob — not a death threat\n"
        "        disk = b.radius + _speed(b.radius) + params.death_zone_buffer\n"
        "        if (split_small > 0.0 and ebm >= spec.SPLIT_MIN_MASS\n"
        "                and (ebm * 0.5) >= ratio * min_my_mass):\n"
        "            disk += split_small * spec.SPLIT_REACH\n"
        "        eaters.append((b.pos[0], b.pos[1], ebm, disk))\n")])


# --- v11ab: caution stack (death-zone buffer + FEED flee margin) -------------
@variant("v11ab")
def _ab(src):
    return _b(_a(src))


# --- v11abg: full caution stack (buffer + flee margin + split-reach veto) ----
@variant("v11abg")
def _abg(src):
    # fields from a, b, g
    src = add_fields(src,
        "\n    # --- v11abg A: mass-adaptive death-zone buffer ---\n"
        "    dz_small_bonus: float = 1.5\n"
        "    dz_small_ref: float = 6.0\n"
        "\n    # --- v11abg B: mass-adaptive FEED flee margin ---\n"
        "    feed_flee_small_bonus: float = 4.0\n"
        "    feed_flee_small_ref: float = 6.0\n"
        "\n    # --- v11abg G: split-reach credit in the veto while tiny ---\n"
        "    dz_split_frac: float = 0.5\n"
        "    dz_split_ref: float = 6.0\n")
    # B: flee margin (two subs)
    src = patch(src, [(
        "    ratio = spec.EAT_SIZE_RATIO\n"
        "    fx = fy = 0.0\n"
        "    fleeing = False\n",
        "    ratio = spec.EAT_SIZE_RATIO\n"
        "    flee_margin = params.feed_flee_margin + params.feed_flee_small_bonus * max(\n"
        "        0.0, 1.0 - my_min_mass / params.feed_flee_small_ref)\n"
        "    fx = fy = 0.0\n"
        "    fleeing = False\n")])
    src = patch(src, [(
        "        if d <= reach + params.feed_flee_margin:\n",
        "        if d <= reach + flee_margin:\n")])
    # A+G combined: one rewrite of the eaters loop
    return patch(src, [(
        "    eaters = []\n"
        "    for b in query.visible_blobs:\n"
        "        ebm = b.radius * b.radius\n"
        "        if ebm < ratio * min_my_mass:\n"
        "            continue  # cannot eat even our smallest blob — not a death threat\n"
        "        disk = b.radius + _speed(b.radius) + params.death_zone_buffer\n"
        "        eaters.append((b.pos[0], b.pos[1], ebm, disk))\n",
        "    # v11abg: while tiny, widen the no-go disk (dz_small_bonus) AND credit a\n"
        "    # split-capable eater part of its 8.9u split reach (dz_split_frac); both\n"
        "    # ramp to 0 by their ref mass so a grown bot is unconstrained.\n"
        "    dz_buf = params.death_zone_buffer + params.dz_small_bonus * max(\n"
        "        0.0, 1.0 - min_my_mass / params.dz_small_ref)\n"
        "    split_small = params.dz_split_frac * max(\n"
        "        0.0, 1.0 - min_my_mass / params.dz_split_ref)\n"
        "    eaters = []\n"
        "    for b in query.visible_blobs:\n"
        "        ebm = b.radius * b.radius\n"
        "        if ebm < ratio * min_my_mass:\n"
        "            continue  # cannot eat even our smallest blob — not a death threat\n"
        "        disk = b.radius + _speed(b.radius) + dz_buf\n"
        "        if (split_small > 0.0 and ebm >= spec.SPLIT_MIN_MASS\n"
        "                and (ebm * 0.5) >= ratio * min_my_mass):\n"
        "            disk += split_small * spec.SPLIT_REACH\n"
        "        eaters.append((b.pos[0], b.pos[1], ebm, disk))\n")])


def main():
    for name, fn in VARIANTS.items():
        out = fn(SRC)
        # sanity: must still parse
        import ast
        ast.parse(out)
        p = REPO / f"dist/potential_field_{name}.py"
        p.write_text(out)
        print(f"wrote {p.relative_to(REPO)} ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
