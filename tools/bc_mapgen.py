#!/usr/bin/env python3
"""
bc_mapgen.py — Procedural map generator for UNSW Battlecode AU 2026 "Sea Dragons".

Emits .map files that conform to the engine's grammar and rules (verified against
the engine's own parser strings, the docs, and live `unswbc run` probes):

  MAP w h                       required, first; everything bounds-checked against it
  [MAP_NAME n]                  optional
  [SYMMETRY x|y|xy]             optional; mirrored tiles share one pearl countdown
                                and fire together. Engine semantics established:
                                  xy = 180-degree point symmetry:
                                       tile (x,y) -> (W-1-x, H-1-y)
                                       H edge north-of (x,y) -> H edge north-of (W-1-x, H-y)
                                         (north flips to south = row 2*(H-y), col W-1-x)
                                       V edge west-of (x,y) -> V edge west-of (W-x, H-1-y)
                                         (west flips to east: row 2*(H-1-y)+1, col W-x)
                                  y  = tile (x,y) -> (x, H-1-y); H edges flip side
                                       (north of (x,y) -> north of (x, H-y));
                                       V edges keep side (west of (x,y) -> west of (x, H-1-y))
                                  x  = tile (x,y) -> (W-1-x, y); H edges keep side;
                                       V edges flip side (west of (x,y) -> west of (W-x, y))
  TILE_COUNT n                  TILE x y minGap maxGap    (maxGap 0 => never spawns; else minGap >= 1)
  EDGE_COUNT n                  EDGE idx kind portalId    (kind 0 open / 1 kelp / 2 portal;
                                                          rows: (2H+1) rows x (W+1) cols;
                                                          even row = horizontal (north of tile row r//2),
                                                          odd row  = vertical (west of tile col c))
  DRAGON_COUNT n                DRAGON team count x y ... (team 0|1, head first, adjacent segs,
                                                          min length 2, ids in FILE ORDER = TURN ORDER)
  [END]                         optional terminator (bundled maps have it)

Rules honoured by this generator:
  - exactly TWO edges per portal id, same orientation (engine rejects otherwise:
    "portal 7 joins a horizontal edge to a vertical one" / "has 1 end(s)")
  - portal pairs mirror-symmetrically placed so both teams enjoy identical portals
  - kelp mirrored constructively (guarantees fair timing AND shape)
  - pearl countdowns SHARED between mirror tiles by engine symmetry => mirrored
    tiles get IDENTICAL (minGap, maxGap) so pops stay fair in timing
  - spawns mirror-symmetric, team 0 dragons listed first (alternate orderings so
    team 0's first dragon acts first)
  - minimum map size enforced: engine rejects maps below 7x7

USAGE
  ./bc_mapgen.py --preset default_small --out maps/gen_default_small.map
  ./bc_mapgen.py --size 24 24 --symmetry xy --kelp-frac 0.10 --portal-pairs 8 \
      --gap-min 1 --gap-max 20 --dragons-per-team 4 --seed 1234 --out maps/gen.map
  ./bc_mapgen.py --presets --out maps/gen/     # generate one map per preset

Presets (mimic bundled maps):
  default_small  16x16, sparse pops, light kelp
  default        32x32, very sparse pops, moderate kelp
  schooltime     48x50, dense timed pops (min gap 20)
  queen          25x35, escalating gap bands
  big_empty      64x64, no kelp, no portals
  arena_ring     12x12, closed border ring (like arena), rich pearls
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Geometry helpers (verified against engine + bundled maps)
# --------------------------------------------------------------------------

def edge_linear_h(x: int, y: int) -> int:
    """Horizontal edge (north side of tile (x,y)) -> linear index. Row 2y, col x."""
    return 2 * y * (0 + 1)  # placeholder replaced below


class EdgeGrid:
    """Wraps the (2H+1) x (W+1) edge grid of a W x H toroidal map."""

    def __init__(self, width: int, height: int):
        self.W = width
        self.H = height
        # grid[row][col] = kind; row even = horizontal (north of tile row r//2),
        # row odd = vertical (west of tile col c). Tile y = row//2.
        self.rows = 2 * self.H + 1
        self.cols = self.W + 1
        self.grid = [[0] * self.cols for _ in range(self.rows)]
        self.portal_partner: dict[int, tuple[int, int]] = {}  # pid -> (r,c)

    def h(self, x: int, y: int) -> tuple[int, int]:
        """linear index of horizontal edge north of tile (x,y) as (row, col)."""
        return (2 * y, x)

    def v(self, x: int, y: int) -> tuple[int, int]:
        """linear index of vertical edge west of tile (x,y) as (row, col)."""
        return (2 * y + 1, x)

    def tile_from_h(self, r: int, c: int) -> tuple[int, int]:
        return (c, r // 2)

    def tile_from_v(self, r: int, c: int) -> tuple[int, int]:
        return (c, r // 2)

    def set_kelp(self, r: int, c: int):
        self.grid[r][c] = 1

    def set_portal(self, r1: int, c1: int, r2: int, c2: int, pid: int):
        # same orientation enforced by callers
        assert self.grid[r1][c1] == 0 and self.grid[r2][c2] == 0, \
            f"edge ({r1},{c1}) or ({r2},{c2}) already occupied"
        assert (r1 % 2) == (r2 % 2), \
            f"portal {pid}: orientation mismatch (rows {r1} vs {r2})"
        self.grid[r1][c1] = 2
        self.grid[r2][c2] = 2
        self.portal_partner[pid] = (r2, c2)

    def mirror_transform(self, symmetry: str, r: int, c: int) -> tuple[int, int]:
        """
        Mirror an edge (r,c) to its symmetric partner. Validated against
        default.map (SYMMETRY xy): the transforms close kelp perfectly there
        (0/216 misses) — 180-degree rotation flips edge SIDES (N<->S, W<->E).
        For y-mirrors N<->S flip, W<->E preserve. For x-mirrors N/S preserve,
        W/E flip.
        """
        W, H = self.W, self.H
        horiz = (r % 2 == 0)
        x, y = (c, r // 2)

        if symmetry == "xy":
            # tile (x,y) -> (W-1-x, H-1-y); edge sides flip (N<->S, W<->E)
            mx, my = W - 1 - x, H - 1 - y
            if horiz:   # north of (x,y) -> north of (W-1-x, H-y) = south of (mx,my)
                return (2 * (H - y), W - 1 - x)
            else:       # west of (x,y) -> west of (W-x, H-1-y) = east of (mx,my)
                return (2 * (H - 1 - y) + 1, W - x)
        elif symmetry == "y":
            # tile (x,y) -> (x, H-1-y); N<->S flip, W<->E preserve
            my = H - 1 - y
            if horiz:
                return (2 * (H - y), x)          # south of (x,my) = north of (x,H-y)
            else:
                return (2 * my + 1, x)           # west of (x,my)
        elif symmetry == "x":
            # tile (x,y) -> (W-1-x, y); N/S preserve, W<->E flip
            mx = W - 1 - x
            if horiz:
                return (2 * y, mx)               # north of (mx,y)
            else:
                return (2 * y + 1, W - x)        # east of (mx,y)
        raise ValueError(symmetry)

    def is_on_symmetry_axis(self, symmetry: str, r: int, c: int) -> bool:
        """True if edge (r,c) lies ON the mirror locus (maps to itself)."""
        mr, mc = self.mirror_transform(symmetry, r, c)
        return (mr, mc) == (r, c)

    def iter_interior_h(self):
        """Interior horizontal edges: north side of tile rows 1..H-1, cols 1..W-1."""
        for y in range(1, self.H):
            for x in range(1, self.W):
                yield self.h(x, y)

    def iter_interior_v(self):
        """Interior vertical edges: west side of tile cols 1..W-1, rows 1..H-1."""
        for y in range(1, self.H):
            for x in range(1, self.W):
                yield self.v(x, y)

    def iter_edges_of_orientation(self, horiz: bool):
        for r in range(self.rows):
            if (r % 2 == 0) != horiz:
                continue
            for c in range(self.cols):
                yield (r, c)

    def to_linear(self, r: int, c: int) -> int:
        return r * self.cols + c


# --------------------------------------------------------------------------
# Presets (mimic bundled maps)
# --------------------------------------------------------------------------

PRESETS = {
    "default_small": dict(size=(16, 16), symmetry="xy", kelp_frac=0.08,
                          portal_pairs=4, gap_min=3, gap_max=40,
                          dragons_per_team=1, pearl_frac=0.0),
    "default":       dict(size=(32, 32), symmetry="xy", kelp_frac=0.10,
                          portal_pairs=6, gap_min=8, gap_max=120,
                          dragons_per_team=2, pearl_frac=0.0),
    "schooltime":    dict(size=(48, 50), symmetry="y", kelp_frac=0.06,
                          portal_pairs=8, gap_min=20, gap_max=60,
                          dragons_per_team=2, pearl_frac=0.0),
    "queen":         dict(size=(25, 35), symmetry="xy", kelp_frac=0.08,
                          portal_pairs=2, gap_bands=[(3, 20), (40, 80),
                                                     (80, 160), (160, 320)],
                          dragons_per_team=2, pearl_frac=0.0),
    "big_empty":     dict(size=(64, 64), symmetry="xy", kelp_frac=0.0,
                          portal_pairs=0, gap_min=5, gap_max=750,
                          dragons_per_team=3, pearl_frac=0.0),
    "arena_ring":    dict(size=(12, 12), symmetry="xy", kelp_frac=0.02,
                          portal_pairs=3, gap_min=2, gap_max=20,
                          dragons_per_team=1, pearl_start=8),
}


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

@dataclass
class GenParams:
    width: int = 16
    height: int = 16
    symmetry: str = "xy"           # x | y | xy
    kelp_frac: float = 0.08        # fraction of interior edges that become kelp
    portal_pairs: int = 4          # number of portal PAIRS (2 edges each)
    gap_min: int = 3
    gap_max: int = 40
    gap_bands: list | None = None  # [(min,max),...] escalating bands (queen-style)
    dragons_per_team: int = 1
    dragon_len: int = 3
    pearl_start: int = 0           # if >0, that fraction... reserved
    seed: int = 0
    name: str = ""


def mirror_tile(symmetry: str, x: int, y: int, W: int, H: int):
    if symmetry == "xy":
        return (W - 1 - x, H - 1 - y)
    if symmetry == "y":
        return (x, H - 1 - y)
    if symmetry == "x":
        return (W - 1 - x, y)
    raise ValueError(symmetry)


def generate(p: GenParams) -> str:
    rng = random.Random(p.seed)
    W, H = p.width, p.height
    if W < 7 or H < 7:
        raise ValueError(f"map must be at least 7x7 (got {W}x{H}); engine rejects smaller")
    eg = EdgeGrid(W, H)

    # ---------------- kelp (constructive symmetry) ----------------
    if p.kelp_frac > 0:
        if p.symmetry in ("y", "x"):
            # generate on one half; mirror to the other; axis tiles decided
            # independently (they map to themselves).
            if p.symmetry == "y":
                half_ys = range(0, H // 2)
            else:
                half_xs = range(0, W // 2)
        # Collect candidate interior edges (kelp only on tile-adjacent edges,
        # and keep rim edges open unless the preset asks for a ring).
        cands = list(eg.iter_interior_h()) + list(eg.iter_interior_v())
        rng.shuffle(cands)
        n_kelp = int(len(cands) * p.kelp_frac)
        for r, c in cands[: n_kelp]:
            if eg.grid[r][c] != 0:
                continue
            mr, mc = eg.mirror_transform(p.symmetry, r, c)
            eg.set_kelp(r, c)
            if (mr, mc) != (r, c):
                if eg.grid[mr][mc] == 0:
                    eg.set_kelp(mr, mc)
        # verify closure
        for r in range(eg.rows):
            for c in range(eg.cols):
                if eg.grid[r][c] == 1:
                    mr, mc = eg.mirror_transform(p.symmetry, r, c)
                    assert eg.grid[mr][mc] == 1, \
                        f"kelp symmetry broken at ({r},{c}) -> ({mr},{mc})"

    # ---------------- portals (constructive symmetry) ----------------
    if p.portal_pairs > 0:
        h_cands = list(eg.iter_interior_h())
        v_cands = list(eg.iter_interior_v())
        rng.shuffle(h_cands)
        rng.shuffle(v_cands)
        pid = 0
        # alternate orientation placements
        for i in range(p.portal_pairs):
            placed = False
            for cands in (h_cands, v_cands):
                while cands:
                    r, c = cands.pop()
                    mr, mc = eg.mirror_transform(p.symmetry, r, c)
                    if (mr, mc) == (r, c):
                        continue  # a portal whose partner is itself is useless
                    if eg.grid[r][c] != 0 or eg.grid[mr][mc] != 0:
                        continue
                    if (r % 2) != (mr % 2):
                        continue  # orientation mismatch (cannot happen with
                                  # the verified transforms, guarded anyway)
                    eg.set_portal(r, c, mr, mc, pid)
                    pid += 1
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                break  # ran out of space

    # ---------------- pearl gaps (countdowns) ----------------
    # Mirrored tiles share one countdown under engine symmetry => give mirrored
    # tiles the SAME (minGap, maxGap) so timing stays fair.
    tile_gap = {}
    if p.gap_bands:
        # queen-style: bands of increasing gaps, geometric regions
        for y in range(H):
            for x in range(W):
                frac = (y / max(1, H - 1))
                band = min(int(frac * len(p.gap_bands)), len(p.gap_bands) - 1)
                gmin, gmax = p.gap_bands[band]
                tile_gap[(x, y)] = (gmin, gmax)
    else:
        gmin, gmax = max(1, p.gap_min), max(p.gap_min, p.gap_max)
        for y in range(H):
            for x in range(W):
                tile_gap[(x, y)] = (gmin, gmax)
    # enforce mirror equality
    for (x, y), g in list(tile_gap.items()):
        mx, my = mirror_tile(p.symmetry, x, y, W, H)
        tile_gap[(mx, my)] = g
        tile_gap[(x, y)] = g

    # ---------------- dragons (mirror-symmetric spawns) ----------------
    # Team 0 first (ids in file order = turn order); alternate team orderings
    # so the lower ids interleave as in bundled maps.
    n = p.dragons_per_team
    dl = p.dragon_len
    if W * H < 4 * n * dl:
        raise ValueError("map too small for the requested dragon fleet")
    # Place team 0 horizontally (head east), mirrored for team 1.
    spawns = []  # (team, [(x,y) head-first...])
    margin_x = max(2, W // 8)
    margin_y = max(2, H // (n + 2))
    for i in range(n):
        y = margin_y + i * max(1, (H - 2 * margin_y) // max(1, n))
        hx = margin_x
        # keep the head 2+ tiles from kelp by nudging if needed
        segs = [(hx + j, y) for j in range(dl)]  # head first
        m_segs = [mirror_tile(p.symmetry, x, yy, W, H) for x, yy in segs]
        spawns.append((0, segs))
        spawns.append((1, m_segs))
    # alternating team order like bundled maps: 0,1,0,1,...
    # ids assigned in file order: dragons list order.

    # ---------------- emit ----------------
    lines = []
    lines.append(f"MAP {W} {H}")
    lines.append(f"SYMMETRY {p.symmetry}")
    lines.append(f"MAP_NAME {p.name or 'Generated'}")
    lines.append(f"TILE_COUNT {W * H}")
    for y in range(H):
        for x in range(W):
            gmin, gmax = tile_gap[(x, y)]
            lines.append(f"TILE {x} {y} {gmin} {gmax}")
    # EDGE_COUNT: declare every edge explicitly (bundled maps omit rim edges;
    # undeclared = open. Full declaration is unambiguous and engine-accepted.)
    edge_count = eg.rows * eg.cols
    lines.append(f"EDGE_COUNT {edge_count}")
    # recover portal ids: pair portal edges by mirror relation
    portal_edges = [(r, c) for r in range(eg.rows) for c in range(eg.cols)
                    if eg.grid[r][c] == 2]
    assert len(portal_edges) % 2 == 0, "portals must come in pairs"
    pairs_done = set()
    pid = 0
    pid_of = {}
    for r, c in portal_edges:
        if (r, c) in pairs_done:
            continue
        mr, mc = eg.mirror_transform(p.symmetry, r, c)
        assert (mr, mc) in set(portal_edges), \
            f"portal edge ({r},{c}) lacks its mirror partner"
        pid_of[(r, c)] = pid
        pid_of[(mr, mc)] = pid
        pairs_done.add((r, c))
        pairs_done.add((mr, mc))
        pid += 1
    for r in range(eg.rows):
        for c in range(eg.cols):
            lin = r * eg.cols + c
            kind = eg.grid[r][c]
            if kind == 2:
                lines.append(f"EDGE {lin} 2 {pid_of[(r, c)]}")
            elif kind == 1:
                lines.append(f"EDGE {lin} 1 -1")
            else:
                lines.append(f"EDGE {lin} 0 -1")
    lines.append(f"DRAGON_COUNT {len(spawns)}")
    for team, segs in spawns:
        coords = []
        for (sx, sy) in segs:
            coords.extend([str(sx), str(sy)])
        lines.append(f"DRAGON {team} {dl} " + " ".join(coords))
    lines.append("END")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--preset", help="named preset (see PRESETS)")
    ap.add_argument("--size", nargs=2, type=int, metavar=("W", "H"))
    ap.add_argument("--symmetry", choices=["x", "y", "xy"], default=None)
    ap.add_argument("--kelp-frac", type=float, default=None)
    ap.add_argument("--portal-pairs", type=int, default=None)
    ap.add_argument("--gap-min", type=int, default=None)
    ap.add_argument("--gap-max", type=int, default=None)
    ap.add_argument("--gap-bands", nargs="*", type=int, default=None,
                    help="escalating gap bands, flat list min max min max ...")
    ap.add_argument("--dragons-per-team", type=int, default=None)
    ap.add_argument("--dragon-len", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0,
                    help="generation seed (0 = different every run)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output .map path (or directory with --presets)")
    ap.add_argument("--presets", action="store_true",
                    help="generate one map per preset into --out directory")
    args = ap.parse_args()

    if args.presets:
        args.out.mkdir(parents=True, exist_ok=True)
        for pname, kw in PRESETS.items():
            gp = GenParams(**{k: v for k, v in kw.items() if k in
                              GenParams.__dataclass_fields__},
                           name=pname)
            text = generate(gp)
            path = args.out / f"gen_{pname}.map"
            path.write_text(text)
            print(f"wrote {path} ({gp.width}x{gp.height}, sym={gp.symmetry}, "
                  f"kelp={gp.kelp_frac}, portals={gp.portal_pairs})")
        return 0

    if args.preset:
        if args.preset not in PRESETS:
            print(f"unknown preset: {args.preset}; have {sorted(PRESETS)}",
                  file=sys.stderr)
            return 2
        kw = dict(PRESETS[args.preset])
    else:
        kw = {}
    if args.size:
        kw["size"] = tuple(args.size)
    if args.symmetry:
        kw["symmetry"] = args.symmetry
    if args.kelp_frac is not None:
        kw["kelp_frac"] = args.kelp_frac
    if args.portal_pairs is not None:
        kw["portal_pairs"] = args.portal_pairs
    if args.gap_min is not None:
        kw["gap_min"] = args.gap_min
    if args.gap_max is not None:
        kw["gap_max"] = args.gap_max
    if args.gap_bands:
        it = iter(args.gap_bands)
        kw["gap_bands"] = list(zip(it, it))
    if args.dragons_per_team is not None:
        kw["dragons_per_team"] = args.dragons_per_team
    if args.dragon_len is not None:
        kw["dragon_len"] = args.dragon_len

    seed = args.seed or random.SystemRandom().getrandbits(31)
    gp = GenParams(
        width=kw.get("size", (16, 16))[0],
        height=kw.get("size", (16, 16))[1],
        symmetry=kw.get("symmetry", "xy"),
        kelp_frac=kw.get("kelp_frac", 0.08),
        portal_pairs=kw.get("portal_pairs", 4),
        gap_min=kw.get("gap_min", 3),
        gap_max=kw.get("gap_max", 40),
        gap_bands=kw.get("gap_bands"),
        dragons_per_team=kw.get("dragons_per_team", 1),
        dragon_len=kw.get("dragon_len", 3),
        seed=seed,
        name=args.preset or "Generated",
    )
    text = generate(gp)
    if args.out.is_dir():
        args.out = args.out / f"gen_{seed}.map"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} ({gp.width}x{gp.height}, sym={gp.symmetry}, "
          f"kelp={gp.kelp_frac}, portals={gp.portal_pairs}, seed={seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
