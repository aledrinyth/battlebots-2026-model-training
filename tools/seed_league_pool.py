#!/usr/bin/env python3
"""Seed a league pool with external frozen clones as PFSP entries (Track 3).

The P6.4 league only ever freezes the *live* training policy into its pool
(``LeaguePool.maybe_snapshot``); there is no built-in way to inject a
pre-existing external checkpoint. This tool writes those checkpoints straight
into the pool's ``league.json`` as :class:`~train.league.PoolEntry` rows, so
the S2 burn starts with the real tournament meta (the R1 clones) in the
opponent pool from update 1 — instead of self-play of a passive champion
against its own passive snapshots.

Idempotent: re-running re-reads the existing ``league.json`` and overwrites
the same ``clone_<stem>`` ids, leaving any live ``snap_v*`` entries untouched.
Entries start at ``p_current=0.5`` — the maximum PFSP weight — so the learner
meets them immediately; as it beats one, PFSP down-weights it automatically.

    python tools/seed_league_pool.py \
        --pool-dir runs/stage2_league/league \
        --clones data/clones/team_49.npz data/clones/team_14.npz \
                 data/clones/team_37.npz data/clones/team_1.npz \
                 data/clones/team_88.npz

Paths are stored as given; pass them repo-relative (the run's cwd is the repo
root, and relative keeps the pool portable laptop↔box).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train.league import LeaguePool, PoolEntry  # noqa: E402


def seed_pool(pool_dir: str | Path, clone_paths: Sequence[str]) -> LeaguePool:
    """Add one ``clone_<stem>`` PoolEntry per path; persist league.json."""
    pool = LeaguePool(pool_dir=pool_dir)  # reads any existing entries
    for path in clone_paths:
        resolved = path if Path(path).is_absolute() else path
        if not (REPO / path).exists() and not Path(path).is_absolute():
            raise FileNotFoundError(
                f"clone weights not found (repo-relative): {path}"
            )
        sid = f"clone_{Path(path).stem}"
        pool.entries[sid] = PoolEntry(
            snapshot_id=sid,
            path=resolved,
            version=0,
            created_s=0.0,
            games=0,
            p_current=0.5,
        )
    pool.save()
    return pool


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool-dir", required=True,
                    help="league dir (e.g. runs/stage2_league/league)")
    ap.add_argument("--clones", nargs="+", required=True,
                    help="exported actor-weights .npz paths (repo-relative)")
    args = ap.parse_args(argv)

    pool = seed_pool(args.pool_dir, args.clones)
    clone_ids = sorted(k for k in pool.entries if k.startswith("clone_"))
    snap_ids = sorted(k for k in pool.entries if not k.startswith("clone_"))
    print(f"[seed] {args.pool_dir}/league.json: "
          f"{len(clone_ids)} clones {clone_ids}"
          + (f" + {len(snap_ids)} live snapshots preserved" if snap_ids else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
