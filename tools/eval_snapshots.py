#!/usr/bin/env python3
"""Evaluate a running league's snapshots vs the meta field → the learning curve.

The coordinator writes an actor-weights ``snap_v*.npz`` into the pool every
snapshot interval (atomic rename, so any file present is complete — safe to read
while the burn runs). Each is the exact deploy policy at that version, so we
score it through the ``frozen:`` NumPy path against the same meta-clone field,
same seeds, and print rank/mass over version — is the run climbing?

Run WHILE training with modest ``--workers`` so it doesn't starve the burn, or
on the laptop after rsyncing the pool dir (zero box impact). The ``.pt``
checkpoints work too (``--kind ckpt``), but snapshots are lighter and are the
weights that actually ship.

    python tools/eval_snapshots.py runs/stage2_heuristic --last 6 --games 120 --workers 16

Reference bars printed alongside: s1_champion 3.63 (old baseline) and neutral 4.5.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train.harness.evaluate import play_games  # noqa: E402

DEFAULT_META = tuple(
    f"frozen:data/clones/team_{t}.npz" for t in (49, 14, 37, 1, 88)
)
_VER = re.compile(r"snap_v0*(\d+)\.npz$")


def _version(path: str) -> int:
    m = _VER.search(path)
    return int(m.group(1)) if m else -1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="run dir (uses <run_dir>/league/snap_v*.npz)")
    ap.add_argument("--kind", choices=["snap", "ckpt"], default="snap")
    ap.add_argument("--last", type=int, default=0, help="only the N newest (0=all)")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=16,
                    help="keep modest while a burn is running (don't oversubscribe)")
    ap.add_argument("--opponents", nargs="+", default=list(DEFAULT_META))
    ap.add_argument("--base-seed", type=int, default=999_983)
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    if args.kind == "snap":
        paths = sorted(glob.glob(str(run / "league" / "snap_v*.npz")), key=_version)
        subj = lambda p: f"frozen:{p}"          # noqa: E731 (npz → deploy path)
    else:
        paths = sorted(glob.glob(str(run / "checkpoints" / "*.pt")), key=_version)
        subj = None                              # .pt loads via --weights (torch)
    if not paths:
        print(f"no {args.kind} files under {run}", file=sys.stderr)
        return 1
    if args.last:
        paths = paths[-args.last:]

    print(f"# {len(paths)} {args.kind}(s) vs meta field "
          f"({args.games} games, seed {args.base_seed}); "
          f"refs: s1_champion 3.63, neutral 4.5")
    print(f"{'version':>8}  {'mean_rank':>9}  {'mean_mass':>9}  {'top_half':>8}  {'wins':>5}")
    for p in paths:
        v = _version(p)
        if args.kind == "snap":
            rows = play_games(None, tuple(args.opponents), args.games, "cpu",
                              args.base_seed, args.workers, scripted_subject=subj(p))
        else:
            rows = play_games(p, tuple(args.opponents), args.games, "cpu",
                              args.base_seed, args.workers)
        n = len(rows)
        mr = sum(r for _, r, _ in rows) / n
        mm = sum(m for _, _, m in rows) / n
        th = sum(1 for _, r, _ in rows if r <= 4) / n
        wr = sum(1 for _, r, _ in rows if r == 1) / n
        print(f"{v:>8}  {mr:>9.3f}  {mm:>9.2f}  {th:>7.1%}  {wr:>5.1%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
