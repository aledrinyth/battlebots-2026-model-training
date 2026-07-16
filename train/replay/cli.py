"""``python -m train.replay`` entry points (impl doc §11).

    python -m train.replay check   <replay.json> [...]      # §7.1 block per file
    python -m train.replay convert <replay.json> --out shards/
    python -m train.replay ingest  --replays match_replays/ --out shards/
    python -m train.replay mine    --shards shards/ [--out notes/replay-mining.md]
    python -m train.replay parity  [--seed 0] [--rounds N]

``ingest`` is idempotent — re-running after new replays arrive is the watch
mode. ``--fast-vision`` (default on) installs the bit-exact NumPy crop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO = Path(__file__).resolve().parent.parent.parent


def _load_catalog(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    return json.loads(Path(path).read_text())


def _print_check(source: str, result, failures) -> None:
    r = result.report
    frac = (
        r["eaten_in_vision"] / r["eaten_total"] if r["eaten_total"] else 1.0
    )
    print(f"--- {source}")
    print(f"engine_version in replay: {r['engine_version']}")
    print(f"rounds (snapshot blocks): {r['rounds']}")
    print(f"(obs, action) rows built: {r['rows']} (move events {r['move_events']})")
    print(f"CanonicalObs.validate ok: {r['rows']}/{r['rows']}"
          if r["obs_validated"] else "CanonicalObs.validate:    DISABLED")
    print(f"heading bins covered:     {r['heading_bins_covered']}/24")
    print(f"split=True rows:          {r['split_rows']}")
    print(f"'stay' (0,0) moves:       {r['stay_rows']}")
    print(f"is_reset frames:          {r['reset_rows']} "
          f"# {r['first_frames']} first-frames + {r['respawns']} respawns")
    print(f"eaten-food-in-vision:     {r['eaten_in_vision']}/{r['eaten_total']}"
          f" ({100 * frac:.2f}%; plain window, telemetry)")
    print(f"eaten-food-in-reach:      {r['eaten_in_reach']}/{r['eaten_total']}"
          f"  # must be 100%")
    print(f"RESULT: {'PASS' if not failures else 'FAIL — ' + '; '.join(failures)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m train.replay",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--no-fast-vision", action="store_true",
                    help="use the pristine engine getters (slower, ground truth)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="§7.1 validation block per replay")
    p_check.add_argument("replays", nargs="+", type=Path)

    p_conv = sub.add_parser("convert", help="convert one replay into a shard")
    p_conv.add_argument("replay", type=Path)
    p_conv.add_argument("--out", type=Path, required=True)
    p_conv.add_argument("--catalog", type=Path, default=None)

    p_ing = sub.add_parser("ingest", help="idempotent batch ingestion")
    p_ing.add_argument("--replays", type=Path,
                       default=REPO / "match_replays")
    p_ing.add_argument("--out", type=Path, required=True)
    p_ing.add_argument("--catalog", type=Path, default=None)
    p_ing.add_argument("--move-quarantined", action="store_true",
                       help="move rejected files to match_replays/_quarantine/")

    p_mine = sub.add_parser("mine", help="per-bot behaviour report")
    p_mine.add_argument("--shards", type=Path, required=True)
    p_mine.add_argument("--out", type=Path, default=None,
                        help="markdown report path")

    p_bc = sub.add_parser("bc-dataset",
                          help="shards → train/bc.py-format npz (whole episodes)")
    p_bc.add_argument("--shards", type=Path, required=True)
    p_bc.add_argument("--out", type=Path, required=True)
    p_bc.add_argument("--min-rank", type=int, default=3,
                      help="keep episodes finishing at or above this rank of 8 "
                           "(default 3 — the winners cut)")
    p_bc.add_argument("--teams", type=int, nargs="+", default=None,
                      help="restrict to these team_ids (clone training)")
    p_bc.add_argument("--exclude-teams", type=int, nargs="+", default=[53],
                      help="never imitate these seats (default: 53 = us)")

    p_par = sub.add_parser("parity", help="record a wrapper match, assert "
                                          "bit-identical reconstruction")
    p_par.add_argument("--seed", type=int, default=0)
    p_par.add_argument("--rounds", type=int, default=None,
                       help="truncate the game (default: full length)")

    args = ap.parse_args(argv)

    if not args.no_fast_vision:
        from train.env import fast_vision
        fast_vision.install()

    if args.cmd == "check":
        from train.replay.convert import check_report, convert_replay
        worst = 0
        for path in args.replays:
            events = json.loads(Path(path).read_text())
            result = convert_replay(events, source=str(path))
            failures = check_report(result)
            _print_check(str(path), result, failures)
            worst = max(worst, 1 if failures else 0)
        return worst

    if args.cmd == "convert":
        from train.replay.ingest import ingest_paths
        report = ingest_paths([args.replay], args.out,
                              catalog=_load_catalog(args.catalog))
        print(report.summary())
        for src, reason in report.quarantined:
            print(f"  QUARANTINED {src}: {reason}")
        return 0 if report.ingested or report.skipped else 1

    if args.cmd == "ingest":
        from train.replay.ingest import ingest_dir
        report = ingest_dir(
            args.replays, args.out,
            catalog=_load_catalog(args.catalog),
            move_quarantined=args.move_quarantined,
        )
        print(report.summary())
        for src, reason in report.quarantined:
            print(f"  QUARANTINED {src}: {reason}")
        return 0

    if args.cmd == "mine":
        from train.replay.mine import to_markdown, write_report
        report = write_report(args.shards, args.out)
        print(to_markdown(report))
        return 0

    if args.cmd == "bc-dataset":
        from train.replay.bc_dataset import write_bc_data
        report = write_bc_data(
            args.shards, args.out,
            min_rank_weight=(8 - args.min_rank) / 7.0,
            exclude_teams=args.exclude_teams,
            teams=args.teams,
        )
        print(f"{report['rows']} rows / {report['episodes']} episodes -> "
              f"{args.out}  (split rate {report['split_rate']:.3%}, "
              f"stay rows {report['stay_rows']})")
        for name, t in report["teams"].items():
            print(f"  {name}: {t['episodes']} episodes, {t['rows']} rows, "
                  f"{t['splits']} splits")
        return 0

    if args.cmd == "parity":
        from train.replay.parity import run_parity_gate
        report = run_parity_gate(seed=args.seed, rounds=args.rounds)
        print(report.summary())
        for m in report.mismatches[:10]:
            print(f"  {m}")
        return 0 if report.passed else 1

    return 2  # pragma: no cover - argparse enforces the subcommand set


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
