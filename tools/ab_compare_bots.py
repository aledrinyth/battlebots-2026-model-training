#!/usr/bin/env python3
"""A/B compare two bot files head-to-head over seeded matches.

Runs 4-vs-4 matches. For each seed it plays TWICE with seating mirrored
(A in slots 0-3 then A in slots 4-7) so any positional/seat advantage cancels.
Each match runs as an ISOLATED subprocess (the engine is not re-entrant in one
process). Aggregates mean final mass, mean rank (1=best), and per-match wins.

Usage:
  ab_compare_bots.py --a "tests/fixtures/bots/greedy 4.py" \
      --b tests/fixtures/bots/greedy4_baseline.py --seeds 0 1 2 3 --rounds 600
"""
from __future__ import annotations
import argparse, json, statistics, subprocess, sys, tempfile, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".venv-bot/bin/python")
RECORDER = str(REPO / "tools/record_reference_match.py")


def run(a: str, b: str, seed: int, rounds: int, a_first: bool):
    if a_first:
        specs = [f"4:{a}", f"4:{b}"]; a_slots = {0, 1, 2, 3}
    else:
        specs = [f"4:{b}", f"4:{a}"]; a_slots = {4, 5, 6, 7}
    fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
    cmd = [PY, RECORDER, "--seed", str(seed), "--rounds", str(rounds),
           "--out", tmp, "--bots", *specs]
    subprocess.run(cmd, cwd=str(REPO), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    trace = json.loads(Path(tmp).read_text()); Path(tmp).unlink(missing_ok=True)
    fm = {int(k): v for k, v in trace["final_masses"].items()}
    rank_of = {pid: i + 1 for i, pid in enumerate(trace["rankings"])}
    a_m = [fm[p] for p in fm if p in a_slots]
    b_m = [fm[p] for p in fm if p not in a_slots]
    a_r = [rank_of[p] for p in fm if p in a_slots]
    b_r = [rank_of[p] for p in fm if p not in a_slots]
    return a_m, b_m, a_r, b_r


def _run_star(job):
    return run(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel match subprocesses (default 1 = sequential)")
    args = ap.parse_args()

    jobs = [(args.a, args.b, seed, args.rounds, a_first)
            for seed in args.seeds for a_first in (True, False)]

    A_mass, B_mass, A_rank, B_rank = [], [], [], []
    a_wins = b_wins = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_run_star, jobs))
    else:
        results = [run(*j) for j in jobs]
    for (a, b, seed, rounds, a_first), (am, bm, ar, br) in zip(jobs, results):
        A_mass += am; B_mass += bm; A_rank += ar; B_rank += br
        if sum(am) > sum(bm): a_wins += 1
        else: b_wins += 1
        print(f"seed {seed} {'A-first' if a_first else 'B-first'}: "
              f"A total={sum(am):6.2f} rank={statistics.mean(ar):.2f} | "
              f"B total={sum(bm):6.2f} rank={statistics.mean(br):.2f}", flush=True)

    print("\n==== AGGREGATE ====")
    print(f"A = {args.a}")
    print(f"B = {args.b}")
    print(f"matches: {a_wins + b_wins}  (A group higher total mass in {a_wins}, B in {b_wins})")
    print(f"A: mean mass={statistics.mean(A_mass):.3f}  mean rank={statistics.mean(A_rank):.3f}  "
          f"max mass={max(A_mass):.2f}")
    print(f"B: mean mass={statistics.mean(B_mass):.3f}  mean rank={statistics.mean(B_rank):.3f}  "
          f"max mass={max(B_mass):.2f}")
    print(f"mass ratio A/B = {statistics.mean(A_mass)/statistics.mean(B_mass):.3f}  "
          f"(>1 means A better)")


if __name__ == "__main__":
    main()
