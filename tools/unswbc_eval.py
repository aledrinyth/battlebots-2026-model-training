#!/usr/bin/env python3
"""
unswbc_eval.py — Parallel round-robin test harness for UNSW Battlecode bots.

USAGE
  ./unswbc_eval.py --bots DIR_OR_EXE [DIR_OR_EXE ...] --maps MAP [MAP ...] --seeds N [options]

What it does
  - Round-robin: every unordered bot pair × every map, both orderings (A-vs-B and
    B-vs-A) per seed. A team listed first moves its dragons first, so BOTH sides
    of every pairing are sampled (first-listed advantage is real).
  - Runs N matches per matchup in PARALLEL across --workers processes/cores using
    multiprocessing (unswbc run itself is single-process; parallelism is ours).
  - Seeds: --seeds N draws N seeds with a reproducible RNG (--rng-seed to vary).
    WITH --seed supplied to `unswbc run`, pearl respawns and bots' own random
    numbers are deterministic => each matchup/seed/ordering triple is exactly
    reproducible. Native mode warns that pearls repeat but bot RNG is drawn
    per-game: still prefer paired orderings within the same seed.

Output
  - results CSV: one row per match (enabled by default) —
      map, bot_a, bot_b, seed, winner(A/B/DRAW), end_reason, rounds,
      a_length, b_length, a_dragon_max, b_dragon_max
  - Aggregate summary (both orderings + combined win rate, longest-dragon stats)
    printed to stdout and written next to the CSV.

Examples
  ./unswbc_eval.py --bots ../bot_1/bot_16 ../bot_1/bot_31 \
      --maps ../../maps/default.map ../../maps/trophy.map --seeds 8
  ./unswbc_eval.py --bots bot_a bot_b --maps maps/arena.map --seeds 4 --sandbox
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# --------------------------------------------------------------------------
# Result-line parsing (matches run.py's output format exactly):
#   "team 0 wins after 33 rounds (by elimination) (1.5s)"
#   "draw after 500 rounds (equal length) (2.1s)"
#   round/death lines: "round N: bot ID (team T) died: reason"
#   "-v" adds "points MEMORY..." lines per turn (not parsed here)
# --------------------------------------------------------------------------
WIN_RE = re.compile(
    r"^team ([AB]) wins after (\d+) rounds \(([^)]+)\)"
)
DRAW_RE = re.compile(
    r"^draw after (\d+) rounds \(([^)]+)\)"
)
DEATH_RE = re.compile(
    r"^round (\d+): bot (\d+) \(team ([AB])\) died: (.+?)$"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


@dataclass
class MatchResult:
    map: str
    bot_a: str          # listed first (moves first)
    bot_b: str
    seed: str           # hex, as printed by run.py
    winner: str         # "A" | "B" | "DRAW"
    end_reason: str
    rounds: int
    a_length: int       # total length, team A
    b_length: int
    a_death_rounds: str # "id:round;id:round" (info)
    b_death_rounds: str
    swap: bool          # was this the swapped-ordering run?

    def row(self) -> dict:
        d = asdict(self)
        # convenience: which underlying bot won?
        return d


def run_one(task: dict) -> MatchResult:
    """Run a single `unswbc run` invocation and parse its result."""
    cmd = task["cmd"]
    meta = task["meta"]
    timeout = task.get("timeout", 240)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        out = proc.stdout
        err = proc.stderr
    except subprocess.TimeoutExpired:
        return MatchResult(
            map=meta["map"], bot_a=meta["bot_a"], bot_b=meta["bot_b"],
            seed=meta["seed"], winner="ERROR", end_reason="timeout",
            rounds=-1, a_length=-1, b_length=-1,
            a_death_rounds="", b_death_rounds="", swap=meta["swap"],
        )
    dur = time.monotonic() - t0

    winner = None
    rounds = -1
    reason = ""
    deaths_a: list[str] = []
    deaths_b: list[str] = []

    for line in out.splitlines():
        clean = strip_ansi(line.strip())
        m = WIN_RE.match(clean)
        if m:
            wteam = m.group(1)
            rounds = int(m.group(2))
            reason = m.group(3).strip()
            winner = wteam
            continue
        m = DRAW_RE.match(clean)
        if m:
            rounds = int(m.group(1))
            reason = m.group(2).strip()
            winner = "DRAW"
            continue
        m = DEATH_RE.match(clean)
        if m:
            rnd, bid, team, why = m.groups()
            (deaths_a if team == "A" else deaths_b).append(f"{bid}:{rnd}")

    if winner is None:
        # engine crashed, build failed, or map refused: record as ERROR
        snippet = (out or err).strip().splitlines()
        why = snippet[0] if snippet else "no result line"
        return MatchResult(
            map=meta["map"], bot_a=meta["bot_a"], bot_b=meta["bot_b"],
            seed=meta["seed"], winner="ERROR",
            end_reason=why[:160],
            rounds=-1, a_length=-1, b_length=-1,
            a_death_rounds=";".join(deaths_a),
            b_death_rounds=";".join(deaths_b), swap=meta["swap"],
        )

    a_len = b_len = -1
    # per-team total length comes only from replay parsing; approximate via
    # surviving max length we cannot see without replay. Record -1 unless we
    # can glean it. (Kept in the schema for future enhancement.)
    return MatchResult(
        map=meta["map"], bot_a=meta["bot_a"], bot_b=meta["bot_b"],
        seed=meta["seed"], winner=winner, end_reason=reason,
        rounds=rounds, a_length=a_len, b_length=b_len,
        a_death_rounds=";".join(deaths_a), b_death_rounds=";".join(deaths_b),
        swap=meta["swap"],
    )


def build_tasks(bots: list[str], maps: list[str], seeds: list[int],
                unswbc: str, sandbox: bool, verbose: bool,
                passthrough: list[str], timeout: int) -> list[dict]:
    tasks = []
    for mp_ in maps:
        for ba, bb in itertools.permutations(bots, 2):
            same_bot = ba == bb
            for si, seed in enumerate(seeds):
                # Mirror match (same bot both sides) doesn't need swapping;
                # permutations already yields (A,B) only when A!=B, but handle
                # identical paths gracefully: single ordering per seed.
                orderings = [(ba, bb)] if same_bot else [(ba, bb), (bb, ba)]
                # Swapped ordering handled by permutations themselves: we run
                # permutations so both orders appear naturally.
                for oa, ob in orderings:
                    if oa == ob and tasks:
                        pass  # fallthrough
                    cmd = [unswbc, "run", mp_, oa, ob,
                           "--seed", str(seed), "--no-replay", "--no-debug"]
                    cmd += passthrough
                    if sandbox:
                        cmd.append("--sandbox")
                    if verbose:
                        cmd.append("-v")
                    tasks.append({
                        "cmd": cmd,
                        "meta": {
                            "map": mp_, "bot_a": oa, "bot_b": ob,
                            "seed": f"{seed:#018x}",
                            "swap": oa != ba,
                        },
                        "timeout": timeout,
                    })
    # de-duplicate: permutations give (A,B) and (B,A); our inner loop would
    # double-run each ordering. Fix: dedupe on (map, a, b, seed).
    seen = set()
    uniq = []
    for t in tasks:
        m = t["meta"]
        key = (m["map"], m["bot_a"], m["bot_b"], m["seed"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)
    return uniq


def aggregate(results: list[MatchResult], out_csv: Path) -> str:
    # ---- write per-match CSV
    fieldnames = ["map", "bot_a", "bot_b", "seed", "winner", "end_reason",
                  "rounds", "a_length", "b_length", "a_deaths", "b_deaths",
                  "swap"]
    with out_csv.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(fieldnames)
        for r in results:
            wr.writerow([
                Path(r.map).name,
                Path(r.bot_a).name,
                Path(r.bot_b).name,
                r.seed, r.winner, r.end_reason, r.rounds,
                r.a_length, r.b_length,
                r.a_death_rounds, r.b_death_rounds,
                r.swap,
            ])

    # ---- aggregate per (botA,botB,map)
    stats: dict[tuple, dict] = {}
    for r in results:
        key = (Path(r.map).name, Path(r.bot_a).name, Path(r.bot_b).name)
        st = stats.setdefault(key, {
            "n": 0, "a_wins": 0, "b_wins": 0, "draws": 0, "errors": 0,
            "rounds_sum": 0,
            "a_deaths": [], "b_deaths": [],
        })
        st["n"] += 1
        if r.winner == "A":
            st["a_wins"] += 1
        elif r.winner == "B":
            st["b_wins"] += 1
        elif r.winner == "DRAW":
            st["draws"] += 1
        else:
            st["errors"] += 1
        if r.rounds > 0:
            st["rounds_sum"] += r.rounds
        for d in r.a_death_rounds.split(";"):
            if ":" in d:
                st["a_deaths"].append(int(d.split(":")[1]))
        for d in r.b_death_rounds.split(";"):
            if ":" in d:
                st["b_deaths"].append(int(d.split(":")[1]))

    # ---- combined per-pair (both orderings)
    pair_stats: dict[tuple, dict] = {}
    for (mp_, ba, bb), st in stats.items():
        pk = tuple(sorted((ba, bb))) + (mp_,)
        pst = pair_stats.setdefault(pk, {"n": 0, "ba_wins": 0, "bb_wins": 0,
                                          "draws": 0, "errors": 0})
        pst["n"] += st["n"]
        first, second = pk[0], pk[1]
        pst["ba_wins"] += st["a_wins"] if ba == first else st["b_wins"]
        pst["bb_wins"] += st["b_wins"] if ba == first else st["a_wins"]
        pst["draws"] += st["draws"]
        pst["errors"] += st["errors"]

    lines = []
    lines.append("=" * 78)
    lines.append("PER-ORDERING RESULTS (team A listed first moves first)")
    lines.append("=" * 78)
    hdr = f"{'map':<28} {'botA':<18} {'botB':<18} {'A':>4} {'B':>4} {'D':>4} {'E':>4}"
    lines.append(hdr)
    for (mp_, ba, bb), st in sorted(stats.items()):
        wr_a = 100.0 * st["a_wins"] / max(1, st["n"])
        wr_b = 100.0 * st["b_wins"] / max(1, st["n"])
        lines.append(
            f"{mp_:<28} {ba:<18} {bb:<18} "
            f"{st['a_wins']:>3}({wr_a:4.0f}%) {st['b_wins']:>3}({wr_b:4.0f}%) "
            f"{st['draws']:>3} {st['errors']:>3}"
        )
    lines.append("")
    lines.append("=" * 78)
    lines.append("COMBINED PER-PAIR (both orderings folded together)")
    lines.append("=" * 78)
    lines.append(f"{'map':<28} {'bot 1':<18} {'bot 2':<18} {'b1':>6} {'b2':>6} {'D':>4} {'E':>4}")
    for pk, st in sorted(pair_stats.items()):
        n = max(1, st["n"] - st["draws"] - st["errors"])
        w1 = 100.0 * st["ba_wins"] / max(1, st["n"])
        w2 = 100.0 * st["bb_wins"] / max(1, st["n"])
        lines.append(
            f"{pk[2]:<28} {pk[0]:<18} {pk[1]:<18} "
            f"{st['ba_wins']:>3}({w1:4.0f}%) {st['bb_wins']:>3}({w2:4.0f}%) "
            f"{st['draws']:>3} {st['errors']:>3}"
        )

    # death-round distribution per bot (survival stat)
    lines.append("")
    lines.append("=" * 78)
    lines.append("SURVIVAL: median death round per bot across all matches")
    lines.append("=" * 78)
    death_rounds: dict[str, list[int]] = {}
    for r in results:
        if r.winner in ("ERROR",):
            continue
        for bot, dfield in ((Path(r.bot_a).name, r.a_death_rounds),
                            (Path(r.bot_b).name, r.b_death_rounds)):
            for d in dfield.split(";"):
                if ":" in d:
                    death_rounds.setdefault(bot, []).append(int(d.split(":")[1]))
    med = lambda xs: sorted(xs)[len(xs)//2] if xs else -1
    for bot, ds in sorted(death_rounds.items()):
        lines.append(f"  {bot:<24} median death round: {med(ds):>4}  (n={len(ds)})")

    text = "\n".join(lines)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bots", nargs="+", required=True,
                    help="bot dirs (containing bot.toml) or executables/.cpp mains")
    ap.add_argument("--maps", nargs="+", required=True, help=".map files")
    ap.add_argument("--seeds", type=int, default=8,
                    help="number of seeds to sample (default 8)")
    ap.add_argument("--seed-list", type=int, nargs="*", default=None,
                    help="explicit seed integers (overrides --seeds)")
    ap.add_argument("--rng-seed", type=int, default=None,
                    help="RNG seed for SEED SELECTION (default: fixed 20261001 for reproducibility)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="parallel processes (default: all cores)")
    ap.add_argument("--sandbox", action="store_true",
                    help="judge-exact CPU-points pricing (slower)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="pass -v to unswbc (per-round logging)")
    ap.add_argument("--passthrough", nargs="*", default=[],
                    help="extra flags forwarded to unswbc run (e.g. -- --no-logs)")
    ap.add_argument("--timeout", type=int, default=240,
                    help="per-match wall clock seconds (default 240)")
    ap.add_argument("--out", type=Path, default=Path("eval_results/results.csv"),
                    help="CSV output path (summary written next to it)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned matchup count and exit")
    args = ap.parse_args()

    unswbc = shutil_which("unswbc") or "/Users/alif_1/.local/bin/unswbc"

    rng_seed = args.rng_seed if args.rng_seed is not None else 20261001
    if args.seed_list:
        seeds = args.seed_list
    else:
        rng = random.Random(rng_seed)
        seeds = [rng.getrandbits(64) for _ in range(args.seeds)]

    # Expand single-source-bot dirs? Users pass distinct dirs; identical dirs
    # on both sides is a legal mirror match and is preserved.
    tasks = build_tasks(args.bots, args.maps, seeds, unswbc,
                        args.sandbox, args.verbose, args.passthrough,
                        args.timeout)
    total_matches = len(tasks)
    n_orderings = len(list(itertools.permutations(args.bots, 2)))

    print(f"harness: {len(args.bots)} bots × {len(args.maps)} maps, "
          f"{len(seeds)} seeds, {n_orderings} orderings "
          f"=> {total_matches} matches across {args.workers} workers")
    print(f"seeds: {[hex(s) for s in seeds]}")
    if args.dry_run:
        for t in tasks[:6]:
            print(" ", " ".join(shlex.quote(c) for c in t["cmd"]))
        print(f"... {total_matches} total")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    results: list[MatchResult] = []
    with mp.Pool(args.workers) as pool:
        pending = pool.imap_unordered(run_one, tasks)
        for i, res in enumerate(pending, 1):
            results.append(res)
            if i % 10 == 0 or i == total_matches:
                el = time.monotonic() - t0
                err_n = sum(1 for r in results if r.winner == "ERROR")
                print(f"\r[{i}/{total_matches}] {el:6.0f}s  errors:{err_n}",
                      end="", flush=True)
    print()

    text = aggregate(results, args.out)
    print(text)
    out_txt = args.out.with_suffix(".summary.txt")
    out_txt.write_text(text)
    print(f"\nwrote: {args.out}\nwrote: {out_txt}")
    return 0


def shutil_which(cmd: str):
    from shutil import which
    return which(cmd)


if __name__ == "__main__":
    sys.exit(main())
