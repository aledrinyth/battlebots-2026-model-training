#!/usr/bin/env python3
"""Paired multiway benchmark for v14 candidate bots vs the shipped lineage.

Every game runs 8 distinct single-file bots (seat-rotated per seed so each bot
sees every seat equally -> paired). For each bot we report, averaged over all
its seat-games:

  meanRank   final ranking (1 = best) -- the tournament objective
  meanFinal  final mass
  meanPeak   peak mass reached at ANY round (the "did it snowball" signal)
  deaths     mean death count (blob fully eaten) -- the "farmed early" signal
  firstDeath mean round of first death ('-' style: higher = survived longer)
  wins       seat-games finished rank 1

Root-cause context (replay_stats over 628 tourney matches): team 53 flatlines
at ~1.2 mass because it never grows -- so meanPeak and firstDeath are the
metrics that actually transfer, not just meanRank in the self-play echo chamber.

Usage: tools/bench_candidates.py [N_SEEDS] [--field a,b,c]
"""
import subprocess, json, tempfile, os, sys, math
from concurrent.futures import ProcessPoolExecutor

REPO = "/Users/alif_1/Documents/battlebots"
PY = f"{REPO}/.venv-bot/bin/python"
REC = "tools/record_reference_match.py"

# 8-bot field: 4 shipped baselines + 4 candidates. All single-file & runnable.
FIELD = [
    ("v8",     "dist/potential_field_v8.py"),
    ("v11",    "dist/potential_field_v11.py"),
    ("v12",    "dist/potential_field_v12.py"),
    ("v13h",   "dist/potential_field_v13h.py"),
    ("v13h2",  "dist/potential_field_v13h.py"),         # 2nd champion seat (noise anchor)
    ("v14h",   "dist/cand/potential_field_v14h.py"),    # early capture, range 9
    ("v14h5",  "dist/cand/potential_field_v14h5.py"),   # early capture, range 5 (safe)
    ("v14h5b", "dist/cand/potential_field_v14h5.py"),   # 2nd v14h5 seat (noise anchor)
]


def _peak_and_deaths(trace):
    """Per-seat peak mass and death count from the round trace.

    mass(player) = sum of blob radius^2 over its blobs in that round's snapshot.
    A death = a round where the player was alive last snapshot and is now not,
    OR (robust) its blob set went empty. We approximate deaths by counting
    alive->dead transitions across snapshots."""
    n = trace["num_players"]
    peak = [0.0] * n
    deaths = [0] * n
    first_death = [None] * n
    prev_alive = [True] * n
    for ri, rnd in enumerate(trace["rounds"]):
        snaps = rnd["snaps"]
        for pid_s, s in snaps.items():
            pid = int(pid_s)
            m = sum(b[3] * b[3] for b in s["blobs"])  # b = [id,x,y,radius,cd]
            if m > peak[pid]:
                peak[pid] = m
            alive = bool(s["alive"]) and len(s["blobs"]) > 0
            if prev_alive[pid] and not alive:
                deaths[pid] += 1
                if first_death[pid] is None:
                    first_death[pid] = ri
            prev_alive[pid] = alive
    return peak, deaths, first_death


def run_one(args):
    seed, shift = args
    order = FIELD[shift:] + FIELD[:shift]
    specs = [f"1:{f[1]}" for f in order]
    fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
    subprocess.run([PY, REC, "--seed", str(seed), "--rounds", "1400", "--out", tmp, "--bots", *specs],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   cwd=REPO)
    t = json.loads(open(tmp).read()); os.unlink(tmp)
    fm = {int(k): v for k, v in t["final_masses"].items()}
    rank = {p: i + 1 for i, p in enumerate(t["rankings"])}
    peak, deaths, fdeath = _peak_and_deaths(t)
    out = {}
    for seat, f in enumerate(order):
        nm = f[0]
        d = out.setdefault(nm, {"final": 0.0, "rank": 0.0, "peak": 0.0,
                                "deaths": 0.0, "fdeath": 0.0, "n": 0, "win": 0})
        d["final"] += fm[seat]
        d["rank"] += rank[seat]
        d["peak"] += peak[seat]
        d["deaths"] += deaths[seat]
        d["fdeath"] += fdeath[seat] if fdeath[seat] is not None else 1400
        d["n"] += 1
        d["win"] += 1 if rank[seat] == 1 else 0
    return out


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    jobs = [(s, s % 8) for s in range(N)]
    agg = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for out in ex.map(run_one, jobs):
            for nm, d in out.items():
                a = agg.setdefault(nm, {"final": 0.0, "rank": 0.0, "peak": 0.0,
                                        "deaths": 0.0, "fdeath": 0.0, "n": 0, "win": 0})
                for k in ("final", "rank", "peak", "deaths", "fdeath", "n", "win"):
                    a[k] += d[k]
    print(f"CANDIDATE MULTIWAY {N} seeds (8 bots/game, seat-rotated, paired):")
    print(f"{'bot':6} {'games':>5} {'meanRank':>9} {'meanFinal':>10} {'meanPeak':>9} "
          f"{'deaths':>7} {'1stDeath':>9} {'wins':>5}")
    for nm in sorted(agg, key=lambda k: agg[k]['rank'] / agg[k]['n']):
        a = agg[nm]; n = a['n']
        print(f"{nm:6} {n:5d} {a['rank']/n:9.3f} {a['final']/n:10.3f} {a['peak']/n:9.3f} "
              f"{a['deaths']/n:7.2f} {a['fdeath']/n:9.0f} {a['win']:5d}")
