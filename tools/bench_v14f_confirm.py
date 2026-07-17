#!/usr/bin/env python3
"""Confirmation: does v14f genuinely beat champion v13h?

Cleaner field than bench_candidates: only the two contenders (v14f, v13h) plus
the three stable baselines (v8, v11, v12). Each contender gets 2 seats, baselines
fill the rest, seat-rotated per seed. Reduces the "many near-identical candidates
reflect off each other" noise of the 8-way candidate field.

Reports the same growth/survival metrics. Head-to-head line at the end: in how
many seeds did v14f's best seat outrank v13h's best seat.
"""
import subprocess, json, tempfile, os, sys
from concurrent.futures import ProcessPoolExecutor

REPO = "/Users/alif_1/Documents/battlebots"
PY = f"{REPO}/.venv-bot/bin/python"
REC = "tools/record_reference_match.py"

# 8 seats: v14f x2, v13h x2, v8, v11, v12, v11 (baselines fill 4)
FIELD = [
    ("v14f", "dist/cand/potential_field_v14f.py"),
    ("v14f", "dist/cand/potential_field_v14f.py"),
    ("v13h", "dist/potential_field_v13h.py"),
    ("v13h", "dist/potential_field_v13h.py"),
    ("v8",   "dist/potential_field_v8.py"),
    ("v11",  "dist/potential_field_v11.py"),
    ("v12",  "dist/potential_field_v12.py"),
    ("v11",  "dist/potential_field_v11.py"),
]


def _peak_deaths(trace):
    n = trace["num_players"]
    peak = [0.0] * n; deaths = [0] * n; prev = [True] * n
    for rnd in trace["rounds"]:
        for pid_s, s in rnd["snaps"].items():
            pid = int(pid_s)
            m = sum(b[3] * b[3] for b in s["blobs"])
            if m > peak[pid]: peak[pid] = m
            alive = bool(s["alive"]) and len(s["blobs"]) > 0
            if prev[pid] and not alive: deaths[pid] += 1
            prev[pid] = alive
    return peak, deaths


def run_one(args):
    seed, shift = args
    order = FIELD[shift:] + FIELD[:shift]
    specs = [f"1:{f[1]}" for f in order]
    fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
    subprocess.run([PY, REC, "--seed", str(seed), "--rounds", "1400", "--out", tmp, "--bots", *specs],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=REPO)
    t = json.loads(open(tmp).read()); os.unlink(tmp)
    fm = {int(k): v for k, v in t["final_masses"].items()}
    rank = {p: i + 1 for i, p in enumerate(t["rankings"])}
    peak, deaths = _peak_deaths(t)
    agg = {}
    best_rank = {}
    for seat, f in enumerate(order):
        nm = f[0]
        d = agg.setdefault(nm, {"final": 0.0, "rank": 0.0, "peak": 0.0, "deaths": 0.0, "n": 0, "win": 0})
        d["final"] += fm[seat]; d["rank"] += rank[seat]; d["peak"] += peak[seat]
        d["deaths"] += deaths[seat]; d["n"] += 1; d["win"] += 1 if rank[seat] == 1 else 0
        best_rank[nm] = min(best_rank.get(nm, 99), rank[seat])
    h2h = 1 if best_rank.get("v14f", 99) < best_rank.get("v13h", 99) else 0
    return agg, h2h


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    jobs = [(s, s % 8) for s in range(N)]
    agg = {}; h2h_wins = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for out, h2h in ex.map(run_one, jobs):
            h2h_wins += h2h
            for nm, d in out.items():
                a = agg.setdefault(nm, {"final": 0.0, "rank": 0.0, "peak": 0.0, "deaths": 0.0, "n": 0, "win": 0})
                for k in ("final", "rank", "peak", "deaths", "n", "win"):
                    a[k] += d[k]
    print(f"CONFIRM {N} seeds (v14f x2, v13h x2, v8/v11/v12/v11):")
    print(f"{'bot':6} {'seatgames':>9} {'meanRank':>9} {'meanFinal':>10} {'meanPeak':>9} {'deaths':>7} {'wins':>5}")
    for nm in sorted(agg, key=lambda k: agg[k]['rank'] / agg[k]['n']):
        a = agg[nm]; n = a['n']
        print(f"{nm:6} {n:9d} {a['rank']/n:9.3f} {a['final']/n:10.3f} {a['peak']/n:9.3f} {a['deaths']/n:7.2f} {a['win']:5d}")
    print(f"\nHEAD-TO-HEAD (best seat per side): v14f beat v13h in {h2h_wins}/{N} seeds "
          f"({100*h2h_wins/N:.0f}%)")
