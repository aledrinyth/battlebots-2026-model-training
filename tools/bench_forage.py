#!/usr/bin/env python3
"""Isolated foraging-efficiency benchmark.

The mass gap that gates everything (v13-loss-diagnosis) originates in rounds
0-400 pure foraging, where we eat ~15 pellets/100r vs winners' ~30 -- and ~80%
of that time nothing threatens us. So this measures ONE thing with no self-play
noise: how much mass a bot accrues when mostly left alone to eat.

Field = 1 candidate seat + 7 DontMove/passive seats (they spawn, sit, and are
too small to threaten), so the candidate forages a near-empty arena. We report
mass at rounds 100/200/300/400 -- the "do we cross mass 5 by ~300 like winners"
question -- averaged over seeds (seat-rotated, paired).

Usage: bench_forage.py [N] --bots name=path,name=path,...
"""
import subprocess, json, tempfile, os, sys, argparse
from concurrent.futures import ProcessPoolExecutor

REPO = "/Users/alif_1/Documents/battlebots"
PY = f"{REPO}/.venv-bot/bin/python"
REC = "tools/record_reference_match.py"
PASSIVE = "tests/fixtures/bots/dont_move.py"


def _mass_at(trace, seat, rounds):
    """Mass of `seat` at each requested round index (last snapshot <= r)."""
    out = {}
    ri_target = sorted(rounds)
    ti = 0
    for ri, rnd in enumerate(trace["rounds"]):
        s = rnd["snaps"].get(str(seat))
        if s is None:
            continue
        m = sum(b[3] * b[3] for b in s["blobs"])
        while ti < len(ri_target) and ri >= ri_target[ti]:
            out[ri_target[ti]] = m
            ti += 1
    for r in rounds:
        out.setdefault(r, 0.0)
    return out


def run_one(args):
    seed, cand_path, cand_name = args
    # candidate in seat 0, passive elsewhere
    specs = [f"1:{cand_path}", f"7:{PASSIVE}"]
    fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
    subprocess.run([PY, REC, "--seed", str(seed), "--rounds", "420", "--out", tmp, "--bots", *specs],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=REPO)
    t = json.loads(open(tmp).read()); os.unlink(tmp)
    return _mass_at(t, 0, [100, 200, 300, 400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=24)
    ap.add_argument("--bots", required=True, help="name=path,name=path,...")
    a = ap.parse_args()
    cands = [b.split("=") for b in a.bots.split(",")]
    print(f"FORAGE (1 forager + 7 passive), {a.n} seeds — mass at round:")
    print(f"{'bot':10} {'r100':>7} {'r200':>7} {'r300':>7} {'r400':>7}")
    for name, path in cands:
        jobs = [(s, path, name) for s in range(a.n)]
        acc = {100: 0.0, 200: 0.0, 300: 0.0, 400: 0.0}
        with ProcessPoolExecutor(max_workers=8) as ex:
            for m in ex.map(run_one, jobs):
                for k in acc:
                    acc[k] += m[k]
        n = a.n
        print(f"{name:10} {acc[100]/n:7.2f} {acc[200]/n:7.2f} {acc[300]/n:7.2f} {acc[400]/n:7.2f}")


if __name__ == "__main__":
    main()
