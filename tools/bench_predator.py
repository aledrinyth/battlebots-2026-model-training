#!/usr/bin/env python3
"""Batch large-predator survivability: champion v13h vs a candidate, PAIRED.

Wraps tools/inject_predator_match.run: for each seed, both bots sit in the SAME
naturally-aged arena and each faces an identically-specced mass-40 split-lunging
predator injected at its vision edge (the "rounded a corner into a big opponent
that fed off-screen" failure mode from the tournament losses). Paired => low
variance, directly comparable.

Reports, per side: death rate in the observation window and mean surviving mass.
Lower death rate + higher surviving mass = better.

Usage:
  bench_predator.py [N_SEEDS] --cand dist/cand/potential_field_v14f.py \
      [--inject-round 400] [--window 120] [--pred-mass 40]
Champion (target0) is always v13h; candidate is target1.
"""
import sys, argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.inject_predator_match import run  # noqa: E402

CHAMP = "dist/potential_field_v13h.py"


def _job(a):
    seed, inj, win, pmass, offset, cand, rcap = a
    try:
        r = run(seed, inj, win, pmass, offset,
                str((REPO / CHAMP).resolve()),
                str((REPO / cand).resolve()), rcap)
        return r
    except Exception as e:
        return {"err": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=48)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--inject-round", type=int, default=400)
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--pred-mass", type=float, default=40.0)
    ap.add_argument("--offset", type=float, default=15.0)
    ap.add_argument("--rounds-cap", type=int, default=1400)
    a = ap.parse_args()

    jobs = [(s, a.inject_round, a.window, a.pred_mass, a.offset, a.cand, a.rounds_cap)
            for s in range(a.n)]
    champ_d = cand_d = 0
    champ_m = cand_m = 0.0
    n_ok = 0
    errs = {}
    # max_tasks_per_child=1: the in-process engine binds CORE_DIRECTORY at import
    # time, so a reused worker would read a stale (deleted) workspace's pipes.
    # A fresh process per task imports the engine against its own workspace.
    with ProcessPoolExecutor(max_workers=8, max_tasks_per_child=1) as ex:
        for r in ex.map(_job, jobs):
            if "err" in r:
                errs[r["err"]] = errs.get(r["err"], 0) + 1
                continue
            n_ok += 1
            champ_d += 1 if r["died0"] else 0
            cand_d += 1 if r["died1"] else 0
            champ_m += r["mass0"]
            cand_m += r["mass1"]
    if errs:
        print(f"[warn] {sum(errs.values())} runs errored:")
        for e, c in sorted(errs.items(), key=lambda x: -x[1])[:5]:
            print(f"   x{c}: {e[:160]}")
    if not n_ok:
        print("no successful runs"); return
    print(f"PREDATOR STRESS  inject@{a.inject_round} window={a.window} "
          f"predMass={a.pred_mass}  seeds={n_ok}")
    print(f"{'bot':22} {'deathRate':>10} {'survMass':>9}")
    print(f"{'v13h (champion)':22} {champ_d/n_ok:10.2%} {champ_m/n_ok:9.3f}")
    print(f"{a.cand.split('/')[-1]:22} {cand_d/n_ok:10.2%} {cand_m/n_ok:9.3f}")
    dd = (champ_d - cand_d) / n_ok
    print(f"\ncandidate death-rate delta vs champion: {dd:+.2%} "
          f"({'BETTER' if dd > 0 else 'WORSE' if dd < 0 else 'tie'})")


if __name__ == "__main__":
    main()
