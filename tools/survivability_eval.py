#!/usr/bin/env python3
"""Score early-game SURVIVABILITY of a candidate bot in full 8-player games.

Runs 8-player games (short — default 400 rounds, but otherwise a full game:
regular respawns, full arena) where the roster is the CANDIDATE plus previous
iterations (default v11 / v10 / v8), two seats each. Each game is scored PER BOT
TYPE by the user's rule:

    reward  = sum of the type's total mass at the end of every ``--checkpoint``
              rounds (100, 200, 300, 400) — a bot that dies is 0 mass at that
              checkpoint, so survival is rewarded directly
    penalty = ``--death-penalty`` for every death (an ``alive`` True->False
              transition; a respawn that dies again counts again)
    score   = reward - penalty

Because every type plays the SAME games, the baselines (v11/v10/v8) calibrate
the scale, so a candidate's score is directly comparable across experiments.

Seats are rotated (shift 0 and 4 = the ab_compare mirror) across each seed so
no type is pinned to one quadrant. Runs games as isolated subprocesses (the
engine is not re-entrant) in a small process pool.

Usage:
  tools/survivability_eval.py --candidate dist/potential_field_v11a.py \
      --label v11a --seeds 0 1 2 3 4 5 6 7 --rounds 400 --death-penalty 3.0
"""
from __future__ import annotations
import argparse, json, os, subprocess, tempfile, statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".venv-bot/bin/python")
RECORDER = str(REPO / "tools/record_reference_match.py")

DEFAULT_BASELINES = {
    "v11": "dist/potential_field_v11.py",
    "v10": "dist/potential_field_v10.py",
    "v8": "dist/potential_field_v8.py",
}


def _mass_at(snap) -> float:
    if snap is None or not snap["alive"]:
        return 0.0
    return sum(b[3] * b[3] for b in snap["blobs"])


def _score_trace(trace: dict, roster: list[str], checkpoints: list[int]) -> dict:
    """Return per-seat records: {seat: {tag, deaths, masses:{cp:mass}}}."""
    R = trace["rounds"]
    n = len(R)
    recs = {}
    for seat, tag in enumerate(roster):
        key = str(seat)
        # deaths: alive True -> False transitions across the whole game
        prev = None
        deaths = 0
        for rr in R:
            s = rr["snaps"].get(key)
            a = None if s is None else s["alive"]
            if prev is True and a is False:
                deaths += 1
            if a is not None:
                prev = a
        masses = {}
        for cp in checkpoints:
            i = min(cp - 1, n - 1)
            masses[cp] = _mass_at(R[i]["snaps"].get(key))
        recs[seat] = {"tag": tag, "deaths": deaths, "masses": masses}
    return recs


def _run_one(args) -> dict:
    label, files, order, seed, rounds, shift, checkpoints = args
    # base roster: 2 of each type, candidate first
    base = [t for t in order for _ in range(2)]
    roster = base[shift:] + base[:shift]
    specs = [f"1:{files[tag]}" for tag in roster]
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    cmd = [PY, RECORDER, "--seed", str(seed), "--rounds", str(rounds),
           "--out", tmp, "--bots", *specs]
    subprocess.run(cmd, cwd=str(REPO), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    trace = json.loads(Path(tmp).read_text())
    Path(tmp).unlink(missing_ok=True)
    return _score_trace(trace, roster, checkpoints)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--death-penalty", type=float, default=3.0)
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=[100, 200, 300, 400])
    ap.add_argument("--baselines", nargs="*", default=None,
                    help="tag=path pairs; default v11/v10/v8")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.baselines:
        baselines = {}
        for kv in args.baselines:
            k, _, v = kv.partition("=")
            baselines[k] = v
    else:
        baselines = dict(DEFAULT_BASELINES)
    baselines_files = {k: str((REPO / v)) for k, v in baselines.items()}
    files = {args.label: str(REPO / args.candidate), **baselines_files}
    order = [args.label] + list(baselines.keys())
    checkpoints = [c for c in args.checkpoints if c <= args.rounds]

    jobs = []
    for seed in args.seeds:
        for shift in (0, 2, 4, 6):
            jobs.append((args.label, files, order,
                         seed, args.rounds, shift, checkpoints))

    all_recs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for recs in ex.map(_run_one, jobs):
            all_recs.append(recs)

    # aggregate per tag across all games+seats
    tags = [args.label] + list(baselines.keys())
    agg = {t: {"deaths": 0, "seat_games": 0,
               "mass": {cp: 0.0 for cp in checkpoints}} for t in tags}
    for recs in all_recs:
        for seat, r in recs.items():
            t = r["tag"]
            agg[t]["deaths"] += r["deaths"]
            agg[t]["seat_games"] += 1
            for cp in checkpoints:
                agg[t]["mass"][cp] += r["masses"][cp]

    print(f"\n==== SURVIVABILITY: {args.label} vs {'/'.join(baselines)} ====")
    print(f"seeds={args.seeds} rounds={args.rounds} games={len(jobs)} "
          f"death_penalty={args.death_penalty} checkpoints={checkpoints}")
    print(f"{'tag':>6} | {'seatgames':>9} | {'deaths':>6} {'d/game':>7} | "
          + " ".join(f'm@{cp:>4}' for cp in checkpoints)
          + f" | {'reward':>8} {'penalty':>8} {'SCORE':>9}")
    results = {}
    for t in tags:
        a = agg[t]
        sg = a["seat_games"] or 1
        reward = sum(a["mass"][cp] for cp in checkpoints)
        penalty = args.death_penalty * a["deaths"]
        score = reward - penalty
        # per-seat-game means for readability
        mean_m = {cp: a["mass"][cp] / sg for cp in checkpoints}
        results[t] = {"deaths": a["deaths"], "seat_games": sg,
                      "reward": reward, "penalty": penalty, "score": score,
                      "score_per_game": score / sg,
                      "deaths_per_game": a["deaths"] / sg,
                      "mean_mass": mean_m}
        print(f"{t:>6} | {sg:>9} | {a['deaths']:>6} {a['deaths']/sg:>7.3f} | "
              + " ".join(f'{mean_m[cp]:>6.2f}' for cp in checkpoints)
              + f" | {reward:>8.1f} {penalty:>8.1f} {score:>9.1f}")

    # headline: candidate vs each baseline on score-per-game and deaths-per-game
    cand = results[args.label]
    print("\n---- candidate vs baselines (per seat-game) ----")
    for t in baselines:
        b = results[t]
        print(f"  {args.label} vs {t}: "
              f"score {cand['score_per_game']:.2f} vs {b['score_per_game']:.2f} "
              f"(Δ {cand['score_per_game']-b['score_per_game']:+.2f})  |  "
              f"deaths/game {cand['deaths_per_game']:.3f} vs {b['deaths_per_game']:.3f} "
              f"(Δ {cand['deaths_per_game']-b['deaths_per_game']:+.3f})")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"label": args.label, "rounds": args.rounds,
             "death_penalty": args.death_penalty, "results": results}, indent=2))
        print(f"\n[json] -> {args.json_out}")


if __name__ == "__main__":
    main()
