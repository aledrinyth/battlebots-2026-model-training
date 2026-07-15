#!/usr/bin/env python3
"""P6.3 — promotion gate: never promote noise.

Adapted from head-to-head SPRT to 8-player FFA: the candidate and the
incumbent play **paired game sets** — identical seeds, identical seat
rotation, identical opponents (``evaluate.play_games`` guarantees game ``g``
is fully determined by ``(base_seed + g, g % 8)``), with only the subject
swapped. The statistic is the paired per-game rank difference

    d_g = rank_candidate(g) − rank_incumbent(g)      (negative ⇒ better)

evaluated **sequentially in blocks**: after each block, a seeded bootstrap CI
on ``mean(d)`` decides —

* promote  when the (1−α) CI sits entirely below 0 (candidate better,
  p < α one-sided);
* reject   when it sits entirely above 0 (incumbent clearly better — stop
  wasting compute, the SPRT-style futility exit);
* continue otherwise, until ``cap`` paired games, then **no promotion**
  (conservative: ties keep the incumbent).

Identical weights ⇒ identical games ⇒ every ``d_g == 0`` ⇒ the CI is the
point {0} and the gate never promotes — the playbook's self-test falls out
of the construction.

Usage:
    tools/promote.py --candidate cand.pt --incumbent inc.pt \
        [--opponents greedy_food cautious chaser coward dont_move] \
        [--block 200] [--cap 2000] [--alpha 0.05] [--workers 16] [--seed 424243]

Exit 0 ⇒ promoted; exit 1 ⇒ not promoted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_OPPONENTS = ("greedy_food", "cautious", "chaser", "coward", "dont_move")


def paired_verdict(diffs: Sequence[float], alpha: float = 0.05,
                   n_boot: int = 10_000, seed: int = 0) -> Dict:
    """Bootstrap CI on the mean paired difference → promote/reject/continue."""
    d = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])  # one-sided at α each way
    if hi < 0.0:
        decision = "promote"
    elif lo > 0.0:
        decision = "reject"
    else:
        decision = "continue"
    return {
        "n": int(len(d)),
        "mean_diff": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "decision": decision,
    }


def promote(
    candidate: str,
    incumbent: str,
    opponents: Sequence[str] = DEFAULT_OPPONENTS,
    block: int = 200,
    cap: int = 2000,
    alpha: float = 0.05,
    device: str = "cpu",
    base_seed: int = 424_243,
    workers: int = 1,
    play_games_fn=None,
) -> Dict:
    """Run the sequential paired gate; return the final verdict dict.

    ``play_games_fn`` is injectable for tests; defaults to the real
    ``train.harness.evaluate.play_games``.
    """
    if play_games_fn is None:
        from train.harness.evaluate import play_games as play_games_fn  # noqa: PLW0127

    diffs: List[float] = []
    played = 0
    history: List[Dict] = []
    while played < cap:
        n = min(block, cap - played)
        # Paired sets: same seed window for both subjects, offset per block.
        seed0 = base_seed + played
        cand = play_games_fn(candidate, tuple(opponents), n, device, seed0, workers)
        inc = play_games_fn(incumbent, tuple(opponents), n, device, seed0, workers)
        for (gc, rc, _mc), (gi, ri, _mi) in zip(cand, inc):
            assert gc == gi, "paired sets fell out of alignment"
            diffs.append(float(rc - ri))
        played += n

        verdict = paired_verdict(diffs, alpha=alpha, seed=base_seed)
        history.append(verdict)
        if verdict["decision"] != "continue":
            break

    final = history[-1]
    return {
        "promoted": final["decision"] == "promote",
        "decision": final["decision"] if final["decision"] != "continue" else "no_promotion_at_cap",
        "games_paired": played,
        "mean_rank_diff": final["mean_diff"],   # negative ⇒ candidate better
        "ci": [final["ci_lo"], final["ci_hi"]],
        "alpha": alpha,
        "blocks": history,
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    ap.add_argument("--block", type=int, default=200)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=424_243)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    res = promote(args.candidate, args.incumbent, tuple(args.opponents),
                  block=args.block, cap=args.cap, alpha=args.alpha,
                  device=args.device, base_seed=args.seed, workers=args.workers)
    for i, b in enumerate(res["blocks"]):
        print(f"[promote] block {i}: n={b['n']} mean_diff={b['mean_diff']:+.3f} "
              f"CI[{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}] -> {b['decision']}")
    print(f"[promote] {'PROMOTED' if res['promoted'] else 'NOT PROMOTED'} "
          f"({res['decision']}) after {res['games_paired']} paired games; "
          f"mean rank diff {res['mean_rank_diff']:+.3f}")
    sys.exit(0 if res["promoted"] else 1)


if __name__ == "__main__":
    _main()
