"""Expert demonstration collector (P3.5 Track D; seeds the v3 §4B BC path).

Runs food-only games with a **scripted expert** (default ``greedy_food``, the
measured mass-12 oracle) acting in the policy seat, and records, per alive
round, exactly what the RL policy would have seen and what the expert chose:

    (flat obs (736,), heading_bin, split, label_ok, is_reset)

The observation goes through the *real* policy path — ``Actor`` → shared
``Featurizer`` → ``flat_obs.flatten`` — so a network trained on these rows sees
the training/deploy distribution bit-for-bit. The label is the expert's move
snapped to the 24-bin act_v1 heading grid, and — deliberately — **the quantized
action is also what gets executed**, so the recorded trajectories live in the
policy's own action space (what a perfect clone would reproduce) rather than the
expert's continuous-heading space. The collector reports the quantized expert's
mean final mass, which doubles as the BC ceiling for this dataset.

A ``(0, 0)`` "stay" move has no heading angle: the row is recorded with
``label_ok=False`` (the BC loss masks its heading term; the split label stays
valid) and bin 0 is executed. Rare in the food-only field.

Numpy-only (no torch): importable by any collection process. CLI:

    python -m train.harness.demos --games 200 --out data/demos_greedy_food.npz
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from bot.adapters.act_v1 import HEADING_BINS, CanonicalAction
from train.harness.actor import Actor
from train.harness.flat_obs import OBS_WIDTH, flatten
from train.opponents.scripted import make_bot


def direction_to_bin(dx: float, dy: float) -> Tuple[int, bool]:
    """Snap a direction vector to the nearest act_v1 heading bin.

    Returns ``(bin, ok)``; ``ok=False`` for the angle-less ``(0, 0)`` stay.
    Bin geometry matches ``act_v1.bin_to_vector`` (bin i at ``2π·i/HEADING_BINS``,
    CCW from +x), so ``direction_to_bin(*bin_to_vector(b)) == (b, True)``.
    """
    if dx == 0.0 and dy == 0.0:
        return 0, False
    ang = math.atan2(dy, dx)
    return int(round(ang * HEADING_BINS / (2.0 * math.pi))) % HEADING_BINS, True


def collect_demos(
    expert: str = "greedy_food",
    games: int = 200,
    base_seed: int = 424_242,
    seat: int = 0,
    opponents: Sequence[str] = ("dont_move",),
    max_rounds: int = 0,
) -> Dict[str, np.ndarray]:
    """Collect expert rollouts; return the dataset as a dict of aligned arrays.

    ``max_rounds`` (0 = full games) caps recorded rounds per game — for tests.
    Rows are chronologically contiguous per game; ``is_reset`` marks each game's
    first frame (and any respawn, though the food-only field has none), which is
    all the recurrent BC trainer needs to zero hidden state correctly.
    """
    obs_rows: List[np.ndarray] = []
    headings: List[int] = []
    splits: List[float] = []
    label_oks: List[bool] = []
    is_resets: List[bool] = []
    final_masses: List[float] = []

    scripted_names = list(opponents)
    for g in range(games):
        bot = make_bot(expert)
        bot.reset()
        scripted = {
            pid: make_bot(scripted_names[(pid - 1) % len(scripted_names)])
            for pid in range(8) if pid != seat
        }
        actor = Actor(seed=base_seed + g, policy_seats={seat}, scripted=scripted,
                      emit_update=False)
        actor.reset()

        recorded = 0
        done = False
        while not done:
            obs = actor.observe()
            if seat in obs:
                q = actor.query(seat)
                move = bot.act(q)
                hbin, ok = direction_to_bin(move.direction.x, move.direction.y)
                obs_rows.append(flatten(obs[seat]))
                headings.append(hbin)
                splits.append(1.0 if move.split else 0.0)
                label_oks.append(ok)
                is_resets.append(bool(obs[seat].is_reset))
                done = actor.step({seat: CanonicalAction(heading_bin=hbin,
                                                         split=bool(move.split))})
                recorded += 1
                if max_rounds and recorded >= max_rounds:
                    break
            else:  # dead gap: scripted seats act, no row
                done = actor.step({})
        final_masses.append(float(actor.final_masses()[seat]))

    return {
        "obs": np.stack(obs_rows).astype(np.float32),
        "heading": np.asarray(headings, dtype=np.int64),
        "split": np.asarray(splits, dtype=np.float32),
        "label_ok": np.asarray(label_oks, dtype=bool),
        "is_reset": np.asarray(is_resets, dtype=bool),
        "final_masses": np.asarray(final_masses, dtype=np.float32),
    }


def save_demos(data: Dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **data)


def _main() -> None:
    p = argparse.ArgumentParser(description="Collect expert demos (P3.5 Track D)")
    p.add_argument("--expert", default="greedy_food")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--base-seed", type=int, default=424_242)
    p.add_argument("--opponents", nargs="+", default=["dont_move"])
    p.add_argument("--out", default="data/demos_greedy_food.npz")
    args = p.parse_args()

    t0 = time.monotonic()
    data = collect_demos(expert=args.expert, games=args.games,
                         base_seed=args.base_seed, opponents=tuple(args.opponents))
    save_demos(data, args.out)
    fm = data["final_masses"]
    print(
        f"[demos] {args.games} games -> {len(data['obs'])} rows in "
        f"{time.monotonic() - t0:.0f}s | quantized-{args.expert} mean final mass "
        f"{fm.mean():.2f} (min {fm.min():.2f}, max {fm.max():.2f}) | "
        f"label_ok {data['label_ok'].mean():.1%} | wrote {args.out}"
    )


if __name__ == "__main__":
    _main()


__all__ = ["collect_demos", "save_demos", "direction_to_bin", "OBS_WIDTH"]
