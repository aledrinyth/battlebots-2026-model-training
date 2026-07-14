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
import multiprocessing as mp
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


def _play_expert_game(expert: str, g: int, base_seed: int, seat: int,
                      opponents: Sequence[str], max_rounds: int) -> Dict[str, np.ndarray]:
    """One seeded expert game → this game's aligned row arrays."""
    obs_rows: List[np.ndarray] = []
    headings: List[int] = []
    splits: List[float] = []
    label_oks: List[bool] = []
    is_resets: List[bool] = []

    scripted_names = list(opponents)
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

    return {
        "obs": np.stack(obs_rows).astype(np.float32),
        "heading": np.asarray(headings, dtype=np.int64),
        "split": np.asarray(splits, dtype=np.float32),
        "label_ok": np.asarray(label_oks, dtype=bool),
        "is_reset": np.asarray(is_resets, dtype=bool),
        "final_mass": np.float32(actor.final_masses()[seat]),
    }


def _mp_collect(args) -> List[Tuple[int, Dict[str, np.ndarray]]]:
    """Worker entry (spawn): play a strided slice of game indices."""
    expert, indices, base_seed, seat, opponents, max_rounds = args
    return [(g, _play_expert_game(expert, g, base_seed, seat, opponents, max_rounds))
            for g in indices]


def _strided_chunks(n: int, workers: int) -> List[List[int]]:
    """Partition ``range(n)`` into ≤``workers`` strided, non-empty chunks."""
    chunks = [list(range(w, n, workers)) for w in range(max(1, workers))]
    return [c for c in chunks if c]


def collect_demos(
    expert: str = "greedy_food",
    games: int = 200,
    base_seed: int = 424_242,
    seat: int = 0,
    opponents: Sequence[str] = ("dont_move",),
    max_rounds: int = 0,
    workers: int = 1,
) -> Dict[str, np.ndarray]:
    """Collect expert rollouts; return the dataset as a dict of aligned arrays.

    ``max_rounds`` (0 = full games) caps recorded rounds per game — for tests.
    Rows are chronologically contiguous per game; ``is_reset`` marks each game's
    first frame (and any respawn — mixed-opponent fields have real deaths),
    which is all the recurrent BC trainer needs to zero hidden state correctly.

    ``workers > 1`` fans games out over processes (each game is fully
    determined by ``base_seed + g``, so the dataset is byte-identical to the
    sequential run — games always concatenate in index order).
    """
    if workers > 1 and games > 1:
        ctx = mp.get_context("spawn")
        jobs = [(expert, chunk, base_seed, seat, tuple(opponents), max_rounds)
                for chunk in _strided_chunks(games, workers)]
        with ctx.Pool(len(jobs)) as pool:
            played = [row for rows in pool.map(_mp_collect, jobs) for row in rows]
        played.sort()  # by game index — order-independent of worker count
        per_game = [d for _g, d in played]
    else:
        per_game = [_play_expert_game(expert, g, base_seed, seat, opponents, max_rounds)
                    for g in range(games)]

    return {
        "obs": np.concatenate([d["obs"] for d in per_game]),
        "heading": np.concatenate([d["heading"] for d in per_game]),
        "split": np.concatenate([d["split"] for d in per_game]),
        "label_ok": np.concatenate([d["label_ok"] for d in per_game]),
        "is_reset": np.concatenate([d["is_reset"] for d in per_game]),
        "final_masses": np.asarray([d["final_mass"] for d in per_game], dtype=np.float32),
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
    p.add_argument("--workers", type=int, default=1,
                   help="parallelize games across N processes (byte-identical "
                        "dataset, ~N× faster; keep cores free if training is running)")
    args = p.parse_args()

    t0 = time.monotonic()
    data = collect_demos(expert=args.expert, games=args.games,
                         base_seed=args.base_seed, opponents=tuple(args.opponents),
                         workers=args.workers)
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
