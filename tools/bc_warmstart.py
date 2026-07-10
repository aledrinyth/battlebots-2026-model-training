"""Behaviour-cloning warm-start (P3.5 Track D) — imitate the greedy_food expert.

The capacity probe (`tools.probe_food_capacity`) established that the deep-set
network CAN read nearest-food direction (≈72% top-1 supervised), so G1's wall is
RL/exploration, not representation. The literature's reliable unlock for exactly
this situation is behaviour cloning: initialise the policy by supervised
imitation of a competent expert, *then* PPO-fine-tune (or, if BC alone is strong
enough, ship it). No replay corpus is needed for Stage-0 — the scripted
`greedy_food` bot (mean final mass ≈ 12 in this field vs the 0.81 floor)
generates unlimited demonstrations through the existing `Actor`.

Pipeline:
  1. Collect: play food-only games with `greedy_food` acting in seat 0; every
     round record the featurized obs the seat saw + the expert's chosen action
     (its `MovePlayer` direction → nearest heading bin, and its split bit).
  2. Fit: the AgarPolicy actor path (deep-set encoders → trunk → GRU-zero-hidden
     → heading + split heads), cross-entropy on the bin + BCE on split.
  3. Save an **evaluate-/resume-compatible** checkpoint (state_dict + arch +
     provenance), so `train.harness.evaluate --weights …` scores it for G1 and
     `train.harness.run --init-weights …` warm-starts PPO from it.

Single-frame (zero-hidden) BC is a deliberate, adequate warm-start: it seeds a
food-steering policy that PPO then makes recurrent/robust. Training-only tool.

    python -m tools.bc_warmstart --games 150 --epochs 20 --out runs/bc_greedy/bc_v1.pt
    python -m train.harness.evaluate --weights runs/bc_greedy/bc_v1.pt \
        --opponents dont_move --games 30 --vs-random
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from bot.adapters.act_v1 import HEADING_BINS
from bot.adapters.obs_v1 import CanonicalObs
from train.harness.actor import Actor
from train.harness.collate import collate_obs
from train.harness.config import TrainConfig
from train.harness.provenance import registry_row
from train.harness.shard import _atomic
from train.harness.trainer import _arch_meta
from train.model import build_policy
from train.opponents.scripted import make_bot

_TWO_PI = 2.0 * math.pi


def _dir_to_bin(move) -> int:
    """Nearest heading bin to a MovePlayer's (world-frame) direction vector."""
    ang = math.atan2(float(move.direction.y), float(move.direction.x))
    return int(round(ang * HEADING_BINS / _TWO_PI)) % HEADING_BINS


def collect_demos(n_games: int, base_seed: int, max_per_game: int = 1500
                  ) -> Tuple[List[CanonicalObs], np.ndarray, np.ndarray]:
    """Play `greedy_food` in seat 0; record (obs, expert heading bin, split).

    Seat 0 is a *policy* seat so the actor featurizes it, but it is driven by a
    separate `greedy_food` expert scoring the same raw query — giving featurized
    obs paired with the expert's label, on the expert's own (competent, food-
    seeking) trajectory. Seats 1–7 are `dont_move` (the Stage-0 food-only field)."""
    from bot.adapters.act_v1 import CanonicalAction

    obs_list: List[CanonicalObs] = []
    headings: List[int] = []
    splits: List[float] = []
    for g in range(n_games):
        expert = make_bot("greedy_food")
        expert.reset()
        scripted = {pid: make_bot("dont_move") for pid in range(1, 8)}
        actor = Actor(seed=base_seed + g, policy_seats={0}, scripted=scripted,
                      emit_update=False)
        actor.reset()
        kept = 0
        done = False
        while not done and kept < max_per_game:
            frame = actor.observe()
            if 0 not in frame:                        # seat 0 dead (respawn gap)
                done = actor.step({})
                continue
            co = frame[0]
            move = expert.act(actor.query(0))         # expert's chosen MovePlayer
            b, sp = _dir_to_bin(move), (1.0 if move.split else 0.0)
            obs_list.append(co)
            headings.append(b)
            splits.append(sp)
            kept += 1
            done = actor.step({0: CanonicalAction(heading_bin=b, split=bool(sp))})
    return (obs_list, np.asarray(headings, dtype=np.int64),
            np.asarray(splits, dtype=np.float32))


def _forward(policy, batch: List[CanonicalObs], device):
    bo = collate_obs(batch, device=device)
    hidden = torch.zeros(len(batch), policy.hidden_size, device=device)
    hl, sl, _v, _h = policy(bo, hidden)
    return hl, sl


def fit_bc(obs_list, headings, splits, *, epochs, lr, batch_size, val_frac,
           seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(obs_list)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    dev = torch.device(device)

    policy = build_policy(TrainConfig()).to(dev)
    opt = torch.optim.Adam(policy.actor_parameters(), lr=lr)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    yh = torch.as_tensor(headings, device=dev)
    ys = torch.as_tensor(splits, device=dev)

    def _acc(idx):
        policy.eval()
        with torch.no_grad():
            c = w1 = 0
            for s in range(0, len(idx), 512):
                ch = idx[s:s + 512]
                hl, _ = _forward(policy, [obs_list[i] for i in ch], dev)
                pred = hl.argmax(-1)
                yt = yh[ch]
                c += int((pred == yt).sum())
                d = (pred - yt).abs()
                w1 += int((torch.minimum(d, HEADING_BINS - d) <= 1).sum())
            return c / len(idx), w1 / len(idx)

    for ep in range(epochs):
        policy.train()
        rng.shuffle(train_idx)
        tot = 0.0
        for s in range(0, len(train_idx), batch_size):
            ch = train_idx[s:s + batch_size]
            hl, sl = _forward(policy, [obs_list[i] for i in ch], dev)
            loss = ce(hl, yh[ch]) + bce(sl, ys[ch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(ch)
        va = _acc(val_idx)
        print(f"[ep {ep+1}/{epochs}] loss={tot/len(train_idx):.3f}  "
              f"val heading top1={va[0]:.1%} within1={va[1]:.1%}", flush=True)
    return policy, opt


def save_checkpoint(path, policy, opt, *, run_name):
    """Write an evaluate-/resume-compatible checkpoint (Checkpointer schema)."""
    cfg = TrainConfig()
    blob = {
        "version": 0,
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "optim_state": opt.state_dict(),
        "arch": _arch_meta(cfg),
        "provenance": {**registry_row(cfg, run_name=run_name, stage="S0-bc"),
                       "version": 0, "source": "bc_warmstart:greedy_food"},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _atomic(path, lambda p: torch.save(blob, p))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="BC warm-start from greedy_food (Track D)")
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--base-seed", type=int, default=31337)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="runs/bc_greedy/bc_v1.pt")
    args = ap.parse_args()

    t0 = time.monotonic()
    print(f"collecting greedy_food demos from {args.games} games…", flush=True)
    obs_list, headings, splits = collect_demos(args.games, args.base_seed)
    print(f"collected {len(obs_list)} frames in {time.monotonic()-t0:.0f}s; "
          f"split rate {splits.mean():.1%}; "
          f"heading spread {int((np.bincount(headings, minlength=HEADING_BINS)>0).sum())}"
          f"/{HEADING_BINS} bins", flush=True)

    policy, opt = fit_bc(obs_list, headings, splits, epochs=args.epochs, lr=args.lr,
                         batch_size=args.batch_size, val_frac=args.val_frac,
                         seed=args.base_seed, device=args.device)
    path = save_checkpoint(args.out, policy, opt, run_name="bc_greedy")
    print(f"\nsaved BC checkpoint → {path}")
    print("eval it for G1:\n"
          f"  python -m train.harness.evaluate --weights {path} "
          f"--opponents dont_move --games 30 --vs-random")


if __name__ == "__main__":
    main()
