"""Representation-capacity probe (P3.5 diagnosis) — CAN the deep-set net read
nearest-food direction at all?

Every G1 run has now failed the same way: healthy training curves (entropy down,
value_loss alive, positive collection reward) but a greedy eval pinned at the
0.81 floor — even the oracle run, where the behaviour policy was *handed* perfect
food-steering demonstrations. The standing hypothesis (docs §4.3/§3b) is a
**representation bottleneck**: the permutation-invariant deep-set pooling dilutes
the nearest-food direction, so the policy can't map obs → "steer at food" and PPO
settles on a constant heading that harvests the exploration reward without
tracking food.

This probe tests that hypothesis directly and cheaply, with **no RL** to confound
it. It is pure supervised learning:

  1. Collect real frames by driving one seat straight at its nearest food (via the
     oracle bin) through actual games — the on-distribution "approaching food"
     frames the deployed bot must handle. Label each frame with that true
     nearest-food heading bin.
  2. Fit a fresh AgarPolicy's actor path (deep-set encoders → trunk → GRU-with-
     zero-hidden → heading head) to predict the bin, cross-entropy, held-out split.
  3. Report validation top-1 (and top-3, within-1-bin) accuracy.

**Verdict.** High val top-1 (say ≥60%, vs the 1/24≈4.2% chance floor and a
within-1-bin bar) ⇒ the architecture CAN represent food-seeking, so the RL
failure is optimization/exploration, not the network → tune PPO / anneal the
oracle / BC-warm-start-then-finetune. Low val top-1 (near chance, or only good on
within-1-bin) ⇒ the encoder genuinely can't read direction → the fix is
representation surgery (Track E: concat the raw nearest-K food rows into the
trunk), and no amount of RL tuning will clear G1.

The forward here is single-frame (zero hidden), so it lower-bounds capacity: the
recurrent policy can only do better with history. Training-only tool; imports
torch; never shipped.

    python -m tools.probe_food_capacity --games 60 --epochs 8
"""

from __future__ import annotations

import argparse
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
from train.harness.explore import OracleGuidance
from train.harness.flat_obs import flatten
from train.model import build_policy
from train.opponents.scripted import make_bot


def collect_frames(n_games: int, base_seed: int, max_per_game: int = 600
                   ) -> Tuple[List[CanonicalObs], np.ndarray]:
    """Sweep seat 0 with held random headings; return (obs, nearest-food-bin labels).

    Collection behaviour is deliberately **decorrelated from the label**: the seat
    holds a uniformly-random heading for ``U[1, 20]`` rounds (temporally-extended
    random — the same coverage mechanism as ActionPersistence), so it translates
    across the arena (the nearest-food direction varies ⇒ balanced labels) while
    its instantaneous heading is independent of where the food is. That closes the
    confound of oracle-driving (velocity ≈ label), forcing the net to read
    direction from the **food encoder**, not from self-velocity. The other seven
    seats are ``dont_move`` (the Stage-0 food-only field). Only food-present frames
    are kept (a no-food frame carries no direction label)."""
    obs_list: List[CanonicalObs] = []
    labels: List[int] = []
    for g in range(n_games):
        rng = np.random.default_rng(base_seed * 100003 + g)
        scripted = {pid: make_bot("dont_move") for pid in range(1, 8)}
        actor = Actor(seed=base_seed + g, policy_seats={0}, scripted=scripted,
                      emit_update=False)
        actor.reset()
        held, counter = int(rng.integers(HEADING_BINS)), 0
        kept = 0
        done = False
        while not done and kept < max_per_game:
            if counter <= 0:                          # start a new held-heading burst
                held = int(rng.integers(HEADING_BINS))
                counter = int(rng.integers(1, 21))
            counter -= 1
            frame = actor.observe()
            if 0 not in frame:                        # seat 0 dead (respawn gap)
                counter = 0                           # drop the hold across respawn
                done = actor.step({})
                continue
            co = frame[0]
            flat = flatten(co)[None, :]               # (1, OBS_WIDTH)
            bins, has_food = OracleGuidance.oracle_bins(flat)
            if bool(has_food[0]):
                obs_list.append(co)
                labels.append(int(bins[0]))
                kept += 1
            done = actor.step({0: _action(held)})     # random sweep, not toward food
    return obs_list, np.asarray(labels, dtype=np.int64)


def _action(bin_idx: int):
    from bot.adapters.act_v1 import CanonicalAction
    return CanonicalAction(heading_bin=bin_idx, split=False)


def _heading_logits(policy, batch: List[CanonicalObs], device) -> torch.Tensor:
    """Actor-path heading logits for a batch of frames, single-frame (zero hidden)."""
    bo = collate_obs(batch, device=device)
    hidden = torch.zeros(len(batch), policy.hidden_size, device=device)
    hl, _sl, _v, _h = policy(bo, hidden)
    return hl


def fit_probe(obs_list: List[CanonicalObs], labels: np.ndarray, *, epochs: int,
              lr: float, batch_size: int, val_frac: float, seed: int, device: str):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(obs_list)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    dev = torch.device(device)

    policy = build_policy(TrainConfig()).to(dev)
    # Optimize the whole actor path (encoders + trunk + gru + heading head).
    opt = torch.optim.Adam(policy.actor_parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    y = torch.as_tensor(labels, device=dev)

    def _eval(idx: np.ndarray):
        policy.eval()
        with torch.no_grad():
            correct = top3 = within1 = 0
            for s in range(0, len(idx), 512):
                chunk = idx[s:s + 512]
                hl = _heading_logits(policy, [obs_list[i] for i in chunk], dev)
                yt = y[chunk]
                pred = hl.argmax(-1)
                correct += int((pred == yt).sum())
                top3 += int((hl.topk(3, dim=-1).indices == yt[:, None]).any(-1).sum())
                d = (pred - yt).abs()
                circ = torch.minimum(d, HEADING_BINS - d)   # circular bin distance
                within1 += int((circ <= 1).sum())
            m = len(idx)
            return correct / m, top3 / m, within1 / m

    for ep in range(epochs):
        policy.train()
        rng.shuffle(train_idx)
        tot = 0.0
        for s in range(0, len(train_idx), batch_size):
            chunk = train_idx[s:s + batch_size]
            hl = _heading_logits(policy, [obs_list[i] for i in chunk], dev)
            loss = loss_fn(hl, y[chunk])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(chunk)
        tr = _eval(train_idx[:2000])
        va = _eval(val_idx)
        print(f"[ep {ep+1}/{epochs}] loss={tot/len(train_idx):.3f}  "
              f"train top1={tr[0]:.1%}  val top1={va[0]:.1%} top3={va[1]:.1%} "
              f"within1={va[2]:.1%}", flush=True)
    return va


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep-set food-direction capacity probe")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--base-seed", type=int, default=4242)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    t0 = time.monotonic()
    print(f"collecting frames from {args.games} games…", flush=True)
    obs_list, labels = collect_frames(args.games, args.base_seed)
    hist = np.bincount(labels, minlength=HEADING_BINS)
    print(f"collected {len(obs_list)} food-present frames in "
          f"{time.monotonic()-t0:.0f}s; label spread over "
          f"{int((hist>0).sum())}/{HEADING_BINS} bins "
          f"(min {hist.min()}, max {hist.max()})", flush=True)
    if len(obs_list) < 200:
        print("WARNING: very few frames; raise --games for a trustworthy read.", flush=True)

    va = fit_probe(obs_list, labels, epochs=args.epochs, lr=args.lr,
                   batch_size=args.batch_size, val_frac=args.val_frac,
                   seed=args.base_seed, device=args.device)
    top1, top3, within1 = va
    print("\n=== CAPACITY VERDICT ===")
    print(f"chance top1 = {1/HEADING_BINS:.1%};  measured val top1 = {top1:.1%}, "
          f"top3 = {top3:.1%}, within1 = {within1:.1%}")
    if top1 >= 0.60:
        print("CAN represent food direction → RL/exploration is the bottleneck, "
              "not the architecture (tune PPO / anneal oracle / BC warm-start).")
    elif within1 >= 0.50:
        print("PARTIAL: reads direction coarsely but not sharply → borderline; "
              "representation surgery (Track E) likely still needed for a crisp argmax.")
    else:
        print("CANNOT represent food direction (near chance) → the deep-set "
              "encoder is the bottleneck → Track E (raw nearest-food rows into trunk).")


if __name__ == "__main__":
    main()
