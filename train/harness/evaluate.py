"""Evaluation (lightweight stand-in for P7.1) — is the trained policy any good?

Training reward and loss **cannot** answer this: the shaped Δmass reward is
gameable, and in the decoupled two-box setup it's staleness-corrupted on top. The
only honest signal is head-to-head play against **held-out** opponents.

This loads a weights file, plays ``n_games`` seat- and seed-randomized 8-player
games with the trained policy in one seat and scripted anchors in the other
seven, and reports **mean rank / win-rate / mean final mass** — using **greedy
(argmax) actions**, exactly how the bot would act at match time. It uses only the
scripted baselines that exist today; the full P7.1 tournament (Elo ladder,
milestone snapshots held out from training) is a later phase.

Competence bars (from the playbook):

* **G1 (minimum "it learned"):** mean final mass ≥ 3× a random policy in the same
  field. Run with ``--vs-random`` to get the ratio directly.
* **S1 (genuinely competent):** beats ``greedy_food`` — i.e. mean rank clearly
  below 4.5 (neutral for 8 players) in a greedy-heavy field, and above greedy's
  own rank head-to-head.
* **Strong:** also tops ``cautious`` and ``chaser``; then Elo vs its own past
  snapshots plateaus (diminishing returns → time to freeze).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from bot.adapters.act_v1 import decode_logits
from train.harness.actor import Actor
from train.harness.collate import collate_obs
from train.harness.config import TrainConfig
from train.model import build_policy
from train.opponents.scripted import make_bot


def _opponent_assignment(policy_seat: int, opponents: Sequence[str]) -> Dict[int, str]:
    """Fill the seven non-policy seats by cycling through ``opponents``."""
    names: Dict[int, str] = {}
    j = 0
    for pid in range(8):
        if pid == policy_seat:
            continue
        names[pid] = opponents[j % len(opponents)]
        j += 1
    return names


def _load_policy(weights_path: Optional[str], device: torch.device,
                 seed: int = 0) -> torch.nn.Module:
    if weights_path is None:  # random-init baseline (the real P3.3 network)
        # Seeded so the baseline is one fixed random policy regardless of how
        # many worker processes build it (and reproducible across invocations).
        torch.manual_seed(seed)
        return build_policy(TrainConfig()).to(device).eval()
    blob = torch.load(weights_path, map_location=device, weights_only=False)
    arch = blob.get("arch", {})
    # Reconstruct the exact architecture the checkpoint was trained with. Older
    # (pre-P3.3) checkpoints have no ``use_skeleton``/``model_*`` keys and were
    # SkeletonPolicy(32, 64) — fall back to that so they still load.
    is_skeleton = bool(arch.get("use_skeleton", "model_version" not in arch))
    cfg = TrainConfig(
        use_skeleton=is_skeleton,
        entity_hidden=int(arch.get("entity_hidden", 32)),
        trunk_hidden=int(arch.get("trunk_hidden", 64)),
        model_entity_hidden=int(arch.get("model_entity_hidden", 64)),
        model_trunk_hidden=int(arch.get("model_trunk_hidden", 256)),
    )
    policy = build_policy(cfg).to(device)
    policy.load_state_dict(blob["state_dict"])
    return policy.eval()


@dataclass
class EvalResult:
    n_games: int
    opponents: List[str]
    mean_rank: float          # 1 (best) … 8 (worst); 4.5 = neutral
    win_rate: float           # fraction finishing 1st
    top_half_rate: float      # fraction finishing in the top 4
    mean_final_mass: float
    rank_hist: List[int]      # counts for ranks 1..8

    def summary(self) -> str:
        hist = " ".join(f"{r+1}:{c}" for r, c in enumerate(self.rank_hist))
        return (
            f"games={self.n_games} vs [{', '.join(self.opponents)}]\n"
            f"  mean_rank={self.mean_rank:.2f} (neutral 4.5)  win_rate={self.win_rate:.1%}  "
            f"top_half={self.top_half_rate:.1%}  mean_final_mass={self.mean_final_mass:.2f}\n"
            f"  rank histogram: {hist}"
        )


def _play_games(policy: torch.nn.Module, opponents: Sequence[str],
                indices: Iterable[int], device: str,
                base_seed: int) -> List[Tuple[int, int, float]]:
    """Play the given game indices; return ``[(g, rank, final_mass), …]``.

    Game ``g`` is fully determined by ``(base_seed + g, g % 8)``, so any
    partition of indices across processes reproduces the sequential results.
    """
    dev = torch.device(device)
    out: List[Tuple[int, int, float]] = []
    with torch.no_grad():
        for g in indices:
            policy_seat = g % 8  # rotate seat for fairness
            scripted = {pid: make_bot(name) for pid, name in _opponent_assignment(policy_seat, opponents).items()}
            actor = Actor(seed=base_seed + g, policy_seats={policy_seat}, scripted=scripted, emit_update=False)
            actor.reset()
            hidden = torch.zeros(1, policy.hidden_size, device=dev)

            done = False
            while not done:
                obs = actor.observe()  # {policy_seat: CanonicalObs} if alive, else {}
                if policy_seat in obs:
                    co = obs[policy_seat]
                    if co.is_reset:
                        hidden = torch.zeros_like(hidden)
                    batched = collate_obs([co], device=dev)
                    hl, sl, _v, hidden = policy(batched, hidden)
                    action = decode_logits(hl[0].cpu().tolist(), float(sl[0].cpu()), 0.0)  # greedy
                    done = actor.step({policy_seat: action})
                else:
                    done = actor.step({})  # policy seat dead this round; scripted seats act

            rank = actor.rankings().index(policy_seat) + 1
            out.append((g, rank, float(actor.final_masses()[policy_seat])))
    return out


def _mp_play(args) -> List[Tuple[int, int, float]]:
    """Worker entry (spawn): load the policy once, play a strided slice of games."""
    weights_path, opponents, indices, device, base_seed = args
    torch.set_num_threads(1)  # one game at a time per worker; don't oversubscribe
    policy = _load_policy(weights_path, torch.device(device), seed=base_seed)
    return _play_games(policy, opponents, indices, device, base_seed)


def _strided_chunks(n: int, workers: int) -> List[List[int]]:
    """Partition ``range(n)`` into ≤``workers`` strided, non-empty chunks."""
    chunks = [list(range(w, n, workers)) for w in range(max(1, workers))]
    return [c for c in chunks if c]


def evaluate(
    weights_path: Optional[str],
    opponents: Sequence[str] = ("greedy_food", "cautious", "chaser", "coward", "dont_move"),
    n_games: int = 100,
    device: str = "cpu",
    base_seed: int = 999_983,
    workers: int = 1,
) -> EvalResult:
    """Play ``n_games`` and score the (single) policy seat's finishing position.

    ``workers > 1`` fans the games out over that many processes (games are
    independent and seeded per-index, so the results are identical to the
    sequential run — only wall-clock changes).
    """
    if workers > 1 and n_games > 1:
        ctx = mp.get_context("spawn")
        jobs = [(weights_path, tuple(opponents), chunk, device, base_seed)
                for chunk in _strided_chunks(n_games, workers)]
        with ctx.Pool(len(jobs)) as pool:
            played = [row for rows in pool.map(_mp_play, jobs) for row in rows]
    else:
        policy = _load_policy(weights_path, torch.device(device), seed=base_seed)
        played = _play_games(policy, opponents, range(n_games), device, base_seed)

    played.sort()  # by game index — order-independent of worker count
    ranks = [r for _g, r, _m in played]
    masses = [m for _g, _r, m in played]

    ra = np.array(ranks)
    return EvalResult(
        n_games=n_games,
        opponents=list(opponents),
        mean_rank=float(ra.mean()),
        win_rate=float((ra == 1).mean()),
        top_half_rate=float((ra <= 4).mean()),
        mean_final_mass=float(np.mean(masses)),
        rank_hist=[int((ra == r).sum()) for r in range(1, 9)],
    )


def _main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained policy vs scripted baselines")
    p.add_argument("--weights", default=None, help="weights_v*.pt file (omit for a random baseline)")
    p.add_argument("--opponents", nargs="+",
                   default=["greedy_food", "cautious", "chaser", "coward", "dont_move"])
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--workers", type=int, default=1,
                   help="parallelize games across N processes (identical results, "
                        "~N× faster; keep a few cores free if training is running)")
    p.add_argument("--vs-random", action="store_true",
                   help="also run a random policy in the same games and print the mass ratio (the G1 bar)")
    args = p.parse_args()

    res = evaluate(args.weights, tuple(args.opponents), args.games, args.device,
                   workers=args.workers)
    print("TRAINED:" if args.weights else "RANDOM (no --weights given):")
    print(res.summary())

    if args.vs_random and args.weights:
        base = evaluate(None, tuple(args.opponents), args.games, args.device,
                        workers=args.workers)
        ratio = res.mean_final_mass / max(base.mean_final_mass, 1e-9)
        print("\nRANDOM baseline:")
        print(base.summary())
        print(f"\nmass ratio trained/random = {ratio:.2f}×   (G1 wants ≥ 3×)")


if __name__ == "__main__":
    _main()


__all__ = ["evaluate", "EvalResult"]
