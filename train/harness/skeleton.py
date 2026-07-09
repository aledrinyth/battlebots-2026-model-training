"""End-to-end skeleton driver (P3.1 acceptance: "moves observations end-to-end").

This is the minimal realization of the chosen custom actor–learner substrate:

* N :class:`~train.harness.actor.Actor` instances = N concurrent games, each a mix
  of policy-driven and scripted seats.
* Every round, the driver gathers **every alive policy seat across all games** into
  one flat batch, collates it into torch tensors, and runs **one** central
  :class:`~train.harness.policy_stub.SkeletonPolicy` forward pass (the "single GPU
  inference process" of the design — here just one process/device).
* Per-seat recurrent hidden state is kept centrally in a
  :class:`HiddenStateStore` keyed by ``(game, seat)``; the featurizer's
  ``is_reset`` flag drives per-row zeroing at the policy (first frame / respawn).
* Chosen actions flow back through the shared :class:`ActionAdapter` into each
  actor's wrapper.

It does **no learning** — no PPO, no gradients (that is P3.2/P3.3). It exists to
prove the plumbing carries observations from wrapper → featurizer → batched
policy → action adapter → wrapper without impedance mismatch, and to report the
batch sizes / reset events / dead-player gaps it handled so the P3.1 decision is
backed by a run, not an argument.

Run it directly::

    python -m train.harness.skeleton --games 8 --rounds 200
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from bot.adapters.act_v1 import CanonicalAction, decode_logits
from train.env.wrapper import WrapperEnv  # noqa: F401  (re-exported convenience)
from train.harness.actor import Actor
from train.harness.collate import collate_obs
from train.harness.policy_stub import SkeletonPolicy
from train.opponents.scripted import make_bot

Key = Tuple[int, int]  # (game_index, seat_id)


class HiddenStateStore:
    """Central per-seat recurrent state, keyed by ``(game, seat)``.

    Unknown keys read back as a zero vector, so a brand-new or just-respawned seat
    starts clean without the caller special-casing it. (The policy *also* zeros
    the input on ``is_reset`` rows — belt and suspenders; the store keeps whatever
    the GRU last produced so a live seat carries its memory across rounds.)
    """

    def __init__(self, hidden_size: int, device: torch.device | str = "cpu") -> None:
        self.hidden_size = hidden_size
        self.device = device
        self._state: Dict[Key, torch.Tensor] = {}

    def gather(self, keys: Sequence[Key]) -> torch.Tensor:
        """Stack the hidden rows for ``keys`` into ``(len(keys), hidden_size)``."""
        rows = [
            self._state.get(k, torch.zeros(self.hidden_size, device=self.device))
            for k in keys
        ]
        return torch.stack(rows, dim=0)

    def scatter(self, keys: Sequence[Key], hidden: torch.Tensor) -> None:
        """Write back the post-step hidden rows for ``keys``."""
        detached = hidden.detach()
        for i, k in enumerate(keys):
            self._state[k] = detached[i]


@dataclass
class SkeletonStats:
    """What a skeleton run observed — the evidence the plumbing worked."""

    games: int
    rounds_requested: int
    rounds_advanced: int = 0
    forward_calls: int = 0
    frames: int = 0                 # total policy-seat observations processed
    max_batch: int = 0
    batch_sizes: List[int] = field(default_factory=list)
    resets: int = 0                 # is_reset frames (first-frame + respawns)
    respawns: int = 0               # is_reset on a seat seen before (post-gap)
    games_finished: int = 0
    invalid_batches: int = 0        # alive-set != batch (must stay 0)

    @property
    def mean_batch(self) -> float:
        return sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0

    def summary(self) -> str:
        return (
            f"games={self.games} rounds_advanced={self.rounds_advanced} "
            f"forward_calls={self.forward_calls} frames={self.frames} "
            f"mean_batch={self.mean_batch:.1f} max_batch={self.max_batch} "
            f"resets={self.resets} respawns={self.respawns} "
            f"games_finished={self.games_finished} invalid_batches={self.invalid_batches}"
        )


def _default_scripted(policy_seats: set[int], roster: int) -> Dict[int, object]:
    """Fill every non-policy seat with a rotating mix of scripted anchors."""
    anchors = ["greedy_food", "cautious", "chaser", "coward", "dont_move"]
    scripted: Dict[int, object] = {}
    j = 0
    for pid in range(roster):
        if pid in policy_seats:
            continue
        scripted[pid] = make_bot(anchors[j % len(anchors)])
        j += 1
    return scripted


def build_actors(
    games: int,
    policy_seats: Sequence[int] = (0, 1),
    base_seed: int = 1000,
    roster: int = 8,
) -> List[Actor]:
    """Construct ``games`` actors, each with the given policy seats + scripted rest."""
    pset = set(policy_seats)
    actors: List[Actor] = []
    for g in range(games):
        actors.append(
            Actor(
                seed=base_seed + g,
                policy_seats=pset,
                scripted=_default_scripted(pset, roster),
                emit_update=False,
            )
        )
    return actors


def run_skeleton(
    games: int = 8,
    rounds: int = 200,
    policy_seats: Sequence[int] = (0, 1),
    base_seed: int = 1000,
    device: str = "cpu",
    policy: Optional[SkeletonPolicy] = None,
    split_threshold: float = 0.0,
) -> SkeletonStats:
    """Run the end-to-end skeleton for ``rounds`` rounds over ``games`` games.

    Returns the :class:`SkeletonStats` gathered — non-zero ``frames`` and
    ``forward_calls`` with ``invalid_batches == 0`` is the P3.1 acceptance signal.
    """
    dev = torch.device(device)
    if policy is None:
        policy = SkeletonPolicy()
    policy = policy.to(dev)
    policy.eval()

    actors = build_actors(games, policy_seats, base_seed)
    for a in actors:
        a.reset()

    hidden = HiddenStateStore(policy.hidden_size, dev)
    seen_keys: set[Key] = set()
    stats = SkeletonStats(games=games, rounds_requested=rounds)

    with torch.no_grad():
        for _ in range(rounds):
            # ---- gather every alive policy seat across all live games ----
            keys: List[Key] = []
            obs_list = []
            for gi, actor in enumerate(actors):
                if actor.done:
                    continue
                seat_obs = actor.observe()
                # invariant: the batch's seats for this game == its alive policy seats
                if set(seat_obs) != actor.alive_policy_seats():
                    stats.invalid_batches += 1
                for seat, obs in seat_obs.items():
                    keys.append((gi, seat))
                    obs_list.append(obs)

            if not obs_list:
                break  # every game finished

            # ---- one central batched forward pass ----
            batched = collate_obs(obs_list, device=dev)
            h_in = hidden.gather(keys)
            heading_logits, split_logits, _value, h_out = policy(batched, h_in)
            hidden.scatter(keys, h_out)

            stats.forward_calls += 1
            b = len(keys)
            stats.frames += b
            stats.batch_sizes.append(b)
            stats.max_batch = max(stats.max_batch, b)

            # reset / respawn bookkeeping (drives GRU reset; audited here)
            reset_flags = batched.is_reset.tolist()
            for k, is_reset in zip(keys, reset_flags):
                if is_reset:
                    stats.resets += 1
                    if k in seen_keys:
                        stats.respawns += 1
                seen_keys.add(k)

            # ---- decode via the shared adapter, scatter back to actors ----
            heads_cpu = heading_logits.cpu().tolist()
            splits_cpu = split_logits.cpu().tolist()
            per_actor: Dict[int, Dict[int, CanonicalAction]] = {}
            for (gi, seat), h_logits, s_logit in zip(keys, heads_cpu, splits_cpu):
                action = decode_logits(h_logits, s_logit, split_threshold)
                per_actor.setdefault(gi, {})[seat] = action

            advanced = False
            for gi, actor in enumerate(actors):
                if actor.done:
                    continue
                done = actor.step(per_actor.get(gi, {}))
                advanced = True
                if done:
                    stats.games_finished += 1
            if advanced:
                stats.rounds_advanced += 1

    return stats


# --- multiprocessing seam smoke (de-risks P3.2's shared-memory transport) ----

def actor_worker(
    seed: int,
    policy_seats: Sequence[int],
    scripted_names: Mapping[int, str],
    n_frames: int,
    queue,
) -> None:
    """Run one actor in a worker process, streaming featurized frames over ``queue``.

    This is the P3.2 transport in miniature: the child is **numpy/CPU-only** (it
    never imports torch), featurizes with the shared module, and puts
    :class:`CanonicalObs` payloads on a queue for the central learner. Policy
    seats are driven by a fixed dummy action here (the child has no policy — that
    lives centrally); the point is only that observations cross the process
    boundary intact. A ``None`` sentinel marks end-of-stream.
    """
    import sys

    from train.harness.actor import Actor as _Actor
    from train.opponents.scripted import make_bot as _make_bot

    scripted = {int(pid): _make_bot(name) for pid, name in scripted_names.items()}
    actor = _Actor(seed=seed, policy_seats=policy_seats, scripted=scripted, emit_update=False)
    actor.reset()

    dummy = CanonicalAction(heading_bin=0, split=False)
    sent = 0
    for _ in range(n_frames):
        if actor.done:
            break
        seat_obs = actor.observe()
        queue.put(
            {
                "round": actor.env.state.round,
                "seats": {seat: obs for seat, obs in seat_obs.items()},
                # proof the actor path pulled in no torch (design invariant)
                "torch_imported": "torch" in sys.modules,
            }
        )
        sent += 1
        actor.step({seat: dummy for seat in seat_obs})
    queue.put(None)  # sentinel


def _main() -> None:
    parser = argparse.ArgumentParser(description="P3.1 end-to-end harness skeleton")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--policy-seats", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    stats = run_skeleton(
        games=args.games,
        rounds=args.rounds,
        policy_seats=tuple(args.policy_seats),
        base_seed=args.seed,
        device=args.device,
    )
    print(stats.summary())


if __name__ == "__main__":
    _main()


__all__ = [
    "HiddenStateStore",
    "SkeletonStats",
    "build_actors",
    "run_skeleton",
    "actor_worker",
]
