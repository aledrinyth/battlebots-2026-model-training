"""Temporally-extended exploration (P3.5) — action persistence for collection.

Per-step independent action sampling *dithers*: a food-only agent jitters in
place, covers almost no ground, and so almost never stumbles onto the food it
must eat to get any reward signal — the exploration wall that stalls Stage-0 at
small batch. The fix from the literature (Dabney, Ostrovski & Barreto 2020,
"Temporally-Extended ε-Greedy Exploration", arXiv:2006.01782; reported essential
for Agar.io pellet-eating by the AgarCL platform) is dead simple: **occasionally
commit to the sampled action for a random number of steps.** Those short bursts
of persistence sweep the arena, surface the eats, and bootstrap food-seeking.

:class:`ActionPersistence` wraps a step's freshly-sampled action. With
probability ``persist_prob`` a *free* stream starts a burst — it holds its
current action for ``U[1, max_duration]`` more rounds; a holding stream repeats
its held action and decrements. Because a held action is not a fresh sample of
this step's distribution, its collection-time log-prob is **recomputed** under
the current policy (:func:`train.ppo.log_prob_of`) — that is the ``old`` logp PPO
stores, and PPO's clip ratio absorbs the mild off-policyness (the same tolerance
that handles two-box staleness).

**Training-only.** This perturbs only the *behaviour* policy while collecting
rollouts; the learned policy is unchanged, and the deployed bot samples/argmaxes
fresh every round — so match parity is untouched. Disabled (``persist_prob=0``)
it is never constructed, and the collection path is byte-identical to before.
Deterministic: its RNG is seeded from the run's ``base_seed``.
"""

from __future__ import annotations

import numpy as np
import torch

from train.ppo import log_prob_of

_SEED_MIX = 0x9E3779B9  # decorrelate this RNG from the env/policy seeds


class ActionPersistence:
    """Per-stream action-hold state for temporally-extended exploration."""

    def __init__(self, n_rows: int, persist_prob: float, max_duration: int,
                 base_seed: int) -> None:
        if not (0.0 < persist_prob <= 1.0):
            raise ValueError(f"persist_prob must be in (0,1], got {persist_prob}")
        if max_duration < 1:
            raise ValueError(f"max_duration must be >= 1, got {max_duration}")
        self.n = int(n_rows)
        self.p = float(persist_prob)
        self.maxd = int(max_duration)
        self.rng = np.random.default_rng((int(base_seed) ^ _SEED_MIX) & 0xFFFFFFFF)
        self.counter = np.zeros(self.n, dtype=np.int64)   # remaining hold rounds
        self.hold_h = np.zeros(self.n, dtype=np.int64)    # held heading bin
        self.hold_s = np.zeros(self.n, dtype=np.float32)  # held split flag

    def apply(self, heading_logits, split_logit, heading, split, logp, is_reset):
        """Apply persistence to a step's fresh action; return ``(heading, split, logp)``.

        ``heading``/``split``/``logp`` are the fresh-sample tensors; ``is_reset``
        is a length-``n`` bool array/tensor (a stream's first frame after a
        respawn — its held action must not carry across the gap). Returns tensors
        on the same devices as the inputs, with the log-prob recomputed for any
        stream whose action was overridden by a hold.
        """
        dev_h, dev_s = heading.device, split.device
        h = heading.detach().cpu().numpy().astype(np.int64).copy()
        s = split.detach().cpu().numpy().astype(np.float32).copy()
        reset = (is_reset.detach().cpu().numpy() if torch.is_tensor(is_reset)
                 else np.asarray(is_reset)).astype(bool)

        self.counter[reset] = 0                    # drop holds across a respawn
        held = self.counter > 0
        # holding streams: repeat the held action, tick the clock down
        h[held] = self.hold_h[held]
        s[held] = self.hold_s[held]
        self.counter[held] -= 1
        # free streams: this fresh action may start a new burst
        free = ~held
        start = free & (self.rng.random(self.n) < self.p)
        n_start = int(start.sum())
        if n_start:
            self.hold_h[start] = h[start]
            self.hold_s[start] = s[start]
            self.counter[start] = self.rng.integers(1, self.maxd + 1, size=n_start)

        if not held.any():
            return heading, split, logp   # nothing overridden ⇒ fresh logp is exact

        new_h = torch.as_tensor(h, device=dev_h)
        new_s = torch.as_tensor(s, dtype=split.dtype, device=dev_s)
        new_logp = log_prob_of(heading_logits, split_logit, new_h, new_s.float())
        return new_h, new_s, new_logp



# --- oracle-guided bursts (escalation Track C) -------------------------------

# Flat-obs column offsets of the nearest-food features, derived from the frozen
# obs_v1 dims (wire order: self_vec, food, food_mask, … — see
# train.harness.flat_obs._SPEC; cross-checked against flat_obs in the tests).
# food rows are distance-sorted, so row 0 is the nearest food; fields (dx, dy,
# dist) are world-frame deltas from the view center under a positive scale, so
# atan2(dy, dx) is the world heading toward the food — the same frame act_v1's
# bin geometry uses (bin i at angle 2π·i/HEADING_BINS, CCW from +x).
from bot.adapters.act_v1 import HEADING_BINS
from bot.adapters.obs_v1 import FOOD_DIM, FOOD_MAX, SELF_DIM

_FOOD0_DX = SELF_DIM
_FOOD0_DY = SELF_DIM + 1
_FOOD_MASK0 = SELF_DIM + FOOD_MAX * FOOD_DIM

_ORACLE_SEED_MIX = 0x85EBCA6B  # decorrelate from env/policy/persistence RNGs


class OracleGuidance:
    """Nearest-food oracle bursts — targeted exploration (P3.5 Track C).

    Random persistence buys *coverage*; at Stage-0 scale that surfaced eats but
    the policy gradient still had to discover "steer at food" from undirected
    bursts. This guidance makes the discovery free: with probability
    ``oracle_prob`` a free stream starts a burst of ``U[1, max_duration]``
    rounds during which its action is overridden with the heading bin pointing
    at its **current** nearest food (re-aimed every round; split forced 0). The
    oracle needs no actor-side changes — the flat obs already carries the
    nearest food's (dx, dy) — and rows with no visible food keep their sampled
    action (the burst clock still ticks).

    Like :class:`ActionPersistence`, this perturbs only the *behaviour* policy
    during collection: overridden actions are stored with their log-prob
    recomputed under the current policy (:func:`train.ppo.log_prob_of`), so
    PPO's importance ratio is exact and the clip absorbs the off-policyness.
    The deployed bot and the (always oracle-free, greedy) eval are untouched —
    a G1 pass earned with this bootstrap is still an honest pass. Disabled
    (``oracle_prob=0``) it is never constructed. When combined with
    ``ActionPersistence``, apply this **last** — the oracle wins on its rows.
    """

    def __init__(self, n_rows: int, oracle_prob: float, max_duration: int,
                 base_seed: int) -> None:
        if not (0.0 < oracle_prob <= 1.0):
            raise ValueError(f"oracle_prob must be in (0,1], got {oracle_prob}")
        if max_duration < 1:
            raise ValueError(f"max_duration must be >= 1, got {max_duration}")
        self.n = int(n_rows)
        self.p = float(oracle_prob)
        self.maxd = int(max_duration)
        self.rng = np.random.default_rng((int(base_seed) ^ _ORACLE_SEED_MIX) & 0xFFFFFFFF)
        self.counter = np.zeros(self.n, dtype=np.int64)  # remaining guided rounds

    @staticmethod
    def oracle_bins(obs: np.ndarray):
        """Vectorized ``(bins, has_food)``: nearest-food heading bin per row."""
        dx = obs[:, _FOOD0_DX].astype(np.float64)
        dy = obs[:, _FOOD0_DY].astype(np.float64)
        has_food = obs[:, _FOOD_MASK0] > 0.5
        ang = np.arctan2(dy, dx)  # (-π, π], world frame
        bins = np.rint(ang * (HEADING_BINS / (2.0 * np.pi))).astype(np.int64) % HEADING_BINS
        return bins, has_food

    def apply(self, heading_logits, split_logit, heading, split, logp, is_reset, obs):
        """Override guided rows' actions with the food oracle; recompute logp.

        Unlike a persistence hold (which repeats from the round *after* the
        burst starts), an oracle burst acts on the oracle immediately on its
        start round — the drawn duration counts this round.
        """
        reset = (is_reset.detach().cpu().numpy() if torch.is_tensor(is_reset)
                 else np.asarray(is_reset)).astype(bool)
        self.counter[reset] = 0                    # a respawn ends the burst
        free = self.counter <= 0
        start = free & (self.rng.random(self.n) < self.p)
        n_start = int(start.sum())
        if n_start:
            self.counter[start] = self.rng.integers(1, self.maxd + 1, size=n_start)
        guided = self.counter > 0
        self.counter[guided] -= 1
        if not guided.any():
            return heading, split, logp

        obs_np = obs if isinstance(obs, np.ndarray) else np.asarray(obs)
        bins, has_food = self.oracle_bins(obs_np)
        override = guided & has_food               # no food in vision ⇒ keep sample
        if not override.any():
            return heading, split, logp

        h = heading.detach().cpu().numpy().astype(np.int64).copy()
        s = split.detach().cpu().numpy().astype(np.float32).copy()
        h[override] = bins[override]
        s[override] = 0.0                          # the oracle never splits
        new_h = torch.as_tensor(h, device=heading.device)
        new_s = torch.as_tensor(s, dtype=split.dtype, device=split.device)
        new_logp = log_prob_of(heading_logits, split_logit, new_h, new_s.float())
        return new_h, new_s, new_logp


__all__ = ["ActionPersistence", "OracleGuidance"]
