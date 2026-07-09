"""P3.3 — the real recurrent policy network + asymmetric critic.

This is the model from ``plan_training_v1_policy_only.md`` §3.2 / §4.3 that
replaces the throwaway :class:`~train.harness.policy_stub.SkeletonPolicy` wiring
stub. It keeps the **exact** call contract the P3.2 transport already depends on::

    policy(obs: BatchedObs, hidden) -> (heading_logits, split_logit, value, new_hidden)

so nothing in ``vec_actor`` / ``learner`` / ``rollout`` / ``shard`` / the GAE
changes when the network is swapped in (see ``notes/P3.2-vectorized-actors.md``
§1 "What P3.2 delivers vs. what P3.3 adds").

Architecture (actor / deployment path — the only thing that ships):

* per-category **deep-set encoders**: a shared 2-layer MLP per entity
  (food / enemy / own / virus) → masked **mean+max** pool → ``(B, 2·h)`` each;
* concat with the self/global vector → **trunk MLP (256)** → **GRU(256)**;
* heads: **policy** = 24-bin heading logits (``HEADING_BINS``) + a Bernoulli
  split logit; **value** = scalar.

Float32 throughout, target **300k–800k actor-path parameters** (asserted by
:func:`assert_actor_param_budget`). The default sizes (``entity_hidden=64``,
``trunk_hidden=256``) land at ~555k.

**Asymmetric critic (train-only, §4.3).** The value head additionally consumes a
privileged *full-map* encoding (all players' blobs, all food, all viruses) that
the actor never sees. This is legal because the critic is **discarded at
deployment** — only the actor path is exported to ``weights.npz`` (P5.1). The
``forward`` accepts an optional :class:`PrivilegedObs`; when it is ``None`` (the
current transport, and always at inference) the critic falls back to a symmetric
value from the shared recurrent features, so the value target is still learned.
The privileged full-map slab is the "extra buffer field" hook P3.2 §1 anticipates
— wiring it through shared memory is downstream plumbing, not a network change.

The recurrent-reset contract is honored exactly as in the stub: rows flagged
``obs.is_reset`` (first frame of a game or first frame after a respawn gap) have
their incoming hidden state **zeroed before** the GRU step. Unit-tested in
``tests/test_model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn

from bot.adapters.act_v1 import HEADING_BINS
from bot.adapters.obs_v1 import (
    ENEMY_DIM,
    FOOD_DIM,
    OWN_DIM,
    SELF_DIM,
    VIRUS_DIM,
)
from train.harness.collate import BatchedObs

# --- version + actor-parameter budget (playbook P3.3) -----------------------

#: Bumped whenever the network *architecture* changes (registry pairs it with a
#: checkpoint, alongside the obs/act adapter versions).
MODEL_VERSION = "agar_v1"

#: The playbook's actor-path budget: "parameter count printed and asserted
#: 300k–800k (actor path)". Only the deployment path counts — the critic's
#: privileged encoder + value head are train-only and excluded.
ACTOR_PARAM_MIN = 300_000
ACTOR_PARAM_MAX = 800_000


# --- privileged full-map state (asymmetric critic input, train-only) --------

#: Up to 8 players × MAX_BLOB_COUNT(16) blobs on the map.
PRIV_BLOB_MAX = 128
#: (x, y, radius, is_self, is_enemy) — all normalized/relative, train-only.
PRIV_BLOB_DIM = 5
#: All food pellets on the map (``spec.FOOD_COUNT``).
PRIV_FOOD_MAX = 160
#: (x, y).
PRIV_FOOD_DIM = 2
#: All viruses (``spec.VIRUS_COUNT`` = 6; slot for 8).
PRIV_VIRUS_MAX = 8
#: (x, y, radius).
PRIV_VIRUS_DIM = 3


@dataclass(frozen=True)
class PrivilegedObs:
    """Batched full-map state for the asymmetric critic (leading dim ``B``).

    Mirrors the :class:`~train.harness.collate.BatchedObs` masking convention
    (``*_mask`` is ``1.0`` for a real entity, ``0.0`` for padding). Never touches
    the actor path and is never exported — training only.
    """

    blobs: torch.Tensor        # (B, PRIV_BLOB_MAX, PRIV_BLOB_DIM)
    blobs_mask: torch.Tensor   # (B, PRIV_BLOB_MAX)
    food: torch.Tensor         # (B, PRIV_FOOD_MAX, PRIV_FOOD_DIM)
    food_mask: torch.Tensor    # (B, PRIV_FOOD_MAX)
    virus: torch.Tensor        # (B, PRIV_VIRUS_MAX, PRIV_VIRUS_DIM)
    virus_mask: torch.Tensor   # (B, PRIV_VIRUS_MAX)

    @property
    def batch_size(self) -> int:
        return int(self.blobs.shape[0])

    @classmethod
    def zeros(cls, batch: int, device: torch.device | str = "cpu") -> "PrivilegedObs":
        """An all-padding privileged slab — the symmetric-fallback input."""
        f = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        return cls(
            blobs=f(batch, PRIV_BLOB_MAX, PRIV_BLOB_DIM),
            blobs_mask=f(batch, PRIV_BLOB_MAX),
            food=f(batch, PRIV_FOOD_MAX, PRIV_FOOD_DIM),
            food_mask=f(batch, PRIV_FOOD_MAX),
            virus=f(batch, PRIV_VIRUS_MAX, PRIV_VIRUS_DIM),
            virus_mask=f(batch, PRIV_VIRUS_MAX),
        )

    def to(self, device: torch.device | str) -> "PrivilegedObs":
        return PrivilegedObs(
            blobs=self.blobs.to(device),
            blobs_mask=self.blobs_mask.to(device),
            food=self.food.to(device),
            food_mask=self.food_mask.to(device),
            virus=self.virus.to(device),
            virus_mask=self.virus_mask.to(device),
        )


# --- shared building blocks -------------------------------------------------

def masked_mean_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean+max pool over the entity axis.

    ``x`` is ``(B, K, D)``, ``mask`` is ``(B, K)`` with ``1.0`` for real
    entities. Returns ``(B, 2D)`` (mean concat max). Padding rows are excluded
    from the mean (divide by the real count, floored at 1) and pushed to the
    dtype min before the max so they never win; an all-padding row yields zeros
    for both halves. Identical semantics to the P3.1 stub's pool, so obs↔policy
    parity is preserved.
    """
    m = mask.unsqueeze(-1)  # (B, K, 1)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
    mean = (x * m).sum(dim=1) / count  # (B, D)

    neg_inf = torch.finfo(x.dtype).min
    masked = x.masked_fill(m == 0.0, neg_inf)
    mx = masked.max(dim=1).values  # (B, D)
    any_real = mask.sum(dim=1, keepdim=True) > 0.0
    mx = torch.where(any_real, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=-1)


class EntityEncoder(nn.Module):
    """Shared 2-layer MLP per entity → masked mean+max pool → ``(B, 2·hidden)``.

    This is the deep-set encoder of plan v1 §3.2: permutation-invariant over the
    entity axis, so entity input order never matters (the featurizer's K-nearest
    ordering only affects *which* entities are kept, not the encoding).
    """

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, k, d = x.shape
        enc = self.mlp(x.reshape(b * k, d)).reshape(b, k, -1)
        return masked_mean_max(enc, mask)


# --- the policy -------------------------------------------------------------

class AgarPolicy(nn.Module):
    """Recurrent entity-set policy with an asymmetric critic (P3.3).

    Args:
        entity_hidden: width of each per-category actor entity encoder (plan
            v1 §3.2: "64–128 units"). Default 64.
        trunk_hidden: width of the trunk **and** the GRU hidden state (§3.2:
            "Trunk MLP (256) → GRU (256)"). Default 256. Exposed as
            :attr:`hidden_size` — the transport sizes its buffers from it.
        critic_entity_hidden: width of the privileged (critic-only) encoders.
        critic_feat: width of the pooled privileged feature fed to the value
            head alongside the recurrent features.
    """

    def __init__(
        self,
        entity_hidden: int = 64,
        trunk_hidden: int = 256,
        critic_entity_hidden: int = 64,
        critic_feat: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_size = trunk_hidden
        self.critic_feat = critic_feat

        # -- actor / shared trunk (the deployment path) ----------------------
        self.food_enc = EntityEncoder(FOOD_DIM, entity_hidden)
        self.enemy_enc = EntityEncoder(ENEMY_DIM, entity_hidden)
        self.own_enc = EntityEncoder(OWN_DIM, entity_hidden)
        self.virus_enc = EntityEncoder(VIRUS_DIM, entity_hidden)

        concat_dim = (
            SELF_DIM
            + self.food_enc.out_dim
            + self.enemy_enc.out_dim
            + self.own_enc.out_dim
            + self.virus_enc.out_dim
        )
        self.trunk = nn.Sequential(
            nn.Linear(concat_dim, trunk_hidden),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(trunk_hidden, trunk_hidden)

        self.heading_head = nn.Linear(trunk_hidden, HEADING_BINS)
        self.split_head = nn.Linear(trunk_hidden, 1)

        # -- critic (train-only; discarded at deployment) --------------------
        # Privileged deep-set encoders over the full map + a small MLP → a
        # critic feature that the value head reads alongside the recurrent
        # features. When no privileged slab is supplied the feature is zeroed
        # (symmetric fallback), so the value head input width is constant.
        self.priv_blob_enc = EntityEncoder(PRIV_BLOB_DIM, critic_entity_hidden)
        self.priv_food_enc = EntityEncoder(PRIV_FOOD_DIM, critic_entity_hidden)
        self.priv_virus_enc = EntityEncoder(PRIV_VIRUS_DIM, critic_entity_hidden)
        priv_concat = (
            self.priv_blob_enc.out_dim
            + self.priv_food_enc.out_dim
            + self.priv_virus_enc.out_dim
        )
        self.priv_mlp = nn.Sequential(
            nn.Linear(priv_concat, critic_feat),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(trunk_hidden + critic_feat, 1)

    # -- names of the actor (deployment) submodules ----------------------------
    #: Everything on the path that ships; used by :meth:`actor_parameters` and
    #: the param-budget assertion, and (later, P5.1) by weight export.
    _ACTOR_MODULES = (
        "food_enc", "enemy_enc", "own_enc", "virus_enc",
        "trunk", "gru", "heading_head", "split_head",
    )

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the actor/deployment-path parameters (critic excluded)."""
        for name in self._ACTOR_MODULES:
            yield from getattr(self, name).parameters()

    def initial_hidden(
        self, batch_size: int, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    # -- forward ---------------------------------------------------------------

    def _actor_features(self, obs: BatchedObs, hidden: torch.Tensor) -> torch.Tensor:
        """Deep-set encode → trunk → one GRU step, with per-row reset.

        Returns the new hidden/recurrent feature ``(B, hidden_size)``. Rows where
        ``obs.is_reset`` is true have their incoming hidden zeroed *before* the
        GRU step (fresh episode-segment for that seat).
        """
        keep = (~obs.is_reset).to(hidden.dtype).unsqueeze(-1)  # (B, 1)
        hidden = hidden * keep

        feats = torch.cat(
            [
                obs.self_vec,
                self.food_enc(obs.food, obs.food_mask),
                self.enemy_enc(obs.enemy, obs.enemy_mask),
                self.own_enc(obs.own, obs.own_mask),
                self.virus_enc(obs.virus, obs.virus_mask),
            ],
            dim=-1,
        )
        trunk = self.trunk(feats)
        return self.gru(trunk, hidden)

    def _critic_feature(
        self, batch: int, device: torch.device, privileged: Optional[PrivilegedObs]
    ) -> torch.Tensor:
        """The privileged critic feature ``(B, critic_feat)`` — zeros if none."""
        if privileged is None:
            return torch.zeros(batch, self.critic_feat, device=device)
        pooled = torch.cat(
            [
                self.priv_blob_enc(privileged.blobs, privileged.blobs_mask),
                self.priv_food_enc(privileged.food, privileged.food_mask),
                self.priv_virus_enc(privileged.virus, privileged.virus_mask),
            ],
            dim=-1,
        )
        return self.priv_mlp(pooled)

    def forward(
        self,
        obs: BatchedObs,
        hidden: torch.Tensor,
        privileged: Optional[PrivilegedObs] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One recurrent step.

        Args:
            obs: batched observation, ``B`` rows (the *actor's* local view).
            hidden: ``(B, hidden_size)`` incoming recurrent state, row-aligned to
                ``obs``; zeroed per-row where ``obs.is_reset``.
            privileged: optional full-map state for the **critic only**. ``None``
                at inference and in the current transport (symmetric-value
                fallback). Never affects the actor heads.

        Returns:
            ``(heading_logits[B, HEADING_BINS], split_logit[B], value[B],
            new_hidden[B, hidden_size])``.
        """
        new_hidden = self._actor_features(obs, hidden)

        heading_logits = self.heading_head(new_hidden)
        split_logit = self.split_head(new_hidden).squeeze(-1)

        critic_feat = self._critic_feature(
            new_hidden.shape[0], new_hidden.device, privileged
        )
        value = self.value_head(
            torch.cat([new_hidden, critic_feat], dim=-1)
        ).squeeze(-1)

        return heading_logits, split_logit, value, new_hidden


# --- parameter accounting ---------------------------------------------------

def count_actor_parameters(policy: AgarPolicy) -> int:
    """Total actor/deployment-path parameters (the budgeted count)."""
    return sum(p.numel() for p in policy.actor_parameters())


def count_total_parameters(policy: nn.Module) -> int:
    """Total parameters including the train-only critic (for logging)."""
    return sum(p.numel() for p in policy.parameters())


def assert_actor_param_budget(
    policy: AgarPolicy,
    lo: int = ACTOR_PARAM_MIN,
    hi: int = ACTOR_PARAM_MAX,
) -> int:
    """Assert the actor path is within ``[lo, hi]`` (playbook P3.3). Returns the
    count so callers can print it."""
    n = count_actor_parameters(policy)
    if not (lo <= n <= hi):
        raise AssertionError(
            f"actor-path parameter count {n:,} outside budget [{lo:,}, {hi:,}]; "
            f"adjust entity_hidden / trunk_hidden"
        )
    return n


# --- factory ----------------------------------------------------------------

def build_policy(cfg) -> nn.Module:
    """Construct the training policy for a :class:`TrainConfig`.

    Returns the real :class:`AgarPolicy` by default; ``cfg.use_skeleton`` selects
    the P3.1 :class:`~train.harness.policy_stub.SkeletonPolicy` wiring stub (kept
    for the P3.2 transport tests and cheap smoke runs). Both honor the same
    ``policy(BatchedObs, hidden)`` contract, so the learner/worker code is
    identical either way.
    """
    if getattr(cfg, "use_skeleton", False):
        from train.harness.policy_stub import SkeletonPolicy

        return SkeletonPolicy(
            entity_hidden=cfg.entity_hidden, trunk_hidden=cfg.trunk_hidden
        )
    return AgarPolicy(
        entity_hidden=cfg.model_entity_hidden,
        trunk_hidden=cfg.model_trunk_hidden,
    )


__all__ = [
    "MODEL_VERSION",
    "ACTOR_PARAM_MIN",
    "ACTOR_PARAM_MAX",
    "PRIV_BLOB_MAX",
    "PRIV_BLOB_DIM",
    "PRIV_FOOD_MAX",
    "PRIV_FOOD_DIM",
    "PRIV_VIRUS_MAX",
    "PRIV_VIRUS_DIM",
    "PrivilegedObs",
    "masked_mean_max",
    "EntityEncoder",
    "AgarPolicy",
    "count_actor_parameters",
    "count_total_parameters",
    "assert_actor_param_budget",
    "build_policy",
]
