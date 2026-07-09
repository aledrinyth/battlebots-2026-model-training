"""SkeletonPolicy — the throwaway wiring model for the P3.1 spike.

Its *only* job is to prove observations move end-to-end: consume a
:class:`~train.harness.collate.BatchedObs`, run one recurrent step, and emit the
exact head shapes the action side expects (``HEADING_BINS`` heading logits + a
split logit), plus a value scalar. It is deliberately small and unoptimized.

**This is not the P3.3 network.** P3.3 delivers ``train/model.py`` with the real
per-category deep-set encoders, the 300k–800k-parameter budget assertion, and
the asymmetric critic that additionally reads privileged full-map state. The
architecture here (masked mean+max pooling per entity category → trunk →
``GRUCell`` → heads) is the same *shape* so the P3.1 skeleton exercises the real
data path, but the sizes and the critic are stubbed.

The one piece of real semantics it must get right — because the whole custom
harness is being chosen partly to control it — is **per-row recurrent reset**:
seats flagged ``is_reset`` (first frame of a game or first frame after a respawn
gap) must have their incoming hidden state zeroed before the GRU step. That is
implemented and unit-tested here.
"""

from __future__ import annotations

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


def _masked_mean_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean+max pool over the entity axis.

    ``x`` is ``(B, K, D)``, ``mask`` is ``(B, K)`` with 1.0 for real entities.
    Returns ``(B, 2D)`` (mean concat max). Padding rows are excluded from the
    mean (divide by the real count, floored at 1) and pushed to ``-inf`` before
    the max so they never win; an all-padding row yields zeros for both halves.
    """
    m = mask.unsqueeze(-1)  # (B, K, 1)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
    mean = (x * m).sum(dim=1) / count  # (B, D)

    neg_inf = torch.finfo(x.dtype).min
    masked = x.masked_fill(m == 0.0, neg_inf)
    mx = masked.max(dim=1).values  # (B, D)
    # Rows with zero real entities have max == neg_inf everywhere → zero them.
    any_real = (mask.sum(dim=1, keepdim=True) > 0.0)
    mx = torch.where(any_real, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=-1)


class _EntityEncoder(nn.Module):
    """Shared 2-layer MLP per entity → masked mean+max pool → ``(B, 2·hidden)``."""

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
        return _masked_mean_max(enc, mask)


class SkeletonPolicy(nn.Module):
    """Minimal recurrent policy for the P3.1 end-to-end skeleton.

    Args:
        entity_hidden: width of each per-category entity encoder.
        trunk_hidden: width of the trunk and the GRU hidden state.
    """

    def __init__(self, entity_hidden: int = 32, trunk_hidden: int = 64) -> None:
        super().__init__()
        self.hidden_size = trunk_hidden

        self.food_enc = _EntityEncoder(FOOD_DIM, entity_hidden)
        self.enemy_enc = _EntityEncoder(ENEMY_DIM, entity_hidden)
        self.own_enc = _EntityEncoder(OWN_DIM, entity_hidden)
        self.virus_enc = _EntityEncoder(VIRUS_DIM, entity_hidden)

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
        self.value_head = nn.Linear(trunk_hidden, 1)

    def initial_hidden(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(
        self, obs: BatchedObs, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One recurrent step.

        Args:
            obs: batched observation, ``B`` rows.
            hidden: ``(B, hidden_size)`` incoming recurrent state, row-aligned to
                ``obs``. Rows where ``obs.is_reset`` is true are zeroed *before*
                the GRU step (fresh episode-segment for that seat).

        Returns:
            ``(heading_logits[B, HEADING_BINS], split_logit[B], value[B],
            new_hidden[B, hidden_size])``.
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
        new_hidden = self.gru(trunk, hidden)

        heading_logits = self.heading_head(new_hidden)
        split_logit = self.split_head(new_hidden).squeeze(-1)
        value = self.value_head(new_hidden).squeeze(-1)
        return heading_logits, split_logit, value, new_hidden


__all__ = ["SkeletonPolicy"]
