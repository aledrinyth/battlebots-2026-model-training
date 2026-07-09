"""Collation — the actor→learner boundary (P3.1).

Actors emit per-seat :class:`~bot.adapters.obs_v1.CanonicalObs` (numpy, produced
by the shared featurizer). The learner needs those stacked into batched torch
tensors for one GPU forward pass over *all* alive policy seats across *all*
concurrent games. That stacking is the seam this module owns.

Keeping it in one tiny module means the custom-harness decision (P3.1) rests on a
contract that is trivially inspectable: a list of ``CanonicalObs`` in, a
:class:`BatchedObs` of torch tensors out, shapes fixed by ``obs_v1``. P3.2 swaps
the transport underneath (multiprocessing + shared memory) but keeps this exact
tensor layout, so the learner code never changes.

The stack is done in numpy first (cheap, contiguous) and handed to torch with a
single ``torch.from_numpy`` per field — no per-row python-side tensor creation on
the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from bot.adapters.obs_v1 import CanonicalObs
from train.harness.flat_obs import FIELD_OFFSETS, OBS_WIDTH


@dataclass(frozen=True)
class BatchedObs:
    """A batch of featurized frames as torch tensors (leading dim ``B``).

    Field shapes mirror ``obs_v1`` with a batch axis prepended. ``is_reset`` is a
    boolean ``(B,)`` tensor the learner uses to zero the recurrent hidden state
    for seats on their first frame or first post-respawn frame.
    """

    self_vec: torch.Tensor   # (B, SELF_DIM)          float32
    food: torch.Tensor       # (B, FOOD_MAX, FOOD_DIM) float32
    food_mask: torch.Tensor  # (B, FOOD_MAX)          float32
    enemy: torch.Tensor      # (B, ENEMY_MAX, ENEMY_DIM)
    enemy_mask: torch.Tensor # (B, ENEMY_MAX)
    own: torch.Tensor        # (B, OWN_MAX, OWN_DIM)
    own_mask: torch.Tensor   # (B, OWN_MAX)
    virus: torch.Tensor      # (B, VIRUS_MAX, VIRUS_DIM)
    virus_mask: torch.Tensor # (B, VIRUS_MAX)
    is_reset: torch.Tensor   # (B,)                    bool

    @property
    def batch_size(self) -> int:
        return int(self.self_vec.shape[0])

    def to(self, device: torch.device | str) -> "BatchedObs":
        """Move every tensor to ``device`` (returns a new :class:`BatchedObs`)."""
        return BatchedObs(
            self_vec=self.self_vec.to(device),
            food=self.food.to(device),
            food_mask=self.food_mask.to(device),
            enemy=self.enemy.to(device),
            enemy_mask=self.enemy_mask.to(device),
            own=self.own.to(device),
            own_mask=self.own_mask.to(device),
            virus=self.virus.to(device),
            virus_mask=self.virus_mask.to(device),
            is_reset=self.is_reset.to(device),
        )


def collate_obs(
    batch: Sequence[CanonicalObs], device: torch.device | str = "cpu"
) -> BatchedObs:
    """Stack a non-empty sequence of :class:`CanonicalObs` into a :class:`BatchedObs`.

    All arrays are already fixed-shape (obs_v1 pads to the caps), so this is a
    plain ``np.stack`` per field followed by one ``torch.from_numpy`` — no ragged
    handling, no per-entity work.
    """
    if len(batch) == 0:
        raise ValueError("collate_obs got an empty batch")

    def _stack(field: str) -> torch.Tensor:
        arr = np.stack([getattr(o, field) for o in batch], axis=0)
        return torch.from_numpy(np.ascontiguousarray(arr)).to(device)

    is_reset = torch.tensor(
        [bool(o.is_reset) for o in batch], dtype=torch.bool, device=device
    )
    return BatchedObs(
        self_vec=_stack("self_vec"),
        food=_stack("food"),
        food_mask=_stack("food_mask"),
        enemy=_stack("enemy"),
        enemy_mask=_stack("enemy_mask"),
        own=_stack("own"),
        own_mask=_stack("own_mask"),
        virus=_stack("virus"),
        virus_mask=_stack("virus_mask"),
        is_reset=is_reset,
    )


def build_batched_obs(
    flat: np.ndarray, is_reset: np.ndarray, device: torch.device | str = "cpu"
) -> BatchedObs:
    """Vectorized ``(B, OBS_WIDTH)`` slab → :class:`BatchedObs` (the P3.2 hot path).

    The learner reads the whole shared-memory obs slab (one contiguous
    ``float32`` array, ``B = num_actors × seats`` rows produced by
    ``flat_obs.flatten``) and turns it into batched torch tensors in one shot:
    slice each field's columns, reshape to the batched entity shape, wrap in a
    single ``torch.from_numpy`` per field. No per-row Python loop and no
    per-entity object construction — the exact tensor layout ``collate_obs``
    produces, so the learner/policy code is identical whether obs arrive as a
    list of :class:`CanonicalObs` (skeleton) or as a flat slab (vectorized).

    ``is_reset`` is a ``(B,)`` array (any dtype; cast to bool) — the recurrent
    reset flag per row.
    """
    if flat.ndim != 2 or flat.shape[1] != OBS_WIDTH:
        raise ValueError(f"expected (B, {OBS_WIDTH}); got {flat.shape}")
    b = flat.shape[0]

    def _field(name: str) -> torch.Tensor:
        start, stop, shape = FIELD_OFFSETS[name]
        arr = flat[:, start:stop].reshape((b, *shape))
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device)

    return BatchedObs(
        self_vec=_field("self_vec"),
        food=_field("food"),
        food_mask=_field("food_mask"),
        enemy=_field("enemy"),
        enemy_mask=_field("enemy_mask"),
        own=_field("own"),
        own_mask=_field("own_mask"),
        virus=_field("virus"),
        virus_mask=_field("virus_mask"),
        is_reset=torch.from_numpy(np.ascontiguousarray(is_reset).astype(bool)).to(device),
    )


__all__ = ["BatchedObs", "collate_obs", "build_batched_obs"]
