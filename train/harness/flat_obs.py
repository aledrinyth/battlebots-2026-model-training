"""Flat obs codec — the actor↔learner wire format (P3.2).

An actor process featurizes a frame into a :class:`~bot.adapters.obs_v1.CanonicalObs`
(nine numpy arrays) and must hand it to the central learner across a
shared-memory slab. Ragged structs do not live in a flat ``float32`` buffer, so
this module defines a **fixed-width, fixed-order flattening** of ``CanonicalObs``
into a single ``(OBS_WIDTH,)`` vector and back.

The layout offsets are derived from the ``obs_v1`` dims (never hard-coded), so if
the schema ever changes the codec follows automatically and the parity test
catches any mismatch.

**This module is numpy-only — it must not import torch.** Actor processes import
it to write their slab rows, and actors are the match-legal, torch-free surface
(and spawn-fast for it). The *torch-side* batched reader that turns a whole
``(B, OBS_WIDTH)`` slab into a :class:`~train.harness.collate.BatchedObs` lives in
``collate.build_batched_obs`` (the learner's hot path), next to ``BatchedObs``
itself. :func:`unflatten` (single-row, back to ``CanonicalObs``) exists for the
roundtrip test and debugging; it is not on the hot path.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from bot.adapters.obs_v1 import (
    CanonicalObs,
    ENEMY_DIM,
    ENEMY_MAX,
    FOOD_DIM,
    FOOD_MAX,
    OWN_DIM,
    OWN_MAX,
    SELF_DIM,
    VIRUS_DIM,
    VIRUS_MAX,
)

# --- layout (derived from obs_v1 dims) --------------------------------------
# Each entry: (field name, flat length, unflattened shape). Order is the wire
# order; concatenating the lengths gives OBS_WIDTH.
_SPEC: List[Tuple[str, int, tuple]] = [
    ("self_vec", SELF_DIM, (SELF_DIM,)),
    ("food", FOOD_MAX * FOOD_DIM, (FOOD_MAX, FOOD_DIM)),
    ("food_mask", FOOD_MAX, (FOOD_MAX,)),
    ("enemy", ENEMY_MAX * ENEMY_DIM, (ENEMY_MAX, ENEMY_DIM)),
    ("enemy_mask", ENEMY_MAX, (ENEMY_MAX,)),
    ("own", OWN_MAX * OWN_DIM, (OWN_MAX, OWN_DIM)),
    ("own_mask", OWN_MAX, (OWN_MAX,)),
    ("virus", VIRUS_MAX * VIRUS_DIM, (VIRUS_MAX, VIRUS_DIM)),
    ("virus_mask", VIRUS_MAX, (VIRUS_MAX,)),
]

# Precompute [start, stop) offsets once at import.
_OFFSETS: dict[str, Tuple[int, int, tuple]] = {}
_cursor = 0
for _name, _len, _shape in _SPEC:
    _OFFSETS[_name] = (_cursor, _cursor + _len, _shape)
    _cursor += _len

#: Total flat width of one featurized frame.
OBS_WIDTH: int = _cursor


def flatten(obs: CanonicalObs, out: np.ndarray | None = None) -> np.ndarray:
    """Serialize ``obs`` into a ``(OBS_WIDTH,)`` float32 vector.

    If ``out`` is given (e.g. a view into a shared-memory slab row) it is filled
    in place and returned; otherwise a fresh array is allocated.
    """
    if out is None:
        out = np.empty(OBS_WIDTH, dtype=np.float32)
    for name, (start, stop, _shape) in _OFFSETS.items():
        out[start:stop] = getattr(obs, name).reshape(-1)
    return out


def unflatten(vec: np.ndarray) -> CanonicalObs:
    """Inverse of :func:`flatten` for a single row (test/debug path)."""
    kw = {}
    for name, (start, stop, shape) in _OFFSETS.items():
        kw[name] = np.ascontiguousarray(vec[start:stop].reshape(shape), dtype=np.float32)
    return CanonicalObs(is_reset=False, **kw)


#: Field → (start, stop, unflattened shape) offsets, exported so the torch-side
#: reader (``collate.build_batched_obs``) shares this one layout definition.
FIELD_OFFSETS = _OFFSETS


__all__ = ["OBS_WIDTH", "FIELD_OFFSETS", "flatten", "unflatten"]
