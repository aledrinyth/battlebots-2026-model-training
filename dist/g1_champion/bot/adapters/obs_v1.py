"""Canonical observation schema — version ``obs_v1`` (P2.1).

This module is the **frozen boundary** on the observation side of the pipeline:

    QueryMovePlayer --[Featurizer]--> CanonicalObs (obs_v1) --> PolicyCore

It defines *only* the schema — the exact tensor shapes, the per-tensor field
layout, the normalization convention, and the padding/mask convention. The
logic that fills these tensors lives in ``bot.featurizer`` (P2.2), the single
module both the trainer and the shipped bot import so they can never disagree.

Full field-by-field documentation is in ``notes/canonical-spec.md``; the tables
there and the constants here are kept in lock-step by ``tests/test_featurizer``.

**Normalization convention (one rule).** Every spatial quantity — relative
positions, radii, frame-diff velocities, wall distances — is expressed in units
of the *half vision window* ``half = vision_size / 2`` for the current frame.
Because ``vision_size`` grows sub-linearly with the player's own size
(``get_player_vision_size``), this makes the whole observation egocentric **and**
scale-adaptive: as the agent grows, the world zooms out with it and the same
tactical geometry maps to the same numbers. The one absolute-scale cue the agent
would otherwise lose is re-injected as ``self_vec[0]`` (``vision_size`` itself).

All arrays are ``float32``. Padding rows are all-zero and flagged by a companion
mask (``1.0`` real, ``0.0`` padding). Outputs are clipped to the bounds below so
every value is finite and range-bounded regardless of input (property-tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bot import spec

# --- version tag ------------------------------------------------------------

#: Bumped (obs_v2, …) whenever the tensor layout below changes. Recorded in the
#: model registry alongside every checkpoint so an old policy is always paired
#: with the adapter it was trained against.
OBS_VERSION = "obs_v1"

# --- tensor shapes ----------------------------------------------------------

#: Self / global feature vector width (see ``notes/canonical-spec.md`` §self).
SELF_DIM = 16

#: Nearest-K food pellets. 160 exist (``spec.FOOD_COUNT``); we keep the 64
#: closest. Fields: (dx, dy, dist).
FOOD_MAX = 64
FOOD_DIM = 3

#: Nearest-K *enemy* blobs (own blobs never appear here — ``visible_blobs`` is
#: enemies only). Up to 7×16 = 112 can exist; we keep the 32 closest.
#: Fields: (dx, dy, radius, vel_x, vel_y, merge_cooldown, can_eat_me,
#: split_threat, i_can_eat).
ENEMY_MAX = 32
ENEMY_DIM = 9

#: Own blobs, from ``you.blobs`` (never K-truncated — at most MAX_BLOB_COUNT).
#: Fields: (dx, dy, radius, merge_cooldown, can_split).
OWN_MAX = spec.MAX_BLOB_COUNT  # 16
OWN_DIM = 5

#: Nearest-K viruses. 6 exist (``spec.VIRUS_COUNT``); slot for 8.
#: Fields: (dx, dy, dist, radius, is_dangerous).
VIRUS_MAX = 8
VIRUS_DIM = 5

# --- clipping bounds (guarantee finite, range-bounded outputs) --------------
# These are generous — for genuinely in-vision entities the normalized values
# sit within roughly [-1, 1]; the clips only ever bind on pathological inputs
# (e.g. an own blob flung far outside the window) so NaN/inf can never escape.

POS_CLIP = 8.0   # relative positions, distances, wall distances (half-units)
VEL_CLIP = 4.0   # frame-diff velocities (half-units / frame)
RAD_CLIP = 8.0   # radii (half-units)


@dataclass(frozen=True)
class CanonicalObs:
    """One featurized frame, in the canonical layout above.

    ``*_mask`` arrays are ``1.0`` for a real entity and ``0.0`` for padding, so a
    deep-set encoder can masked-pool without seeing padding. ``is_reset`` is
    ``True`` on the first frame of an episode-segment for this seat — the game's
    very first query, or the first query after a respawn gap (dead players are
    not queried, so the round number jumps). The caller resets the GRU hidden
    state whenever it sees ``is_reset`` (recorded design decision, P2.2), and on
    that frame all velocity estimates are zero (no previous frame to diff).
    """

    self_vec: np.ndarray      # (SELF_DIM,)
    food: np.ndarray          # (FOOD_MAX, FOOD_DIM)
    food_mask: np.ndarray     # (FOOD_MAX,)
    enemy: np.ndarray         # (ENEMY_MAX, ENEMY_DIM)
    enemy_mask: np.ndarray    # (ENEMY_MAX,)
    own: np.ndarray           # (OWN_MAX, OWN_DIM)
    own_mask: np.ndarray      # (OWN_MAX,)
    virus: np.ndarray         # (VIRUS_MAX, VIRUS_DIM)
    virus_mask: np.ndarray    # (VIRUS_MAX,)
    is_reset: bool

    def validate(self) -> None:
        """Assert exact shapes, dtypes, finiteness and mask bounds.

        Cheap enough to call in tests and in a debug build; not on the hot path.
        """
        checks = [
            ("self_vec", self.self_vec, (SELF_DIM,)),
            ("food", self.food, (FOOD_MAX, FOOD_DIM)),
            ("food_mask", self.food_mask, (FOOD_MAX,)),
            ("enemy", self.enemy, (ENEMY_MAX, ENEMY_DIM)),
            ("enemy_mask", self.enemy_mask, (ENEMY_MAX,)),
            ("own", self.own, (OWN_MAX, OWN_DIM)),
            ("own_mask", self.own_mask, (OWN_MAX,)),
            ("virus", self.virus, (VIRUS_MAX, VIRUS_DIM)),
            ("virus_mask", self.virus_mask, (VIRUS_MAX,)),
        ]
        for name, arr, shape in checks:
            assert arr.shape == shape, f"{name}: shape {arr.shape} != {shape}"
            assert arr.dtype == np.float32, f"{name}: dtype {arr.dtype} != float32"
            assert np.all(np.isfinite(arr)), f"{name}: non-finite value"
        for name, mask in (
            ("food_mask", self.food_mask),
            ("enemy_mask", self.enemy_mask),
            ("own_mask", self.own_mask),
            ("virus_mask", self.virus_mask),
        ):
            assert np.all((mask == 0.0) | (mask == 1.0)), f"{name}: non-binary mask"


__all__ = [
    "OBS_VERSION",
    "SELF_DIM",
    "FOOD_MAX",
    "FOOD_DIM",
    "ENEMY_MAX",
    "ENEMY_DIM",
    "OWN_MAX",
    "OWN_DIM",
    "VIRUS_MAX",
    "VIRUS_DIM",
    "POS_CLIP",
    "VEL_CLIP",
    "RAD_CLIP",
    "CanonicalObs",
]
