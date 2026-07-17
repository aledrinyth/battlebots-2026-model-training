"""Canonical action schema — version ``act_v1`` (P2.1).

The **frozen boundary** on the action side of the pipeline:

    PolicyCore --> CanonicalAction (act_v1) --[ActionAdapter]--> MovePlayer

The policy emits a discrete heading bin plus a split bit; the adapter turns that
into the engine's ``MovePlayer`` (a unit ``DirectionModel`` vector + split bool).

This file freezes the *interface* (the bin count, the canonical action struct,
the bin→vector geometry). The exhaustive validity-fuzz proof that no emitted
move can ever trip ``MoveValidator`` is P2.3; the geometry it certifies lives
here so P2.1 downstream (the policy head width = ``HEADING_BINS``) is pinned now.

Only ``lib`` is imported (match-legal surface); never ``engine`` or ``train``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lib.interface.events.moves.move_player import MovePlayer
from lib.models.penguin_model import DirectionModel

#: Bumped (act_v2, …) if the action surface changes (more bins, an eject action,
#: continuous heading, …). Recorded per checkpoint in the registry.
ACT_VERSION = "act_v1"

#: Number of discrete heading bins the policy chooses among. Evenly spaced on the
#: full circle starting at +x (bin 0 → 0°/east), counter-clockwise. 24 bins ⇒
#: 15° resolution, plenty for a heading-controlled agent and cheap to argmax.
HEADING_BINS = 24

#: Precomputed unit vectors, one per bin (bin i at angle ``2π·i/HEADING_BINS``).
#: Kept as a plain tuple so the hot path does no trig and no allocation.
_BIN_VECTORS: tuple[tuple[float, float], ...] = tuple(
    (math.cos(2.0 * math.pi * i / HEADING_BINS), math.sin(2.0 * math.pi * i / HEADING_BINS))
    for i in range(HEADING_BINS)
)


def bin_to_vector(heading_bin: int) -> tuple[float, float]:
    """Unit ``(x, y)`` heading for a bin index (``0 <= heading_bin < HEADING_BINS``)."""
    return _BIN_VECTORS[heading_bin]


def _argmax(values) -> int:
    """Index of the largest element (lowest index on ties — deterministic).

    Works on any indexable sequence (Python list, tuple, or ``np.ndarray``) so
    the adapter never forces a NumPy dependency and both the torch trainer and
    the NumPy bot can call it.
    """
    best_i = 0
    best_v = values[0]
    for i in range(1, len(values)):
        v = values[i]
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def decode_logits(heading_logits, split_logit, split_threshold: float = 0.0) -> "CanonicalAction":
    """Greedy decode of a policy's raw outputs into a :class:`CanonicalAction`.

    ``heading_logits`` is a length-``HEADING_BINS`` sequence (argmax → bin);
    ``split_logit`` is a scalar (split iff ``> split_threshold``; the default 0.0
    treats it as a pre-sigmoid logit). Kept sampling-free: the shipped bot acts
    greedily (P5.2), and the trainer samples upstream and passes the chosen
    ``CanonicalAction`` directly.
    """
    if len(heading_logits) != HEADING_BINS:
        raise ValueError(
            f"expected {HEADING_BINS} heading logits, got {len(heading_logits)}"
        )
    return CanonicalAction(
        heading_bin=_argmax(heading_logits),
        split=bool(float(split_logit) > split_threshold),
    )


@dataclass(frozen=True)
class CanonicalAction:
    """What the policy commits to each turn: a heading bin and whether to split.

    Deliberately backend-agnostic — the trainer produces one of these from its
    torch heads, the shipped bot produces one from the NumPy forward pass, and
    both hand it to the *same* ``ActionAdapter`` so training and match time emit
    identical wire moves.
    """

    heading_bin: int
    split: bool

    def __post_init__(self) -> None:
        if not (0 <= self.heading_bin < HEADING_BINS):
            raise ValueError(
                f"heading_bin {self.heading_bin} out of range [0, {HEADING_BINS})"
            )


class ActionAdapter:
    """Canonical action → engine ``MovePlayer``. Stateless; safe to share."""

    version = ACT_VERSION
    heading_bins = HEADING_BINS

    @staticmethod
    def to_move(player_id: int, action: CanonicalAction) -> MovePlayer:
        """Build the validated ``MovePlayer`` for ``player_id`` from ``action``.

        The heading is always a non-zero unit vector (a bin always maps to a point
        on the unit circle), so ``DirectionModel``'s "exactly one representation,
        both x and y present" validator is satisfied by construction — this is the
        property P2.3 fuzzes exhaustively.

        The ``_BIN_VECTORS`` table is a compile-time constant of ``cos``/``sin``
        values, so ``(x, y)`` is inherently finite; the explicit ``isfinite``
        guard is belt-and-suspenders for engine v2026.1.10's
        ``FiniteBaseModel(allow_inf_nan=False)`` — it converts any future
        non-finite heading (e.g. a continuous-heading act_v2) into a clear
        adapter-boundary error instead of a pydantic ``ValidationError`` raised
        deep in wire serialization mid-match.
        """
        x, y = _BIN_VECTORS[action.heading_bin]
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(
                f"non-finite heading ({x}, {y}) for bin {action.heading_bin}; "
                "a MovePlayer with NaN/Inf is rejected by the engine "
                "(allow_inf_nan=False)"
            )
        return MovePlayer(
            player_id=int(player_id),
            direction=DirectionModel(x=x, y=y),
            split=bool(action.split),
        )

    @staticmethod
    def logits_to_move(
        player_id: int, heading_logits, split_logit, split_threshold: float = 0.0
    ) -> MovePlayer:
        """Decode raw policy outputs straight to a ``MovePlayer`` (the bot's hot
        path): ``decode_logits`` then :meth:`to_move`."""
        return ActionAdapter.to_move(
            player_id, decode_logits(heading_logits, split_logit, split_threshold)
        )


__all__ = [
    "ACT_VERSION",
    "HEADING_BINS",
    "bin_to_vector",
    "decode_logits",
    "CanonicalAction",
    "ActionAdapter",
]
