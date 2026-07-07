"""P1.5 — NumPy-vectorized vision loops for the vendored engine (hot-loop opt).

## Why this exists

P1.4 measured the in-process training path at **34 rounds/s/core**, tripping the
plan's hard floor (<100 ⇒ P1.5 mandatory). The profile is unambiguous: physics
(`commit_round`) alone runs at ~300 rounds/s, but building each round's
`QueryMovePlayer` observations collapses it ~9×. The culprit is not pydantic — it
is the vendored engine's vision test recomputing the **observer's view-center and
vision-size once per entity**:

    get_visible_food(pid):
        for food in map.foods:                 # 160 foods
            player_can_see_point(pid, ...)      # -> get_player_view_center(pid)
                                                #    get_player_vision_size(pid)

`get_player_view_center` (player.x, player.y) does two mass-weighted `sum()`
passes over the player's blobs; `get_player_vision_size` does another `sum()` plus
a `math.pow`. Both are **constant for a player within a round**, yet the original
recomputes them for all 160 foods (+ viruses + every enemy blob) of every player,
every round — 553,600 redundant recomputations over a 400-round profile.

## What this does

Replaces `GameState.get_visible_food / get_visible_blobs / get_visible_viruses`
with versions that:

1. compute the observer's view-center and half-vision **once** (via the engine's
   own unmodified `get_player_view_center` / `get_player_vision_size`, so those
   scalars are bit-identical to the original), then
2. run the membership test over all entity positions at once with NumPy.

Two membership tests are used, matching the engine as of v2026.1.11:
- **food** — point-in-box (`is_in_vision`): `abs(target - observer) <= half`.
- **viruses & player blobs** — circle-vs-box (`is_circle_in_vision`): the circle
  is visible when it overlaps the vision box, so an entity is now visible from
  *outside* the window by up to its radius (the v2026.1.7–.9 change). See
  `_circle_in_vision_mask`.

## Bit-exactness (the P1.5 acceptance)

Both tests are evaluated on Python floats (IEEE-754 float64) by the engine.
NumPy evaluates the identical elementwise expressions on float64 arrays — same
subtraction, same `abs`, same `max(., 0)`, same squared-distance comparison — so
every visibility boolean is identical. The observer scalars come from the
engine's own methods (unchanged accumulation order), and emitted models / list
order are preserved exactly (foods in `map.foods` order; blobs in
player-then-`sorted_blobs` order). `tests/test_fast_vision.py` asserts this
equivalence directly against the pristine engine over full games;
`tests/test_wrapper_parity.py` (P1.3) keeps the physics bit-exact independently.

The vendored submodule is left **byte-identical to its pinned commit** — this
override is the auditable "patch", carried as one reviewable file in the training
tree rather than a dirty submodule working copy (strictly more auditable: the pin
verifies clean). Only the observation/vision path is touched; `commit_round`
physics (including the parity-critical eat loops) is untouched.
"""

from __future__ import annotations

import numpy as np

from engine.state.game_state import GameState
from lib.models.blob_model import VisibleBlobModel

# Saved originals (populated by install(); used by uninstall() and the tests that
# compare fast-vs-pristine output).
_ORIGINALS: dict[str, object] = {}


def _circle_in_vision_mask(xs, ys, rs, center_x, center_y, half):
    """Vectorized mirror of ``GameState.is_circle_in_vision``.

    A circle of radius ``r`` at ``(x, y)`` is visible when the squared distance
    from its center to the vision box (edge-clamped per axis) is within ``r²`` —
    i.e. the circle overlaps the box even if its center sits outside it. This is
    the v2026.1.7–.9 "visible from outside the screen" rule for viruses and
    player blobs. The per-element ops match the engine's scalar float64
    expression exactly, so every visibility boolean is bit-identical.
    """
    dx_outside = np.maximum(np.abs(xs - center_x) - half, 0.0)
    dy_outside = np.maximum(np.abs(ys - center_y) - half, 0.0)
    return dx_outside * dx_outside + dy_outside * dy_outside <= rs * rs


def _visible_food_fast(self: GameState, player_id: int):
    player = self.players[player_id]
    if not player.alive:
        return []
    foods = self.map.foods
    if not foods:
        return []
    # Compute the observer's frame ONCE (bit-identical to the engine's per-entity
    # recomputation — state is constant across the loop).
    center_x, center_y = self.get_player_view_center(player_id)
    half = self.get_player_vision_size(player_id) / 2

    n = len(foods)
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    for i, food in enumerate(foods):
        pos = food.pos
        xs[i] = pos[0]
        ys[i] = pos[1]
    mask = (np.abs(xs - center_x) <= half) & (np.abs(ys - center_y) <= half)
    return [food for food, visible in zip(foods, mask) if visible]


def _visible_viruses_fast(self: GameState, player_id: int):
    player = self.players[player_id]
    if not player.alive:
        return []
    viruses = self.map.viruses
    if not viruses:
        return []
    center_x, center_y = self.get_player_view_center(player_id)
    half = self.get_player_vision_size(player_id) / 2

    n = len(viruses)
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    rs = np.empty(n, dtype=np.float64)
    for i, virus in enumerate(viruses):
        pos = virus.pos
        xs[i] = pos[0]
        ys[i] = pos[1]
        rs[i] = virus.radius
    mask = _circle_in_vision_mask(xs, ys, rs, center_x, center_y, half)
    return [virus for virus, visible in zip(viruses, mask) if visible]


def _visible_blobs_fast(self: GameState, player_id: int):
    if not self.players[player_id].alive:
        return []
    center_x, center_y = self.get_player_view_center(player_id)
    half = self.get_player_vision_size(player_id) / 2

    # Gather (owner, blob) in the *exact* order the original emits them:
    # players in dict order (self excluded, dead excluded), blobs sorted by id.
    owners = []
    blobs = []
    for other in self.players.values():
        if other.id == player_id or not other.alive:
            continue
        for blob in other.sorted_blobs():
            owners.append(other)
            blobs.append(blob)
    if not blobs:
        return []

    n = len(blobs)
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    rs = np.empty(n, dtype=np.float64)
    for i, blob in enumerate(blobs):
        xs[i] = blob.x
        ys[i] = blob.y
        rs[i] = blob.radius
    mask = _circle_in_vision_mask(xs, ys, rs, center_x, center_y, half)
    return [
        blob._to_visible_model(player_id=owner.id, team_id=owner.team_id)
        for owner, blob, visible in zip(owners, blobs, mask)
        if visible
    ]


def _public_visible_blobs_fast(self: GameState, player_id: int):
    """Vectorized mirror of v2026.1.14's ``GameState.get_public_visible_blobs``.

    v2026.1.14 sends the *public* view to players (the "leaked info" fix): enemy
    ``blob_id`` is re-indexed to a per-target ordinal (``0, 1, 2…`` among that
    player's **visible** blobs, in ``sorted_blobs`` order). The engine's public
    method re-implements the visibility loop (it does **not** call
    ``get_visible_blobs``), so it would bypass this hot-loop optimization and pay
    the per-blob view-center recomputation P1.5 removed. This mirror keeps the
    speedup (observer frame computed once) and re-indexes exactly as the engine,
    so the query the trainer builds is byte-identical to the live match's.
    """
    if not self.players[player_id].alive:
        return []
    center_x, center_y = self.get_player_view_center(player_id)
    half = self.get_player_vision_size(player_id) / 2

    result: list[VisibleBlobModel] = []
    for other in self.players.values():
        if other.id == player_id or not other.alive:
            continue
        blobs = other.sorted_blobs()
        if not blobs:
            continue
        n = len(blobs)
        xs = np.empty(n, dtype=np.float64)
        ys = np.empty(n, dtype=np.float64)
        rs = np.empty(n, dtype=np.float64)
        for i, blob in enumerate(blobs):
            xs[i] = blob.x
            ys[i] = blob.y
            rs[i] = blob.radius
        mask = _circle_in_vision_mask(xs, ys, rs, center_x, center_y, half)
        # Per-target ordinal, incremented only for visible blobs (engine order).
        public_blob_id = 0
        for blob, visible in zip(blobs, mask):
            if visible:
                result.append(
                    VisibleBlobModel(
                        player_id=other.id,
                        team_id=other.team_id,
                        blob_id=public_blob_id,
                        pos=(blob.x, blob.y),
                        radius=blob.radius,
                        merge_cooldown=blob.merge_cooldown,
                    )
                )
                public_blob_id += 1
    return result


def install() -> None:
    """Patch the vision getters onto `GameState` (idempotent).

    ``get_visible_{food,viruses,blobs}`` are the raw (stable-id) getters; the
    v2026.1.14 public food/virus getters reuse the patched raw ones internally so
    they inherit the speedup, but ``get_public_visible_blobs`` re-implements its
    loop, so we patch it explicitly with :func:`_public_visible_blobs_fast`.
    """
    if _ORIGINALS:
        return
    _ORIGINALS["get_visible_food"] = GameState.get_visible_food
    _ORIGINALS["get_visible_viruses"] = GameState.get_visible_viruses
    _ORIGINALS["get_visible_blobs"] = GameState.get_visible_blobs
    GameState.get_visible_food = _visible_food_fast
    GameState.get_visible_viruses = _visible_viruses_fast
    GameState.get_visible_blobs = _visible_blobs_fast
    # v2026.1.14 public re-indexed blobs (what the live query now sends).
    if hasattr(GameState, "get_public_visible_blobs"):
        _ORIGINALS["get_public_visible_blobs"] = GameState.get_public_visible_blobs
        GameState.get_public_visible_blobs = _public_visible_blobs_fast


def uninstall() -> None:
    """Restore the pristine engine getters (used by the equivalence test)."""
    if not _ORIGINALS:
        return
    GameState.get_visible_food = _ORIGINALS.pop("get_visible_food")
    GameState.get_visible_viruses = _ORIGINALS.pop("get_visible_viruses")
    GameState.get_visible_blobs = _ORIGINALS.pop("get_visible_blobs")
    if "get_public_visible_blobs" in _ORIGINALS:
        GameState.get_public_visible_blobs = _ORIGINALS.pop("get_public_visible_blobs")


def is_installed() -> bool:
    return bool(_ORIGINALS)
