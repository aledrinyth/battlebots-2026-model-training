# Progression annotation (reconstructed): Training and deployment must use the same normalization and entity selection rules.
"""The featurizer (P2.2) — the *one* observation module.

Both the trainer and the shipped bot import ``Featurizer`` from here, so the
features a policy is trained on are, by construction, byte-identical to the
features it acts on at match time. This kills the trainer/inference-disagreement
bug class outright.

What it does, per query (all rules verified against the engine — see
``notes/mechanics-spec.md``):

* **Egocentric + vision-normalized.** Everything is measured relative to
  ``view_center`` (the vision-window center) and divided by the half vision
  window ``half = vision_size / 2`` (``bot.adapters.obs_v1`` normalization rule).
  Note ``view_center`` is the mass centroid **clamped** to keep the window on-map
  (engine v2026.1.10), so it equals ``you.pos`` except near a wall, where the
  own-blob offset it induces usefully encodes wall proximity; centering on the
  viewport (not ``you.pos``) keeps every *visible* entity within ``[-1, 1]``.
* **K-nearest entity selection** with a **deterministic** tie-break: entities are
  ordered by ``(distance, id)`` where ``id`` is the id the engine assigns in the
  query (``food_id`` / ``virus_id`` / ``(player_id, blob_id)``). As of engine
  **v2026.1.14** these are **per-frame public ordinals** (the "leaked info" fix —
  ``get_public_visible_*`` re-indexes them ``0, 1, 2…`` each query rather than
  exposing the engine's stable internal ids), but they remain **unique within a
  frame**, so the order is still a pure function of the *set* of visible entities
  and invariant to the order the engine listed them in.
* **Frame-difference velocity estimation** for enemy blobs. Enemies' split eject
  momentum (``eject_vx/vy``) is *hidden* from observations, so a recently-split
  enemy carries large invisible velocity that only a frame diff can recover —
  this is **mandatory**. Because v2026.1.14 re-indexes ``blob_id`` per frame, a
  blob **cannot** be matched across frames by id; the :class:`EntityTracker`
  instead matches each enemy blob to the nearest same-player blob in the previous
  frame within one tick's plausible travel (``_VEL_MATCH_RADIUS``), greedily and
  deterministically. A blob with no such predecessor (new, just-split, or
  re-entered vision) correctly gets zero velocity.
* **Relational, verified-mechanics features** — never raw ids:
  ``merge_cooldown`` (exposed for enemies too: >0 ⇒ just-split, can't re-merge);
  ``can_eat_me`` / ``i_can_eat`` using the *exact* rule (**mass** ratio
  ``EAT_SIZE_RATIO`` as of engine v2026.1.12 **and** center-coverage);
  ``split_threat`` computed against
  the enemy's **split reach** (``radius + SPLIT_REACH`` ≈ +8.9 units), not its
  body radius; per-blob ``can_split`` (mass ≥ ``SPLIT_MIN_MASS``); virus
  ``is_dangerous`` (any own blob mass > ``VIRUS_POP_MASS_THRESHOLD`` ⇒ contact
  shatters it).

**Reset semantics.** Dead players are never queried, so on respawn the round
number jumps. The featurizer detects that gap (and the very first frame), clears
the tracker, zeros all velocities that frame, and flags ``is_reset`` so the
caller resets the GRU hidden state (recorded design decision). One
:class:`Featurizer` instance serves one seat; parallel games use one each.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from bot import spec
from bot.adapters import obs_v1
from bot.adapters.obs_v1 import (
    ENEMY_DIM,
    ENEMY_MAX,
    FOOD_DIM,
    FOOD_MAX,
    OWN_DIM,
    OWN_MAX,
    POS_CLIP,
    RAD_CLIP,
    SELF_DIM,
    VEL_CLIP,
    VIRUS_DIM,
    VIRUS_MAX,
    CanonicalObs,
)


def _clip(value: float, bound: float) -> float:
    """Scalar clip to ``[-bound, bound]`` (own tiny helper — no NumPy per value)."""
    if value > bound:
        return bound
    if value < -bound:
        return -bound
    return value


#: Max plausible single-tick blob displacement, used to gate cross-frame velocity
#: matching (v2026.1.14: ids are per-frame ordinals, so we match by position). A
#: blob moves at most ``movement_speed(radius) ≤ BASE_PLAYER_SPEED`` per tick, plus
#: up to ``SPLIT_EJECT_SPEED`` of residual eject momentum right after a split. Any
#: candidate previous blob farther than this cannot be the same physical blob, so
#: we leave the current blob's velocity at zero rather than fabricate a match.
_VEL_MATCH_RADIUS = spec.BASE_PLAYER_SPEED + spec.SPLIT_EJECT_SPEED
_VEL_MATCH_R2 = _VEL_MATCH_RADIUS * _VEL_MATCH_RADIUS


class EntityTracker:
    """Holds the previous frame's visible enemy-blob positions **grouped by
    player**, for frame-difference velocity estimation.

    Since engine v2026.1.14 re-indexes ``blob_id`` per frame (per-target public
    ordinals), a blob can no longer be matched across frames by id. We instead
    keep each player's visible-blob positions and let :meth:`Featurizer` match
    each current blob to the nearest same-player blob from the previous frame.
    Only blobs visible in the *previous* frame can yield a velocity; a blob that
    was out of vision (new, or re-entered) gets zero velocity — we genuinely
    could not observe its motion while it was gone. Cleared on our death/respawn
    (the :class:`Featurizer` calls :meth:`clear` on a round gap).
    """

    __slots__ = ("_by_player",)

    def __init__(self) -> None:
        # player_id -> list of (x, y) for that player's blobs visible last frame.
        self._by_player: Dict[int, List[Tuple[float, float]]] = {}

    def clear(self) -> None:
        self._by_player.clear()

    def prev_positions(self, player_id: int) -> Optional[List[Tuple[float, float]]]:
        return self._by_player.get(player_id)

    def commit(self, by_player: Dict[int, List[Tuple[float, float]]]) -> None:
        """Replace the tracked set with exactly this frame's visible enemy blobs.

        Replacing (rather than updating) is deliberate: a blob absent this frame
        is dropped, so if it reappears later it correctly starts from zero
        velocity instead of diffing across the invisible gap.
        """
        self._by_player = by_player


class Featurizer:
    """Turns a ``QueryMovePlayer`` into a :class:`~bot.adapters.obs_v1.CanonicalObs`.

    Stateful across a single seat's episode (holds the velocity tracker and the
    last round number for respawn-gap detection). Call :meth:`reset` to force a
    fresh episode; otherwise resets happen automatically on a round-number gap.
    """

    obs_version = obs_v1.OBS_VERSION

    def __init__(self) -> None:
        self.tracker = EntityTracker()
        self._last_round: Optional[int] = None

    def reset(self) -> None:
        """Forget all cross-frame state (velocity tracker + round watermark)."""
        self.tracker.clear()
        self._last_round = None

    # -- main entry point ---------------------------------------------------

    def featurize(self, query) -> CanonicalObs:
        """Featurize one alive-player query into the canonical layout."""
        is_reset = self._last_round is None or query.round != self._last_round + 1
        if is_reset:
            self.tracker.clear()

        cx, cy = query.view_center
        half = query.vision_size / 2.0
        inv_half = 1.0 / half if half > 0.0 else 0.0

        own_blobs = query.you.blobs
        own_stats = self._own_stats(own_blobs)
        own_max_r = own_stats["max_r"]

        self_vec = self._self_vector(query, cx, cy, inv_half, own_stats)
        own, own_mask = self._own_tensor(own_blobs, cx, cy, inv_half)
        food, food_mask = self._food_tensor(query.visible_food, cx, cy, inv_half)
        virus, virus_mask = self._virus_tensor(
            query.visible_viruses, cx, cy, inv_half, own_stats["max_mass"]
        )
        enemy, enemy_mask = self._enemy_tensor(
            query.visible_blobs, cx, cy, inv_half, own_max_r, is_reset
        )

        self._last_round = query.round
        return CanonicalObs(
            self_vec=self_vec,
            food=food,
            food_mask=food_mask,
            enemy=enemy,
            enemy_mask=enemy_mask,
            own=own,
            own_mask=own_mask,
            virus=virus,
            virus_mask=virus_mask,
            is_reset=is_reset,
        )

    # -- self / global vector ----------------------------------------------

    @staticmethod
    def _own_stats(own_blobs: Sequence) -> dict:
        radii = [b.radius for b in own_blobs]
        masses = [r * r for r in radii]
        cds = [b.merge_cooldown for b in own_blobs]
        # An alive player always has ≥1 blob (dead ⇒ never queried), but guard
        # anyway so the featurizer can never divide by an empty sequence.
        if not radii:
            return {
                "n": 0, "max_r": 0.0, "min_r": 0.0, "max_mass": 0.0,
                "min_cd": 0, "max_cd": 0, "can_split": 0.0, "can_pop": 0.0,
                "player_radius": 0.0,
            }
        max_mass = max(masses)
        return {
            "n": len(radii),
            "max_r": max(radii),
            "min_r": min(radii),
            "max_mass": max_mass,
            "min_cd": min(cds),
            "max_cd": max(cds),
            "can_split": 1.0 if max_mass >= spec.SPLIT_MIN_MASS else 0.0,
            "can_pop": 1.0 if max_mass > spec.VIRUS_POP_MASS_THRESHOLD else 0.0,
            "player_radius": math.sqrt(sum(masses)),
        }

    def _self_vector(self, query, cx, cy, inv_half, s: dict) -> np.ndarray:
        cd_norm = 1.0 / spec.SPLIT_COOLDOWN_FRAMES
        vec = np.zeros(SELF_DIM, dtype=np.float32)
        vec[0] = query.vision_size / spec.VISION_SIZE - 1.0  # absolute-scale cue (≥0)
        vec[1] = s["n"] / spec.MAX_BLOB_COUNT
        vec[2] = _clip(s["player_radius"] * inv_half, RAD_CLIP)
        vec[3] = _clip(s["max_r"] * inv_half, RAD_CLIP)
        vec[4] = _clip(s["min_r"] * inv_half, RAD_CLIP)
        vec[5] = s["min_cd"] * cd_norm
        vec[6] = s["max_cd"] * cd_norm
        vec[7] = s["can_split"]
        vec[8] = s["can_pop"]
        vec[9] = _clip(cx * inv_half, POS_CLIP)                       # dist to left wall
        vec[10] = _clip((spec.ARENA_SIZE - cx) * inv_half, POS_CLIP)  # to right wall
        vec[11] = _clip(cy * inv_half, POS_CLIP)                      # to bottom wall
        vec[12] = _clip((spec.ARENA_SIZE - cy) * inv_half, POS_CLIP)  # to top wall
        vec[13] = query.round / spec.MAX_ROUNDS
        vec[14] = min(len(query.visible_blobs), ENEMY_MAX) / ENEMY_MAX
        vec[15] = min(len(query.visible_food), FOOD_MAX) / FOOD_MAX
        return vec

    # -- own blobs ----------------------------------------------------------

    def _own_tensor(self, own_blobs, cx, cy, inv_half):
        arr = np.zeros((OWN_MAX, OWN_DIM), dtype=np.float32)
        mask = np.zeros(OWN_MAX, dtype=np.float32)
        cd_norm = 1.0 / spec.SPLIT_COOLDOWN_FRAMES
        # Own blobs are canonically ordered by blob_id (the engine's own
        # sorted_blobs order); never more than MAX_BLOB_COUNT of them.
        blobs = sorted(own_blobs, key=lambda b: b.blob_id)[:OWN_MAX]
        for i, b in enumerate(blobs):
            bx, by = b.pos
            mass = b.radius * b.radius
            arr[i, 0] = _clip((bx - cx) * inv_half, POS_CLIP)
            arr[i, 1] = _clip((by - cy) * inv_half, POS_CLIP)
            arr[i, 2] = _clip(b.radius * inv_half, RAD_CLIP)
            arr[i, 3] = b.merge_cooldown * cd_norm
            arr[i, 4] = 1.0 if mass >= spec.SPLIT_MIN_MASS else 0.0
            mask[i] = 1.0
        return arr, mask

    # -- food ---------------------------------------------------------------

    def _food_tensor(self, visible_food, cx, cy, inv_half):
        arr = np.zeros((FOOD_MAX, FOOD_DIM), dtype=np.float32)
        mask = np.zeros(FOOD_MAX, dtype=np.float32)
        # (distance², id, dx, dy) — squared distance keeps the K-nearest sort
        # exact and cheap; id (food_id) breaks ties deterministically.
        scored: List[Tuple[float, int, float, float]] = []
        for f in visible_food:
            fx, fy = f.pos
            dx, dy = fx - cx, fy - cy
            scored.append((dx * dx + dy * dy, f.food_id, dx, dy))
        scored.sort()
        for i, (d2, _fid, dx, dy) in enumerate(scored[:FOOD_MAX]):
            arr[i, 0] = _clip(dx * inv_half, POS_CLIP)
            arr[i, 1] = _clip(dy * inv_half, POS_CLIP)
            arr[i, 2] = _clip(math.sqrt(d2) * inv_half, POS_CLIP)
            mask[i] = 1.0
        return arr, mask

    # -- viruses ------------------------------------------------------------

    def _virus_tensor(self, visible_viruses, cx, cy, inv_half, own_max_mass):
        arr = np.zeros((VIRUS_MAX, VIRUS_DIM), dtype=np.float32)
        mask = np.zeros(VIRUS_MAX, dtype=np.float32)
        # A virus is dangerous to *me* iff contact would shatter one of my blobs,
        # i.e. any own blob has mass > the pop threshold (mass > 2.7 as of v2026.1.11).
        dangerous = 1.0 if own_max_mass > spec.VIRUS_POP_MASS_THRESHOLD else 0.0
        scored: List[Tuple[float, int, float, float, float]] = []
        for v in visible_viruses:
            vx, vy = v.pos
            dx, dy = vx - cx, vy - cy
            scored.append((dx * dx + dy * dy, v.virus_id, dx, dy, v.radius))
        scored.sort()
        for i, (d2, _vid, dx, dy, r) in enumerate(scored[:VIRUS_MAX]):
            arr[i, 0] = _clip(dx * inv_half, POS_CLIP)
            arr[i, 1] = _clip(dy * inv_half, POS_CLIP)
            arr[i, 2] = _clip(math.sqrt(d2) * inv_half, POS_CLIP)
            arr[i, 3] = _clip(r * inv_half, RAD_CLIP)
            arr[i, 4] = dangerous
            mask[i] = 1.0
        return arr, mask

    # -- enemy blobs --------------------------------------------------------

    def _estimate_velocities(
        self, visible_blobs, is_reset
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Per-frame velocities keyed by this frame's ``(player_id, blob_id)``.

        v2026.1.14 makes ``blob_id`` a per-frame ordinal, so we match by position:
        for each player, greedily pair current blobs with the previous frame's
        blobs of the same player, nearest-first, each previous blob used once, and
        only within ``_VEL_MATCH_RADIUS``. Pairs are ordered by
        ``(distance², current blob_id, previous index)`` — a deterministic key
        that does not depend on the order the engine listed the blobs, so the
        result is permutation-invariant. Unmatched current blobs (new / just-split
        / re-entered vision) are simply absent from the map ⇒ zero velocity.
        """
        vel: Dict[Tuple[int, int], Tuple[float, float]] = {}
        if is_reset:
            return vel

        # Group this frame's blobs by player: player_id -> [(blob_id, x, y)].
        cur_by_player: Dict[int, List[Tuple[int, float, float]]] = {}
        for b in visible_blobs:
            cur_by_player.setdefault(b.player_id, []).append((b.blob_id, b.pos[0], b.pos[1]))

        for pid, cur in cur_by_player.items():
            prev = self.tracker.prev_positions(pid)
            if not prev:
                continue
            pairs: List[Tuple[float, int, int]] = []
            for bid, bx, by in cur:
                for pi, (px, py) in enumerate(prev):
                    d2 = (bx - px) * (bx - px) + (by - py) * (by - py)
                    if d2 <= _VEL_MATCH_R2:
                        pairs.append((d2, bid, pi))
            pairs.sort()
            # index current blobs by blob_id for lookup during greedy assignment
            cur_by_id = {bid: (bx, by) for bid, bx, by in cur}
            used_cur: set = set()
            used_prev: set = set()
            for d2, bid, pi in pairs:
                if bid in used_cur or pi in used_prev:
                    continue
                used_cur.add(bid)
                used_prev.add(pi)
                bx, by = cur_by_id[bid]
                px, py = prev[pi]
                vel[(pid, bid)] = (bx - px, by - py)
        return vel

    def _enemy_tensor(self, visible_blobs, cx, cy, inv_half, own_max_r, is_reset):
        arr = np.zeros((ENEMY_MAX, ENEMY_DIM), dtype=np.float32)
        mask = np.zeros(ENEMY_MAX, dtype=np.float32)
        cd_norm = 1.0 / spec.SPLIT_COOLDOWN_FRAMES
        eat_ratio = spec.EAT_SIZE_RATIO
        split_reach = spec.SPLIT_REACH
        own_mass = own_max_r * own_max_r

        # Position-based cross-frame velocity match (ids are per-frame ordinals).
        vel = self._estimate_velocities(visible_blobs, is_reset)

        # Score every visible enemy blob; keep the K nearest by (distance, id).
        scored: List[Tuple[float, Tuple[int, int], object, float, float]] = []
        new_positions: Dict[int, List[Tuple[float, float]]] = {}
        for b in visible_blobs:
            bx, by = b.pos
            dx, dy = bx - cx, by - cy
            key = (b.player_id, b.blob_id)
            new_positions.setdefault(b.player_id, []).append((bx, by))
            scored.append((dx * dx + dy * dy, key, b, dx, dy))
        scored.sort(key=lambda t: (t[0], t[1]))

        for i, (d2, key, b, dx, dy) in enumerate(scored[:ENEMY_MAX]):
            dist = math.sqrt(d2)
            bx, by = b.pos
            er = b.radius
            em = er * er

            # frame-diff velocity (zero on reset frames / first sighting / no match)
            vx, vy = vel.get(key, (0.0, 0.0))

            # exact verified eat rules (MASS ratio AND center-coverage) — engine
            # v2026.1.12 made _can_eat_blob a mass ratio: eater.mass ≥ ratio·target.mass
            # (⇔ radius ratio √ratio). Center-coverage (target center within eater
            # disc) is unchanged: dist ≤ eater.radius.
            can_eat_me = 1.0 if (em >= eat_ratio * own_mass and dist <= er) else 0.0
            i_can_eat = 1.0 if (own_mass >= eat_ratio * em and dist <= own_max_r) else 0.0

            # split-reach threat: enemy can split (mass ≥ 2), a resulting half
            # (mass em/2) could still eat me by mass ratio, and I sit within its reach.
            split_threat = 0.0
            if em >= spec.SPLIT_MIN_MASS and (em * 0.5) >= eat_ratio * own_mass:
                gap = dist - er
                if gap <= split_reach:
                    split_threat = 1.0 - max(0.0, gap) / split_reach

            arr[i, 0] = _clip(dx * inv_half, POS_CLIP)
            arr[i, 1] = _clip(dy * inv_half, POS_CLIP)
            arr[i, 2] = _clip(er * inv_half, RAD_CLIP)
            arr[i, 3] = _clip(vx * inv_half, VEL_CLIP)
            arr[i, 4] = _clip(vy * inv_half, VEL_CLIP)
            arr[i, 5] = b.merge_cooldown * cd_norm
            arr[i, 6] = can_eat_me
            arr[i, 7] = split_threat
            arr[i, 8] = i_can_eat
            mask[i] = 1.0

        # Record this frame's positions for next frame's velocity diff.
        self.tracker.commit(new_positions)
        return arr, mask


__all__ = ["Featurizer", "EntityTracker"]
