"""P2.1 + P2.2 acceptance: the canonical adapter boundaries and the featurizer.

Covers the P2.2 acceptance checklist directly:

* unit tests — empty vision, dead/first frame, max-entity frames, respawn (reset)
  frame, entities entering/leaving vision (velocity), and each verified-mechanics
  feature (can-eat / can-be-eaten, split-reach threat, virus danger, cooldown);
* property tests — translation invariance of the egocentric entity tensors, and
  permutation invariance of the featurized output (post-K-nearest, id tie-break).

Deterministic randomized versions of the two properties always run (so they gate
in ``make ci`` under ``.venv-bot``); the ``hypothesis`` versions run when that
library is present (``.venv-train``).

Plus a P2.1 boundary check that ``ActionAdapter`` emits model-valid moves for
every heading bin (the full stateful ``MoveValidator`` fuzz is P2.3).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from bot import spec
from bot.adapters import act_v1, obs_v1
from bot.adapters.act_v1 import ActionAdapter, CanonicalAction
from bot.featurizer import Featurizer

from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.blob_model import BlobModel, VisibleBlobModel
from lib.models.food_model import FoodModel
from lib.models.player_model import PlayerModel
from lib.models.virus_model import VirusModel


# --- synthetic query builders ----------------------------------------------

def own_blob(blob_id, x, y, radius, cd=0):
    return BlobModel(blob_id=blob_id, pos=(x, y), radius=radius, merge_cooldown=cd)


def enemy_blob(player_id, blob_id, x, y, radius, cd=0, team_id=None):
    return VisibleBlobModel(
        player_id=player_id,
        team_id=player_id if team_id is None else team_id,
        blob_id=blob_id,
        pos=(x, y),
        radius=radius,
        merge_cooldown=cd,
    )


def make_query(
    *,
    round=0,
    own=((0, 30.0, 30.0, 1.0, 0),),
    enemies=(),
    foods=(),
    viruses=(),
    view_center=None,
    vision_size=20.0,
    pid=0,
    rankings=None,
):
    """Assemble a ``QueryMovePlayer`` from compact tuples.

    ``own``/``enemies``: ``(blob_id, x, y, radius[, cd])`` (enemies prefix a
    ``player_id``). ``foods``: ``(food_id, x, y)``. ``viruses``:
    ``(virus_id, x, y, radius)``. ``view_center`` defaults to the mass-weighted
    centroid of ``own`` (as the engine sets it).
    """
    own_blobs = tuple(own_blob(*b) for b in own)
    masses = [b.radius * b.radius for b in own_blobs]
    total_mass = sum(masses) or 1.0
    if view_center is None:
        cx = sum(b.pos[0] * m for b, m in zip(own_blobs, masses)) / total_mass
        cy = sum(b.pos[1] * m for b, m in zip(own_blobs, masses)) / total_mass
        view_center = (cx, cy)
    you = PlayerModel(
        player_id=pid,
        team_id=pid,
        pos=view_center,
        radius=math.sqrt(total_mass),
        alive=True,
        blobs=own_blobs,
    )
    if rankings is None:
        # own + unique enemy player ids, largest-radius-first (mirrors the
        # engine's get_rankings ordering). Value is unused by current adapters,
        # but must be a valid Sequence[int].
        radii = {pid: math.sqrt(total_mass)}
        for e in enemies:
            radii[e[0]] = max(radii.get(e[0], 0.0), e[4])
        rankings = [p for p, _ in sorted(radii.items(), key=lambda kv: kv[1], reverse=True)]
    return QueryMovePlayer(
        update={},
        round=round,
        rankings=rankings,
        you=you,
        view_center=view_center,
        vision_size=vision_size,
        visible_food=[FoodModel(food_id=f[0], pos=(f[1], f[2])) for f in foods],
        visible_blobs=[enemy_blob(*e) for e in enemies],
        visible_viruses=[
            VirusModel(virus_id=v[0], pos=(v[1], v[2]), radius=v[3]) for v in viruses
        ],
    )


# --- basic shape / emptiness ------------------------------------------------

def test_empty_vision():
    co = Featurizer().featurize(make_query())
    co.validate()
    assert co.is_reset is True
    assert co.food_mask.sum() == 0
    assert co.enemy_mask.sum() == 0
    assert co.virus_mask.sum() == 0
    assert co.own_mask.sum() == 1  # you always have at least one blob


def test_shapes_match_schema():
    co = Featurizer().featurize(make_query(enemies=((1, 0, 32, 30, 1.0),), foods=((0, 31, 30),)))
    assert co.self_vec.shape == (obs_v1.SELF_DIM,)
    assert co.food.shape == (obs_v1.FOOD_MAX, obs_v1.FOOD_DIM)
    assert co.enemy.shape == (obs_v1.ENEMY_MAX, obs_v1.ENEMY_DIM)
    assert co.own.shape == (obs_v1.OWN_MAX, obs_v1.OWN_DIM)
    assert co.virus.shape == (obs_v1.VIRUS_MAX, obs_v1.VIRUS_DIM)


def test_max_entity_frame_saturates_masks():
    # More entities than slots: masks saturate exactly at the K limits and only
    # the nearest survive.
    foods = tuple((i, 30.0 + 0.01 * i, 30.0) for i in range(200))
    enemies = tuple((1 + i % 7, i, 30.0 + 0.02 * i, 31.0, 0.5) for i in range(120))
    viruses = tuple((i, 30.0, 30.0 + 0.03 * i, 1.5) for i in range(12))
    co = Featurizer().featurize(make_query(foods=foods, enemies=enemies, viruses=viruses))
    co.validate()
    assert int(co.food_mask.sum()) == obs_v1.FOOD_MAX
    assert int(co.enemy_mask.sum()) == obs_v1.ENEMY_MAX
    assert int(co.virus_mask.sum()) == obs_v1.VIRUS_MAX


def test_knn_keeps_nearest():
    # 100 foods on a ray; only the 64 nearest kept, and rows are distance-sorted.
    foods = tuple((i, 30.0 + 0.1 * (i + 1), 30.0) for i in range(100))
    co = Featurizer().featurize(make_query(foods=foods))
    kept = int(co.food_mask.sum())
    assert kept == obs_v1.FOOD_MAX
    dists = co.food[:kept, 2]
    assert np.all(np.diff(dists) >= -1e-6), "food rows not distance-sorted"
    # the farthest kept must be closer than the nearest dropped (the 65th pellet)
    assert dists.max() <= (0.1 * (obs_v1.FOOD_MAX + 1)) / (20.0 / 2.0) + 1e-6


# --- reset / respawn semantics ---------------------------------------------

def test_first_frame_is_reset_and_zero_velocity():
    f = Featurizer()
    co = f.featurize(make_query(round=0, enemies=((1, 0, 35, 30, 1.0),)))
    assert co.is_reset is True
    row = co.enemy[0]
    assert row[3] == 0.0 and row[4] == 0.0  # vel_x, vel_y


def test_round_gap_triggers_reset():
    f = Featurizer()
    f.featurize(make_query(round=10, enemies=((1, 0, 35, 30, 1.0),)))
    co_next = f.featurize(make_query(round=11, enemies=((1, 0, 36, 30, 1.0),)))
    assert co_next.is_reset is False  # consecutive → no reset
    co_gap = f.featurize(make_query(round=45, enemies=((1, 0, 36, 30, 1.0),)))
    assert co_gap.is_reset is True  # respawn gap → reset, velocity re-zeroed
    assert co_gap.enemy[0, 3] == 0.0 and co_gap.enemy[0, 4] == 0.0


# --- velocity estimation ----------------------------------------------------

def test_velocity_from_frame_diff():
    f = Featurizer()
    half = 10.0
    f.featurize(make_query(round=0, enemies=((1, 0, 35.0, 30.0, 1.0),), vision_size=2 * half))
    co = f.featurize(make_query(round=1, enemies=((1, 0, 36.5, 30.7, 1.0),), vision_size=2 * half))
    assert co.is_reset is False
    # enemy moved (+1.5, +0.7) absolute ⇒ normalized by half
    np.testing.assert_allclose(co.enemy[0, 3], 1.5 / half, atol=1e-5)
    np.testing.assert_allclose(co.enemy[0, 4], 0.7 / half, atol=1e-5)


def test_velocity_zero_for_entering_blob():
    # A blob absent last frame (left/re-entered vision) starts at zero velocity.
    f = Featurizer()
    f.featurize(make_query(round=0, enemies=((1, 0, 35.0, 30.0, 1.0),)))
    co = f.featurize(make_query(round=1, enemies=(
        (1, 0, 36.0, 30.0, 1.0),   # was visible → has velocity
        (2, 0, 40.0, 30.0, 1.0),   # new this frame → zero velocity
    )))
    # find rows by (player_id) via distance ordering: blob (1) is nearer
    # row0 = player1 (moved +1), row1 = player2 (new)
    assert co.enemy[0, 3] != 0.0
    assert co.enemy[1, 3] == 0.0 and co.enemy[1, 4] == 0.0


def test_velocity_survives_blob_id_reindex():
    """v2026.1.14 re-indexes ``blob_id`` per frame, so the same physical blob can
    change id between frames. Velocity must be recovered by **position**, not id —
    id-based tracking would report zero here (the crux of the .14 migration)."""
    f = Featurizer()
    half = 10.0
    # frame 0: the blob is reported as (player 1, blob_id 1) at (40, 40)
    f.featurize(make_query(round=0, enemies=((1, 1, 40.0, 40.0, 1.0),), vision_size=2 * half))
    # frame 1: SAME physical blob at (40.5, 40.3), but re-indexed to blob_id 0
    co = f.featurize(make_query(round=1, enemies=((1, 0, 40.5, 40.3, 1.0),), vision_size=2 * half))
    assert co.is_reset is False
    np.testing.assert_allclose(co.enemy[0, 3], 0.5 / half, atol=1e-5)
    np.testing.assert_allclose(co.enemy[0, 4], 0.3 / half, atol=1e-5)


def test_velocity_zero_when_no_plausible_predecessor():
    """A blob farther than one tick's plausible travel from any previous
    same-player blob gets zero velocity — the position matcher must not fabricate
    a match to a different blob that happens to share the (re-indexed) id."""
    f = Featurizer()
    half = 10.0
    f.featurize(make_query(round=0, enemies=((1, 0, 30.0, 30.0, 1.0),), vision_size=2 * half))
    # 5 units away in one tick exceeds _VEL_MATCH_RADIUS (~2.7) ⇒ cannot be the
    # same blob ⇒ zero velocity, not a spurious (+5, 0).
    co = f.featurize(make_query(round=1, enemies=((1, 0, 35.0, 30.0, 1.0),), vision_size=2 * half))
    assert co.enemy[0, 3] == 0.0 and co.enemy[0, 4] == 0.0


# --- verified-mechanics features -------------------------------------------

def test_can_eat_flags_bigger_enemy():
    # me r=1.0 at centre; enemy r=2.0 within its own disc ⇒ it can eat me, not vice versa.
    co = Featurizer().featurize(make_query(
        own=((0, 30.0, 30.0, 1.0, 0),),
        enemies=((1, 0, 31.5, 30.0, 2.0),),
    ))
    row = co.enemy[0]
    assert row[6] == 1.0  # can_eat_me
    assert row[8] == 0.0  # i_can_eat


def test_can_eat_flags_smaller_enemy():
    co = Featurizer().featurize(make_query(
        own=((0, 30.0, 30.0, 2.0, 0),),
        enemies=((1, 0, 31.5, 30.0, 1.0),),
    ))
    row = co.enemy[0]
    assert row[6] == 0.0
    assert row[8] == 1.0  # i_can_eat


def test_can_eat_uses_mass_ratio_not_radius():
    # Engine v2026.1.12: eating is a MASS ratio (eater.mass ≥ 1.2·target.mass),
    # i.e. radius ratio √1.2 ≈ 1.095 — NOT a 1.2× radius ratio. An enemy at
    # r=1.15 (mass 1.3225) vs me r=1.0 (mass 1.0): 1.3225 ≥ 1.2·1.0 ✓ under the
    # mass rule (can_eat_me=1), but 1.15 < 1.2·1.0 would be 0 under the old radius
    # rule. Center-coverage holds (dist 0.5 ≤ both radii).
    co = Featurizer().featurize(make_query(
        own=((0, 30.0, 30.0, 1.0, 0),),
        enemies=((1, 0, 30.5, 30.0, 1.15),),
    ))
    assert co.enemy[0, 6] == 1.0  # can_eat_me — mass rule (would be 0 under radius rule)
    assert co.enemy[0, 8] == 0.0  # i_can_eat — I don't out-mass it by 1.2×

    # Symmetric: me r=1.15 out-masses enemy r=1.0 by 1.3225 ≥ 1.2 ⇒ i_can_eat=1.
    co2 = Featurizer().featurize(make_query(
        own=((0, 30.0, 30.0, 1.15, 0),),
        enemies=((1, 0, 30.5, 30.0, 1.0),),
    ))
    assert co2.enemy[0, 8] == 1.0  # i_can_eat — mass rule (would be 0 under radius rule)
    assert co2.enemy[0, 6] == 0.0  # can_eat_me


def test_split_threat_within_and_beyond_reach():
    # enemy r=2.0 (mass 4 ≥ 2, can split; half-mass 2.0 ≥ 1.2×my mass=1.0).
    near = Featurizer().featurize(make_query(
        own=((0, 30.0, 30.0, 1.0, 0),),
        enemies=((1, 0, 33.0, 30.0, 2.0),),  # dist 3, gap 1 < reach ⇒ threat > 0
    ))
    far = Featurizer().featurize(make_query(
        own=((0, 30.0, 30.0, 1.0, 0),),
        enemies=((1, 0, 55.0, 30.0, 2.0),),  # dist 25, gap 23 > reach ⇒ threat 0
    ))
    assert near.enemy[0, 7] > 0.0
    assert far.enemy[0, 7] == 0.0


def test_split_threat_zero_when_enemy_cannot_split():
    # enemy r=1.0 (mass 1 < SPLIT_MIN_MASS) can't split ⇒ no split threat.
    co = Featurizer().featurize(make_query(
        own=((0, 30.0, 30.0, 0.5, 0),),
        enemies=((1, 0, 31.0, 30.0, 1.0),),
    ))
    assert co.enemy[0, 7] == 0.0


def test_enemy_merge_cooldown_exposed():
    co = Featurizer().featurize(make_query(enemies=((1, 0, 32.0, 30.0, 1.0, 9),)))
    np.testing.assert_allclose(co.enemy[0, 5], 9.0 / spec.SPLIT_COOLDOWN_FRAMES, atol=1e-6)


def test_virus_danger_flag():
    big = Featurizer().featurize(make_query(
        own=((0, 30.0, 30.0, 1.7, 0),),  # mass 2.89 > 2.7 ⇒ virus shatters me
        viruses=((0, 32.0, 30.0, 1.5),),
    ))
    small = Featurizer().featurize(make_query(
        own=((0, 30.0, 30.0, 1.0, 0),),  # mass 1.0 < 2.7 ⇒ safe
        viruses=((0, 32.0, 30.0, 1.5),),
    ))
    assert big.virus[0, 4] == 1.0
    assert small.virus[0, 4] == 0.0


def test_self_vector_flags():
    co = Featurizer().featurize(make_query(
        own=((0, 30.0, 30.0, 1.7, 3), (1, 31.0, 30.0, 0.5, 0)),
    ))
    # can_split: largest mass 2.89 ≥ 2.0 ⇒ 1; can_pop: 2.89 > 2.7 ⇒ 1
    assert co.self_vec[7] == 1.0
    assert co.self_vec[8] == 1.0
    # num own blobs / 16
    np.testing.assert_allclose(co.self_vec[1], 2.0 / spec.MAX_BLOB_COUNT, atol=1e-6)
    # own blob rows: canonical order by blob_id, can_split per blob
    assert co.own_mask.sum() == 2
    assert co.own[0, 4] == 1.0  # blob 0, mass 2.89 ≥ 2
    assert co.own[1, 4] == 0.0  # blob 1, mass 0.25 < 2


# --- properties: deterministic (always run) --------------------------------

def _entity_arrays(co):
    return (co.food, co.food_mask, co.enemy, co.enemy_mask,
            co.own, co.own_mask, co.virus, co.virus_mask)


def _assert_rows_close(a, b, atol=1e-4):
    """Compare two entity tensors as an order-independent multiset of rows.

    The entity tensors feed a **permutation-invariant** deep-set encoder (masked
    mean+max pool — see ``test_deep_set_permutation_invariance``), so the real
    property under a world translation is that the *set* of selected entity rows
    is unchanged, not their exact slot order. Two **distinct but exactly
    equidistant** entities can swap K-nearest slots under a ULP-level rounding of
    the view center (e.g. two enemies symmetric about the centroid: their squared
    egocentric distances tie to the last bit, and which one the `(d², id)` sort
    puts first flips when the center carries a `~1e-15` rounding error). That
    reorder is benign — the pooled encoding, and thus every downstream logit, is
    byte-identical either way. Canonicalizing row order before the value
    comparison tolerates that float tie while still catching any *genuine*
    translation variance (wrong normalization, absolute coordinates leaking in),
    which would change the row *values* a sort cannot hide.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.ndim == 1:  # (K,) mask: a reorder can't change it; sort defensively
        np.testing.assert_allclose(np.sort(a), np.sort(b), atol=atol)
        return
    # (K, D) entity tensor: match each `a` row to its nearest unused `b` row.
    # A plain lexicographic sort is *not* robust here — a real row with a
    # ~1e-16 component can sort before/after the all-zero padding rows depending
    # on that component's sign, misaligning them. Greedy nearest-row matching
    # under the Chebyshev metric is: counterpart rows sit < atol apart while any
    # non-counterpart (incl. padding vs an active row) differs by O(0.1), so the
    # match is unambiguous and still fails loudly on any genuine value drift.
    used = np.zeros(len(b), dtype=bool)
    for row in a:
        dist = np.abs(b - row).max(axis=1)
        dist[used] = np.inf
        j = int(np.argmin(dist))
        if dist[j] > atol:  # no acceptable partner ⇒ report the closest one
            np.testing.assert_allclose(row, b[j], atol=atol)
        used[j] = True


def _random_scene(rng):
    n_food = rng.randint(0, 90)
    n_enemy = rng.randint(0, 40)
    n_virus = rng.randint(0, 10)
    foods = [(i, rng.uniform(20, 40), rng.uniform(20, 40)) for i in range(n_food)]
    enemies = [
        (1 + rng.randrange(7), i, rng.uniform(20, 40), rng.uniform(20, 40),
         rng.uniform(0.4, 3.0), rng.randrange(0, 19))
        for i in range(n_enemy)
    ]
    viruses = [(i, rng.uniform(20, 40), rng.uniform(20, 40), 1.5) for i in range(n_virus)]
    own = [(0, 30.0, 30.0, rng.uniform(0.9, 3.0), 0)]
    return own, foods, enemies, viruses


def test_permutation_invariance_deterministic():
    rng = random.Random(20240705)
    for _ in range(200):
        own, foods, enemies, viruses = _random_scene(rng)
        base = Featurizer().featurize(
            make_query(own=tuple(own), foods=tuple(foods),
                       enemies=tuple(enemies), viruses=tuple(viruses)))
        sf, se, sv = list(foods), list(enemies), list(viruses)
        rng.shuffle(sf); rng.shuffle(se); rng.shuffle(sv)
        shuf = Featurizer().featurize(
            make_query(own=tuple(own), foods=tuple(sf),
                       enemies=tuple(se), viruses=tuple(sv)))
        np.testing.assert_array_equal(base.self_vec, shuf.self_vec)
        for a, b in zip(_entity_arrays(base), _entity_arrays(shuf)):
            np.testing.assert_array_equal(a, b)


def test_translation_invariance_deterministic():
    # Egocentric entity tensors are invariant under a world translation (walls are
    # deliberately absolute, so self_vec wall-distances are excluded).
    rng = random.Random(99)
    for _ in range(200):
        own, foods, enemies, viruses = _random_scene(rng)
        tx, ty = rng.uniform(-8, 8), rng.uniform(-8, 8)
        base = Featurizer().featurize(
            make_query(own=tuple(own), foods=tuple(foods),
                       enemies=tuple(enemies), viruses=tuple(viruses)))
        own_t = [(b[0], b[1] + tx, b[2] + ty, b[3], b[4]) for b in own]
        foods_t = [(f[0], f[1] + tx, f[2] + ty) for f in foods]
        enemies_t = [(e[0], e[1], e[2] + tx, e[3] + ty, e[4], e[5]) for e in enemies]
        viruses_t = [(v[0], v[1] + tx, v[2] + ty, v[3]) for v in viruses]
        shifted = Featurizer().featurize(
            make_query(own=tuple(own_t), foods=tuple(foods_t),
                       enemies=tuple(enemies_t), viruses=tuple(viruses_t)))
        for a, b in zip(_entity_arrays(base), _entity_arrays(shifted)):
            _assert_rows_close(a, b)


# --- properties: hypothesis (run where available) --------------------------
# Guarded by a plain import (not module-level importorskip, which would skip the
# always-run deterministic tests above too). The @given tests are only *defined*
# when hypothesis is importable — present in .venv-train, absent in .venv-bot,
# where the deterministic versions above already gate the same properties.

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover - depends on venv
    HAS_HYPOTHESIS = False


if HAS_HYPOTHESIS:
    _coord = st.floats(min_value=20.0, max_value=40.0, allow_nan=False, allow_infinity=False)
    _radius = st.floats(min_value=0.4, max_value=3.0, allow_nan=False, allow_infinity=False)

    @st.composite
    def _scenes(draw):
        n_food = draw(st.integers(0, 80))
        n_enemy = draw(st.integers(0, 40))
        n_virus = draw(st.integers(0, 10))
        foods = [(i, draw(_coord), draw(_coord)) for i in range(n_food)]
        enemies = [
            (1 + draw(st.integers(0, 6)), i, draw(_coord), draw(_coord),
             draw(_radius), draw(st.integers(0, 18)))
            for i in range(n_enemy)
        ]
        viruses = [(i, draw(_coord), draw(_coord), 1.5) for i in range(n_virus)]
        own = (0, 30.0, 30.0, draw(_radius), 0)
        return (own,), foods, enemies, viruses

    @settings(max_examples=150, deadline=None)
    @given(_scenes())
    def test_outputs_finite_and_bounded(scene):
        own, foods, enemies, viruses = scene
        co = Featurizer().featurize(make_query(
            own=own, foods=tuple(foods), enemies=tuple(enemies), viruses=tuple(viruses)))
        co.validate()  # shapes, dtype, finiteness, binary masks
        assert np.abs(co.enemy[:, :5]).max(initial=0.0) <= max(
            obs_v1.POS_CLIP, obs_v1.VEL_CLIP, obs_v1.RAD_CLIP) + 1e-6

    @settings(max_examples=150, deadline=None)
    @given(_scenes(), st.randoms(use_true_random=False))
    def test_permutation_invariance_property(scene, rng):
        own, foods, enemies, viruses = scene
        base = Featurizer().featurize(make_query(
            own=own, foods=tuple(foods), enemies=tuple(enemies), viruses=tuple(viruses)))
        sf, se, sv = list(foods), list(enemies), list(viruses)
        rng.shuffle(sf); rng.shuffle(se); rng.shuffle(sv)
        shuf = Featurizer().featurize(make_query(
            own=own, foods=tuple(sf), enemies=tuple(se), viruses=tuple(sv)))
        np.testing.assert_array_equal(base.self_vec, shuf.self_vec)
        for a, b in zip(_entity_arrays(base), _entity_arrays(shuf)):
            np.testing.assert_array_equal(a, b)

    @settings(max_examples=150, deadline=None)
    @given(_scenes(),
           st.floats(-8, 8, allow_nan=False), st.floats(-8, 8, allow_nan=False))
    def test_translation_invariance_property(scene, tx, ty):
        own, foods, enemies, viruses = scene
        base = Featurizer().featurize(make_query(
            own=own, foods=tuple(foods), enemies=tuple(enemies), viruses=tuple(viruses)))
        own_t = tuple((b[0], b[1] + tx, b[2] + ty, b[3], b[4]) for b in own)
        foods_t = [(f[0], f[1] + tx, f[2] + ty) for f in foods]
        enemies_t = [(e[0], e[1], e[2] + tx, e[3] + ty, e[4], e[5]) for e in enemies]
        viruses_t = [(v[0], v[1] + tx, v[2] + ty, v[3]) for v in viruses]
        shifted = Featurizer().featurize(make_query(
            own=own_t, foods=tuple(foods_t), enemies=tuple(enemies_t), viruses=tuple(viruses_t)))
        for a, b in zip(_entity_arrays(base), _entity_arrays(shifted)):
            _assert_rows_close(a, b)


# --- P2.1 action-adapter boundary ------------------------------------------

def test_action_adapter_all_bins_model_valid():
    # Every heading bin × split builds a model-valid MovePlayer (a non-zero unit
    # vector satisfying DirectionModel). Full stateful MoveValidator fuzz is P2.3.
    for b in range(act_v1.HEADING_BINS):
        for split in (False, True):
            mv = ActionAdapter.to_move(3, CanonicalAction(heading_bin=b, split=split))
            assert mv.player_id == 3
            assert mv.split is split
            x, y = mv.direction.to_vector()
            np.testing.assert_allclose(math.hypot(x, y), 1.0, atol=1e-9)


def test_action_adapter_rejects_out_of_range_bin():
    with pytest.raises(ValueError):
        CanonicalAction(heading_bin=act_v1.HEADING_BINS, split=False)
