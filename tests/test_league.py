"""P6.2 acceptance: pool growth, PFSP sampling, anchor floor, frozen seats.

Playbook clauses under test:
* pool grows on schedule (injectable clock; interval gating; version dedup);
* the sampling histogram matches the PFSP weights;
* a dummy "always-loses" entry is down-weighted automatically;
* scripted anchors always hold ≥10% of seats (and 100% while the pool is empty);
* a frozen snapshot actually plays — through the match-time NumPy path — as a
  drop-in scripted seat in a real wrapper game.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from train.league import ANCHOR, PFSP_FLOOR, POOL, LeaguePool

REPO = Path(__file__).resolve().parent.parent
FIXTURE_WEIGHTS = REPO / "tests" / "fixtures" / "adversarial" / "weights_random.npz"


def _pool(tmp_path, **kw) -> LeaguePool:
    return LeaguePool(pool_dir=tmp_path / "pool", **kw)


def _fresh_policy():
    torch = pytest.importorskip("torch")
    from train.model import AgarPolicy
    torch.manual_seed(0)
    return AgarPolicy()


# --- snapshots on schedule -----------------------------------------------------

def test_pool_grows_on_schedule(tmp_path) -> None:
    pool = _pool(tmp_path, snapshot_interval_s=1800.0)
    policy = _fresh_policy()

    assert pool.maybe_snapshot(policy, version=10, now=0.0) is not None    # bootstrap
    assert pool.maybe_snapshot(policy, version=20, now=900.0) is None      # too soon
    assert pool.maybe_snapshot(policy, version=30, now=1800.0) is not None # on the dot
    assert pool.maybe_snapshot(policy, version=30, now=3600.0) is None     # same version
    assert len(pool.entries) == 2
    for e in pool.entries.values():
        assert Path(e.path).exists()

    # Registry round-trips: a fresh LeaguePool on the same dir sees the same state.
    reloaded = _pool(tmp_path)
    assert set(reloaded.entries) == set(pool.entries)
    assert reloaded.maybe_snapshot(policy, version=40, now=1900.0) is None  # clock persisted


def test_snapshot_is_actor_path_only(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    pool = _pool(tmp_path)
    entry = pool.maybe_snapshot(_fresh_policy(), version=1, now=0.0)
    with np.load(entry.path) as z:
        assert not any(k.startswith("value_head") for k in z.files)  # critic stays home


# --- PFSP ------------------------------------------------------------------------

def test_pfsp_histogram_matches_weights(tmp_path) -> None:
    pool = _pool(tmp_path)
    # Hand-craft entries at known win-rates (no snapshots needed for sampling).
    from train.league import PoolEntry
    for sid, p in [("a", 0.5), ("b", 0.8), ("c", 0.95)]:
        pool.entries[sid] = PoolEntry(snapshot_id=sid, path=f"/pool/{sid}.npz",
                                      version=0, created_s=0.0, p_current=p)
    weights = pool.pfsp_weights()
    rng = random.Random(0)
    seats = pool.sample_opponents(4, rng)  # warmup shape check
    assert len(seats) == 4

    counts: Counter = Counter()
    n_draws = 30_000
    for _ in range(n_draws):
        for kind, ref in pool.sample_opponents(2, rng):  # 1 anchor + 1 pool each call
            if kind == POOL:
                counts[Path(ref).stem] += 1
    total = sum(counts.values())
    wsum = sum(weights.values())
    for sid in weights:
        expected = weights[sid] / wsum
        observed = counts[sid] / total
        assert abs(observed - expected) < 0.02, (sid, observed, expected)


def test_always_loses_entry_is_down_weighted(tmp_path) -> None:
    pool = _pool(tmp_path, ema=0.2)
    from train.league import PoolEntry
    pool.entries["healthy"] = PoolEntry("healthy", "/p/h.npz", 0, 0.0, p_current=0.5)
    pool.entries["pushover"] = PoolEntry("pushover", "/p/p.npz", 0, 0.0, p_current=0.5)

    for _ in range(40):  # the current learner beats the pushover every game
        pool.record_result("pushover", current_won=True)

    w = pool.pfsp_weights()
    assert w["pushover"] < 0.03            # decayed to ~the exploration floor
    assert w["healthy"] == pytest.approx(0.25 + PFSP_FLOOR)
    assert w["healthy"] / w["pushover"] > 8  # sampling collapses toward informative foes
    assert pool.entries["pushover"].games == 40


# --- anchor floor ------------------------------------------------------------------

def test_anchors_hold_their_floor(tmp_path) -> None:
    pool = _pool(tmp_path)
    rng = random.Random(1)
    # Empty pool ⇒ anchors take every seat.
    assert all(k == ANCHOR for k, _ in pool.sample_opponents(4, rng))

    from train.league import PoolEntry
    pool.entries["a"] = PoolEntry("a", "/p/a.npz", 0, 0.0)
    for n_seats in (1, 2, 4, 7):
        for _ in range(200):
            seats = pool.sample_opponents(n_seats, rng, total_seats=8)
            n_anchor = sum(1 for k, _ in seats if k == ANCHOR)
            assert len(seats) == n_seats
            assert n_anchor >= 1  # ceil(0.10 × 8) = 1, always


def test_anchor_names_come_from_the_configured_set(tmp_path) -> None:
    pool = _pool(tmp_path, anchors=("greedy_food", "chaser"))
    rng = random.Random(2)
    names = set()
    for _ in range(100):
        for k, ref in pool.sample_opponents(3, rng):
            if k == ANCHOR:
                names.add(ref)
    assert names <= {"greedy_food", "chaser"}


# --- frozen seats actually play -----------------------------------------------------

def test_frozen_bot_plays_a_real_game_seat() -> None:
    pytest.importorskip("engine")
    from train.harness.actor import Actor
    from train.opponents.frozen import FrozenPolicyBot
    from train.opponents.scripted import make_bot

    frozen = FrozenPolicyBot(FIXTURE_WEIGHTS)
    scripted = {1: frozen, **{pid: make_bot("dont_move") for pid in range(2, 8)}}
    actor = Actor(seed=5, policy_seats={0}, scripted=scripted, emit_update=False)
    actor.reset()

    from bot.adapters.act_v1 import CanonicalAction
    for _ in range(25):
        obs = actor.observe()
        moves = {0: CanonicalAction(heading_bin=0, split=False)} if 0 in obs else {}
        if actor.step(moves):
            break
    # The frozen seat survived 25 real rounds through the NumPy match path
    # and kept recurrent state (hidden updated in place each act()).
    assert actor.env.state.players[1].alive
    import numpy as np
    assert np.isfinite(frozen._hidden).all()
    assert np.abs(frozen._hidden).sum() > 0  # the GRU actually integrated frames

    frozen.reset()
    assert np.abs(frozen._hidden).sum() == 0
