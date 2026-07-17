"""Seeding external clones into a league pool (Track 3).

Pins that seeded entries survive the LeaguePool read/write round-trip, get
sampled as opponents, coexist with live snapshots, and that the sampler emits
the ``pool:clone_<stem>`` plan string the coordinator folds results by.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from train.league import ANCHOR, POOL, PFSP_FLOOR, LeaguePool, PoolEntry  # noqa: E402
from seed_league_pool import seed_pool  # noqa: E402

CLONES = [f"data/clones/team_{t}.npz" for t in (49, 14, 37, 1, 88)]
HAVE_CLONES = all((REPO / c).exists() for c in CLONES)


def test_seed_writes_entries_at_max_pfsp_weight(tmp_path):
    pool_dir = tmp_path / "league"
    # use fixture weights so the test runs without the real clones present
    fixture = "tests/fixtures/adversarial/weights_random.npz"
    pool = seed_pool(pool_dir, [fixture, fixture.replace(".npz", ".npz")])
    # both map to the same clone_<stem> id (idempotent overwrite)
    assert [k for k in pool.entries if k.startswith("clone_")] == ["clone_weights_random"]
    e = pool.entries["clone_weights_random"]
    assert e.p_current == 0.5 and e.games == 0
    assert e.path == fixture

    # persisted and re-read identically by a fresh LeaguePool
    reread = LeaguePool(pool_dir=pool_dir)
    assert "clone_weights_random" in reread.entries
    # p·(1−p)+floor is maximal at p=0.5 → seeded clones are sampled first
    w = reread.pfsp_weights()
    assert w["clone_weights_random"] == pytest.approx(0.25 + PFSP_FLOOR, abs=1e-6)


def test_seed_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="clone weights not found"):
        seed_pool(tmp_path / "league", ["data/clones/nope.npz"])


def test_seeded_clone_is_sampled_and_plan_round_trips(tmp_path):
    """The sampler must seat a seeded clone and label it pool:<sid> — the exact
    string the coordinator folds PFSP results by."""
    pool_dir = tmp_path / "league"
    seed_pool(pool_dir, ["tests/fixtures/adversarial/weights_random.npz"])

    from train.harness.league_seats import LeagueSeatSampler

    sampler = LeagueSeatSampler(league_dir=pool_dir, anchors=("greedy_food",),
                                anchor_min_frac=0.10, base_seed=1)
    # draw many games; a p=0.5 clone must take pool seats and be labelled right
    seen_pool = False
    for g in range(40):
        _bots, plan = sampler.sample(stream_index=0, game_counter=g, seats=(4, 5, 6, 7))
        for spec in plan.values():
            if spec.startswith("pool:"):
                assert spec == "pool:clone_weights_random"
                seen_pool = True
    assert seen_pool, "seeded clone never sampled into an opponent seat"


def test_live_snapshots_coexist_with_seeded_clones(tmp_path):
    pool_dir = tmp_path / "league"
    seed_pool(pool_dir, ["tests/fixtures/adversarial/weights_random.npz"])
    pool = LeaguePool(pool_dir=pool_dir)
    # simulate a live snapshot landing alongside the clone
    pool.entries["snap_v000010"] = PoolEntry("snap_v000010", "/p/s.npz", 10, 1.0)
    pool.save()
    reread = LeaguePool(pool_dir=pool_dir)
    assert set(reread.entries) == {"clone_weights_random", "snap_v000010"}
    # folding a clone result updates only its p_current (no crash on custom id)
    reread.record_result("clone_weights_random", current_won=True)
    assert reread.entries["clone_weights_random"].p_current > 0.5


@pytest.mark.skipif(not HAVE_CLONES, reason="real clone npzs not present")
def test_seeds_the_five_real_clones(tmp_path):
    pool = seed_pool(tmp_path / "league", CLONES)
    ids = sorted(k for k in pool.entries if k.startswith("clone_"))
    assert ids == ["clone_team_1", "clone_team_14", "clone_team_37",
                   "clone_team_49", "clone_team_88"]
