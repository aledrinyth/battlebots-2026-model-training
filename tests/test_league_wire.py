"""P6.4 acceptance — league wiring: config identity, seat sampling at rollover,
result records, learner-side folding/snapshots, and the collapse alerts.

The clauses under test:

* **Config identity is preserved.** The league knobs are hash-inert at their
  defaults — the LIVE ``stage1_bcft`` box run (hash pinned below) must resume
  across this code change (the two-box rsync rule). ``league_dir`` never
  hashes (operational); ``league: true`` is a genuinely new experiment.
* **Seat sampling** is deterministic per (stream, game), anchors-only while the
  pool is empty, seats frozen snapshots once entries exist, shares loaded
  weights across seats without sharing hidden state, and degrades (never
  raises) on unreadable registry/snapshots — an actor must not break the
  barrier protocol because a file was mid-write.
* **Rollover integration**: GameRunner reseats opponents per game via the
  sampler and emits a GameResult (rankings + masses + plan) at every game end.
* **Folding**: the coordinator reads results incrementally (offsets persist
  across restarts — an EMA is never double-folded), records one PFSP outcome
  per (policy seat, pool seat) pair, and snapshots on its cadence.
* **Alerts**: each collapse signature fires on a synthetic collapse and stays
  quiet on a healthy run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURE_WEIGHTS = REPO / "tests" / "fixtures" / "adversarial" / "weights_random.npz"

# Config hashes recorded 2026-07-17, BEFORE the P6.4 fields were added.
# If these fail after an *intentional* engine-pin or adapter bump, re-pin them;
# any other failure means checkpoint resume of every existing run just broke.
STAGE1_BCFT_HASH = "47b4c63d62829931"       # the live S1 box run
G1_CHAMPION_HASH = "8f19bff4240be0dd"       # dist/g1_champion provenance
DEFAULTS_HASH = "336d5e6c3611cd2c"


# --------------------------------------------------------------------------- #
# Config identity                                                              #
# --------------------------------------------------------------------------- #


def test_league_knobs_are_hash_inert_at_defaults():
    pytest.importorskip("torch")
    from train.harness.config import TrainConfig
    from train.harness.provenance import config_dict, config_hash

    assert config_hash(TrainConfig()) == DEFAULTS_HASH
    base = config_dict(TrainConfig(), for_hash=True)
    assert not any(k.startswith("league") for k in base)

    on = config_dict(TrainConfig(league=True, league_dir="/x"), for_hash=True)
    assert on["league"] is True
    assert "league_dir" not in on  # operational, never part of the identity
    assert config_hash(TrainConfig(league=True, league_dir="/x")) == \
        config_hash(TrainConfig(league=True, league_dir="/elsewhere"))


def test_live_run_hashes_unchanged():
    pytest.importorskip("torch")
    yaml = pytest.importorskip("yaml")  # noqa: F841
    from train.harness.provenance import config_hash
    from train.harness.run import RunConfig

    s1 = RunConfig.load(str(REPO / "configs" / "stage1_bcft.yaml"))
    assert config_hash(s1.base_train_config()) == STAGE1_BCFT_HASH
    g1 = RunConfig.load(str(REPO / "configs" / "stage0_foodonly_rig_bcft.yaml"))
    assert config_hash(g1.base_train_config()) == G1_CHAMPION_HASH


def test_league_config_validation():
    pytest.importorskip("torch")
    from train.harness.config import TrainConfig

    with pytest.raises(ValueError, match="non-policy seat"):
        TrainConfig(league=True, league_dir="/x",
                    policy_seats=tuple(range(8))).validate()
    with pytest.raises(ValueError, match="league_dir"):
        TrainConfig(league=True, policy_seats=(0, 1, 2, 3)).validate()
    TrainConfig(league=True, league_dir="/x", policy_seats=(0, 1, 2, 3)).validate()


# --------------------------------------------------------------------------- #
# GameResult                                                                   #
# --------------------------------------------------------------------------- #


def _result(**over):
    from train.harness.league_seats import GameResult
    kw = dict(
        stream_index=3, game_counter=7, seed=123,
        rankings=[4, 0, 1, 5, 2, 6, 7, 3],
        final_masses={p: float(10 - p) for p in range(8)},
        plan={0: "policy", 1: "policy", 2: "policy", 3: "policy",
              4: "pool:snap_v000010", 5: "anchor:cautious",
              6: "anchor:greedy_food", 7: "pool:snap_v000020"},
    )
    kw.update(over)
    return GameResult(**kw)


def test_game_result_roundtrip_and_views():
    from train.harness.league_seats import GameResult

    res = _result()
    back = GameResult.from_json(res.to_json())
    assert back == res
    assert back.rank_of()[4] == 1 and back.rank_of()[3] == 8
    assert back.policy_seats() == [0, 1, 2, 3]
    assert back.pool_seats() == [(4, "snap_v000010"), (7, "snap_v000020")]
    assert ("5", "cautious") not in back.anchor_seats()  # pids back to ints
    assert (5, "cautious") in back.anchor_seats()


# --------------------------------------------------------------------------- #
# Seat sampling                                                                #
# --------------------------------------------------------------------------- #


def _write_registry(league_dir: Path, entries: dict) -> None:
    league_dir.mkdir(parents=True, exist_ok=True)
    (league_dir / "league.json").write_text(json.dumps(
        {"entries": entries, "last_snapshot_s": 0.0}))


def _entry(sid: str, path: str, p: float = 0.5) -> dict:
    return {"snapshot_id": sid, "path": path, "version": 1,
            "created_s": 0.0, "games": 0, "p_current": p}


def test_sampler_anchors_only_without_registry(tmp_path):
    from train.harness.league_seats import LeagueSeatSampler

    s = LeagueSeatSampler(tmp_path / "league", ("greedy_food", "cautious"), 0.10, 99)
    bots, plan = s.sample(0, 0, (4, 5, 6, 7))
    assert set(bots) == {4, 5, 6, 7}
    assert all(spec.startswith("anchor:") for spec in plan.values())


def test_sampler_is_deterministic_per_game(tmp_path):
    from train.harness.league_seats import LeagueSeatSampler

    league = tmp_path / "league"
    _write_registry(league, {"snap_v000001": _entry("snap_v000001", str(FIXTURE_WEIGHTS))})
    mk = lambda: LeagueSeatSampler(league, ("greedy_food", "cautious"), 0.10, 42)  # noqa: E731
    _, plan_a = mk().sample(3, 17, (4, 5, 6, 7))
    _, plan_b = mk().sample(3, 17, (4, 5, 6, 7))
    assert plan_a == plan_b  # same (stream, game) => same draw, across processes
    plans = {json.dumps(mk().sample(3, g, (4, 5, 6, 7))[1], sort_keys=True)
             for g in range(12)}
    assert len(plans) > 1  # the draw actually varies game to game


def test_sampler_seats_pool_entries_sharing_weights(tmp_path):
    from train.harness.league_seats import LeagueSeatSampler
    from train.opponents.frozen import FrozenPolicyBot

    league = tmp_path / "league"
    _write_registry(league, {"snap_v000001": _entry("snap_v000001", str(FIXTURE_WEIGHTS))})
    s = LeagueSeatSampler(league, ("greedy_food",), 0.10, 7)
    # Draw until the single pool entry holds >= 2 seats in one game (PFSP is
    # with replacement; with one entry every non-anchor seat is its).
    for g in range(20):
        bots, plan = s.sample(0, g, (4, 5, 6, 7))
        frozen = [b for b in bots.values() if isinstance(b, FrozenPolicyBot)]
        if len(frozen) >= 2:
            break
    else:
        pytest.fail("pool entry never took two seats in 20 games")
    assert all(plan[pid] == "pool:snap_v000001"
               for pid, b in bots.items() if isinstance(b, FrozenPolicyBot))
    # Shared loaded weights (one NumpyPolicy), independent hidden state.
    assert frozen[0]._policy is frozen[1]._policy
    assert frozen[0]._hidden is not frozen[1]._hidden
    assert len(s._policies) == 1  # the LRU held one load for both seats


def test_sampler_degrades_never_raises(tmp_path):
    from train.harness.league_seats import LeagueSeatSampler

    league = tmp_path / "league"
    # Unreadable snapshot path: seat falls back to an anchor.
    _write_registry(league, {"snap_v000009": _entry("snap_v000009", str(tmp_path / "gone.npz"))})
    s = LeagueSeatSampler(league, ("greedy_food", "cautious"), 0.10, 5)
    bots, plan = s.sample(0, 0, (4, 5, 6, 7))
    assert set(bots) == {4, 5, 6, 7}
    assert all(spec.startswith("anchor:") for spec in plan.values())

    # Corrupt registry afterward: sampler keeps the last-known entries.
    (league / "league.json").write_text("{ not json")
    import os
    os.utime(league / "league.json")  # force a different mtime
    bots2, _ = s.sample(0, 1, (4, 5, 6, 7))
    assert set(bots2) == {4, 5, 6, 7}


# --------------------------------------------------------------------------- #
# GameRunner rollover integration (short games via a class-level round cap)    #
# --------------------------------------------------------------------------- #


class _CountingSampler:
    """Stub sampler: dont_move seats, records every (stream, game) it filled."""

    def __init__(self):
        self.calls = []

    def sample(self, stream_index, game_counter, seats):
        from train.opponents.scripted import make_bot
        self.calls.append((stream_index, game_counter))
        return ({pid: make_bot("dont_move") for pid in seats},
                {pid: "anchor:dont_move" for pid in seats})


@pytest.fixture
def short_games(monkeypatch):
    pytest.importorskip("engine")
    from engine.state.game_state import GameState
    monkeypatch.setattr(GameState, "is_game_over", lambda self: self.round + 1 >= 20)


def test_gamerunner_reseats_and_reports_per_game(short_games):
    pytest.importorskip("engine")
    import numpy as np
    from train.harness.game_runner import GameRunner

    sampler = _CountingSampler()
    sink = []
    runner = GameRunner(
        stream_index=2, policy_seats=(0, 1, 2, 3), scripted_by_seat={},
        base_seed=100, mass_scale=1.0, reward_clip=10.0,
        seat_sampler=sampler, result_sink=sink.append,
    )
    heading = np.zeros(4, dtype=np.float32)
    split = np.zeros(4, dtype=np.float32)
    for _ in range(65):  # 3+ short games
        runner.observe()
        runner.step(heading, split)

    assert sampler.calls[:4] == [(2, 0), (2, 1), (2, 2), (2, 3)]
    assert len(sink) >= 3
    first = sink[0]
    assert first.game_counter == 0 and first.stream_index == 2
    assert sorted(first.rankings) == list(range(8))       # a full ranking
    assert set(first.final_masses) == set(range(8))
    assert [p for p, s in sorted(first.plan.items()) if s == "policy"] == [0, 1, 2, 3]
    assert all(first.plan[p] == "anchor:dont_move" for p in range(4, 8))
    assert [r.game_counter for r in sink[:3]] == [0, 1, 2]


def test_gamerunner_without_league_is_unchanged(short_games):
    """sampler=None keeps the static scripted mapping; no sink, no records."""
    pytest.importorskip("engine")
    import numpy as np
    from train.harness.game_runner import GameRunner

    runner = GameRunner(
        stream_index=0, policy_seats=(0, 1, 2, 3),
        scripted_by_seat={4: "greedy_food", 5: "cautious", 6: "chaser", 7: "coward"},
        base_seed=100, mass_scale=1.0, reward_clip=10.0,
    )
    names = {pid: bot.name for pid, bot in runner.actor.scripted.items()}
    assert names == {4: "greedy_food", 5: "cautious", 6: "chaser", 7: "coward"}
    heading = np.zeros(4, dtype=np.float32)
    split = np.zeros(4, dtype=np.float32)
    for _ in range(25):  # crosses one rollover without a sink
        runner.observe()
        runner.step(heading, split)
    assert runner._game_counter >= 1


# --------------------------------------------------------------------------- #
# Learner-side folding + snapshots                                             #
# --------------------------------------------------------------------------- #


def test_coordinator_folds_per_pair_and_persists_offsets(tmp_path):
    pytest.importorskip("torch")
    from train.harness.league_seats import ResultsWriter
    from train.harness.league_wire import LeagueCoordinator

    league = tmp_path / "league"
    coord = LeagueCoordinator(league, ("greedy_food",), ema=0.1)
    # Seed a pool entry by hand (no snapshot needed for folding).
    from train.league import PoolEntry
    coord.pool.entries["snap_v000010"] = PoolEntry(
        "snap_v000010", str(league / "snap_v000010.npz"), 10, 0.0)
    coord.pool.entries["snap_v000020"] = PoolEntry(
        "snap_v000020", str(league / "snap_v000020.npz"), 20, 0.0)

    w = ResultsWriter(league, "actor_000")
    # rankings: pool seat 4 wins; policy 0,1,2 next; pool seat 7 mid; policy 3 last-ish.
    w(_result())
    w.close()

    class _NoSnap:  # snapshots not under test here
        def state_dict(self):  # pragma: no cover
            raise AssertionError("must not snapshot")

    coord.pool.snapshot_interval_s = math.inf
    coord.pool._last_snapshot_s = 0.0
    metrics = coord.step(_NoSnap(), version=1, now=1.0)

    # Pairs vs snap_v000010 (rank 1): every policy seat loses -> p_current sinks.
    e10 = coord.pool.entries["snap_v000010"]
    assert e10.games == 4
    assert e10.p_current == pytest.approx(0.5 * 0.9**4)
    # Pairs vs snap_v000020 (seat 7, rank 7): policy ranks 2,3,5 win; 8 loses.
    e20 = coord.pool.entries["snap_v000020"]
    assert e20.games == 4
    assert e20.p_current > 0.5  # net-positive vs the current learner
    assert metrics["league_games_folded"] == 1.0
    assert metrics["league_policy_rank"] == pytest.approx((2 + 3 + 5 + 8) / 4)
    assert metrics["league_pair_winrate"] == pytest.approx(3 / 8)

    # Offsets persisted: a fresh coordinator on the same dir does NOT refold,
    # and the cumulative games counter survives the restart (metrics'
    # league_games_total stays monotone over a kill+resume).
    coord2 = LeagueCoordinator(league, ("greedy_food",), ema=0.1)
    assert coord2._read_new_results() == []
    assert coord2.games_folded_total == 1


def test_coordinator_leaves_partial_lines_pending(tmp_path):
    pytest.importorskip("torch")
    from train.harness.league_wire import LeagueCoordinator

    league = tmp_path / "league"
    (league / "results").mkdir(parents=True)
    f = league / "results" / "actor_000.jsonl"
    full = _result().to_json()
    f.write_text(full + "\n" + full[: len(full) // 2])  # torn tail, no newline

    coord = LeagueCoordinator(league, ("greedy_food",))
    got = coord._read_new_results()
    assert len(got) == 1                      # the torn line waits
    f.write_text(full + "\n" + full + "\n")   # writer completes it later
    got2 = coord._read_new_results()
    assert len(got2) == 1                     # only the completed tail, once


def test_coordinator_snapshots_on_cadence_and_registry_feeds_sampler(tmp_path):
    torch = pytest.importorskip("torch")
    from train.harness.league_seats import LeagueSeatSampler
    from train.harness.league_wire import LeagueCoordinator
    from train.model import AgarPolicy

    torch.manual_seed(0)
    policy = AgarPolicy()
    league = tmp_path / "league"
    coord = LeagueCoordinator(league, ("greedy_food",), snapshot_interval_s=100.0)

    m0 = coord.step(policy, version=1, now=0.0)      # bootstrap: pool was empty
    assert m0["league_pool_size"] == 1.0
    m1 = coord.step(policy, version=5, now=50.0)     # too soon
    assert m1["league_pool_size"] == 1.0
    m2 = coord.step(policy, version=9, now=100.0)    # on the dot
    assert m2["league_pool_size"] == 2.0

    # The registry the coordinator saved is exactly what an actor-side sampler
    # reads: pool seats appear, playing through the NumPy path.
    from train.opponents.frozen import FrozenPolicyBot
    s = LeagueSeatSampler(league, ("greedy_food",), 0.10, 1)
    for g in range(10):
        bots, plan = s.sample(0, g, (4, 5, 6, 7))
        if any(isinstance(b, FrozenPolicyBot) for b in bots.values()):
            break
    else:
        pytest.fail("sampler never seated a pool snapshot from the live registry")


# --------------------------------------------------------------------------- #
# Alerts                                                                       #
# --------------------------------------------------------------------------- #


def test_entropy_floor_alert():
    from train.harness.monitor import AlertConfig, check_entropy_floor

    cfg = AlertConfig(entropy_floor=0.05, entropy_window=10)
    healthy = [0.8] * 50
    assert check_entropy_floor(healthy, cfg) is None
    collapsed = [0.8] * 40 + [0.01] * 10
    alert = check_entropy_floor(collapsed, cfg)
    assert alert is not None and alert.severity == "critical"
    assert check_entropy_floor([0.01] * 5, cfg) is None  # too few points


def test_pool_domination_alert():
    from train.harness.monitor import AlertConfig, check_pool_domination

    cfg = AlertConfig(domination_p=0.95, domination_min_games=30)
    entries = {
        "a": {"games": 100, "p_current": 0.99},
        "b": {"games": 50, "p_current": 0.97},
        "c": {"games": 3, "p_current": 0.10},   # under-probed: ignored
    }
    alert = check_pool_domination(entries, cfg)
    assert alert is not None and alert.name == "pool_domination"
    entries["b"]["p_current"] = 0.6              # one informative foe remains
    assert check_pool_domination(entries, cfg) is None
    assert check_pool_domination({}, cfg) is None


def test_mass_plateau_alert():
    from train.harness.monitor import AlertConfig, check_mass_plateau

    cfg = AlertConfig(plateau_window=20, plateau_min_gain=0.0)
    growing = [float(i) for i in range(100)]
    assert check_mass_plateau(growing, cfg) is None
    flat = [float(i) for i in range(60)] + [60.0] * 40
    alert = check_mass_plateau(flat, cfg)
    assert alert is not None and alert.severity == "warning"
    assert check_mass_plateau([5.0] * 30, cfg) is None  # < 2 windows of data


def test_summarize_results_and_check_alerts_end_to_end(tmp_path):
    from train.harness import monitor
    from train.harness.league_seats import ResultsWriter

    league = tmp_path / "league"
    w = ResultsWriter(league, "actor_000")
    for _ in range(3):
        w(_result())
    w.close()

    results = monitor.load_results(league)
    assert len(results) == 3
    s = monitor.summarize_results(results)
    assert s["games"] == 3
    assert s["policy_mean_rank"] == pytest.approx((2 + 3 + 5 + 8) / 4)
    assert s["vs_anchor"]["cautious"] == pytest.approx(2 / 4)  # anchor seat 5, rank 4
    assert s["vs_pool"]["snap_v000010"] == 0.0                 # it won the game
    assert s["seat_share"] == {"anchor": 6, "pool": 6}

    alerts = monitor.check_alerts(
        [{"entropy": 0.9}] * 50, {}, monitor.policy_mass_series(results))
    assert alerts == []


# --------------------------------------------------------------------------- #
# End-to-end: the whole league loop closes                                     #
# --------------------------------------------------------------------------- #


def test_local_collector_league_loop_closes(tmp_path, short_games):
    """demos→fold→snapshot→frozen-seat, through the real collection path.

    In-process (LocalCollector shares GameRunner with the actor processes), so
    the short-game cap applies: games finish, results land in the worker's
    jsonl, the coordinator folds them and snapshots the live policy, and the
    next rollover seats the snapshot as a FrozenPolicyBot.
    """
    torch = pytest.importorskip("torch")
    from train.harness.config import TrainConfig
    from train.harness.league_wire import coordinator_from_config
    from train.harness.rollout_worker import LocalCollector
    from train.model import build_policy
    from train.opponents.frozen import FrozenPolicyBot

    cfg = TrainConfig(
        games_per_worker=2, policy_seats=(0, 1, 2, 3),
        scripted_names=("greedy_food", "cautious"), rollout_len=24,
        league=True, league_dir=str(tmp_path / "league"),
        league_snapshot_interval_s=1800.0,
    )
    cfg.validate()
    torch.manual_seed(cfg.base_seed)
    policy = build_policy(cfg)
    policy.eval()
    collector = LocalCollector(cfg, policy, worker_id=0)
    coord = coordinator_from_config(cfg)

    coord.step(policy, version=0, now=0.0)  # bootstrap snapshot (pool empty)
    assert len(coord.pool.entries) == 1

    for _ in range(3):  # ≥ one 20-round game per stream at rollout 24
        collector.collect_rollout()

    results_file = tmp_path / "league" / "results" / "worker_000.jsonl"
    assert results_file.exists() and results_file.stat().st_size > 0

    metrics = coord.step(policy, version=1, now=10.0)
    assert metrics["league_games_folded"] >= 2  # both streams rolled over
    assert metrics["league_games_total"] == metrics["league_games_folded"]
    assert "league_policy_rank" in metrics

    # The snapshot is being probed: either it already took seats (recorded
    # games) or the next games seat it now that the registry is hot.
    seated = any(e.games > 0 for e in coord.pool.entries.values())
    if not seated:
        for runner in collector.runners:
            bots, _plan = runner.seat_sampler.sample(99, 0, (4, 5, 6, 7))
            if any(isinstance(b, FrozenPolicyBot) for b in bots.values()):
                seated = True
                break
    assert seated


def test_run_manager_league_run_and_resume(tmp_path):
    """RunManager + real actor processes with league on: the pool bootstraps,
    metrics carry the league scalars, and a resume neither re-hashes nor
    re-snapshots (interval gate persisted in league.json)."""
    pytest.importorskip("torch")
    from train.harness.run import RunConfig, run_from_config

    run_cfg = RunConfig.from_dict({
        "name": "league_smoke", "out_dir": str(tmp_path / "runs"),
        "total_updates": 2, "checkpoint_interval_s": 10_000.0,
        "device": "cpu",
        "train": {
            "num_actors": 1, "policy_seats": [0, 1, 2, 3],
            "scripted_names": ["greedy_food", "cautious"],
            "rollout_len": 2, "league": True,
        },
    })
    out = run_from_config(run_cfg)
    assert out["final_version"] == 2

    root = tmp_path / "runs" / "league_smoke"
    league = root / "league"
    assert (league / "league.json").exists()          # defaulted under the run dir
    reg = json.loads((league / "league.json").read_text())
    assert len(reg["entries"]) == 1                   # bootstrap snapshot only
    snaps = list(league.glob("snap_v*.npz"))
    assert len(snaps) == 1

    recs = [json.loads(l) for l in (root / "metrics.jsonl").read_text().splitlines()]
    assert all("league_pool_size" in r for r in recs)
    assert recs[-1]["league_pool_size"] == 1.0

    # Resume: same config re-invoked continues (hash equality) and the pool
    # neither double-bootstraps nor forgets its clock.
    run_cfg.total_updates = 4
    out2 = run_from_config(run_cfg)
    assert out2["final_version"] == 4
    reg2 = json.loads((league / "league.json").read_text())
    assert set(reg2["entries"]) == set(reg["entries"])
