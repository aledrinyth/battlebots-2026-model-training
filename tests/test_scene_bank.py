"""P4.2 acceptance: adversarial scene bank content + bot-path timing gates.

The committed bank (``tests/fixtures/adversarial/scenes.json.gz``) holds the
exact wire payloads the engine would send on engineered worst-case frames.
The replay gate runs each through the bot's full match-time path
(parse → featurize → forward → serialize) against thresholds read from
``notes/benchmarks.md`` (per-turn budget) and the plan targets (P99 < 3 ms,
max < 100 ms, forward ≤ 1 ms — P5.2's gate formalized).

Weights: ``weights_random.npz`` — a seeded random-init export of the
*production* architecture (timing is identical to trained weights; no
training artifact needed in CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("engine")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import scene_bank  # noqa: E402  (path set above)

EXPECTED_SCENES = {
    "max_entities", "empty_vision", "respawn", "final_round",
    "organic_early", "organic_mid", "organic_late",
}


@pytest.fixture(scope="module")
def bank() -> dict:
    return scene_bank.load_bank()


@pytest.fixture(scope="module")
def report(bank: dict) -> dict:
    return scene_bank.replay_bank(bank, scene_bank.WEIGHTS_PATH)


# --- bank content -------------------------------------------------------------

def test_bank_has_every_scene(bank: dict) -> None:
    names = {s["name"] for s in bank["scenes"]}
    assert names == EXPECTED_SCENES
    for s in bank["scenes"]:
        assert len(s["frames"]) == scene_bank.FRAMES_PER_SCENE


def test_max_entities_frame_is_actually_dense(bank: dict) -> None:
    from lib.config.arena import MAX_BLOB_COUNT
    from lib.interface.queries.query_move import QueryMovePlayer

    scene = next(s for s in bank["scenes"] if s["name"] == "max_entities")
    q = QueryMovePlayer.model_validate_json(scene["frames"][0])
    assert len(q.visible_blobs) >= 80        # 7 enemies split toward 16 blobs each
    assert len(q.visible_food) >= 100        # dense food carpet
    assert len(q.you.blobs) == MAX_BLOB_COUNT
    assert len(q.visible_viruses) >= 1


def test_empty_vision_frame_is_actually_empty(bank: dict) -> None:
    from lib.interface.queries.query_move import QueryMovePlayer

    scene = next(s for s in bank["scenes"] if s["name"] == "empty_vision")
    for raw in scene["frames"]:
        q = QueryMovePlayer.model_validate_json(raw)
        assert not q.visible_food and not q.visible_blobs and not q.visible_viruses
        assert q.you.blobs  # we are alive and alone


def test_respawn_frame_has_round_gap_and_backlog(bank: dict) -> None:
    from lib.interface.queries.query_move import QueryMovePlayer

    scene = next(s for s in bank["scenes"] if s["name"] == "respawn")
    q0 = QueryMovePlayer.model_validate_json(scene["frames"][0])
    # The first post-respawn query: a gap vs the pre-death round and the
    # censored event backlog accumulated while dead.
    assert q0.round - scene["pre_death_round"] > 1
    assert len(q0.update) > 0
    q1 = QueryMovePlayer.model_validate_json(scene["frames"][1])
    assert q1.round == q0.round + 1  # then consecutive again


def test_final_round_frames_reach_the_last_round(bank: dict) -> None:
    from lib.config.arena import MAX_ROUNDS
    from lib.interface.queries.query_move import QueryMovePlayer

    scene = next(s for s in bank["scenes"] if s["name"] == "final_round")
    rounds = [QueryMovePlayer.model_validate_json(f).round for f in scene["frames"]]
    assert rounds == [MAX_ROUNDS - 3, MAX_ROUNDS - 2, MAX_ROUNDS - 1]


# --- timing gates (the P4.2 acceptance) ----------------------------------------

def test_budget_is_read_from_benchmarks_not_hardcoded() -> None:
    budget = scene_bank.read_bot_side_budget_ms()
    assert 0.0 < budget < 8000.0


def test_bot_path_meets_all_timing_gates(report: dict) -> None:
    assert report["ok"], report["gates"]
    assert report["frames_timed"] >= 60  # 7 scenes × 3 frames × 3 recorded reps


def test_forward_pass_meets_p52_gate(report: dict) -> None:
    # P5.2 formalized: NumPy forward ≤ 1 ms/turn on adversarial frames.
    assert report["stages_ms"]["forward"]["p99"] < scene_bank.FORWARD_BUDGET_MS


def test_every_stage_reported(report: dict) -> None:
    for stage in scene_bank.STAGES:
        st = report["stages_ms"][stage]
        assert st["max"] >= st["p99"] >= st["p50"] > 0.0


# --- generator self-checks (engineered states still engineer) ------------------

def test_generator_max_entities_smoke() -> None:
    pytest.importorskip("train.env.wrapper")
    scene = scene_bank.gen_max_entities(seed=7)  # different seed than the bank
    assert len(scene["frames"]) == scene_bank.FRAMES_PER_SCENE


def test_generator_empty_vision_smoke() -> None:
    pytest.importorskip("train.env.wrapper")
    scene = scene_bank.gen_empty_vision(seed=7)
    assert len(scene["frames"]) == scene_bank.FRAMES_PER_SCENE
