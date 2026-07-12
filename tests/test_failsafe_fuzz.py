"""P4.3 acceptance (fuzz half): malformed observations never crash the bot.

Every fuzz case runs the *real* per-turn pipeline — kit pydantic parse →
``bot.featurizer`` → ``bot.inference`` forward → ``ActionAdapter`` →
serialize — wrapped in ``bot.safety.safe_move`` exactly as P5.4's main loop
will. Acceptance: **zero uncaught exceptions and 100 % valid moves** (every
emitted payload passes the kit-validator mirror in ``bundle_scan``).

Mutation corpus: the P4.2 scene bank's real wire payloads, deformed —
missing fields, nulls, wrong types, truncation, zero entities, zero
``vision_size``, non-finite floats, absurd magnitudes — plus raw garbage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("engine")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import bundle_scan  # noqa: E402
import scene_bank  # noqa: E402

from bot.adapters.act_v1 import ActionAdapter  # noqa: E402
from bot.featurizer import Featurizer  # noqa: E402
from bot.inference import NumpyPolicy  # noqa: E402
from bot.safety import safe_move, zero_move, flee_to_center_move  # noqa: E402

PLAYER_ID = 0


# --- the pipeline under test (P5.4's per-turn body) -----------------------------

@pytest.fixture(scope="module")
def policy() -> NumpyPolicy:
    return NumpyPolicy.load(str(scene_bank.WEIGHTS_PATH))


@pytest.fixture(scope="module")
def base_frames() -> list[str]:
    bank = scene_bank.load_bank()
    return [f for s in bank["scenes"] for f in s["frames"]]


def run_turn(raw: str, policy: NumpyPolicy) -> str:
    """One full guarded turn: raw wire bytes in → move payload out."""
    from lib.interface.queries.query_move import QueryMovePlayer

    featurizer = Featurizer()
    hidden = policy.initial_hidden()

    query = None
    try:
        query = QueryMovePlayer.model_validate_json(raw)
    except Exception:
        pass  # safe_move falls back to zero_move without a query

    def compute():
        if query is None:
            raise ValueError("unparseable query")
        obs = featurizer.featurize(query)
        action, _ = policy.act(obs, hidden)
        return ActionAdapter.to_move(PLAYER_ID, action)

    return safe_move(compute, PLAYER_ID, query=query).model_dump_json()


# --- mutation corpus -------------------------------------------------------------

def _mutations(raw: str) -> list[tuple[str, str]]:
    """(label, payload) deformations of one real frame."""
    d = json.loads(raw)
    out: list[tuple[str, str]] = []

    def variant(label: str, obj) -> None:
        out.append((label, json.dumps(obj)))

    for field in ("you", "round", "view_center", "vision_size",
                  "visible_food", "visible_blobs", "visible_viruses", "update"):
        m = dict(d)
        m.pop(field, None)
        variant(f"missing_{field}", m)

    m = dict(d); m["visible_food"] = []; m["visible_blobs"] = []; m["visible_viruses"] = []
    variant("zero_entities", m)

    m = dict(d); m["vision_size"] = 0.0
    variant("zero_vision_size", m)

    m = dict(d); m["vision_size"] = None
    variant("null_vision_size", m)

    m = dict(d); m["you"] = None
    variant("null_you", m)

    m = json.loads(raw); m["you"]["blobs"] = []
    variant("no_own_blobs", m)

    m = dict(d); m["round"] = 10**9
    variant("huge_round", m)

    m = dict(d); m["round"] = "not_a_number"
    variant("round_wrong_type", m)

    m = json.loads(raw)
    if m["visible_food"]:
        m["visible_food"][0]["pos"] = [1e308, -1e308]
        variant("absurd_food_pos", m)

    # Non-finite floats: json.dumps would need allow_nan; hand-splice instead
    # (the engine's FiniteBaseModel must reject these at parse).
    spliced = raw.replace('"vision_size":', '"vision_size":NaN,"_x":', 1)
    out.append(("nan_vision_size", spliced))

    out.append(("truncated", raw[: len(raw) // 2]))
    return out


RAW_GARBAGE = [
    ("empty_string", ""),
    ("not_json", "hello, engine"),
    ("json_scalar", "42"),
    ("json_empty_object", "{}"),
    ("json_array", "[1,2,3]"),
    ("null", "null"),
    ("binaryish", "\x00\x01\x02\xff"),
]


# --- the acceptance --------------------------------------------------------------

def test_clean_frames_all_emit_valid_moves(base_frames, policy) -> None:
    for raw in base_frames:
        payload = run_turn(raw, policy)
        bundle_scan.validate_emitted_move(payload, PLAYER_ID)


def test_fuzzed_frames_never_crash_and_always_answer(base_frames, policy) -> None:
    cases = 0
    for raw in base_frames:
        for label, payload_in in _mutations(raw):
            payload_out = run_turn(payload_in, policy)  # must not raise
            bundle_scan.validate_emitted_move(payload_out, PLAYER_ID)
            cases += 1
    assert cases >= 200, f"fuzz corpus unexpectedly small: {cases}"


def test_raw_garbage_never_crashes(policy) -> None:
    for label, payload_in in RAW_GARBAGE:
        payload_out = run_turn(payload_in, policy)
        bundle_scan.validate_emitted_move(payload_out, PLAYER_ID)


# --- fallback semantics -----------------------------------------------------------

def test_fallback_is_zero_move_without_a_query() -> None:
    payload = run_turn("not json at all", NumpyPolicy.load(str(scene_bank.WEIGHTS_PATH)))
    move = json.loads(payload)
    assert move["direction"]["x"] == 0.0 and move["direction"]["y"] == 0.0


def test_fallback_flees_to_center_with_a_query(base_frames) -> None:
    from lib.interface.queries.query_move import QueryMovePlayer

    query = QueryMovePlayer.model_validate_json(base_frames[0])

    def exploding_compute():
        raise RuntimeError("forward pass exploded")

    move = safe_move(exploding_compute, PLAYER_ID, query=query)
    expect = flee_to_center_move(query, PLAYER_ID)
    assert (move.direction.x, move.direction.y) == (expect.direction.x, expect.direction.y)
    bundle_scan.validate_emitted_move(move.model_dump_json(), PLAYER_ID)


def test_safe_move_rejects_wrong_player_id_result() -> None:
    bad = zero_move(PLAYER_ID + 1)
    move = safe_move(lambda: bad, PLAYER_ID)
    assert move.player_id == PLAYER_ID  # converted to fallback, not passed through
