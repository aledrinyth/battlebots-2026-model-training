"""P2.3 acceptance: no emitted move can ever trigger ``InvalidMoveException``.

The exact ban path a submission faces (``player_connection._query_move``):

    move.model_dump_json()  →  MovePlayer.model_validate_json(...)  →  MoveValidator.validate(...)

so this suite round-trips every candidate move through JSON *and* the kit's real
``MoveValidator`` against a live game state, exactly as the grader would.

Acceptance:
* all 24 heading bins × split ∈ {0, 1} validate;
* 10 000 random logits decode to 100 % valid moves;
* serialized message length ≪ 4096 chars (the wire cap).
"""

from __future__ import annotations

import numpy as np
import pytest

from bot.adapters.act_v1 import ActionAdapter, CanonicalAction, HEADING_BINS, decode_logits

from engine.interface.io.input_validator import MoveValidator
from lib.interface.events.moves.move_player import MovePlayer

from train.env.wrapper import WrapperEnv

# The engine's wire cap for a message the bot sends (io_config.MAX_CHARACTERS_READ).
MAX_CHARACTERS_READ = 4096


@pytest.fixture(scope="module")
def live_query():
    """A real alive-player query + a validator bound to its game state."""
    env = WrapperEnv(seed=7)
    obs = env.reset()
    pid = next(iter(obs))
    return env, pid, obs[pid], MoveValidator(env.state)


def _assert_ban_safe(move: MovePlayer, query, validator: MoveValidator, pid: int) -> int:
    """Round-trip a move exactly as the engine does; return its wire length.

    Raises if the move would be rejected (parse error or validator error).
    """
    payload = move.model_dump_json()
    parsed = MovePlayer.model_validate_json(payload)   # engine-side parse
    validator.validate(parsed, query, pid)             # engine-side validation
    return len(payload)


def test_all_bins_x_split_validate(live_query):
    _env, pid, query, validator = live_query
    for b in range(HEADING_BINS):
        for split in (False, True):
            move = ActionAdapter.to_move(pid, CanonicalAction(heading_bin=b, split=split))
            length = _assert_ban_safe(move, query, validator, pid)
            assert length < MAX_CHARACTERS_READ


def test_fuzz_random_logits_all_valid(live_query):
    _env, pid, query, validator = live_query
    rng = np.random.default_rng(20240705)
    max_len = 0
    N = 10_000
    for _ in range(N):
        heading_logits = rng.normal(0.0, 5.0, size=HEADING_BINS).astype(np.float32)
        split_logit = float(rng.normal(0.0, 5.0))
        move = ActionAdapter.logits_to_move(pid, heading_logits, split_logit)
        max_len = max(max_len, _assert_ban_safe(move, query, validator, pid))
    # 100% valid == we reached here without an exception across all N.
    assert max_len < MAX_CHARACTERS_READ
    # a MovePlayer is tiny — sanity that we're nowhere near the cap
    assert max_len < 200


def test_nan_inf_logits_still_emit_finite_valid_move(live_query):
    """Engine v2026.1.10's ``FiniteBaseModel(allow_inf_nan=False)`` rejects any
    heading carrying NaN/Inf. The adapter must never emit one even from a
    degenerate policy output: ``_argmax`` skips NaN (every ``v > best`` is False)
    and the bin table is a finite ``cos``/``sin`` constant, so the emitted
    ``DirectionModel`` is always finite and passes the engine's parse+validate."""
    import math

    _env, pid, query, validator = live_query

    logits = [float("nan")] * HEADING_BINS
    logits[5] = 1.0  # a single finite winner
    move = ActionAdapter.logits_to_move(pid, logits, split_logit=float("nan"))
    assert math.isfinite(move.direction.x) and math.isfinite(move.direction.y)
    _assert_ban_safe(move, query, validator, pid)  # engine parse+validate accepts it

    # even all-NaN logits + an Inf split logit fall back to a finite bin, never crash
    degenerate = ActionAdapter.logits_to_move(
        pid, [float("nan")] * HEADING_BINS, split_logit=float("inf")
    )
    assert math.isfinite(degenerate.direction.x) and math.isfinite(degenerate.direction.y)
    _assert_ban_safe(degenerate, query, validator, pid)


def test_message_length_far_under_cap(live_query):
    _env, pid, query, _validator = live_query
    move = ActionAdapter.to_move(pid, CanonicalAction(heading_bin=13, split=True))
    assert len(move.model_dump_json()) < 200


# --- decode_logits unit behavior -------------------------------------------

def test_decode_logits_argmax_and_split_threshold():
    logits = [0.0] * HEADING_BINS
    logits[7] = 9.0
    assert decode_logits(logits, split_logit=1.0).heading_bin == 7
    assert decode_logits(logits, split_logit=1.0).split is True
    assert decode_logits(logits, split_logit=-1.0).split is False
    # custom threshold
    assert decode_logits(logits, split_logit=0.3, split_threshold=0.5).split is False


def test_decode_logits_argmax_ties_lowest_index():
    logits = [0.0] * HEADING_BINS
    logits[3] = logits[9] = 5.0  # tie
    assert decode_logits(logits, split_logit=0.0).heading_bin == 3


def test_decode_logits_wrong_width_raises():
    with pytest.raises(ValueError):
        decode_logits([0.0] * (HEADING_BINS - 1), split_logit=0.0)


def test_decode_logits_accepts_numpy():
    logits = np.full(HEADING_BINS, -1.0, dtype=np.float32)
    logits[19] = 2.0
    act = decode_logits(logits, split_logit=np.float32(0.7))
    assert act.heading_bin == 19 and act.split is True
