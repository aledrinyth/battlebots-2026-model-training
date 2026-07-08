"""P1.3 acceptance: the training wrapper **is** the competition environment.

Ground truth is produced by `tools/record_reference_match.py`, which runs the
engine's own `GameEngine` — seeded — against eight **real bot subprocesses over
real named pipes** (the exact grader wiring; only the 0.1 s/round sleep is
removed). Because `WrapperEnv` drives that same `StateMutator.commit_round` and
issues an identical `random` call sequence, replaying a recorded game's moves
through `WrapperEnv.reset(seed)` must reproduce every recorded per-round snapshot
**bit-for-bit** (spawns, splits, eats, deaths, respawns, regen all align via the
shared seed — no state injection needed).

Committed traces: `tests/fixtures/parity/trace_{0..4}.json.gz` — five full
1400-round games with the cautious/dont_move reference lineup (seed 0's game
already includes real deaths). The default CI test checks a bounded prefix of all
five (fast); the `slow` test replays all five in full (the literal ≥5-full-games
acceptance).

Regenerate after a vendored-engine pin bump:
    for s in 0 1 2 3 4; do \
      .venv-bot/bin/python tools/record_reference_match.py \
        --seed $s --out tests/fixtures/parity/trace_$s.json.gz; done
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from lib.interface.events.moves.move_player import MovePlayer
from lib.models.penguin_model import DirectionModel

from train.env.wrapper import WrapperEnv

FIXTURES = Path(__file__).parent / "fixtures" / "parity"
TRACES = sorted(FIXTURES.glob("trace_*.json.gz"))

# Bit-exact is expected (identical code + identical RNG stream); the tiny
# tolerance only guards against a stray last-ulp libm difference and is far
# tighter than the 1e-9 the acceptance allows.
POS_ATOL = 1e-9

CI_PREFIX_ROUNDS = 150  # default (fast) test replays this many rounds of each game


def _load(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode())


def _build_move(entry) -> MovePlayer:
    pid, dirrepr, split = entry
    if dirrepr["degrees"] is not None:
        direction = DirectionModel(degrees=dirrepr["degrees"])
    else:
        direction = DirectionModel(x=dirrepr["x"], y=dirrepr["y"])
    return MovePlayer(player_id=pid, direction=direction, split=split)


def _replay_and_compare(trace: dict, max_rounds: int | None) -> dict:
    """Replay a recorded trace through the wrapper; return a stats dict. Raises on
    any semantic divergence (blob-id set, alive/dead, cooldown) and on any float
    beyond POS_ATOL."""
    env = WrapperEnv(seed=trace["seed"], emit_update=False)
    obs = env.reset()

    assert env.turn_order == trace["turn_order"], (
        f"turn_order diverged: {env.turn_order} vs {trace['turn_order']}"
    )

    rounds = trace["rounds"]
    if max_rounds is not None:
        rounds = rounds[:max_rounds]

    max_abs = 0.0
    compared_blobs = 0
    for r, rd in enumerate(rounds):
        # The set of players the engine queried (alive) must match the wrapper's.
        trace_movers = sorted(m[0] for m in rd["moves"])
        assert sorted(obs.keys()) == trace_movers, (
            f"round {r}: alive set diverged wrapper={sorted(obs)} trace={trace_movers}"
        )
        moves = {m[0]: _build_move(m) for m in rd["moves"]}
        obs, done = env.step(moves)

        for pid_str, snap in rd["snaps"].items():
            pid = int(pid_str)
            player = env.state.players[pid]
            assert player.alive == snap["alive"], (
                f"round {r} p{pid}: alive {player.alive} vs recorded {snap['alive']}"
            )
            wrap_blobs = {
                b.blob_id: (b.x, b.y, b.radius, b.merge_cooldown)
                for b in player.blobs.values()
            }
            snap_blobs = {b[0]: (b[1], b[2], b[3], b[4]) for b in snap["blobs"]}
            assert set(wrap_blobs) == set(snap_blobs), (
                f"round {r} p{pid}: blob ids {sorted(wrap_blobs)} vs {sorted(snap_blobs)}"
            )
            for bid, (wx, wy, wr, wcd) in wrap_blobs.items():
                sx, sy, sr, scd = snap_blobs[bid]
                assert wcd == scd, (
                    f"round {r} p{pid} b{bid}: merge_cooldown {wcd} vs {scd}"
                )
                for a, b in ((wx, sx), (wy, sy), (wr, sr)):
                    d = abs(a - b)
                    max_abs = max(max_abs, d)
                    assert d <= POS_ATOL, (
                        f"round {r} p{pid} b{bid}: float diverged by {d:.2e} (> {POS_ATOL:.0e})"
                    )
                compared_blobs += 1

        if done:
            break

    return {"rounds": r + 1, "compared_blobs": compared_blobs, "max_abs": max_abs}


def test_parity_fixtures_present() -> None:
    assert TRACES, (
        "no parity traces found — run tools/record_reference_match.py "
        "(see tests/test_wrapper_parity.py docstring)"
    )
    assert len(TRACES) >= 5, f"acceptance needs >=5 games; found {len(TRACES)}"


@pytest.mark.skipif(not TRACES, reason="no parity traces recorded")
@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_wrapper_matches_runner_prefix(path: Path) -> None:
    """Default CI: bit-exact parity over the first CI_PREFIX_ROUNDS rounds."""
    stats = _replay_and_compare(_load(path), max_rounds=CI_PREFIX_ROUNDS)
    assert stats["compared_blobs"] > 0
    assert stats["max_abs"] <= POS_ATOL


@pytest.mark.slow
@pytest.mark.skipif(not TRACES, reason="no parity traces recorded")
@pytest.mark.parametrize("path", TRACES, ids=lambda p: p.stem)
def test_wrapper_matches_runner_full_game(path: Path) -> None:
    """Literal acceptance: bit-exact parity over the full 1400-round game."""
    stats = _replay_and_compare(_load(path), max_rounds=None)
    assert stats["rounds"] >= 1000, f"trace unexpectedly short: {stats['rounds']} rounds"
    assert stats["max_abs"] <= POS_ATOL
