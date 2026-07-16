"""R0 parity — THE gate (impl doc §7.2, CI-blocking).

Records a seeded in-process match through ``WrapperEnv`` (P1.3-proven equal
to the real seeded pipe-runner engine; its ``_build_query`` mirrors the live
v2026.1.14 ``PlayerConnection`` payload), featurizes every live query, dumps
the engine's own ``private_event_history`` in the tournament JSON encoding,
reconverts it, and asserts every ``CanonicalObs`` **bit-identical** per seat
per round — across deaths, respawns, viruses, and splits.

Also proves the gate *can* fail (a perturbed snapshot is caught) and that
reconstruction does not depend on the fast-vision patch (pristine getters
produce the same bits).
"""

from __future__ import annotations

import pytest

from train.env import fast_vision
from train.replay.parity import (
    record_wrapper_match,
    run_parity_gate,
    verify_parity,
)


@pytest.fixture(autouse=True)
def _restore_fast_vision():
    yield
    # other suites (and WrapperEnv itself) expect the patch installed
    fast_vision.install()


def test_parity_full_game_bit_identical():
    report = run_parity_gate(seed=0)
    assert report.passed, report.mismatches[:5]
    assert report.frames > 8000            # a full 1400-round, 8-seat game
    assert report.deaths > 0               # predators did their job
    assert report.resets > 8               # first-frames + real respawns
    assert report.summary().startswith("parity PASS")


def test_parity_second_seed():
    report = run_parity_gate(seed=7)
    assert report.passed, report.mismatches[:5]
    assert report.deaths > 0 and report.resets > 8


def test_parity_survives_pristine_getters():
    """Reconstruction must not silently depend on the fast-vision patch:
    record with the wrapper (patch installed), then reconvert against the
    engine's original getters."""
    capture = record_wrapper_match(seed=3, rounds=240)
    fast_vision.uninstall()
    report = verify_parity(capture)
    assert report.passed, report.mismatches[:5]
    assert report.frames > 1000


def test_parity_gate_actually_detects_corruption():
    """Test the test: nudging one recorded snapshot position must fail the
    bit-compare (a gate that cannot fail proves nothing)."""
    capture = record_wrapper_match(seed=5, rounds=60)
    for e in capture.events:
        if e["event_type"] == "event_player_moved" and e["alive"] and e["blobs"]:
            e["blobs"][0]["pos"][0] += 1e-3
            break
    report = verify_parity(capture)
    assert not report.passed
    assert any("differs" in m for m in report.mismatches)


@pytest.mark.slow
def test_parity_many_seeds_slow():
    for seed in range(10):
        report = run_parity_gate(seed=seed)
        assert report.passed, (seed, report.mismatches[:3])
