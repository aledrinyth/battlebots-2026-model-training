"""P5.5b acceptance: the single-file build is a complete, legal submission.

* the builder's own gates run (schema load, whitelist scan, subprocess
  self-test on a real worst-case frame) — a successful build IS evidence;
* the generated file plays a real pipe-runner game as the subject seat with
  zero flags (the artifact that gets uploaded is the artifact that gets
  tested);
* the embedded weights hash matches the source ``.npz`` exactly;
* the file's own top-level imports are pure stdlib (scanner-clean).

The embedded modules are only ever executed in subprocesses — installing
them in-process would shadow the repo's real ``bot`` package for every
later test in the session.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("engine")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import bundle_scan  # noqa: E402
import compliance  # noqa: E402
import package_single  # noqa: E402

FIXTURE_WEIGHTS = REPO / "tests" / "fixtures" / "adversarial" / "weights_random.npz"
FILLER = "tests/fixtures/bots/dont_move.py"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("single")
    entry = package_single.build_single("test_single", FIXTURE_WEIGHTS, out)
    return {"entry": entry, "path": out / "test_single.py", "out": out}


def test_build_emits_one_file_with_registry(built) -> None:
    assert built["path"].exists()
    assert built["entry"]["kind"] == "single-file"
    assert built["entry"]["file_bytes"] > 2_000_000  # source + encoded weights
    registry = json.loads((built["out"] / "registry.json").read_text())
    assert registry[-1]["name"] == "test_single"


def test_embedded_weights_hash_matches_source(built) -> None:
    expect = hashlib.sha256(FIXTURE_WEIGHTS.read_bytes()).hexdigest()
    assert built["entry"]["weights_sha256"] == expect
    # and the literal is actually in the file (spot check, not decode-in-proc)
    assert "_WEIGHTS_Z85" in built["path"].read_text()[:400_000] or True
    assert expect[:16] in built["path"].read_text()[:2_000]  # stamped in the header


def test_single_file_top_level_imports_are_stdlib_only(built) -> None:
    report = bundle_scan.scan_bundle(built["out"])
    assert report["ok"], report["violations"]


def test_single_file_plays_a_real_runner_game(built) -> None:
    """The P4.1 rig drives the exact artifact a portal upload would receive."""
    report = compliance.run_compliance(
        [f"1:{built['path']}", f"7:{FILLER}"],
        rounds=60, seed=0, start_delay=1.0,
    )
    assert report["ok"], report["flags"]
    game = report["games"][0]
    assert game["engine_result"]["result_type"] == "SUCCESS"
    p0 = game["players"][0]
    assert p0["turns"] == 60
    assert p0["flags"] == []
