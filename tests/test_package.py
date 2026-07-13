"""P5.5 acceptance: the packaged bundle runs from a clean directory.

* the packager builds the documented layout, the scan gate holds, and the
  registry records provenance;
* the bundle — from a directory outside the repo — plays a real-runner game
  through the compliance rig with zero flags (the packaged bot resolves its
  own weights, imports only bundle-local + whitelisted modules);
* bad weights fail at package time, never on the grader.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("engine")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import compliance  # noqa: E402
import package  # noqa: E402

FILLER = "tests/fixtures/bots/dont_move.py"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("dist")
    entry = package.package("test_bundle", package.REPO / "tests" / "fixtures"
                            / "adversarial" / "weights_random.npz", out)
    return {"entry": entry, "root": out / "test_bundle", "out": out}


def test_bundle_layout_and_registry(bundle) -> None:
    root = bundle["root"]
    assert (root / "main.py").exists()
    assert (root / "bot" / "main.py").exists()
    assert (root / "bot" / "weights.npz").exists()
    assert (root / "bot" / "adapters" / "act_v1.py").exists()
    assert not list(root.rglob("__pycache__"))
    assert not list(root.rglob("*.pt"))

    registry = json.loads((bundle["out"] / "registry.json").read_text())
    entry = registry[-1]
    assert entry["name"] == "test_bundle"
    assert len(entry["weights_sha256"]) == 64
    assert entry["manifest"]["obs_version"] == "obs_v1"
    assert entry["git_commit"]


def test_bundle_scan_gate_holds_on_output(bundle) -> None:
    import bundle_scan
    assert bundle_scan.scan_bundle(bundle["root"])["ok"]


def test_bundle_plays_through_the_real_runner_from_a_clean_dir(bundle) -> None:
    """The playbook's acceptance: bundle + runner, nothing from the repo tree."""
    report = compliance.run_compliance(
        [f"1:{bundle['root'] / 'main.py'}", f"7:{FILLER}"],
        rounds=60, seed=0, start_delay=1.0,
    )
    assert report["ok"], report["flags"]
    game = report["games"][0]
    assert game["engine_result"]["result_type"] == "SUCCESS"
    assert game["players"][0]["turns"] == 60
    assert game["players"][0]["flags"] == []


def test_bad_weights_fail_at_package_time(bundle, tmp_path) -> None:
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not an npz")
    with pytest.raises(Exception):
        package.package("bad_bundle", bad, tmp_path / "dist")
    assert not (tmp_path / "dist" / "bad_bundle").exists()
