"""P1.4 smoke coverage: the benchmark tool runs and the report exists.

The throughput path (a) is fast and subprocess-free — smoke-tested in CI. The
engine-tax path (b) spawns eight real bot subprocesses over pipes; it is
`slow`-marked (opt-in) to keep CI fast and non-flaky. The authoritative numbers
live in `notes/benchmarks.md`, regenerated with `tools/bench_env.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import bench_env  # noqa: E402  (path set above)


def test_throughput_smoke() -> None:
    stats = bench_env.bench_throughput(rounds=20)
    for key in ("physics_only_rps", "step_fast_rps", "step_full_rps", "cores_detected"):
        assert key in stats
    assert stats["physics_only_rps"] > 0
    assert stats["step_fast_rps"] > 0
    # physics-only is the ceiling; obs-building only slows it down
    assert stats["physics_only_rps"] >= stats["step_fast_rps"]
    assert stats["cores_detected"] >= 1


def test_budget_derivation_math() -> None:
    # 2 ms engine tax → (8*0.75 - 1400*0.002)/1400 s ≈ 2.286 ms bot-side/turn
    b = bench_env.derive_budget(0.002)
    assert b["cumulative_budget_s"] == 8
    assert abs(b["usable_s"] - 6.0) < 1e-9
    assert abs(b["bot_side_per_turn_ms"] - ((6.0 - 1400 * 0.002) / 1400 * 1e3)) < 1e-9


def test_report_present_and_well_formed() -> None:
    report = REPO / "notes" / "benchmarks.md"
    assert report.exists(), "run tools/bench_env.py to generate notes/benchmarks.md"
    text = report.read_text()
    assert "In-process training throughput" in text
    assert "Match-clock decomposition" in text
    assert "Bot-side per-turn average budget" in text


@pytest.mark.slow
def test_engine_tax_runs_end_to_end() -> None:
    """Opt-in: the real-pipe engine-tax measurement produces sane per-turn numbers."""
    stats = bench_env.bench_engine_tax(tax_rounds=60)
    assert stats["turns_measured"] > 0
    assert stats["mean_s"] > 0
    assert stats["max_s"] >= stats["p50_s"] > 0
