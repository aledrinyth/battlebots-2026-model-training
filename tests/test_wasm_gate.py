"""Gate G0 (P0.3): wasmtime instantiation is fast enough for the pipe-open window.

Threshold (from the playbook): a cold instantiate must land under **500 ms** so
it fits inside the engine's 3 s pipe-open alarm alongside the bot's imports.
Fail ⇒ v2 Phases 9–10 (the C→WASM core) are cancelled and v1 is the plan.

Skips cleanly if wasmtime isn't installed (it's a v2-only dependency).
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2" / "wasm"))

wasmtime = pytest.importorskip("wasmtime", reason="wasmtime is a v2-only dependency")
from wasm_core import DEFAULT_WASM, load_add  # noqa: E402

GATE_MS = 500.0  # hard budget from P0.3
COLD_RUNS = 10


def test_hello_wasm_exists_and_is_valid() -> None:
    assert DEFAULT_WASM.exists(), "run v2/wasm/build_wasm.py to produce hello.wasm"
    assert DEFAULT_WASM.read_bytes()[:4] == b"\x00asm"


def test_add_export_is_correct() -> None:
    add, _ = load_add()
    assert add(2, 3) == 5
    assert add(0, 0) == 0
    assert add(-1, 1) == 0
    assert add(1000, 337) == 1337


def test_cold_instantiation_under_gate() -> None:
    # Each load_add() builds a fresh Engine + Module + Instance = a cold start,
    # exactly what the bot pays once at launch.
    timings = [load_add()[1] for _ in range(COLD_RUNS)]
    worst = max(timings)
    median = statistics.median(timings)
    print(
        f"\n[G0] cold instantiation over {COLD_RUNS} runs: "
        f"median={median:.3f} ms worst={worst:.3f} ms (gate {GATE_MS} ms)"
    )
    # Assert on the worst case with the full gate; it should clear it by orders
    # of magnitude for a module this small.
    assert worst < GATE_MS, f"instantiation {worst:.1f} ms exceeds G0 budget {GATE_MS} ms"
