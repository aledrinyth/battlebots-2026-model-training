"""Instantiate hello.wasm and expose `add` (gate G0 / P0.3).

Kept dependency-light (wasmtime only) so it can run inside a match submission
before the pipe loop opens. `load_add` returns the callable plus the wall-clock
instantiation cost, which is the number gate G0 is measured against.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

import wasmtime

DEFAULT_WASM = Path(__file__).parent / "hello.wasm"


def load_add(wasm_path: Path = DEFAULT_WASM) -> tuple[Callable[[int, int], int], float]:
    """Load `hello.wasm` and return (add, instantiation_ms).

    Instantiation here means the full cold cost that a real bot pays at startup:
    Engine construction + module compile (the dominant term) + Instance wiring.
    """
    start = perf_counter()
    engine = wasmtime.Engine()
    store = wasmtime.Store(engine)
    module = wasmtime.Module(engine, wasm_path.read_bytes())
    instance = wasmtime.Instance(store, module, [])
    raw_add = instance.exports(store)["add"]
    elapsed_ms = (perf_counter() - start) * 1000.0

    def add(a: int, b: int) -> int:
        return raw_add(store, a, b)

    return add, elapsed_ms
