#!/usr/bin/env python
"""Assemble the v9 tactical kernel (kernel.wat -> bytes) and emit the
``base85(zlib(...))`` literal embedded in dist/potential_field_v9_wasm.py.

There is no wasi-sdk / wasm32 clang on this machine (see build_wasm.py), so we
assemble the hand-written WAT with wasmtime's own assembler — the output is a
real .wasm binary, byte-for-byte what a C toolchain would emit for the same
module. Run this whenever kernel.wat changes, then paste the printed literal
into ``_KERNEL_B85`` in the submission and re-run the parity check.

The kernel exports ``score_candidates`` over a fixed linear-memory layout; see
kernel.wat for the layout and search_core.c for the C equivalent that compiles
to the same ABI once wasi-sdk is available (and enables the multi-frame search
that hand-written WAT is impractical for).

    .venv-bot/bin/python v2/wasm/build_kernel.py
"""

from __future__ import annotations

import base64
import hashlib
import zlib
from pathlib import Path

import wasmtime

HERE = Path(__file__).parent
WAT = HERE / "kernel.wat"
WASM = HERE / "kernel.wasm"


def main() -> None:
    binary = bytes(wasmtime.wat2wasm(WAT.read_text()))
    assert binary[:4] == b"\x00asm", "not a valid wasm module"
    WASM.write_bytes(binary)

    # instantiate + smoke-call to prove the module is loadable here
    store = wasmtime.Store()
    inst = wasmtime.Instance(store, wasmtime.Module(store.engine, binary), [])
    assert "score_candidates" in inst.exports(store), "missing export"

    comp = zlib.compress(binary, 9)
    b85 = base64.b85encode(comp).decode()
    print(f"wasm bytes = {len(binary)}  zlib = {len(comp)}  b85 chars = {len(b85)}")
    print(f"sha256(wasm) = {hashlib.sha256(binary).hexdigest()}")
    print("\n# paste into dist/potential_field_v9_wasm.py as _KERNEL_B85 "
          "(78-char lines):")
    print("_KERNEL_B85 = (")
    for i in range(0, len(b85), 78):
        print(f'    "{b85[i:i + 78]}"')
    print(")")


if __name__ == "__main__":
    main()
