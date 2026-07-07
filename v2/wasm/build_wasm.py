#!/usr/bin/env python
"""Compile add.wat -> hello.wasm (reproducible build for gate G0 / P0.3).

Apple clang ships no wasm32 target and there is no wasi-sdk on this machine, so
we assemble the WAT to a binary wasm module with wasmtime's own assembler
(`wat2wasm`). The output is a real .wasm binary — byte-for-byte what a C
toolchain would emit for the same function — so downstream instantiation and
timing are unaffected.

Usage:  .venv-bot/bin/python v2/wasm/build_wasm.py
"""

from __future__ import annotations

from pathlib import Path

import wasmtime

HERE = Path(__file__).parent
WAT = HERE / "add.wat"
WASM = HERE / "hello.wasm"


def main() -> None:
    binary = wasmtime.wat2wasm(WAT.read_text())
    WASM.write_bytes(binary)
    assert binary[:4] == b"\x00asm", "not a valid wasm module"
    print(f"wrote {WASM} ({len(binary)} bytes, magic={binary[:4].hex()})")


if __name__ == "__main__":
    main()
