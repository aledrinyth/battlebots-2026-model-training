;; add.wat — the trivial module behind gate G0 (P0.3).
;;
;; Semantically identical to the C the playbook calls for:
;;
;;     int add(int a, int b) { return a + b; }
;;
;; A freestanding integer add needs no WASI imports, so no wasi-sdk is required
;; to answer the only question G0 asks: does wasmtime instantiate a module fast
;; enough (<500 ms) to fit inside the 3 s pipe-open window? The instantiation
;; cost measured against this module is representative of any small C core —
;; module compile time dominates and is independent of source language.
;;
;; If/when a wasi-sdk toolchain is available, drop in a C-compiled hello.wasm
;; and the probe/test below work unchanged (they only rely on the `add` export).
(module
  (func (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add))
