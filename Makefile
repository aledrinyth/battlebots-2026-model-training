# Reconstructed stage gate; full historical Makefile arrives at stage 15.
PY ?= python3.12
.PHONY: test
test:
	$(PY) -m pytest tests/test_bench_env.py tests/test_protocol_survival.py tests/test_reference_fixtures.py tests/test_spec_constants.py tests/test_wasm_gate.py tests/test_wrapper.py tests/test_wrapper_parity.py -q
