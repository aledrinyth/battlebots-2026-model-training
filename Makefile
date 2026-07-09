# Reconstructed stage gate; full historical Makefile arrives at stage 15.
PY ?= python3.12
.PHONY: test
test:
	$(PY) -m pytest tests/test_action_adapter.py tests/test_bench_env.py tests/test_fast_vision.py tests/test_featurizer.py tests/test_flat_obs.py tests/test_greedy_bot.py tests/test_model.py tests/test_opponents.py tests/test_ppo_learn.py tests/test_protocol_survival.py tests/test_reference_fixtures.py tests/test_rollout_gae.py tests/test_spec_constants.py tests/test_wasm_gate.py tests/test_wrapper.py tests/test_wrapper_parity.py -q
