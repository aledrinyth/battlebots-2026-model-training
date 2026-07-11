# CI wiring (P0.5). The authoritative, locally-runnable gate is `make ci`.
# The GitHub Actions workflow (.github/workflows/ci.yml) runs the same targets
# so local and remote stay identical.
#
# Override the interpreter if your venv lives elsewhere:
#   make ci PY=/path/to/python

PY ?= $(CURDIR)/.venv-bot/bin/python
# NB: don't name this LINT — GNU make ships a builtin default `LINT = lint`
# that `?=` won't override. Derive the linter from PY's bin dir.
LINTER ?= $(dir $(PY))lint-imports

.DEFAULT_GOAL := help
.PHONY: help ci ci-full lint test test-kit wasm venvs

help:
	@echo "Targets:"
	@echo "  make ci        - the gate every change must pass: lint + tests"
	@echo "  make ci-full   - ci + the vendored engine's own test suite"
	@echo "  make lint      - import-linter (bot must not import train/torch)"
	@echo "  make test      - unit + property + gate tests (tests/)"
	@echo "  make test-kit  - the pinned engine's own pytest suite"
	@echo "  make wasm      - (re)build v2/wasm/hello.wasm (gate G0)"
	@echo "  make venvs     - build .venv-bot (v2 deps) and .venv-train"

## ---- the gate ----------------------------------------------------------
# Runs on every change. As they land, the featurizer-parity test (P2.2) and the
# compliance smoke test (P4/P5) are plain pytest files under tests/ and are
# picked up here automatically — no Makefile change needed.
ci: lint test

ci-full: lint test test-kit

## ---- individual checks -------------------------------------------------
lint:
	$(LINTER)

test:
	$(PY) -m pytest tests/ -q

test-kit:
	cd vendor/agario-public && $(PY) -m pytest tests/ -q

wasm:
	$(PY) v2/wasm/build_wasm.py

## ---- environment bootstrap --------------------------------------------
venvs:
	python3.12 -m venv .venv-bot
	.venv-bot/bin/pip install --upgrade pip
	.venv-bot/bin/pip install -r requirements-bot-v2.txt
	python3.12 -m venv .venv-train
	.venv-train/bin/pip install --upgrade pip
	.venv-train/bin/pip install -r requirements-train.txt
