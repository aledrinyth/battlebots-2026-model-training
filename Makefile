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
.PHONY: help ci ci-full lint test test-kit wasm venvs compliance-smoke compliance-full

# The seat under compliance test. Until P5.5 packages the real bot this is the
# strongest scripted stand-in; point it at the bundle's main.py when it exists.
COMPLIANCE_BOT ?= tests/fixtures/bots/greedy.py
FILLER_BOT     ?= tests/fixtures/bots/dont_move.py

help:
	@echo "Targets:"
	@echo "  make ci                - the gate every change must pass: lint + tests + compliance smoke"
	@echo "  make ci-full           - ci + the vendored engine's own test suite"
	@echo "  make lint              - import-linter (bot must not import train/torch)"
	@echo "  make test              - unit + property + gate tests (tests/)"
	@echo "  make test-kit          - the pinned engine's own pytest suite"
	@echo "  make compliance-smoke  - P4.4 smoke: 1 capped real-runner game + scene bank + bundle scan"
	@echo "  make compliance-full   - P4.4 full: 10 full-length games + all checks (pre-submission gate)"
	@echo "  make wasm              - (re)build v2/wasm/hello.wasm (gate G0)"
	@echo "  make venvs             - build .venv-bot (v2 deps) and .venv-train"

## ---- the gate ----------------------------------------------------------
# Runs on every change. The pytest suite already exercises the compliance /
# scene-bank / fuzz code paths; compliance-smoke additionally gates the shipped
# CLIs end-to-end (exit codes included).
ci: lint test compliance-smoke

ci-full: lint test test-kit compliance-smoke

## ---- P4.4 compliance gates ----------------------------------------------
# Smoke: one capped game through the real pipe runner + the P4.2 scene-bank
# timing gates + the P4.3 bundle scan. ~30 s.
compliance-smoke:
	$(PY) tools/compliance.py --bots 1:$(COMPLIANCE_BOT) 7:$(FILLER_BOT) \
		--rounds 60 --start-delay 1
	$(PY) tools/scene_bank.py replay
	$(PY) tools/bundle_scan.py bot

# Full: ten full-length seeded games (subject in half the seats for pressure),
# all checks. Run nightly and BEFORE ANY SUBMISSION.
compliance-full:
	$(PY) tools/compliance.py --bots 4:$(COMPLIANCE_BOT) 4:$(FILLER_BOT) \
		--games 10 --start-delay 1
	$(PY) tools/scene_bank.py replay
	$(PY) tools/bundle_scan.py bot

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
