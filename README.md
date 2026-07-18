# Battlebots

Need a bot that grows early and survives actual opponents. The training score alone clearly isn't enough.

## Where I'm at

C/WAT/WASM search-kernel experiment alongside the old tiny feasibility probe.

Need build reproducibility, parity, and budget checks before routing live actions through it. The kernel files being here doesn't prove a search policy is shipped. Leave the Python controller as the useful baseline while this gets checked.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 39. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This groups uncommitted work into a logical stage. Git reference: `605697b`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
