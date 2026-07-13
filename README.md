# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

Packager exists. Build a self-contained bundle with its weights and provenance.

Don't hand-copy modules and hope they match. Scan and gate the assembled output. Random-init bundle is for a smoke run. Champion bundle needs the actual exported checkpoint.

Still need to check what the portal accepts. A directory working locally doesn't establish the upload format.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 23. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `3b302b9`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
