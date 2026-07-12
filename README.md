# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

Phase-four checkpoint. Compliance pieces are in; leave the next run focused on the main loop and packaging.

Open items: weight resolution, startup behavior, and the time ledger. Keep this short. The code hasn't gained a new controller in this step.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 20. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `0617248`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
