# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

S1 attempt failed, retry path under review. Capture enough state to continue without guessing which weights are being converted.

Food DR, league pieces, and promotion gate are present. Don't claim a new champion until the next candidate actually clears the gate. Mostly a planning pause in this stage.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 30. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `e566570`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
