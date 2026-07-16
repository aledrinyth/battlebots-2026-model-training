# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

Tournament diagnosis. S1's friendly anchors were too friendly.

Look at real replays: growth timing, deaths, split behavior, who actually wins. The plan now is replay reconstruction plus winner-directed BC and a less flattering opponent field. No point polishing the old acceptance score if it doesn't measure the problem.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 33. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `03cc737`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
