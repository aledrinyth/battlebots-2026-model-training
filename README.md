# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

Compliance harness added. Measure through the real runner and use its verdict.

Need both single-turn and cumulative timing cases. A bot can be fast most of the time and still exceed the game budget. Keep the intentional sleeper fixtures; they're how the harness proves it can catch that failure.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. Use the existing tests for the path being changed. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 16. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `c7fe6c2`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
