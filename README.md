# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

Bundle scan + safe fallback. Malformed input shouldn't turn one bad observation into a dead process.

Scan forbidden imports before shipping. Fuzz the guarded turn. If the neural path fails, return the legal fallback and let the match continue. Don't swallow problems during training just because the deployed bot has to survive them.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. Use the existing tests for the path being changed. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 18. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `5b99d7e`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
