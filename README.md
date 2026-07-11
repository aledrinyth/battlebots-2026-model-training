# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

Initial integrated milestone: export trained weights, run the NumPy policy, check it against the training model. Deployment must stay torch-free.

The first actual Git commit already contains a lot of work; the earlier folders break it down for this progression. This folder reaches that committed source.

Next is compliance and an actual submission path. A good checkpoint isn't enough if the runner kills the process.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. Use the existing tests for the path being changed. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 15. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
