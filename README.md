# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

S1 recorded as passed. Historical champion: `stage1_bcft`, `ckpt_v001470`.

Save that identity, then check deployment separately. Passing the training-side gate isn't the same as doing well in the tournament. Next annoying task is the single-file portal requirement.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 31. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `be04ba1`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
