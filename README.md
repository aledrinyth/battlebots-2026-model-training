# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

Record the stage-zero champion before losing track of which checkpoint was which.

The original record names `g1_rig_bcft`, `ckpt_v000123`, and a 13.1x result. That's a historical recorded result, not a fresh benchmark of this folder. Keep the identifier useful; don't turn it into a promise about every opponent field.

Main loop next.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 21. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `e66f823`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
