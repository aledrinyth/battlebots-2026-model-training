# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

League pool and PFSP sampling. Freeze checkpoints, sample opponents, keep each seat's recurrent state separate.

Hard-but-learnable opponents should get attention. A snapshot pool that only grows forever is another disk leak, so cap retention. Need the wiring into actual rollouts next; a league manager sitting unused doesn't train anything.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 28. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `33279cf`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
