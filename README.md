# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

Compliance wired into CI. Quick smoke locally, full gate before a real submission.

Run `make ci` in the bot environment. Full runner checks are slower; budget time for them. Build artifact first, gate that artifact. Testing the source and then shipping some different generated file is an easy way to fool myself.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 19. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `d9fd0ad`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
