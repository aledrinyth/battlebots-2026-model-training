# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

Real bot entry point now. Query in, guarded NumPy turn, move out.

Weight lookup must be predictable: explicit override, packaged weights, then fixture fallback for development. Keep recurrent state between turns, reset where the game contract needs it. Track elapsed time; startup and feature work count too.

Need to test through the protocol, not only call `act()` in a unit test.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 22. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `2b5de80`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
