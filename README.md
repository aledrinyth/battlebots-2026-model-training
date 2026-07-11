# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

Worst-case scene bank now. Average inference speed is the flattering number.

Try dense visibility, lots of blobs, awkward frames. Time the actual bot path, including feature work. Random fixture weights are for correctness/timing checks, not a competitive champion.

Keep timing tests out of a noisy pile of simultaneous benchmarks when investigating a failure.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. Use the existing tests for the path being changed. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 17. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `25151ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
