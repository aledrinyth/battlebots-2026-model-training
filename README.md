# Battlebots

Training pipeline on one side, NumPy match bot on the other. Current focus: make the evaluation and submission path trustworthy.

## Where I'm at

Portal takes one Python file. Single-file builder added; this changes packaging, not the policy objective.

Need identical actions before and after flattening/embedding, legal imports, and the actual file through compliance. Payload size/startup can matter. Don't accidentally benchmark the package and upload the single-file version without checking it.

Later portal scanner behavior may still force a flatter layout.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 32. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `8ddb27c`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
