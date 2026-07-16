# Battlebots

Need a bot that grows early and survives actual opponents. The training score alone clearly isn't enough.

## Where I'm at

Replay bootstrap and league wiring. Bigger step.

Reconstruct observations from recordings, preserve identity/provenance, check parity before trusting labels. Build a dataset, then BC toward winners. A recording is not automatically a faithful training row.

League snapshots must reach the rollout workers and seats, with versioned config and metrics. Meta opponents are the next missing yardstick. Raw replays/shards are external artifacts here; regenerate or restore them when running this path.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 34. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. Source follows the original Git milestone; documentation is consolidated here. Git reference: `605697b`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
