# Battlebots

Need a bot that grows early and survives actual opponents. The training score alone clearly isn't enough.

## Where I'm at

Seed the league with meta clones instead of waiting for self-play to invent useful opposition. Keep live snapshots when adding clone entries; rerunning the seeder should be safe.

S2 heuristic config is staged, runbook included. Config present does not mean training was launched. Need clone weights under the expected paths; those generated datasets aren't copied into these source folders.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 37. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This groups uncommitted work into a logical stage. Git reference: `605697b`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
