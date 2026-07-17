# Battlebots

Need a bot that grows early and survives actual opponents. The training score alone clearly isn't enough.

## Where I'm at

Candidate/ensemble/survivability tools. Several experiments; don't crown every candidate just because it has a file.

Paired seeds, swap seats, inspect early growth AND deaths. Parallel match runs help, but don't mix them with sensitive timing gates. Keep deployed version labels straight. If a change improves a passive-opponent score but dies faster against clones, that's probably not the next submission.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 38. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This groups uncommitted work into a logical stage. Git reference: `605697b`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
