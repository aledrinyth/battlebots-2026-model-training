# Battlebots

Need a bot that grows early and survives actual opponents. The training score alone clearly isn't enough.

## Where I'm at

Flat builders for the portal. The earlier package injection approach can upset the import scanner, so put the needed definitions into a plain single-file namespace.

Dependency order matters. Strip internal package imports carefully, preserve constants, embed NN weights, and compare generated actions against the source bot. For the heuristic, keep the generated output small and scan it too.

Gate the exact file. This note keeps coming back because it's easy to skip.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 36. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This groups uncommitted work into a logical stage. Git reference: `605697b`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
