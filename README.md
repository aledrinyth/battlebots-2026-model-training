# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Engine is now in `vendor/agario-public`, constants in `bot/spec.py`. Keep the engine version pinned. Seriously, don't adjust our constants to make a test green without checking the engine first.

Reference fixtures are in `tests/fixtures/`. Plan is to compare against something real before building a fast training wrapper. Python 3.12; install the engine from the local vendor directory, not an editable install. The old environment had trouble with editable package paths.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

---

Progression copy 02. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
