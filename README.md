# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Two-box path: CPU actors produce shards, learner consumes them and writes weights back. File exchange is easier to debug than a fragile distributed service right now.

Version the shards and weights. Reject stale data beyond the configured limit. Publish finished files atomically, keep retention bounded, and make the worker shut down on error rather than wait forever. Start with a tiny smoke run. Then scale.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 13. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
