# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Custom actor/learner harness. A seat needs its own state; treating an eight-player game as one ordinary single-agent env is going to make a mess.

Shared memory for observations. Batch policy calls in the learner. Keep a stub policy until the plumbing is predictable. TODO: actor failure must wake the learner, otherwise a bad worker just looks like training stopped.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 09. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
