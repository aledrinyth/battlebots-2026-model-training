# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Start small. Two packages: `bot` is what can go into a match, `train` is where the expensive stuff lives. Separate requirements because I don't want torch sneaking into a submission.

First jobs: get the engine pinned, run one boring reference game, check the actual protocol. Model choice can wait. The import boundary matters more right now than another architecture sketch.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

---

Progression copy 01. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
