# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Logging/checkpoint pass. Boring, necessary.

Save the effective configuration, seed, versions, and model/optimizer state. A checkpoint I can't resume is just a large souvenir. Need metrics in a form I can inspect without the dashboard being alive.

Also keep the translation test correction in mind: tolerate equivalent row order; don't hide actual value changes.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 12. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
