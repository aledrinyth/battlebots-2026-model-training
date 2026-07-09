# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Fix the row comparison, leave the features alone.

The encoder pools sets, so row order isn't the property we care about. Compare the same rows with multiplicity preserved. A set is wrong because duplicate rows disappear. Production checks need tolerance; the small example uses exact tuples so the failure is obvious.

This is a test correction, not evidence that the policy suddenly got better. Anyway, back to the harness.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 08. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
