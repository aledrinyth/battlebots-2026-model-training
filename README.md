# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Shared featurizer and action adapter now, plus some scripted opponents. Same features in training and the actual bot. No separate clever inference-side normalization.

Food baseline first. Cautious/chaser opponents after that. A learned policy beating a stationary bot won't tell me much.

One annoying test issue: equally distant entities can swap rows after a translation because of tiny floating-point differences. The row-comparison example shows the bad assertion. Don't fix a test problem by changing the frozen feature contract.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 07. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
