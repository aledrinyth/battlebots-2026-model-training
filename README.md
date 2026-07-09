# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Network + PPO + recurrent rollouts. This is the part worth being fussy about.

Death ends a live segment. The respawn gap is inactive. Bootstrap must be masked at death, the GAE accumulator must clear over the inactive gap, and inactive rows must stay out of the loss. Otherwise one life trains on the next life's reward and the numbers still look perfectly respectable.

Keep hidden resets explicit. Check the hand-computed rollout cases before burning time on a long run.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 10. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
