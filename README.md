# Battlebots

Need a bot that grows early and survives actual opponents. The training score alone clearly isn't enough.

## Where I'm at

Potential-field controller and defensive rails. Food/prey pull, threats/viruses/walls push. FEED then HUNT; every seat gets its own controller state.

Split-to-kill needs three checks: the child can eat the target, the target is reachable, and an enemy can't immediately eat that child. Don't just crank aggression because one highlight looked good.

Frozen and railed opponent names go through the same registry. Versioned opponents load their exact `dist/` files, so those files stay in the snapshot. Rails exist; don't infer they're deployed everywhere.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 35. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This groups uncommitted work into a logical stage. Git reference: `605697b`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
