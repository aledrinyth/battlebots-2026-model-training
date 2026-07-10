# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Stage-zero configs, demos, BC warm-start, exploration checks. Try to learn food collection before asking the policy to invent competent combat.

Need to distinguish "never sees the right state" from "sees it but can't learn the action". Scripted demonstrations help with that. Shaping is a training knob, not a license to change evaluation.

Several configs here. Don't pretend all of them won. Keep the failed directions useful as diagnostics.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 14. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
