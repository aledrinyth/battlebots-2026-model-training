# Battlebots

Need a bot that grows early and survives actual opponents. The training score alone clearly isn't enough.

## Where I'm at

Current handoff. Source, deployed bots, conversion weights, and research experiments are together. Still too many directions, so pick the next measurement before the next implementation.

Phase-expert idea: net for rounds 0–400, another for 400–800, heuristic for 800–1400. BC warm-start then PPO per expert. Phase end mass as an objective; no critic bootstrap across a phase boundary. Reset recurrent state at the switch. For later-phase starts mix replay states with our own earlier expert's end states, otherwise the handoff gets trained on a world it never sees.

This is a plan, not an implemented composite. First check per-phase data volume and whether the expert beats the heuristic in paired full games against meta clones. If it doesn't, keep the heuristic. GPU/JAX/Triton notes were research directions, not proof of a fast working backend.

Loose ends: clone/checkpoint paths, early survival, portal gate on the final artifact. Also that incomplete `CurrentBot.py` fixture still exists. Don't silently call its syntax error a new regression.

## Running things

Python 3.12, separate bot/training environments. Install the local engine from `vendor/agario-public` through the requirement files. `make ci` is the regular gate; full compliance belongs before submission. Replay inbox, generated datasets, and run checkpoints are outside these portable snapshots. Restoring those paths is part of setting up a training run.

Known loose end: `tests/fixtures/bots/CurrentBot.py` has an empty main guard and won't parse. Preserved from the source.

---

Progression copy 40. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This groups uncommitted work into a logical stage. Git reference: `605697b`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
