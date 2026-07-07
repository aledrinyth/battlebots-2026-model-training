# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Wrapper is in. Use the engine's own state and mutator, not a second implementation of physics that slowly drifts away.

Important bit: only alive seats receive observations. Missing a move for a live seat is an error; dead seats aren't asked. Reset must seed the engine RNG before game construction. Turn order and round advancement have to follow the real runner too.

Next: parity traces. Optimization can wait until there is something to compare it with. The fast-vision install is deliberately disabled in this reconstructed step.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

---

Progression copy 04. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
