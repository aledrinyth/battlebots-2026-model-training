# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Fast vision enabled. Compute the observer's frame once, then do the entity checks in arrays. Physics still belongs to the engine.

Circle/window fix: find the nearest point on the box to the circle center; compare squared distance with squared radius. Inclusive boundary. Food keeps its point test. The little visibility example now checks inside, outside, and touching cases.

Need exact membership AND ordering to match, not merely roughly equal visible counts. That's where a "harmless" optimization can change the whole trajectory.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

---

Progression copy 06. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
