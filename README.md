# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Parity and a benchmark now. Need to know whether the wrapper actually agrees with the pipe runner, not just whether both return plausible scores.

Visibility snag: checking a blob's center against the window misses circles whose edges overlap it. Food is a point, blobs and viruses aren't. `progression_examples/visibility.py` reproduces that mistake in isolation. It isn't old production source recovered from Git.

Don't touch the eat loops yet. Profile first.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

---

Progression copy 05. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
