# Battlebots

Build the training pipeline around the competition engine, then ship a bot that can survive the runner. Keep the match-side dependencies small.

## Where I'm at

Wire protocol first, fancy search later. The tiny WASM probe is just a feasibility check, not a search engine.

Need the bot to read a query, produce the right move object, and survive the runner. Keep stdout for the protocol. Diagnostics elsewhere. If a Python-only path is enough for now, use it and move on.

## Running things

Python 3.12. Start with the requirements for the part being exercised. This is an intermediate build; later imports and integration gates may still be missing. Check the available tests before assuming the whole pipeline is runnable.

---

Progression copy 03. README rewritten for this staged collection; first-person wording is editorial, not a recovered diary. This intermediate stage is reconstructed. Git reference: `1dd62ef`. Detailed provenance/file hashes stay in `SNAPSHOT.json`.
