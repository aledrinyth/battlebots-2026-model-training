"""Replay bootstrap (v3 Phase R0) — tournament replays → supervised dataset.

Turns a directory of engine replay JSON files into a versioned,
quality-labelled dataset of ``(obs_736, heading_bin, split, rank_weight,
bot_id, …)`` rows, with a bit-identical parity gate and a streaming ingester.
See ``plan_training_v3_replay_bootstrap_impl.md`` for the design and
``notes/tourney-replay-stats.md`` for the engine-version compatibility audit.

Everything in this package is **torch-free** (numpy + ``bot.*`` + the vendored
engine) and runs in both venvs; the BC consumer (R2) lives elsewhere.
"""

from train.replay.reconstruct import ReplayError, Walker  # noqa: F401
from train.replay.convert import ConvertResult, convert_replay  # noqa: F401
