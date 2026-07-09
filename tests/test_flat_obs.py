"""P3.2 — flat-obs codec parity (``train.harness.flat_obs`` + ``collate``).

The actor↔learner shared-memory transport flattens each ``CanonicalObs`` into a
fixed-width vector and the learner rebuilds a batched tensor from the whole slab.
Both directions must be lossless and must agree with the P3.1 ``collate_obs``
path, or the trainer and the shipped bot would silently see different tensors —
the exact bug the shared-adapter boundary exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # train-side test; skipped in the torch-free bot venv

from bot.featurizer import Featurizer
from train.env.wrapper import WrapperEnv
from train.harness.collate import build_batched_obs, collate_obs
from train.harness.flat_obs import OBS_WIDTH, flatten, unflatten


def _sample_obs(n: int = 6):
    """A handful of real featurized frames from a live wrapper game."""
    env = WrapperEnv(seed=7, emit_update=False)
    queries = env.reset()
    feats = {pid: Featurizer() for pid in queries}
    out = []
    for _ in range(4):
        for pid, q in queries.items():
            out.append(feats[pid].featurize(q))
            if len(out) >= n:
                return out
        from train.env.wrapper import make_move
        moves = {pid: make_move(pid, 1.0, 0.0) for pid in queries}
        queries, done = env.step(moves)
        if done:
            break
    return out


def test_flatten_width_matches_schema():
    from bot.adapters.obs_v1 import (
        ENEMY_DIM, ENEMY_MAX, FOOD_DIM, FOOD_MAX,
        OWN_DIM, OWN_MAX, SELF_DIM, VIRUS_DIM, VIRUS_MAX,
    )
    expected = (
        SELF_DIM
        + FOOD_MAX * FOOD_DIM + FOOD_MAX
        + ENEMY_MAX * ENEMY_DIM + ENEMY_MAX
        + OWN_MAX * OWN_DIM + OWN_MAX
        + VIRUS_MAX * VIRUS_DIM + VIRUS_MAX
    )
    assert OBS_WIDTH == expected


def test_flatten_unflatten_roundtrip():
    for obs in _sample_obs():
        vec = flatten(obs)
        assert vec.shape == (OBS_WIDTH,)
        assert vec.dtype == np.float32
        back = unflatten(vec)
        for name in ("self_vec", "food", "food_mask", "enemy", "enemy_mask",
                     "own", "own_mask", "virus", "virus_mask"):
            np.testing.assert_array_equal(getattr(obs, name), getattr(back, name), err_msg=name)


def test_flatten_into_out_buffer_is_a_view_write():
    obs = _sample_obs(1)[0]
    buf = np.full(OBS_WIDTH, 99.0, dtype=np.float32)
    ret = flatten(obs, out=buf)
    assert ret is buf
    np.testing.assert_array_equal(buf[: obs.self_vec.shape[0]], obs.self_vec)


def test_build_batched_obs_matches_collate():
    """The slab reader (flat → BatchedObs) must equal the list collate path."""
    batch = _sample_obs(5)
    flat = np.stack([flatten(o) for o in batch], axis=0)
    is_reset = np.array([o.is_reset for o in batch])

    from_flat = build_batched_obs(flat, is_reset)
    from_list = collate_obs(batch)

    for name in ("self_vec", "food", "food_mask", "enemy", "enemy_mask",
                 "own", "own_mask", "virus", "virus_mask"):
        a = getattr(from_flat, name).numpy()
        b = getattr(from_list, name).numpy()
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_build_batched_obs_rejects_wrong_width():
    with pytest.raises(ValueError):
        build_batched_obs(np.zeros((3, OBS_WIDTH + 1), dtype=np.float32), np.zeros(3))
