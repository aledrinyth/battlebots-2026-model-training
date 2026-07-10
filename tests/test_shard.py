"""P3.2 (two-box) — shard + weights file I/O (``train.harness.shard``).

The filesystem is the transport between the CPU actor box and the GPU learner
box, so the serialization must be lossless, atomic (no half-written/mid-rsync
file is ever consumed), and mergeable (many small worker shards → one large PPO
batch).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from train.harness.policy_stub import SkeletonPolicy
from train.harness.rollout import RolloutBuffer
from train.harness.shard import (
    archive_shard,
    latest_weights_version,
    list_shards,
    load_weights_into,
    merge_shards,
    prune_weights,
    read_shard,
    write_shard,
    write_weights,
)


def _filled_buffer(T=4, R=6, H=8, seed=0):
    rng = np.random.default_rng(seed)
    b = RolloutBuffer(T, R, H)
    b.obs[:] = rng.standard_normal(b.obs.shape).astype(np.float32)
    b.is_reset[:] = rng.random((T, R)) < 0.2
    b.active[:] = rng.random((T, R)) < 0.9
    b.heading[:] = rng.integers(0, 24, (T, R))
    b.split[:] = (rng.random((T, R)) < 0.5).astype(np.float32)
    b.logp[:] = rng.standard_normal((T, R)).astype(np.float32)
    b.value[:] = rng.standard_normal((T, R)).astype(np.float32)
    b.reward[:] = rng.standard_normal((T, R)).astype(np.float32)
    b.done[:] = (rng.random((T, R)) < 0.1).astype(np.float32)
    b.hidden0 = torch.randn(R, H)
    return b


def test_shard_roundtrip(tmp_path):
    b = _filled_buffer()
    last = np.arange(b.R, dtype=np.float32)
    path = write_shard(str(tmp_path), worker_id=3, seq=7, buffer=b,
                       last_values=last, policy_version=42, extra_meta={"games": 3})
    assert path.endswith("shard_003_00000007.npz")

    s = read_shard(path)
    for name in ("obs", "is_reset", "active", "heading", "split", "logp", "value", "reward", "done"):
        np.testing.assert_array_equal(getattr(s.buffer, name), getattr(b, name), err_msg=name)
    np.testing.assert_array_equal(s.buffer.hidden0.numpy(), b.hidden0.numpy())
    np.testing.assert_array_equal(s.last_values, last)
    assert s.meta["policy_version"] == 42
    assert s.meta["worker_id"] == 3
    assert s.meta["games"] == 3


def test_write_is_atomic_no_tmp_left(tmp_path):
    b = _filled_buffer()
    write_shard(str(tmp_path), 0, 0, b, np.zeros(b.R, np.float32), 0)
    assert glob.glob(os.path.join(str(tmp_path), "*.tmp")) == []
    assert len(list_shards(str(tmp_path))) == 1


def test_merge_shards_concatenates_rows(tmp_path):
    b1 = _filled_buffer(R=6, seed=1)
    b2 = _filled_buffer(R=10, seed=2)
    p1 = write_shard(str(tmp_path), 0, 0, b1, np.ones(6, np.float32), policy_version=5)
    p2 = write_shard(str(tmp_path), 1, 0, b2, np.full(10, 2.0, np.float32), policy_version=3)

    merged, last, min_v = merge_shards([read_shard(p1), read_shard(p2)])
    assert merged.R == 16
    assert min_v == 3  # worst-case staleness
    np.testing.assert_array_equal(merged.obs[:, :6], b1.obs)
    np.testing.assert_array_equal(merged.obs[:, 6:], b2.obs)
    np.testing.assert_array_equal(last[:6], 1.0)
    np.testing.assert_array_equal(last[6:], 2.0)


def test_archive_shard_moves_it(tmp_path):
    b = _filled_buffer()
    p = write_shard(str(tmp_path), 0, 0, b, np.zeros(b.R, np.float32), 0)
    consumed = str(tmp_path / "consumed")
    archive_shard(p, consumed)
    assert not os.path.exists(p)
    assert len(list_shards(str(tmp_path))) == 0
    assert os.path.exists(os.path.join(consumed, os.path.basename(p)))


def test_weights_roundtrip_and_versioning(tmp_path):
    torch.manual_seed(0)
    src = SkeletonPolicy(entity_hidden=8, trunk_hidden=16)
    write_weights(str(tmp_path), src, version=0, arch_meta={"trunk_hidden": 16})
    write_weights(str(tmp_path), src, version=5, arch_meta={"trunk_hidden": 16})
    assert latest_weights_version(str(tmp_path)) == 5

    torch.manual_seed(999)  # different init
    dst = SkeletonPolicy(entity_hidden=8, trunk_hidden=16)
    # weights differ before load
    k = next(iter(src.state_dict()))
    assert not torch.equal(src.state_dict()[k], dst.state_dict()[k])
    v = load_weights_into(dst, str(tmp_path))
    assert v == 5
    for key in src.state_dict():
        torch.testing.assert_close(src.state_dict()[key], dst.state_dict()[key])


def test_prune_weights_keeps_last_k(tmp_path):
    torch.manual_seed(0)
    p = SkeletonPolicy(entity_hidden=4, trunk_hidden=8)
    for v in range(6):
        write_weights(str(tmp_path), p, version=v, arch_meta={})
    prune_weights(str(tmp_path), keep=2)
    remaining = sorted(int(os.path.basename(f)[len("weights_v"):-3])
                       for f in glob.glob(os.path.join(str(tmp_path), "weights_v*.pt")))
    assert remaining == [4, 5]


def test_latest_weights_none_when_empty(tmp_path):
    assert latest_weights_version(str(tmp_path)) is None
    assert load_weights_into(SkeletonPolicy(entity_hidden=4, trunk_hidden=8), str(tmp_path)) is None
