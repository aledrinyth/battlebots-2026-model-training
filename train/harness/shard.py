"""Shard + weights file I/O (P3.2, two-box) — the rsync/scp "wire".

For a **two-machine** setup (CPU actor box + GPU learner box) the transport is
the filesystem: the CPU box writes trajectory **shards** and the GPU box writes
**weight** files; you move them between boxes with ``rsync``/``scp`` (or a shared
filesystem / object store). This module is the serialization contract for both,
with **atomic writes** (write to ``*.tmp`` then ``os.replace``) so a half-written
or mid-transfer file is never picked up.

* A **shard** (``shard_<worker>_<seq>.npz``) is one collected rollout: every
  :class:`~train.harness.rollout.RolloutBuffer` array + the GAE bootstrap
  ``last_values`` + a small JSON meta blob (crucially the ``policy_version`` the
  data was collected under, so the learner can measure policy lag / staleness).
* A **weights** file (``weights_v<NNNNNN>.pt``) is a versioned policy state dict +
  architecture metadata. The consumer picks the highest version present.

Everything here is safe to run on either box; only :func:`write_weights` /
:func:`load_weights_into` touch torch (learner + local-inference actor, both on
training boxes — never the shipped match bot).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from train.harness.rollout import RolloutBuffer

SHARD_GLOB = "shard_*.npz"
_WEIGHTS_RE = "weights_v*.pt"

# Buffer arrays carried in a shard (name → dtype is implied by the buffer).
_ARRAYS = (
    "obs", "is_reset", "active", "heading", "split", "logp", "value", "reward", "done",
)


# --- atomic write helper ----------------------------------------------------

def _atomic(path: str, write_fn) -> str:
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)  # atomic on POSIX within a filesystem
    return path


# --- shards -----------------------------------------------------------------

def write_shard(
    out_dir: str,
    worker_id: int,
    seq: int,
    buffer: RolloutBuffer,
    last_values: np.ndarray,
    policy_version: int,
    extra_meta: Optional[Dict] = None,
    compress: bool = False,
) -> str:
    """Serialize one collected rollout to ``shard_<worker>_<seq>.npz`` (atomic).

    ``compress=True`` uses ``np.savez_compressed`` — the obs array is mostly zero
    padding (masked entity slots), so it shrinks several-fold. Trades CPU on the
    (128-core) actor box for network bandwidth when shards are rsync'd to a
    separate learner box. ``read_shard`` handles both transparently.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "worker_id": int(worker_id),
        "seq": int(seq),
        "policy_version": int(policy_version),
        "rollout_len": int(buffer.T),
        "rows": int(buffer.R),
        "hidden_size": int(buffer.H),
        **(extra_meta or {}),
    }
    payload = {name: getattr(buffer, name) for name in _ARRAYS}
    payload["hidden0"] = buffer.hidden0.detach().cpu().numpy()
    payload["last_values"] = np.asarray(last_values, dtype=np.float32)
    payload["meta_json"] = np.array(json.dumps(meta))

    saver = np.savez_compressed if compress else np.savez

    def _write(p: str) -> None:
        # Pass an open handle: np.savez(name) auto-appends ".npz" to a name that
        # doesn't end in it, which would defeat the atomic-rename tmp path.
        with open(p, "wb") as fh:
            saver(fh, **payload)

    path = os.path.join(out_dir, f"shard_{worker_id:03d}_{seq:08d}.npz")
    return _atomic(path, _write)


@dataclass
class Shard:
    buffer: RolloutBuffer
    last_values: np.ndarray
    meta: Dict
    path: str


def read_shard(path: str) -> Shard:
    """Load a shard back into a :class:`RolloutBuffer` + bootstrap + meta."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        buf = RolloutBuffer(int(meta["rollout_len"]), int(meta["rows"]), int(meta["hidden_size"]))
        for name in _ARRAYS:
            getattr(buf, name)[:] = z[name]
        buf.hidden0 = torch.from_numpy(np.ascontiguousarray(z["hidden0"])).float()
        last_values = np.array(z["last_values"], dtype=np.float32)
    return Shard(buffer=buf, last_values=last_values, meta=meta, path=path)


def merge_shards(shards: List["Shard"]) -> Tuple[RolloutBuffer, np.ndarray, int]:
    """Concatenate several shards along the row axis into one PPO batch.

    Lets many small workers (few games each) feed one large-batch update on the
    GPU: ``R_merged = Σ shard.rows``. All shards must share ``rollout_len`` and
    ``hidden_size``. Returns ``(buffer, last_values, min_policy_version)`` — the
    min version is the worst-case staleness of the merged batch.
    """
    if not shards:
        raise ValueError("merge_shards got no shards")
    T = shards[0].buffer.T
    H = shards[0].buffer.H
    for s in shards:
        if s.buffer.T != T or s.buffer.H != H:
            raise ValueError("cannot merge shards with differing rollout_len/hidden_size")

    R = sum(s.buffer.R for s in shards)
    merged = RolloutBuffer(T, R, H)
    for name in _ARRAYS:
        getattr(merged, name)[:] = np.concatenate([getattr(s.buffer, name) for s in shards], axis=1)
    merged.hidden0 = torch.cat([s.buffer.hidden0 for s in shards], dim=0)
    last_values = np.concatenate([s.last_values for s in shards], axis=0).astype(np.float32)
    min_version = min(int(s.meta["policy_version"]) for s in shards)
    return merged, last_values, min_version


def prune_weights(out_dir: str, keep: int) -> None:
    """Delete all but the ``keep`` highest-version weight files (disk hygiene)."""
    files = _weight_files(out_dir)
    for _v, path in files[:-keep] if keep > 0 else []:
        try:
            os.remove(path)
        except OSError:
            pass


def clear_weights(out_dir: str) -> int:
    """Remove *every* weight file in ``out_dir``. Used by the trainer's ``--fresh``
    restart so the freshly-published ``v0`` is genuinely the latest version (else
    old higher-version checkpoints would shadow it for `latest_weights_version`).
    Returns the number removed."""
    files = _weight_files(out_dir)
    for _v, path in files:
        try:
            os.remove(path)
        except OSError:
            pass
    return len(files)


def list_shards(in_dir: str) -> List[str]:
    """Complete shard files in ``in_dir``, oldest first (by name = worker,seq)."""
    if not os.path.isdir(in_dir):
        return []
    return sorted(glob.glob(os.path.join(in_dir, SHARD_GLOB)))


def archive_shard(path: str, consumed_dir: str) -> None:
    """Move a consumed shard aside so it is not reprocessed (atomic rename)."""
    os.makedirs(consumed_dir, exist_ok=True)
    os.replace(path, os.path.join(consumed_dir, os.path.basename(path)))


# --- weights ----------------------------------------------------------------

def write_weights(out_dir: str, policy, version: int, arch_meta: Dict) -> str:
    """Serialize a versioned policy state dict to ``weights_v<NNNNNN>.pt`` (atomic)."""
    os.makedirs(out_dir, exist_ok=True)
    blob = {
        "version": int(version),
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "arch": dict(arch_meta),
    }
    path = os.path.join(out_dir, f"weights_v{version:06d}.pt")
    return _atomic(path, lambda p: torch.save(blob, p))


def _weight_files(in_dir: str) -> List[Tuple[int, str]]:
    out = []
    for p in glob.glob(os.path.join(in_dir, _WEIGHTS_RE)):
        base = os.path.basename(p)
        try:
            v = int(base[len("weights_v"):-len(".pt")])
        except ValueError:
            continue
        out.append((v, p))
    return sorted(out)


def latest_weights_version(in_dir: str) -> Optional[int]:
    files = _weight_files(in_dir)
    return files[-1][0] if files else None


def load_weights_into(policy, in_dir: str, map_location="cpu") -> Optional[int]:
    """Load the highest-version weights in ``in_dir`` into ``policy``. Returns the
    version loaded, or ``None`` if no weights file is present yet."""
    files = _weight_files(in_dir)
    if not files:
        return None
    version, path = files[-1]
    blob = torch.load(path, map_location=map_location, weights_only=False)
    policy.load_state_dict(blob["state_dict"])
    return int(blob["version"])


__all__ = [
    "write_shard", "read_shard", "list_shards", "archive_shard", "merge_shards", "Shard",
    "write_weights", "latest_weights_version", "load_weights_into",
    "prune_weights", "clear_weights", "SHARD_GLOB",
]
