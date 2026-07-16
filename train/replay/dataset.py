"""Sharded ``.npz`` dataset + manifest/index (impl doc §8).

One shard per replay: ``replay_<match_id>_<hash8>.npz`` with a JSON sidecar
manifest, mirroring ``train/harness/shard.py`` conventions (atomic write to
``*.tmp`` then ``os.replace``). A top-level ``index.json`` holds the interned
bot table, the processed-set (content hash → shard) that makes ingestion
idempotent, per-bot row counts, and the corpus episode counter.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from train.harness import flat_obs
from train.replay.convert import ConvertResult
from train.replay.identity import BotRegistry
from train.replay.labels import rank_weight
from train.replay.schema import ROW_DTYPES

INDEX_NAME = "index.json"
SHARD_GLOB = "replay_*.npz"
_INDEX_VERSION = 1


# --- atomic write helpers ----------------------------------------------------


def _atomic_bytes(path: Path, write_fn) -> Path:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)
    return path


def _atomic_json(path: Path, payload: dict) -> Path:
    return _atomic_bytes(
        path, lambda p: p.write_text(json.dumps(payload, indent=1, sort_keys=True))
    )


# --- row stamping -------------------------------------------------------------


def stamp_rows(
    result: ConvertResult,
    registry: BotRegistry,
    episode_base: int,
) -> Tuple[Dict[str, np.ndarray], Dict[int, dict], int]:
    """Add the corpus-level columns to a replay's converted arrays.

    Returns ``(arrays, seat_table, episodes_used)``: per-row ``rank_weight``
    (linear from the seat's engine-exact final rank), ``bot_id`` (interned
    via the registry), and ``episode`` (``episode_base`` + a dense per-seat
    offset — one episode per ``(replay, seat)``; respawn boundaries stay
    visible through ``is_reset``).
    """
    n_players = len(result.seats)
    seat_ids = sorted(result.seats)
    episode_of = {pid: episode_base + i for i, pid in enumerate(seat_ids)}
    weight_of = {
        pid: rank_weight(result.seats[pid].final_rank, n_players)
        for pid in seat_ids
    }
    bot_of = {
        pid: registry.id_for_team(result.seats[pid].team_id) for pid in seat_ids
    }

    seat_col = result.arrays["seat"]
    arrays = dict(result.arrays)
    arrays["rank_weight"] = np.asarray(
        [weight_of[pid] for pid in seat_col], dtype=ROW_DTYPES["rank_weight"]
    )
    arrays["bot_id"] = np.asarray(
        [bot_of[pid] for pid in seat_col], dtype=ROW_DTYPES["bot_id"]
    )
    arrays["episode"] = np.asarray(
        [episode_of[pid] for pid in seat_col], dtype=ROW_DTYPES["episode"]
    )

    seat_table = {
        pid: {
            **asdict(result.seats[pid]),
            "bot": registry.name_for_team(result.seats[pid].team_id),
            "bot_id": bot_of[pid],
            "rank_weight": weight_of[pid],
            "episode": episode_of[pid],
        }
        for pid in seat_ids
    }
    return arrays, seat_table, len(seat_ids)


# --- shards -------------------------------------------------------------------


def shard_name(match_id: str, content_hash: str) -> str:
    return f"replay_{match_id}_{content_hash[:8]}.npz"


def write_replay_shard(
    out_dir: Path,
    result: ConvertResult,
    content_hash: str,
    registry: BotRegistry,
    episode_base: int,
    match_id: str,
) -> Tuple[Path, Dict[str, np.ndarray], Dict[int, dict], int]:
    """Stamp + write one replay's shard and sidecar manifest (both atomic)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, seat_table, episodes_used = stamp_rows(result, registry, episode_base)
    _validate_arrays(arrays)

    npz_path = out_dir / shard_name(match_id, content_hash)
    _atomic_bytes(
        npz_path,
        lambda p: np.savez_compressed(_npz_handle(p), **arrays),
    )

    manifest = {
        "source": result.source,
        "match_id": match_id,
        "content_hash": content_hash,
        "engine_version": result.engine_version,
        "rounds": result.rounds,
        "rows": result.n_rows,
        "winner_pid": result.winner_pid,
        "winner_team": result.seats[result.winner_pid].team_id,
        "seats": {str(pid): meta for pid, meta in seat_table.items()},
        "report": result.report,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _atomic_json(npz_path.with_suffix(".json"), manifest)
    return npz_path, arrays, seat_table, episodes_used


def _npz_handle(path: Path):
    # np.savez_compressed appends ".npz" unless the target already ends in it,
    # which would defeat the atomic-rename tmp path — hand it an open file.
    return open(path, "wb")


def _validate_arrays(arrays: Dict[str, np.ndarray]) -> None:
    n = arrays["obs"].shape[0]
    if arrays["obs"].shape[1] != flat_obs.OBS_WIDTH:
        raise ValueError(
            f"obs width {arrays['obs'].shape[1]} != {flat_obs.OBS_WIDTH}"
        )
    for name, dtype in ROW_DTYPES.items():
        arr = arrays[name]
        if arr.dtype != dtype:
            raise ValueError(f"{name}: dtype {arr.dtype} != {np.dtype(dtype)}")
        if arr.shape[0] != n:
            raise ValueError(f"{name}: {arr.shape[0]} rows != obs {n}")


def read_shard(path: Path) -> Tuple[Dict[str, np.ndarray], dict]:
    """Load one shard + its sidecar manifest."""
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    manifest = json.loads(Path(path).with_suffix(".json").read_text())
    return arrays, manifest


def list_shards(out_dir: Path) -> List[Path]:
    return sorted(Path(out_dir).glob(SHARD_GLOB))


# --- index --------------------------------------------------------------------


def empty_index() -> dict:
    return {
        "version": _INDEX_VERSION,
        "bots": {},           # name -> interned id
        "replays": {},        # content_hash -> {source, match_id, shard, rows, engine_version}
        "quarantined": {},    # content_hash -> {source, reason}
        "rows_per_bot": {},   # name -> row count
        "next_episode": 0,
    }


def load_index(out_dir: Path) -> dict:
    path = Path(out_dir) / INDEX_NAME
    if not path.exists():
        return empty_index()
    index = json.loads(path.read_text())
    if index.get("version") != _INDEX_VERSION:
        raise ValueError(f"{path}: unsupported index version {index.get('version')}")
    return index


def save_index(out_dir: Path, index: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return _atomic_json(out / INDEX_NAME, index)


def registry_from_index(index: dict, catalog: Optional[dict] = None) -> BotRegistry:
    return BotRegistry(names=index.get("bots", {}), catalog=catalog)
