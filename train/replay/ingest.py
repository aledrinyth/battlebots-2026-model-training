"""Idempotent streaming ingestion: replays → shards + index (impl doc §9).

Replays arrive over days; re-running ``ingest_dir`` after each batch is the
watch mode. Guarantees:

* **Version guard is hard.** A replay outside the audited compatible set
  (schema.COMPATIBLE_ENGINE_VERSIONS — see notes/tourney-replay-stats.md §3)
  is quarantined, never silently mixed. The impl doc's exact-pin match was
  widened to that audited set after the .13→.15 diffs were read; pass
  ``allowed_versions`` to tighten or extend deliberately.
* **Idempotent** via a processed-set keyed by file content hash (sha256);
  re-running skips everything already converted, and a re-appearing file
  with identical bytes never duplicates rows.
* **Resumable**: a corrupt/malformed/aborted replay is recorded in the
  index's quarantine table (and optionally moved to ``_quarantine/``) and
  never blocks the rest.
* A shard is only registered after `check_report` passes — §7.1 acceptance
  is enforced on **every** ingested replay, not just the validation one.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from train.replay import dataset
from train.replay.convert import check_report, convert_replay
from train.replay.reconstruct import ReplayError
from train.replay.schema import COMPATIBLE_ENGINE_VERSIONS

QUARANTINE_DIR = "_quarantine"


@dataclass
class IngestReport:
    ingested: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    quarantined: List[Tuple[str, str]] = field(default_factory=list)  # (source, reason)
    rows_added: int = 0

    def summary(self) -> str:
        return (
            f"{len(self.ingested)} ingested (+{self.rows_added} rows), "
            f"{len(self.skipped)} already present, "
            f"{len(self.quarantined)} quarantined"
        )


def _match_id(path: Path) -> str:
    return path.name.replace("-replay.json", "").replace(".json", "")


def _quarantine(
    report: IngestReport,
    index: dict,
    path: Path,
    content_hash: str,
    reason: str,
    move: bool,
) -> None:
    report.quarantined.append((str(path), reason))
    index["quarantined"][content_hash] = {"source": path.name, "reason": reason}
    if move:
        qdir = path.parent / QUARANTINE_DIR
        qdir.mkdir(exist_ok=True)
        shutil.move(str(path), qdir / path.name)


def ingest_paths(
    paths: Iterable[Path],
    out_dir: Path,
    allowed_versions: Optional[frozenset] = None,
    catalog: Optional[Dict[str, str]] = None,
    move_quarantined: bool = False,
    validate_obs: bool = True,
) -> IngestReport:
    """Convert every not-yet-ingested replay in ``paths`` into shards."""
    out_dir = Path(out_dir)
    allowed = allowed_versions or COMPATIBLE_ENGINE_VERSIONS
    index = dataset.load_index(out_dir)
    registry = dataset.registry_from_index(index, catalog)
    report = IngestReport()

    for path in paths:
        path = Path(path)
        raw = path.read_bytes()
        content_hash = hashlib.sha256(raw).hexdigest()

        known = index["replays"].get(content_hash)
        if known and (out_dir / known["shard"]).exists():
            report.skipped.append(str(path))
            continue
        if content_hash in index["quarantined"]:
            report.skipped.append(str(path))
            continue

        try:
            events = json.loads(raw)
        except json.JSONDecodeError as ex:
            _quarantine(report, index, path, content_hash,
                        f"unparseable JSON: {ex}", move_quarantined)
            continue
        if not isinstance(events, list) or not events:
            _quarantine(report, index, path, content_hash,
                        "top level is not a non-empty event list", move_quarantined)
            continue

        version = events[0].get("engine_version") if isinstance(events[0], dict) else None
        if version not in allowed:
            _quarantine(
                report, index, path, content_hash,
                f"engine {version!r} not in compatible set {sorted(allowed)}",
                move_quarantined,
            )
            continue

        try:
            result = convert_replay(events, source=str(path), validate_obs=validate_obs)
        except (ReplayError, KeyError, TypeError, ValueError) as ex:
            _quarantine(report, index, path, content_hash,
                        f"conversion failed: {ex}", move_quarantined)
            continue

        failures = check_report(result)
        if failures:
            _quarantine(report, index, path, content_hash,
                        "; ".join(failures), move_quarantined)
            continue

        match_id = _match_id(path)
        shard_path, arrays, seat_table, episodes_used = dataset.write_replay_shard(
            out_dir,
            result,
            content_hash,
            registry,
            episode_base=index["next_episode"],
            match_id=match_id,
        )
        index["next_episode"] += episodes_used
        index["bots"] = registry.to_table()
        index["replays"][content_hash] = {
            "source": path.name,
            "match_id": match_id,
            "shard": shard_path.name,
            "rows": result.n_rows,
            "engine_version": result.engine_version,
        }
        for meta in seat_table.values():
            name = meta["bot"]
            index["rows_per_bot"][name] = (
                index["rows_per_bot"].get(name, 0) + meta["n_rows"]
            )
        report.ingested.append(str(path))
        report.rows_added += result.n_rows
        # Persist after every replay so an interrupted run resumes cleanly.
        dataset.save_index(out_dir, index)

    dataset.save_index(out_dir, index)
    return report


def ingest_dir(
    replay_dir: Path,
    out_dir: Path,
    **kwargs,
) -> IngestReport:
    replay_dir = Path(replay_dir)
    paths = sorted(
        p for p in replay_dir.glob("*.json") if p.parent.name != QUARANTINE_DIR
    )
    return ingest_paths(paths, out_dir, **kwargs)
