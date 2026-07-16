"""Per-bot behaviour mining over ingested shards (impl doc §10).

Everything is computed from the dataset rows themselves (obs columns are
decoded via ``flat_obs.FIELD_OFFSETS``), so the report works on any shard
directory without the source replays. Feeds reward-shaping decisions and
flags degenerate dominant strategies to target with exploiters; the
``winners`` slice (rank 1 seats only) is the BC-toward-winners preview.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from bot.adapters import act_v1
from bot.adapters.obs_v1 import ENEMY_DIM, ENEMY_MAX, VIRUS_DIM, VIRUS_MAX
from train.harness import flat_obs
from train.replay import dataset


def _col(arrs: Dict[str, np.ndarray], name: str) -> np.ndarray:
    start, stop, shape = flat_obs.FIELD_OFFSETS[name]
    return arrs["obs"][:, start:stop].reshape((-1,) + shape)


def _heading_vecs() -> np.ndarray:
    return np.asarray(
        [act_v1.bin_to_vector(i) for i in range(act_v1.HEADING_BINS)],
        dtype=np.float64,
    )


_BIN_VECS = _heading_vecs()


def _bot_stats(rows: Dict[str, np.ndarray]) -> dict:
    n = rows["obs"].shape[0]
    enemy = _col(rows, "enemy")            # (n, ENEMY_MAX, ENEMY_DIM)
    enemy_mask = _col(rows, "enemy_mask")  # (n, ENEMY_MAX)
    virus = _col(rows, "virus")
    virus_mask = _col(rows, "virus_mask")
    self_vec = _col(rows, "self_vec")

    has_heading = rows["has_heading"].astype(bool)
    bins = rows["heading_bin"].astype(np.int64)

    # heading persistence: same bin as the previous row of the same episode
    # (both rows real headings, not crossing a reset boundary)
    order = np.lexsort((rows["round"], rows["episode"]))
    ep, bn, hh = rows["episode"][order], bins[order], has_heading[order]
    rs = rows["is_reset"][order].astype(bool)
    prev_same = (
        (ep[1:] == ep[:-1]) & ~rs[1:] & hh[1:] & hh[:-1]
    )
    persistence = (
        float((bn[1:][prev_same] == bn[:-1][prev_same]).mean())
        if prev_same.any()
        else float("nan")
    )

    # aggression proxy: of frames with ≥1 eatable enemy in view (i_can_eat,
    # feature 8), how often does the commanded heading point within 60° of
    # the nearest such enemy?
    eatable = (enemy[:, :, 8] > 0.5) & (enemy_mask > 0.5)
    has_prey = eatable.any(axis=1)
    approach_rate = float("nan")
    idx = np.flatnonzero(has_prey & has_heading)
    if idx.size:
        d2 = np.where(eatable[idx], (enemy[idx, :, 0] ** 2 + enemy[idx, :, 1] ** 2), np.inf)
        nearest = d2.argmin(axis=1)
        dx = enemy[idx, nearest, 0]
        dy = enemy[idx, nearest, 1]
        norm = np.hypot(dx, dy)
        ok = norm > 0
        hv = _BIN_VECS[bins[idx]]
        cos = np.where(
            ok, (hv[:, 0] * dx + hv[:, 1] * dy) / np.where(ok, norm, 1.0), 1.0
        )
        approach_rate = float((cos >= math.cos(math.radians(60))).mean())

    # virus behaviour: fraction of frames with a *dangerous* virus (feature 4)
    # within a quarter of the half-window
    vis_virus = virus_mask > 0.5
    dangerous_near = (
        (virus[:, :, 4] > 0.5) & vis_virus & (virus[:, :, 2] < 0.25)
    ).any(axis=1)

    # wall proximity: self_vec[9..13] hold the *view-center* wall distances
    # (normalized by half-vision), whose floor is exactly 1.0 — the window
    # clamps at half — so "pinned against a wall" reads as ≈ 1.0.
    wall_min = self_vec[:, 9:13].min(axis=1)

    return {
        "rows": int(n),
        "episodes": int(np.unique(rows["episode"]).size),
        "split_rate": float(rows["split"].mean()) if n else float("nan"),
        "splits": int(rows["split"].sum()),
        "stay_rate": float((~has_heading).mean()) if n else float("nan"),
        "heading_persistence": persistence,
        "prey_in_view_rate": float(has_prey.mean()) if n else float("nan"),
        "approach_rate_when_prey": approach_rate,
        "dangerous_virus_near_rate": float(dangerous_near.mean()) if n else float("nan"),
        "wall_hug_rate": float((wall_min <= 1.02).mean()) if n else float("nan"),
        "mean_rank_weight": float(rows["rank_weight"].mean()) if n else float("nan"),
    }


def mine_shards(shard_dir: Path) -> dict:
    """Aggregate per-bot behaviour stats over every shard in ``shard_dir``."""
    shard_dir = Path(shard_dir)
    index = dataset.load_index(shard_dir)
    id_to_name = {v: k for k, v in index.get("bots", {}).items()}

    per_bot: Dict[str, List[Dict[str, np.ndarray]]] = {}
    matches_per_bot: Dict[str, int] = {}
    for path in dataset.list_shards(shard_dir):
        arrays, _manifest = dataset.read_shard(path)
        for bot_id in np.unique(arrays["bot_id"]):
            name = id_to_name.get(int(bot_id), f"bot_{bot_id}")
            sel = arrays["bot_id"] == bot_id
            per_bot.setdefault(name, []).append(
                {k: v[sel] for k, v in arrays.items()}
            )
            matches_per_bot[name] = matches_per_bot.get(name, 0) + 1

    bots = {}
    for name, chunks in per_bot.items():
        rows = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        stats = _bot_stats(rows)
        stats["matches"] = matches_per_bot[name]
        winners = {k: v[rows["rank_weight"] == 1.0] for k, v in rows.items()}
        stats["winners"] = (
            _bot_stats(winners) if winners["obs"].shape[0] else None
        )
        bots[name] = stats

    return {
        "shards": len(dataset.list_shards(shard_dir)),
        "total_rows": int(sum(b["rows"] for b in bots.values())),
        "bots": dict(
            sorted(bots.items(), key=lambda kv: -kv[1]["mean_rank_weight"])
        ),
    }


def to_markdown(report: dict) -> str:
    lines = [
        "# Replay mining report",
        "",
        f"{report['shards']} shards, {report['total_rows']} rows.",
        "",
        "| bot | matches | rows | mean_rw | split% | stay% | persist"
        " | prey_seen% | approach% | virus_near% | wall% |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in report["bots"].items():
        lines.append(
            f"| {name} | {s['matches']} | {s['rows']} "
            f"| {s['mean_rank_weight']:.2f} | {100 * s['split_rate']:.1f} "
            f"| {100 * s['stay_rate']:.1f} | {s['heading_persistence']:.2f} "
            f"| {100 * s['prey_in_view_rate']:.1f} "
            f"| {100 * s['approach_rate_when_prey']:.1f} "
            f"| {100 * s['dangerous_virus_near_rate']:.1f} "
            f"| {100 * s['wall_hug_rate']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(shard_dir: Path, out_md: Optional[Path] = None) -> dict:
    report = mine_shards(shard_dir)
    json_path = Path(shard_dir) / "mining_report.json"
    json_path.write_text(json.dumps(report, indent=1, sort_keys=True))
    if out_md is not None:
        Path(out_md).write_text(to_markdown(report))
    return report
