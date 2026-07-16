"""Replay shards → BC training arrays (v3 R2 feed; impl doc §13.1).

Selects **whole episodes** — one ``(replay, seat)`` trajectory each — and
reorders their rows time-contiguously, because ``train/bc.py`` replays
windows through the GRU and only episode-contiguous rows give it real
hidden-state dynamics. Selection is therefore per-episode by construction:
``rank_weight`` is constant within an episode, so a threshold never splits
one.

Output is exactly the :mod:`train.harness.demos` dict contract (``obs``
f32, ``heading`` i64, ``split`` f32, ``label_ok`` bool, ``is_reset``
bool) plus a ``weight`` f32 column (per-row ``rank_weight``) that
``train.bc`` uses when present. Numpy-only — buildable in either venv.

Defaults implement the handoff's "BC-toward-winners": every top-3-of-8
episode from every team except ours.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from train.replay import dataset

#: rank 3 of 8 → (8-3)/7. Selecting "top-3 finishes" is the winners cut.
TOP3_OF_8_WEIGHT = 5.0 / 7.0
OUR_TEAM = 53


def build_bc_data(
    shard_dir: Path,
    min_rank_weight: float = TOP3_OF_8_WEIGHT,
    exclude_teams: Iterable[int] = (OUR_TEAM,),
    teams: Optional[Iterable[int]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Assemble BC arrays from every shard in ``shard_dir``.

    ``teams`` (optional) restricts to an allowlist (clone training);
    ``min_rank_weight`` keeps only episodes whose seat finished at or above
    that quality; ``exclude_teams`` drops our own seats (never imitate the
    bot being replaced). Returns ``(arrays, report)``.
    """
    exclude = {int(t) for t in exclude_teams}
    allow = None if teams is None else {int(t) for t in teams}

    chunks = []
    episodes = 0
    per_team: Dict[str, Dict[str, float]] = {}
    for path in dataset.list_shards(shard_dir):
        arrays, manifest = dataset.read_shard(path)
        for pid_str, meta in manifest["seats"].items():
            team = int(meta["team_id"])
            if team in exclude:
                continue
            if allow is not None and team not in allow:
                continue
            if meta["rank_weight"] < min_rank_weight:
                continue
            sel = arrays["episode"] == meta["episode"]
            if not sel.any():
                continue
            rows = {k: v[sel] for k, v in arrays.items()}
            order = np.argsort(rows["round"], kind="stable")
            rows = {k: v[order] for k, v in rows.items()}
            if not rows["is_reset"][0]:
                raise ValueError(
                    f"{path.name} episode {meta['episode']}: first row is not "
                    "a reset frame — episode reordering is broken"
                )
            chunks.append(rows)
            episodes += 1
            t = per_team.setdefault(
                meta["bot"], {"episodes": 0, "rows": 0, "splits": 0}
            )
            t["episodes"] += 1
            t["rows"] += int(rows["obs"].shape[0])
            t["splits"] += int(rows["split"].sum())

    if not chunks:
        raise ValueError(
            f"no episodes selected from {shard_dir} "
            f"(min_rank_weight={min_rank_weight}, teams={allow}, "
            f"exclude={exclude})"
        )

    cat = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    heading = cat["heading_bin"].astype(np.int64)
    label_ok = cat["has_heading"].astype(bool)
    heading[~label_ok] = 0  # masked rows carry a dummy bin, like demos.py
    data = {
        "obs": cat["obs"].astype(np.float32, copy=False),
        "heading": heading,
        "split": cat["split"].astype(np.float32),
        "label_ok": label_ok,
        "is_reset": cat["is_reset"].astype(bool),
        "weight": cat["rank_weight"].astype(np.float32, copy=False),
    }
    n = int(data["obs"].shape[0])
    report = {
        "rows": n,
        "episodes": episodes,
        "split_rows": int(data["split"].sum()),
        "split_rate": float(data["split"].mean()),
        "stay_rows": int((~label_ok).sum()),
        "mean_weight": float(data["weight"].mean()),
        "min_rank_weight": min_rank_weight,
        "teams": dict(sorted(per_team.items(),
                             key=lambda kv: -kv[1]["rows"])),
    }
    return data, report


def write_bc_data(
    shard_dir: Path,
    out_path: Path,
    **kwargs,
) -> Dict:
    """Build + save (compressed npz, atomic) with a JSON sidecar report."""
    data, report = build_bc_data(shard_dir, **kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **data)
    tmp.replace(out_path)
    Path(str(out_path) + ".json").write_text(json.dumps(report, indent=1))
    return report
