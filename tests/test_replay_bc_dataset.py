"""R2 feed: replay shards → BC arrays (episode selection, contiguity,
demos-format contract) + the train/bc.py extensions used by replay BC.

The exporter tests are numpy-only and run in both venvs; the ``train_bc``
weighting/keep-best tests need torch and skip in the bot venv (same
convention as test_demos_bc.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from train.replay.bc_dataset import build_bc_data, write_bc_data
from train.replay.ingest import ingest_dir

from test_replay_reconstruct import synthetic_events


@pytest.fixture()
def shard_dir(tmp_path: Path) -> Path:
    replays = tmp_path / "replays"
    replays.mkdir()
    (replays / "match-1001-replay.json").write_text(json.dumps(synthetic_events()))
    other = synthetic_events()
    other[0]["players"][0]["team_id"] = 53  # our seat wins match 2
    other[0]["players"][1]["team_id"] = 30
    (replays / "match-2002-replay.json").write_text(json.dumps(other))
    out = tmp_path / "shards"
    report = ingest_dir(replays, out)
    assert len(report.ingested) == 2
    return out


def test_winners_cut_selects_whole_episodes(shard_dir: Path):
    # min_rank_weight=1.0 -> rank-1 seats only; team 53 excluded even though
    # it won match 2002
    data, report = build_bc_data(shard_dir, min_rank_weight=1.0)
    assert report["episodes"] == 1
    assert set(report["teams"]) == {"team_10"}
    assert report["rows"] == 5  # p0's full trajectory from match 1001

    # demos-format contract (exact dtypes train/bc.py consumes)
    assert data["obs"].dtype == np.float32 and data["obs"].shape == (5, 736)
    assert data["heading"].dtype == np.int64
    assert data["split"].dtype == np.float32
    assert data["label_ok"].dtype == np.bool_
    assert data["is_reset"].dtype == np.bool_
    assert data["weight"].dtype == np.float32
    assert (data["weight"] == 1.0).all()

    # episode-contiguous: first row is a reset frame, rounds ascend
    assert bool(data["is_reset"][0])
    assert data["split"].sum() == 1  # p0's round-4 split survived selection


def test_team_allowlist_and_threshold(shard_dir: Path):
    # clone cut: team 20 finished rank 2 of 2 -> weight 0 -> needs threshold 0
    data, report = build_bc_data(
        shard_dir, min_rank_weight=0.0, teams=[20], exclude_teams=())
    assert set(report["teams"]) == {"team_20"}
    assert report["rows"] == 3
    # its stay row is masked but keeps the split label
    assert data["label_ok"].sum() == 2

    with pytest.raises(ValueError, match="no episodes selected"):
        build_bc_data(shard_dir, min_rank_weight=1.0, teams=[99])


def test_write_bc_data_roundtrip(tmp_path: Path, shard_dir: Path):
    out = tmp_path / "bc" / "winners.npz"
    report = write_bc_data(shard_dir, out, min_rank_weight=1.0)
    loaded = dict(np.load(out))
    assert loaded["obs"].shape[0] == report["rows"] == 5
    sidecar = json.loads(Path(str(out) + ".json").read_text())
    assert sidecar["teams"]["team_10"]["episodes"] == 1


def test_cli_bc_dataset(tmp_path: Path, shard_dir: Path, capsys):
    from train.replay.cli import main
    out = tmp_path / "winners.npz"
    assert main(["bc-dataset", "--shards", str(shard_dir),
                 "--out", str(out), "--min-rank", "1"]) == 0
    assert "team_10" in capsys.readouterr().out
    assert out.exists()


# --- train_bc extensions (torch; skipped in the bot venv) --------------------


def _tiny_weighted_data(n=256, seed=0):
    torch_mod = pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    return torch_mod, {
        "obs": rng.standard_normal((n, 736)).astype(np.float32),
        "heading": rng.integers(0, 24, n).astype(np.int64),
        "split": (rng.random(n) < 0.1).astype(np.float32),
        "label_ok": np.ones(n, dtype=bool),
        "is_reset": (np.arange(n) % 64 == 0),
        "weight": np.full(n, 0.5, dtype=np.float32),
    }


def test_train_bc_accepts_weight_and_pos_weight():
    _torch, data = _tiny_weighted_data()
    from train.bc import train_bc
    policy, metrics = train_bc(data, epochs=1, window=8, batch_windows=8,
                               split_pos_weight=4.0, log=False)
    assert "val_split_recall" in metrics and "val_split_precision" in metrics
    assert np.isfinite(metrics["train_loss"])


def test_train_bc_keep_best_restores_best_epoch():
    _torch, data = _tiny_weighted_data()
    from train.bc import train_bc
    policy, metrics = train_bc(data, epochs=3, window=8, batch_windows=8,
                               keep_best=True, log=False)
    assert 1 <= metrics["best_epoch"] <= 3
    assert metrics["best_val_adj_acc"] >= 0.0


def test_train_bc_unweighted_path_unchanged():
    """Without a weight column the loss path is the original one — same
    metrics keys, still runs (regression guard for the extension)."""
    _torch, data = _tiny_weighted_data()
    data.pop("weight")
    from train.bc import train_bc
    _policy, metrics = train_bc(data, epochs=1, window=8, batch_windows=8,
                                log=False)
    assert "val_acc" in metrics and "val_adj_acc" in metrics
