"""R0 dataset + ingestion acceptance (impl doc §12).

Shard round-trip, manifest/index integrity, idempotent re-ingest, hard
version quarantine, per-shard §7.1 enforcement, and the property test:
every ingested row unflattens into a valid ``CanonicalObs`` with a legal
heading bin.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from train.harness import flat_obs
from train.replay import dataset
from train.replay.identity import BotRegistry
from train.replay.ingest import ingest_dir, ingest_paths
from train.replay.mine import mine_shards, to_markdown
from train.replay.schema import ROW_DTYPES

from test_replay_reconstruct import synthetic_events


@pytest.fixture()
def replay_dir(tmp_path: Path) -> Path:
    d = tmp_path / "replays"
    d.mkdir()
    (d / "match-1001-replay.json").write_text(json.dumps(synthetic_events()))
    return d


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "shards"


def test_ingest_shard_roundtrip_and_index(replay_dir: Path, out_dir: Path):
    report = ingest_dir(replay_dir, out_dir)
    assert report.summary().startswith("1 ingested (+8 rows)")
    assert not report.quarantined

    shards = dataset.list_shards(out_dir)
    assert len(shards) == 1
    arrays, manifest = dataset.read_shard(shards[0])

    # exact schema
    assert set(arrays) == set(ROW_DTYPES)
    for name, dtype in ROW_DTYPES.items():
        assert arrays[name].dtype == dtype, name
    assert arrays["obs"].shape == (8, flat_obs.OBS_WIDTH)

    # manifest integrity
    assert manifest["match_id"] == "match-1001"
    assert manifest["rows"] == 8
    assert manifest["winner_pid"] == 0 and manifest["winner_team"] == 10
    seats = manifest["seats"]
    assert seats["0"]["bot"] == "team_10" and seats["0"]["final_rank"] == 1
    assert seats["1"]["bot"] == "team_20" and seats["1"]["final_rank"] == 2

    # stamped columns: winner seat weight 1.0, loser 0.0 (2 players)
    w = arrays["rank_weight"]
    assert set(w[arrays["seat"] == 0]) == {1.0}
    assert set(w[arrays["seat"] == 1]) == {0.0}
    # one episode per (replay, seat)
    assert set(arrays["episode"][arrays["seat"] == 0]) == {0}
    assert set(arrays["episode"][arrays["seat"] == 1]) == {1}

    index = dataset.load_index(out_dir)
    assert index["next_episode"] == 2
    assert index["rows_per_bot"] == {"team_10": 5, "team_20": 3}
    assert len(index["replays"]) == 1
    (entry,) = index["replays"].values()
    assert entry["shard"] == shards[0].name and entry["rows"] == 8


def test_reingest_is_idempotent(replay_dir: Path, out_dir: Path):
    first = ingest_dir(replay_dir, out_dir)
    assert len(first.ingested) == 1
    before = dataset.load_index(out_dir)

    second = ingest_dir(replay_dir, out_dir)
    assert second.ingested == [] and len(second.skipped) == 1
    assert dataset.load_index(out_dir) == before
    assert len(dataset.list_shards(out_dir)) == 1


def test_version_quarantine_is_hard(replay_dir: Path, out_dir: Path):
    bad = synthetic_events()
    bad[0]["engine_version"] = "2099.1.1"
    (replay_dir / "match-2002-replay.json").write_text(json.dumps(bad))
    corrupt = replay_dir / "match-3003-replay.json"
    corrupt.write_text("{not json")
    aborted = synthetic_events()
    aborted.insert(3, {"event_type": "event_player_banned"})
    (replay_dir / "match-4004-replay.json").write_text(json.dumps(aborted))

    report = ingest_dir(replay_dir, out_dir)
    assert len(report.ingested) == 1  # only the good one
    reasons = {Path(src).name: reason for src, reason in report.quarantined}
    assert "not in compatible set" in reasons["match-2002-replay.json"]
    assert "unparseable JSON" in reasons["match-3003-replay.json"]
    assert "aborted game" in reasons["match-4004-replay.json"]

    index = dataset.load_index(out_dir)
    assert len(index["quarantined"]) == 3
    assert len(index["replays"]) == 1

    # quarantined files are remembered — the re-run skips them
    rerun = ingest_dir(replay_dir, out_dir)
    assert rerun.ingested == [] and rerun.quarantined == []
    assert len(rerun.skipped) == 4

    # --allow-version-style override rescues a known-good future version
    out2 = out_dir.parent / "shards2"
    report2 = ingest_paths(
        [replay_dir / "match-2002-replay.json"], out2,
        allowed_versions=frozenset({"2099.1.1"}),
    )
    assert len(report2.ingested) == 1


def test_catalog_overlays_names(replay_dir: Path, out_dir: Path):
    ingest_dir(replay_dir, out_dir, catalog={"10": "our_bot"})
    index = dataset.load_index(out_dir)
    assert index["rows_per_bot"] == {"our_bot": 5, "team_20": 3}


def test_identical_bytes_under_new_name_dedupe(replay_dir: Path, out_dir: Path):
    """The processed-set keys on content hash — the same match re-downloaded
    under another filename must not duplicate rows."""
    (replay_dir / "match-9999-replay.json").write_text(
        json.dumps(synthetic_events())
    )
    report = ingest_dir(replay_dir, out_dir)
    assert len(report.ingested) == 1 and len(report.skipped) == 1
    assert len(dataset.list_shards(out_dir)) == 1


def test_episode_numbering_advances_across_replays(replay_dir: Path, out_dir: Path):
    other = synthetic_events()
    other[0]["players"][0]["team_id"] = 30  # genuinely different match bytes
    (replay_dir / "match-5005-replay.json").write_text(json.dumps(other))
    ingest_dir(replay_dir, out_dir)

    episodes = set()
    for path in dataset.list_shards(out_dir):
        arrays, _ = dataset.read_shard(path)
        episodes.update(arrays["episode"].tolist())
    assert episodes == {0, 1, 2, 3}
    assert dataset.load_index(out_dir)["next_episode"] == 4


def test_property_ingested_rows_are_valid(replay_dir: Path, out_dir: Path):
    """Impl doc §12 property: every ingested row unflattens into a valid
    CanonicalObs and carries a legal heading bin."""
    ingest_dir(replay_dir, out_dir)
    (path,) = dataset.list_shards(out_dir)
    arrays, _ = dataset.read_shard(path)
    for i in range(arrays["obs"].shape[0]):
        obs = flat_obs.unflatten(arrays["obs"][i])
        obs.validate()
        hb = int(arrays["heading_bin"][i])
        assert -1 <= hb < 24
        assert (hb == -1) == (arrays["has_heading"][i] == 0)


def test_registry_interning_is_stable():
    reg = BotRegistry(catalog={"53": "us"})
    a = reg.id_for_team(53)
    b = reg.id_for_team(49)
    assert reg.id_for_team(53) == a and reg.id_for_team(49) == b
    assert reg.name_for_team(53) == "us" and reg.name_for_team(49) == "team_49"
    # round-trip through the serialized table
    reg2 = BotRegistry(names=reg.to_table())
    assert reg2.intern("us") == a


def test_mining_report_over_ingested_shards(replay_dir: Path, out_dir: Path):
    ingest_dir(replay_dir, out_dir)
    report = mine_shards(out_dir)
    assert report["total_rows"] == 8
    bots = report["bots"]
    assert set(bots) == {"team_10", "team_20"}
    assert bots["team_10"]["rows"] == 5
    assert bots["team_10"]["mean_rank_weight"] == 1.0
    assert bots["team_10"]["splits"] == 1
    assert bots["team_20"]["stay_rate"] == pytest.approx(1 / 3)
    # winners-only slice exists exactly for the winning seat
    assert bots["team_10"]["winners"] is not None
    assert bots["team_20"]["winners"] is None
    md = to_markdown(report)
    assert "team_10" in md and "| bot |" in md


def test_cli_check_and_ingest(tmp_path: Path, replay_dir: Path, capsys):
    from train.replay.cli import main

    replay = replay_dir / "match-1001-replay.json"
    assert main(["check", str(replay)]) == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out and "rows built: 8" in out

    shards = tmp_path / "cli_shards"
    assert main(["ingest", "--replays", str(replay_dir), "--out", str(shards)]) == 0
    assert "1 ingested" in capsys.readouterr().out
    assert len(dataset.list_shards(shards)) == 1
