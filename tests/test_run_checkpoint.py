"""P3.4 — config discipline, logging, checkpointing, resume.

Covers the P3.4 acceptance directly:

* **Config hash / reproducibility** — a run is identified by a stable hash over
  its algorithmic knobs; operational fields (device, run length) don't perturb
  it, but a real hyper-parameter does.
* **Registry schema** — the checkpoint/registry row carries engine pin + adapter
  versions + config hash + config (the schema pipeline P11.1 hardens).
* **Local logging** — the metrics log appends one record per update and can be
  truncated to a checkpoint so a resume leaves no gap or rewind.
* **Checkpointing** — self-describing (policy + optimizer + arch + provenance),
  prunes to `keep`, and is `evaluate`-loadable.
* **Kill and resume mid-flight; curves continue seamlessly** — two invocations
  against one run dir produce a single contiguous update curve, the second
  invocation genuinely *continues* (does not restart), and `--fresh` restarts.

Train-side (torch + yaml); skipped in the match-legal bot venv.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("yaml")

import torch  # noqa: E402

# Spawned actor children must import ``train`` — put the repo root on PYTHONPATH
# (inherited by the children) so the test is invocation-independent.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = _REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

from train.harness.config import TrainConfig  # noqa: E402
from train.harness.provenance import (  # noqa: E402
    adapter_versions,
    config_hash,
    engine_pin,
    registry_row,
)
from train.harness.run import (  # noqa: E402
    Checkpointer,
    MetricsLogger,
    RunConfig,
    RunManager,
)
from train.model import build_policy  # noqa: E402


# --- config hash / provenance (pure) ---------------------------------------

def test_config_hash_stable_and_selective():
    cfg = TrainConfig(policy_seats=(0,), num_actors=8)
    h = config_hash(cfg)
    assert h == config_hash(cfg)                      # deterministic
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)

    # operational fields must NOT change the experiment identity
    op = dataclasses.replace(cfg, device="cuda", num_updates=99, duration_s=42.0,
                             log_every=5, min_infer_batch=512, barrier_timeout_s=99.0)
    assert config_hash(op) == h

    # a real hyper-parameter MUST change it
    assert config_hash(dataclasses.replace(cfg, lr=1e-2)) != h
    assert config_hash(dataclasses.replace(cfg, policy_seats=(0, 1))) != h
    assert config_hash(dataclasses.replace(cfg, entropy_coef=0.5)) != h


def test_registry_row_has_full_schema():
    cfg = TrainConfig()
    row = registry_row(cfg, run_name="r", stage="S0", extra={"final_version": 7})
    assert row["config_hash"] == config_hash(cfg)
    assert row["engine"] == engine_pin()
    assert row["adapters"] == adapter_versions()
    assert row["engine"]["kit_version"]  # non-empty (agario-kit is installed)
    assert row["adapters"] == {"obs": "obs_v1", "act": "act_v1", "model": "agar_v1"}
    assert row["run_name"] == "r" and row["stage"] == "S0" and row["final_version"] == 7
    assert isinstance(row["config"], dict) and "policy_seats" in row["config"]


# --- RunConfig YAML --------------------------------------------------------

def test_run_config_yaml_roundtrip(tmp_path):
    import yaml
    src = RunConfig(name="t", stage="S0", out_dir=str(tmp_path), total_updates=10,
                    train={"num_actors": 4, "policy_seats": [0], "use_skeleton": True})
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(src.to_dict()))
    got = RunConfig.load(str(p))
    assert got.to_dict() == src.to_dict()
    cfg = got.base_train_config()
    assert cfg.policy_seats == (0,)     # list → tuple coercion
    assert cfg.batch_rows == 4


def test_stage0_config_loads():
    rc = RunConfig.load(os.path.join(_REPO, "configs", "stage0_foodonly.yaml"))
    cfg = rc.base_train_config()
    assert rc.stage == "S0-food-only"
    assert cfg.policy_seats == (0,)
    assert cfg.scripted_names == ("dont_move",)  # food-only: opponents sit still


def test_run_config_rejects_unknown_field():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"name": "t", "bogus": 1})


# --- MetricsLogger ---------------------------------------------------------

def test_metrics_logger_append_and_truncate(tmp_path):
    log = MetricsLogger(str(tmp_path / "metrics.jsonl"))
    for u in range(1, 6):
        log.log({"update": u, "loss": -0.1 * u})
    recs = log.records()
    assert [r["update"] for r in recs] == [1, 2, 3, 4, 5]

    log.truncate_after(3)   # a resume from checkpoint v3 drops the stale tail
    assert [r["update"] for r in log.records()] == [1, 2, 3]

    log.log({"update": 4, "loss": -0.4})   # curve continues seamlessly
    assert [r["update"] for r in log.records()] == [1, 2, 3, 4]


# --- Checkpointer ----------------------------------------------------------

def _tiny_policy_and_opt():
    policy = torch.nn.Linear(4, 3)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    return policy, opt


def test_checkpointer_roundtrip_prune_and_evaluate_shape(tmp_path):
    arch = {"model_version": "agar_v1", "use_skeleton": True}
    prov = {"config_hash": "deadbeef", "engine": engine_pin()}
    ck = Checkpointer(str(tmp_path), arch, prov, keep=2, interval_s=1800.0)

    policy, opt = _tiny_policy_and_opt()
    # interval hasn't elapsed → maybe_save is a no-op; force writes.
    assert ck.maybe_save(1, policy, opt) is None
    for v in (1, 2, 3):
        ck.save(v, policy, opt, extra={"version": v})
    assert ck.latest_version() == 3
    # pruned to the 2 most recent
    remaining = sorted(int(os.path.basename(p)[6:-3]) for p in _pt_files(tmp_path))
    assert remaining == [2, 3]

    blob = ck.load_latest()
    assert blob["version"] == 3
    assert "state_dict" in blob and "optim_state" in blob    # resume-complete
    assert blob["arch"] == arch                              # evaluate reads this
    assert blob["provenance"]["config_hash"] == "deadbeef"


def _pt_files(d):
    import glob
    return glob.glob(os.path.join(str(d), "ckpt_v*.pt"))


# --- kill & resume (spawns actor processes) --------------------------------

def _tiny_run_cfg(out_dir, total_updates):
    # use_skeleton + tiny topology: this exercises the run-management plumbing,
    # not learning. checkpoint_interval_s=0 ⇒ a checkpoint after every update.
    return RunConfig(
        name="rc", stage="test", out_dir=str(out_dir),
        total_updates=total_updates, checkpoint_interval_s=0.0, keep_checkpoints=10,
        log_every=1, device="cpu",
        train={"num_actors": 2, "policy_seats": [0, 1], "rollout_len": 4,
               "use_skeleton": True, "base_seed": 7},
    )


def test_kill_and_resume_curves_continue_seamlessly(tmp_path):
    # Invocation A — first 2 updates.
    outA = RunManager(_tiny_run_cfg(tmp_path, total_updates=2)).run()
    assert outA["updates_this_invocation"] == 2
    assert outA["final_version"] == 2

    log = MetricsLogger(os.path.join(str(tmp_path), "rc", "metrics.jsonl"))
    assert [r["update"] for r in log.records()] == [1, 2]

    # Invocation B — same run dir, resumes and runs 2 more (total 4).
    outB = RunManager(_tiny_run_cfg(tmp_path, total_updates=4)).run()
    assert outB["updates_this_invocation"] == 2      # continued, not restarted
    assert outB["final_version"] == 4
    assert outB["config_hash"] == outA["config_hash"]

    # One contiguous, strictly-increasing curve across the kill — no gap, no rewind.
    updates = [r["update"] for r in log.records()]
    assert updates == [1, 2, 3, 4]

    # run.json registry row reflects the final state.
    import json
    row = json.loads((tmp_path / "rc" / "run.json").read_text())
    assert row["final_version"] == 4 and row["status"] == "done"
    assert row["config_hash"] == outA["config_hash"]


def test_already_complete_is_a_noop(tmp_path):
    RunManager(_tiny_run_cfg(tmp_path, total_updates=2)).run()
    out = RunManager(_tiny_run_cfg(tmp_path, total_updates=2)).run()  # nothing left to do
    assert out["updates_this_invocation"] == 0
    assert out["final_version"] == 2


def test_fresh_restarts_from_zero(tmp_path):
    RunManager(_tiny_run_cfg(tmp_path, total_updates=4)).run()       # v1..v4
    out = RunManager(_tiny_run_cfg(tmp_path, total_updates=2), fresh=True).run()
    assert out["updates_this_invocation"] == 2
    assert out["final_version"] == 2                                 # counter reset
    log = MetricsLogger(os.path.join(str(tmp_path), "rc", "metrics.jsonl"))
    assert [r["update"] for r in log.records()] == [1, 2]           # curve reset


def test_resume_aborts_on_config_hash_mismatch(tmp_path):
    # Write a checkpoint stamped with one config...
    cfgX = TrainConfig(lr=3e-4, policy_seats=(0,), use_skeleton=True)
    rmX = RunManager(RunConfig(name="rc", out_dir=str(tmp_path),
                               train={"lr": 3e-4, "policy_seats": [0], "use_skeleton": True}))
    policy = build_policy(cfgX)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rmX.ckptr.save(1, policy, opt, extra={"version": 1})

    # ...then a run with a *different* hyper-parameter must refuse to resume it.
    rmY = RunManager(RunConfig(name="rc", out_dir=str(tmp_path),
                               train={"lr": 1e-3, "policy_seats": [0], "use_skeleton": True}))
    assert rmY.config_hash != rmX.config_hash
    with pytest.raises(ValueError, match="config_hash"):
        rmY._resume(build_policy(TrainConfig(lr=1e-3, use_skeleton=True)),
                    torch.optim.Adam(build_policy(TrainConfig(use_skeleton=True)).parameters()))
