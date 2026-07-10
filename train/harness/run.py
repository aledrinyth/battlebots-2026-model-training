"""Run manager + single-YAML launcher (P3.4) — logging, checkpointing, resume.

P3.2/P3.3 built the training *loop*; P3.4 makes a run **reproducible and
kill-safe**. This module adds, around the single-box :class:`VecLearner`:

* **Config discipline** — one YAML describes a whole run
  (:class:`RunConfig`); a :func:`~train.harness.provenance.config_hash` over the
  algorithmic knobs identifies the experiment; a ``run.json`` registry row records
  engine pin + adapter versions + config hash + the full config (the schema
  pipeline P11.1 later hardens).
* **Logging** — a local W&B-equivalent: :class:`MetricsLogger` appends one JSON
  line per update (``metrics.jsonl``). No cloud dependency in the match-legal
  repo; a W&B mirror can subscribe to the same records later.
* **Checkpointing** — :class:`Checkpointer` writes ``ckpt_v<NNNNNN>.pt`` on a
  wall-clock cadence (default every 30 min) **and** at run end. Each checkpoint
  is self-describing (policy + optimizer + arch + provenance) and
  ``evaluate``-compatible (it carries ``state_dict`` + ``arch``).
* **Resume** — re-running the same command reloads the latest checkpoint (policy
  **and** optimizer) and continues the update counter; the metrics log is
  truncated to the checkpoint so the curve continues seamlessly with no gap or
  rewind. This is the P3.4 acceptance: *kill and resume mid-flight; curves
  continue seamlessly.*

This wraps the **single-box** learner (a self-contained "one run"); the two-box
:mod:`train.harness.trainer` already resumes from its published ``weights_v*.pt``
for the decoupled rig. Torch/YAML only — a training tool, never shipped.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import torch
import yaml

from train.harness.config import TrainConfig
from train.harness.learner import VecLearner
from train.harness.provenance import config_hash, registry_row
from train.harness.shard import _atomic
from train.harness.trainer import _arch_meta

# Fields of TrainConfig that arrive from YAML as lists but the dataclass declares
# as tuples (order-stable, hashable). Coerced when the base config is built.
_TUPLE_FIELDS = ("policy_seats", "scripted_names")


def _write_text_atomic(path: str, text: str) -> None:
    """Atomically overwrite ``path`` with ``text`` (write tmp → fsync-close → rename)."""
    def _w(tmp: str) -> None:
        with open(tmp, "w") as fh:
            fh.write(text)
    _atomic(path, _w)


# --------------------------------------------------------------------------- #
# Run configuration (the single YAML)                                         #
# --------------------------------------------------------------------------- #

@dataclass
class RunConfig:
    """Everything one training run needs, from a single YAML file.

    ``train`` is a mapping of :class:`TrainConfig` overrides (the algorithmic +
    environment knobs). The run-level fields below are operational — how long to
    run, how often to checkpoint, where to write — and are deliberately *outside*
    the config hash so a resume or a duration change does not re-identify the run.
    """

    name: str
    stage: str = ""
    out_dir: str = "runs"
    total_updates: int = 0          # 0 ⇒ bounded only by duration_s
    duration_s: float = 0.0         # per-invocation wall-clock cap; 0 ⇒ off
    checkpoint_interval_s: float = 1800.0   # 30 min (playbook P3.4)
    keep_checkpoints: int = 5
    log_every: int = 1
    device: str = "cpu"
    #: Optional checkpoint path to **warm-start** the policy from when this run
    #: has no checkpoints of its own yet (P3.5 Track D BC→PPO handoff; the same
    #: seam Phase 6 curriculum stages use to warm-start the previous stage).
    #: Loads ``state_dict`` only — the optimizer starts fresh. Ignored once the
    #: run has its own checkpoint to resume (resume wins). Operational, so it
    #: does not perturb the config hash.
    init_weights: str = ""
    train: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        known = {f for f in cls.__dataclass_fields__}  # noqa: E501
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown RunConfig fields: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path: str) -> "RunConfig":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "stage": self.stage, "out_dir": self.out_dir,
            "total_updates": self.total_updates, "duration_s": self.duration_s,
            "checkpoint_interval_s": self.checkpoint_interval_s,
            "keep_checkpoints": self.keep_checkpoints, "log_every": self.log_every,
            "device": self.device, "init_weights": self.init_weights,
            "train": dict(self.train),
        }

    def base_train_config(self) -> TrainConfig:
        """The :class:`TrainConfig` for this run (num_updates/duration set per-invocation)."""
        over = dict(self.train)
        for k in _TUPLE_FIELDS:
            if over.get(k) is not None and k in over:
                over[k] = tuple(over[k])
        over.setdefault("device", self.device)
        return TrainConfig(**over)


# --------------------------------------------------------------------------- #
# Metrics logging (local, append-only, resume-truncatable)                    #
# --------------------------------------------------------------------------- #

class MetricsLogger:
    """Append one JSON record per update to ``metrics.jsonl`` (a local W&B stand-in)."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()

    def truncate_after(self, version: int) -> None:
        """Drop records with ``update > version`` so a resume continues seamlessly.

        A prior run may have logged a few updates *after* its last checkpoint
        before dying. On resume we reload that checkpoint (``version``) and
        re-run those updates, so the stale tail must go — otherwise the curve
        would double back on itself.
        """
        if not os.path.exists(self.path):
            return
        kept: List[str] = []
        with open(self.path) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(rec.get("update", 0)) <= version:
                    kept.append(line)
        _write_text_atomic(self.path, "\n".join(kept) + ("\n" if kept else ""))

    def records(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


# --------------------------------------------------------------------------- #
# Checkpointing (self-describing, wall-clock cadence, prune)                   #
# --------------------------------------------------------------------------- #

class Checkpointer:
    """Writes/loads ``ckpt_v<NNNNNN>.pt`` — policy + optimizer + arch + provenance."""

    def __init__(
        self,
        ckpt_dir: str,
        arch: Dict[str, Any],
        provenance: Dict[str, Any],
        keep: int = 5,
        interval_s: float = 1800.0,
    ) -> None:
        self.dir = ckpt_dir
        self.arch = dict(arch)
        self.provenance = dict(provenance)
        self.keep = int(keep)
        self.interval_s = float(interval_s)
        os.makedirs(ckpt_dir, exist_ok=True)
        self._last_save = time.monotonic()

    # -- discovery ----------------------------------------------------------
    def _files(self) -> List[tuple]:
        out = []
        for p in glob.glob(os.path.join(self.dir, "ckpt_v*.pt")):
            base = os.path.basename(p)
            try:
                v = int(base[len("ckpt_v"):-len(".pt")])
            except ValueError:
                continue
            out.append((v, p))
        return sorted(out)

    def latest_version(self) -> Optional[int]:
        files = self._files()
        return files[-1][0] if files else None

    def load_latest(self, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
        files = self._files()
        if not files:
            return None
        _v, path = files[-1]
        return torch.load(path, map_location=map_location, weights_only=False)

    # -- writing ------------------------------------------------------------
    def save(self, version: int, policy, optimizer, extra: Optional[Dict] = None) -> str:
        blob = {
            "version": int(version),
            "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
            "optim_state": optimizer.state_dict(),
            "arch": dict(self.arch),
            "provenance": {**self.provenance, **(extra or {})},
        }
        path = os.path.join(self.dir, f"ckpt_v{int(version):06d}.pt")
        _atomic(path, lambda p: torch.save(blob, p))
        self._prune()
        self._last_save = time.monotonic()
        return path

    def maybe_save(self, version: int, policy, optimizer, force: bool = False,
                   extra: Optional[Dict] = None) -> Optional[str]:
        if force or (time.monotonic() - self._last_save) >= self.interval_s:
            return self.save(version, policy, optimizer, extra=extra)
        return None

    def _prune(self) -> None:
        if self.keep <= 0:
            return
        for _v, path in self._files()[:-self.keep]:
            try:
                os.remove(path)
            except OSError:
                pass

    def clear(self) -> int:
        """Delete every checkpoint (used by a ``--fresh`` restart). Returns the count."""
        files = self._files()
        for _v, path in files:
            try:
                os.remove(path)
            except OSError:
                pass
        self._last_save = time.monotonic()
        return len(files)


# --------------------------------------------------------------------------- #
# Run manager                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class RunPaths:
    root: str
    checkpoints: str
    metrics: str
    run_json: str

    @classmethod
    def make(cls, out_dir: str, name: str) -> "RunPaths":
        root = os.path.join(out_dir, name)
        os.makedirs(root, exist_ok=True)
        p = cls(root=root,
                checkpoints=os.path.join(root, "checkpoints"),
                metrics=os.path.join(root, "metrics.jsonl"),
                run_json=os.path.join(root, "run.json"))
        os.makedirs(p.checkpoints, exist_ok=True)
        return p


class RunManager:
    """Owns one run: provenance, logging, checkpointing, resume, the train loop."""

    def __init__(self, run_cfg: RunConfig, fresh: bool = False) -> None:
        self.run_cfg = run_cfg
        self.fresh = fresh
        self.paths = RunPaths.make(run_cfg.out_dir, run_cfg.name)
        self.base_cfg = run_cfg.base_train_config()
        self.config_hash = config_hash(self.base_cfg)
        self.arch = _arch_meta(self.base_cfg)
        self.provenance = registry_row(
            self.base_cfg, run_name=run_cfg.name, stage=run_cfg.stage,
        )
        self.logger = MetricsLogger(self.paths.metrics)
        self.ckptr = Checkpointer(
            self.paths.checkpoints, self.arch, self.provenance,
            keep=run_cfg.keep_checkpoints, interval_s=run_cfg.checkpoint_interval_s,
        )

    # -- registry row -------------------------------------------------------
    def _write_run_json(self, extra: Optional[Dict] = None) -> None:
        row = {**self.provenance, **(extra or {})}
        _write_text_atomic(self.paths.run_json, json.dumps(row, indent=2))

    # -- resume -------------------------------------------------------------
    def _resume(self, policy, optimizer) -> int:
        """Load the latest checkpoint into policy+optimizer; return the update it was at."""
        if self.fresh:
            return 0
        blob = self.ckptr.load_latest()
        if blob is None:
            return 0
        got_hash = blob.get("provenance", {}).get("config_hash")
        if got_hash and got_hash != self.config_hash:
            raise ValueError(
                f"checkpoint config_hash {got_hash} != current {self.config_hash}; "
                f"pass --fresh to start a new run (or restore the matching config)"
            )
        policy.load_state_dict(blob["state_dict"])
        if "optim_state" in blob:
            optimizer.load_state_dict(blob["optim_state"])
        version = int(blob["version"])
        self.logger.truncate_after(version)  # drop any post-checkpoint tail
        print(f"[resume] loaded ckpt v{version} (config_hash={self.config_hash}); continuing",
              flush=True)
        return version

    # -- run ----------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        if self.fresh:
            # Restart this run from scratch: drop old checkpoints so the new v1 is
            # genuinely latest, and clear the metrics curve so it starts clean.
            n = self.ckptr.clear()
            self.logger.truncate_after(0)
            if n:
                print(f"[fresh] cleared {n} checkpoint(s); restarting at update 0", flush=True)
        self._write_run_json(extra={"status": "running"})

        # Build the learner (seeds a fresh policy + Adam), then load any checkpoint.
        # num_updates/duration are set per-invocation from the run totals below.
        global0_holder = {"v": 0}
        remaining = self.run_cfg.total_updates
        # First construct with a placeholder num_updates; corrected after resume.
        learner = VecLearner(replace(self.base_cfg, num_updates=max(remaining, 1),
                                     device=self.run_cfg.device,
                                     log_every=self.run_cfg.log_every))
        global0 = self._resume(learner.policy, learner.optimizer)
        global0_holder["v"] = global0
        if global0 == 0 and self.run_cfg.init_weights:
            # Warm-start (BC → PPO handoff / curriculum stage handoff): policy
            # weights only, fresh optimizer, update counter stays at 0. Once the
            # run has its own checkpoints, resume wins and this is skipped.
            blob = torch.load(self.run_cfg.init_weights, map_location="cpu",
                              weights_only=False)
            learner.policy.load_state_dict(blob["state_dict"])
            print(f"[init] warm-started policy from {self.run_cfg.init_weights}",
                  flush=True)

        if self.run_cfg.total_updates and global0 >= self.run_cfg.total_updates:
            print(f"[run] already at {global0}/{self.run_cfg.total_updates} updates; nothing to do",
                  flush=True)
            self._write_run_json(extra={"status": "done", "final_version": global0})
            return {"final_version": global0, "updates_this_invocation": 0}

        this_updates = (self.run_cfg.total_updates - global0) if self.run_cfg.total_updates else 10**9
        learner.cfg = replace(self.base_cfg, num_updates=this_updates,
                              duration_s=self.run_cfg.duration_s,
                              device=self.run_cfg.device, log_every=self.run_cfg.log_every)
        learner.global_update0 = global0

        t0 = time.monotonic()

        def _on_update(lrn: VecLearner, global_update: int, metrics: Dict[str, float]) -> None:
            rec = {"update": int(global_update),
                   "wall_s": round(time.monotonic() - t0, 3),
                   **{k: float(v) for k, v in metrics.items()}}
            self.logger.log(rec)
            self.ckptr.maybe_save(global_update, lrn.policy, lrn.optimizer,
                                  extra={"version": global_update})

        learner.on_update = _on_update
        stats = learner.train()
        final_version = global0 + stats.updates
        # Always leave a checkpoint at the final state (kill-safe boundary).
        self.ckptr.maybe_save(final_version, learner.policy, learner.optimizer,
                              force=True, extra={"version": final_version})
        self._write_run_json(extra={
            "status": "done", "final_version": final_version,
            "updates_this_invocation": stats.updates,
            "wall_s": round(stats.wall_s, 1),
        })
        print(f"[run] {self.run_cfg.name}: {stats.summary()} final_version={final_version}",
              flush=True)
        return {"final_version": final_version, "updates_this_invocation": stats.updates,
                "config_hash": self.config_hash, "run_dir": self.paths.root}


def run_from_config(run_cfg: RunConfig, fresh: bool = False) -> Dict[str, Any]:
    return RunManager(run_cfg, fresh=fresh).run()


def _main() -> None:
    p = argparse.ArgumentParser(description="P3.4 single-YAML training run launcher")
    p.add_argument("config", help="path to a run YAML (see configs/)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore existing checkpoints and restart from update 0")
    p.add_argument("--updates", type=int, default=None, help="override total_updates")
    p.add_argument("--duration", type=float, default=None, help="override per-invocation duration_s")
    p.add_argument("--device", type=str, default=None, help="override device (cpu/cuda)")
    p.add_argument("--num-actors", type=int, default=None,
                   help="override train.num_actors (batch = num_actors × policy_seats); "
                        "set to roughly (CPU core count minus a few) for headroom")
    p.add_argument("--policy-seats", type=int, nargs="+", default=None,
                   help="override train.policy_seats, e.g. --policy-seats 0 1 2 3 for a "
                        "×4 batch (multi-seat food-only bootstrap). Changes the config hash.")
    p.add_argument("--init-weights", type=str, default=None,
                   help="warm-start checkpoint (state_dict only; ignored on resume)")
    args = p.parse_args()

    run_cfg = RunConfig.load(args.config)
    if args.updates is not None:
        run_cfg.total_updates = args.updates
    if args.duration is not None:
        run_cfg.duration_s = args.duration
    if args.device is not None:
        run_cfg.device = args.device
    if args.num_actors is not None:
        run_cfg.train["num_actors"] = args.num_actors
    if args.policy_seats is not None:
        run_cfg.train["policy_seats"] = list(args.policy_seats)
    if args.init_weights is not None:
        run_cfg.init_weights = args.init_weights

    out = run_from_config(run_cfg, fresh=args.fresh)
    print(json.dumps(out))


if __name__ == "__main__":
    _main()


__all__ = ["RunConfig", "RunManager", "MetricsLogger", "Checkpointer", "RunPaths",
           "run_from_config"]
