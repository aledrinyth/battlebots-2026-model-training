"""Run provenance + registry schema (P3.4; seeds pipeline P11.1).

Every training run must be **reproducible from a config hash** (the playbook's
P3.4 requirement and the adaptation pipeline's discipline). This module is the
*one* place that binds a run to its identity:

* the **config hash** — a stable SHA-256 over the algorithmic + environment knobs
  of a :class:`~train.harness.config.TrainConfig` (operational fields like device
  or run length are deliberately excluded, so the *same experiment* hashes the
  same regardless of how long it ran or what hardware it ran on);
* the **engine pin** — the vendored ``agario-kit`` version + submodule commit, so
  a checkpoint records exactly which physics it was trained against;
* the **adapter / featurizer / model versions** — the frozen ``obs_v1`` / ``act_v1``
  boundary and the network kind, so an old checkpoint can be matched to a
  compatible adapter shim later (pipeline P11.1 "league pool reads adapters
  per-entry").

:func:`registry_row` assembles all of this into the checkpoint-metadata /
model-registry schema that P3.4 "starts now" and P11.1 hardens. Nothing here
imports torch or touches ``bot/`` beyond the two frozen adapter version strings,
so it is cheap and side-effect-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from typing import Any, Dict, Optional

from bot.adapters.act_v1 import ACT_VERSION
from bot.adapters.obs_v1 import OBS_VERSION
from train.harness.config import TrainConfig
from train.model import MODEL_VERSION

# Repo root = two levels up from train/harness/provenance.py.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ENGINE_DIR = os.path.join(_REPO_ROOT, "vendor", "agario-public")

# TrainConfig fields that are *operational* (how long / where the run executes),
# not part of the experiment's algorithmic identity. Excluded from the config
# hash so the same experiment hashes identically across a resume, a device swap,
# or a different stopping point. ``seat_slots`` is derived from ``policy_seats``.
_HASH_EXCLUDE = frozenset(
    {"device", "num_updates", "duration_s", "log_every",
     "min_infer_batch", "barrier_timeout_s", "seat_slots",
     "league_dir"}  # a filesystem location, not part of the experiment identity
)

# Additive knobs introduced *after* hashed runs already existed. At their inert
# default they are dropped from the hash payload, so a pre-existing experiment
# (e.g. the P3.5 ``g1_rig`` / ``g1_rig_ms4`` runs) keeps its identity — and its
# checkpoints keep resuming — across the code update; any non-default value is
# a genuinely different experiment and hashes in. Every future default-off knob
# added to TrainConfig belongs here with its default.
_HASH_IF_NONDEFAULT: Dict[str, Any] = {
    "explore_oracle_prob": 0.0,
    "explore_oracle_max": 8,
    "dr_food_density": 0.0,
    "league": False,
    "league_snapshot_interval_s": 1800.0,
    "league_anchor_min_frac": 0.10,
    "league_ema": 0.1,
}


def config_dict(cfg: TrainConfig, *, for_hash: bool = False) -> Dict[str, Any]:
    """A JSON-able dict of the config. ``for_hash`` drops operational fields.

    Tuples become lists (JSON has no tuples); with ``sort_keys`` at dump time the
    result is a canonical, order-stable representation.
    """
    d = asdict(cfg)
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
    if for_hash:
        d = {k: v for k, v in d.items()
             if k not in _HASH_EXCLUDE
             and not (k in _HASH_IF_NONDEFAULT and v == _HASH_IF_NONDEFAULT[k])}
    return d


def engine_pin() -> Dict[str, str]:
    """The vendored engine's identity: kit version (+ submodule commit if git-visible)."""
    try:
        import importlib.metadata as _md

        kit_version = _md.version("agario-kit")
    except Exception:  # pragma: no cover - metadata always present in our venvs
        kit_version = "unknown"
    return {"kit_version": kit_version, "commit": _git_commit(_ENGINE_DIR)}


def _git_commit(repo_dir: str) -> str:
    """Best-effort short commit of ``repo_dir`` (``unknown`` if git is unavailable)."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        commit = out.stdout.strip()
        return commit or "unknown"
    except Exception:
        return "unknown"


def adapter_versions() -> Dict[str, str]:
    """The frozen observation/action adapter + policy-network versions."""
    return {"obs": OBS_VERSION, "act": ACT_VERSION, "model": MODEL_VERSION}


def config_hash(cfg: TrainConfig) -> str:
    """Stable 16-hex-char hash identifying this *experiment* config.

    Binds the algorithmic knobs to the adapter versions and the engine's kit
    version, so a physics or adapter change (which would silently invalidate an
    old checkpoint) produces a different hash. The submodule commit is recorded
    in :func:`engine_pin` but kept *out* of the hash: it is not always
    git-visible (e.g. an exported tree), and the kit version already pins physics.
    """
    payload = {
        "config": config_dict(cfg, for_hash=True),
        "adapters": adapter_versions(),
        "engine_kit_version": engine_pin()["kit_version"],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def registry_row(
    cfg: TrainConfig,
    *,
    run_name: str,
    stage: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The checkpoint-metadata / model-registry row (P3.4 schema; P11.1 hardens it).

    ``extra`` carries run-time facts a checkpoint should record but the config
    does not fix — e.g. the completed-update count, wall-clock, or (later) a
    promotion Elo and ancestry pointer.
    """
    row: Dict[str, Any] = {
        "run_name": run_name,
        "stage": stage,
        "config_hash": config_hash(cfg),
        "engine": engine_pin(),
        "adapters": adapter_versions(),
        "config": config_dict(cfg),
    }
    if extra:
        row.update(extra)
    return row


__all__ = [
    "config_dict",
    "engine_pin",
    "adapter_versions",
    "config_hash",
    "registry_row",
]
