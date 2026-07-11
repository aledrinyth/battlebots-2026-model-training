"""P5.1 — weight export: checkpoint → ``weights.npz`` for the NumPy bot.

Takes any training checkpoint (a P3.4 ``ckpt_v*.pt``, a BC warm-start ``.pt`` —
anything carrying ``state_dict`` + ``arch``) and writes the **actor path only**
(:data:`~train.model.AgarPolicy._ACTOR_MODULES`; the privileged critic never
ships) as float32 arrays in an ``.npz``, plus a JSON manifest (architecture
dims, bin count, adapter/model versions, source provenance) so the bot-side
loader (:mod:`bot.inference`) can size its buffers and refuse a mismatched
schema.

Array keys are the torch ``state_dict`` keys verbatim (``gru.weight_ih``,
``heading_head.bias``, …) — one naming scheme across both backends, so the
parity test (P5.3) can diff them mechanically.

    python -m train.export --ckpt runs/<run>/checkpoints/ckpt_v000123.pt --out weights.npz
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Tuple

import numpy as np
import torch

from bot.adapters.act_v1 import ACT_VERSION, HEADING_BINS
from bot.adapters.obs_v1 import OBS_VERSION
from train.model import MODEL_VERSION, AgarPolicy

#: npz key holding the JSON manifest (all other keys are weight arrays).
MANIFEST_KEY = "__manifest__"

_ACTOR_PREFIXES = tuple(f"{m}." for m in AgarPolicy._ACTOR_MODULES)


def actor_arrays(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """The actor-path subset of a state dict, as float32 numpy arrays."""
    out = {}
    for key, tensor in state_dict.items():
        if key.startswith(_ACTOR_PREFIXES):
            out[key] = tensor.detach().cpu().to(torch.float32).numpy()
    if not out:
        raise ValueError("no actor-path tensors found — is this an AgarPolicy checkpoint?")
    return out


def build_manifest(arch: Dict[str, Any], provenance: Dict[str, Any],
                   source: str) -> Dict[str, Any]:
    if arch.get("use_skeleton"):
        raise ValueError("refusing to export a SkeletonPolicy checkpoint (test stub)")
    return {
        "model_version": arch.get("model_version", MODEL_VERSION),
        "obs_version": OBS_VERSION,
        "act_version": ACT_VERSION,
        "heading_bins": HEADING_BINS,
        "entity_hidden": int(arch.get("model_entity_hidden", 64)),
        "hidden_size": int(arch.get("model_trunk_hidden", 256)),
        "source": source,
        "config_hash": provenance.get("config_hash", ""),
        "version": provenance.get("version", ""),
    }


def export_checkpoint(ckpt_path: str, out_path: str) -> Dict[str, Any]:
    """Export ``ckpt_path``'s actor weights to ``out_path``; return the manifest."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arrays = actor_arrays(blob["state_dict"])
    manifest = build_manifest(blob.get("arch", {}), blob.get("provenance", {}),
                              source=ckpt_path)
    np.savez(out_path, **arrays,
             **{MANIFEST_KEY: np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)})
    return manifest


def load_exported(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Round-trip loader (tests + the bot-side loader mirrors this)."""
    with np.load(path) as z:
        manifest = json.loads(bytes(z[MANIFEST_KEY]).decode())
        arrays = {k: z[k] for k in z.files if k != MANIFEST_KEY}
    return arrays, manifest


def _main() -> None:
    p = argparse.ArgumentParser(description="Export actor weights for the NumPy bot (P5.1)")
    p.add_argument("--ckpt", required=True, help="checkpoint (.pt with state_dict + arch)")
    p.add_argument("--out", default="weights.npz")
    args = p.parse_args()
    manifest = export_checkpoint(args.ckpt, args.out)
    n = len(load_exported(args.out)[0])
    print(f"[export] {args.ckpt} -> {args.out} ({n} actor arrays)\n"
          f"[export] manifest: {json.dumps(manifest)}")


if __name__ == "__main__":
    _main()


__all__ = ["MANIFEST_KEY", "actor_arrays", "build_manifest", "export_checkpoint",
           "load_exported"]
