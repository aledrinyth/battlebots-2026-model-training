"""P5.2 — pure-NumPy inference: the match-time forward pass.

Hand-rolled mirror of the :class:`~train.model.AgarPolicy` **actor path** —
per-category deep-set encoders (shared 2-layer MLP → masked mean+max pool),
trunk, one ``GRUCell`` step, heading/split heads — consuming the featurizer's
:class:`~bot.adapters.obs_v1.CanonicalObs` directly. No torch, no ``train``
imports (this file ships in the bundle; the import-linter contract enforces
the boundary). Weights come from ``train/export.py``'s ``weights.npz``.

Numerical contract (P5.3): logits agree with the torch actor within float32
accumulation-order tolerance (~1e-5 abs); the *chosen action* is identical
except on documented near-ties. Bit-exactness is neither expected nor required.

The GRU step matches ``torch.nn.GRUCell`` exactly::

    r  = σ(x·Wir^T + bir + h·Whr^T + bhr)
    z  = σ(x·Wiz^T + biz + h·Whz^T + bhz)
    n  = tanh(x·Win^T + bin + r ⊙ (h·Whn^T + bhn))
    h' = (1 − z) ⊙ n + z ⊙ h

with ``weight_ih``/``weight_hh`` stacked ``[r; z; n]`` along dim 0.

Hidden-state policy: :meth:`NumpyPolicy.step` zeroes the incoming hidden when
``obs.is_reset`` is set — the same contract as training, and the featurizer
raises ``is_reset`` on the game's first frame and after every respawn gap, so
the caller just threads ``hidden`` through.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from bot.adapters.act_v1 import ACT_VERSION, HEADING_BINS, CanonicalAction, decode_logits
from bot.adapters.obs_v1 import OBS_VERSION, CanonicalObs

#: Mirrors ``train.export.MANIFEST_KEY`` (no train import allowed here).
MANIFEST_KEY = "__manifest__"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class _Encoder:
    """One deep-set encoder: shared 2-layer MLP → masked mean+max pool."""

    __slots__ = ("w1", "b1", "w2", "b2")

    def __init__(self, arrays: Mapping[str, np.ndarray], prefix: str) -> None:
        # Pre-transpose once so the per-turn path is plain (K,D)@(D,H) matmuls.
        self.w1 = np.ascontiguousarray(arrays[f"{prefix}.mlp.0.weight"].T)
        self.b1 = arrays[f"{prefix}.mlp.0.bias"]
        self.w2 = np.ascontiguousarray(arrays[f"{prefix}.mlp.2.weight"].T)
        self.b2 = arrays[f"{prefix}.mlp.2.bias"]

    def __call__(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        h = np.maximum(h @ self.w2 + self.b2, 0.0)          # (K, H)
        count = mask.sum()
        if count == 0.0:                                     # all padding
            return np.zeros(2 * h.shape[1], dtype=np.float32)
        m = mask[:, None]
        mean = (h * m).sum(axis=0) / count
        mx = np.where(m == 0.0, np.float32(np.finfo(np.float32).min), h).max(axis=0)
        return np.concatenate([mean, mx])


class NumpyPolicy:
    """The deployed policy: ``weights.npz`` in, one ``step`` per turn."""

    def __init__(self, arrays: Mapping[str, np.ndarray],
                 manifest: Mapping[str, Any]) -> None:
        if manifest.get("obs_version") != OBS_VERSION or \
                manifest.get("act_version") != ACT_VERSION or \
                int(manifest.get("heading_bins", 0)) != HEADING_BINS:
            raise ValueError(f"weights manifest {manifest} does not match this bot's "
                             f"adapters ({OBS_VERSION}/{ACT_VERSION}/{HEADING_BINS})")
        self.manifest = dict(manifest)
        self.hidden_size = int(manifest["hidden_size"])

        self.food_enc = _Encoder(arrays, "food_enc")
        self.enemy_enc = _Encoder(arrays, "enemy_enc")
        self.own_enc = _Encoder(arrays, "own_enc")
        self.virus_enc = _Encoder(arrays, "virus_enc")

        self.trunk_w = np.ascontiguousarray(arrays["trunk.0.weight"].T)
        self.trunk_b = arrays["trunk.0.bias"]

        H = self.hidden_size
        w_ih, w_hh = arrays["gru.weight_ih"], arrays["gru.weight_hh"]
        b_ih, b_hh = arrays["gru.bias_ih"], arrays["gru.bias_hh"]
        # Split the stacked [r; z; n] gates once at load.
        self.wir, self.wiz, self.win = (np.ascontiguousarray(w_ih[i * H:(i + 1) * H].T)
                                        for i in range(3))
        self.whr, self.whz, self.whn = (np.ascontiguousarray(w_hh[i * H:(i + 1) * H].T)
                                        for i in range(3))
        self.bir, self.biz, self.bin_ = (b_ih[i * H:(i + 1) * H] for i in range(3))
        self.bhr, self.bhz, self.bhn = (b_hh[i * H:(i + 1) * H] for i in range(3))

        self.head_w = np.ascontiguousarray(arrays["heading_head.weight"].T)
        self.head_b = arrays["heading_head.bias"]
        self.split_w = np.ascontiguousarray(arrays["split_head.weight"].T)
        self.split_b = arrays["split_head.bias"]

    # -- loading --------------------------------------------------------------

    @classmethod
    def load(cls, path) -> "NumpyPolicy":
        """Load from an ``.npz`` path *or* a file-like object (``np.load``
        accepts both — the single-file submission build feeds a BytesIO of
        embedded weight bytes)."""
        with np.load(path) as z:
            manifest: Dict[str, Any] = json.loads(bytes(z[MANIFEST_KEY]).decode())
            arrays = {k: np.ascontiguousarray(z[k], dtype=np.float32)
                      for k in z.files if k != MANIFEST_KEY}
        return cls(arrays, manifest)

    def initial_hidden(self) -> np.ndarray:
        return np.zeros(self.hidden_size, dtype=np.float32)

    # -- the per-turn hot path -------------------------------------------------

    def step(self, obs: CanonicalObs,
             hidden: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """One forward step → ``(heading_logits(24,), split_logit, new_hidden)``.

        Zeroes the incoming ``hidden`` when ``obs.is_reset`` (same contract as
        the torch model — first frame of a game / first frame after respawn).
        """
        if obs.is_reset:
            hidden = np.zeros_like(hidden)

        feats = np.concatenate([
            obs.self_vec,
            self.food_enc(obs.food, obs.food_mask),
            self.enemy_enc(obs.enemy, obs.enemy_mask),
            self.own_enc(obs.own, obs.own_mask),
            self.virus_enc(obs.virus, obs.virus_mask),
        ])
        x = np.maximum(feats @ self.trunk_w + self.trunk_b, 0.0)

        r = _sigmoid(x @ self.wir + self.bir + hidden @ self.whr + self.bhr)
        z = _sigmoid(x @ self.wiz + self.biz + hidden @ self.whz + self.bhz)
        n = np.tanh(x @ self.win + self.bin_ + r * (hidden @ self.whn + self.bhn))
        new_hidden = ((1.0 - z) * n + z * hidden).astype(np.float32)

        heading_logits = new_hidden @ self.head_w + self.head_b
        split_logit = float((new_hidden @ self.split_w + self.split_b)[0])
        return heading_logits, split_logit, new_hidden

    def act(self, obs: CanonicalObs,
            hidden: np.ndarray) -> Tuple[CanonicalAction, np.ndarray]:
        """Greedy decode — exactly how the bot plays (P5.4's hot path)."""
        heading_logits, split_logit, new_hidden = self.step(obs, hidden)
        return decode_logits(heading_logits, split_logit, 0.0), new_hidden


__all__ = ["NumpyPolicy", "MANIFEST_KEY"]
