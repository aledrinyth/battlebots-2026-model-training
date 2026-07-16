"""Behaviour-cloning warm-start (P3.5 Track D; the v3 §4B mechanism, expert-fed).

Fits the real :class:`~train.model.AgarPolicy` to expert demonstrations from
:mod:`train.harness.demos` by supervised learning: cross-entropy on the 24-bin
heading (masked where ``label_ok`` is false) + BCE on the split bit, replayed
**recurrently** — windows of consecutive rows run through the GRU exactly like
PPO's BPTT re-eval, with ``is_reset`` zeroing hidden state inside the policy
forward, so the hidden dynamics the clone learns are the ones it deploys with.

This is simultaneously the project's **representation probe**: if this network
cannot fit ``greedy_food`` *supervised* (low held-out heading accuracy), no
exploration scheme will make PPO discover food-steering — that outcome is
evidence for the Track E architecture change, not for more RL patience.

The output checkpoint carries the standard ``state_dict + arch + provenance``
blob, so it is directly loadable by ``train.harness.evaluate --weights`` (G1
measurement) and by ``train.harness.run``'s ``init_weights`` warm-start seam
(BC → PPO fine-tune handoff — the same seam Phase 6's curriculum stages use).

    python -m train.bc --data data/demos_greedy_food.npz --out data/bc_greedy_food.pt
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from train.harness.collate import build_batched_obs
from train.harness.config import TrainConfig
from train.harness.provenance import registry_row
from train.harness.shard import _atomic
from train.harness.trainer import _arch_meta
from train.model import build_policy


def _window_starts(n_rows: int, window: int) -> np.ndarray:
    """Non-overlapping window start indices covering the dataset."""
    return np.arange(0, n_rows - window + 1, window, dtype=np.int64)


def _run_window(policy, obs, is_reset, starts, window, device):
    """Replay ``len(starts)`` windows through the GRU; return stacked logits.

    Returns ``(heading_logits, split_logits)`` shaped ``(window, B, ·)``.
    Hidden starts at zero per window (standard TBPTT truncation); ``is_reset``
    rows re-zero it inside the policy forward mid-window.
    """
    B = len(starts)
    hidden = torch.zeros(B, policy.hidden_size, device=device)
    hls, sls = [], []
    for t in range(window):
        rows = starts + t
        batched = build_batched_obs(obs[rows], is_reset[rows], device)
        hl, sl, _v, hidden = policy(batched, hidden)
        hls.append(hl)
        sls.append(sl.reshape(-1))
    return torch.stack(hls), torch.stack(sls)


def train_bc(
    data: Dict[str, np.ndarray],
    epochs: int = 3,
    window: int = 32,
    batch_windows: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    val_frac: float = 0.05,
    seed: int = 0,
    log: bool = True,
    split_pos_weight: float = 1.0,
    keep_best: bool = False,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Fit the default AgarPolicy to the demos; return ``(policy, metrics)``.

    ``metrics`` carries the held-out heading top-1 accuracy (``val_acc``) — the
    representation-probe verdict — plus final losses and split-head
    precision/recall (the v3 split-resurrection signal).

    If ``data`` carries a ``weight`` column (replay BC: per-row terminal-rank
    quality), both loss terms are weighted by it. ``split_pos_weight``
    up-weights positive split rows in the BCE (replay split base rates are
    1–2 %, where an unweighted head can sit silent below 0.5 forever).
    ``keep_best=True`` restores the epoch with the best ``val_adj_acc``
    afterward (the S1 lesson: epochs past a flat validation curve buy
    memorization, not skill), recording it as ``best_epoch``.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    obs = data["obs"]
    heading = torch.as_tensor(data["heading"])
    split = torch.as_tensor(data["split"])
    label_ok = torch.as_tensor(data["label_ok"].astype(np.float32))
    is_reset = data["is_reset"]
    weight = (
        torch.as_tensor(data["weight"].astype(np.float32))
        if "weight" in data else None
    )

    starts = _window_starts(len(obs), window)
    rng.shuffle(starts)
    n_val = max(1, int(len(starts) * val_frac))
    val_starts, train_starts = starts[:n_val], starts[n_val:]

    cfg = TrainConfig()  # the default (real) architecture — what PPO trains
    policy = build_policy(cfg).to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(reduction="none")
    bce = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(float(split_pos_weight), device=dev),
    )

    def _loss_on(hl, sl, rows_t):
        h = heading[rows_t].to(dev)
        s = split[rows_t].to(dev)
        ok = label_ok[rows_t].to(dev)
        w = weight[rows_t].to(dev) if weight is not None else None
        okw = ok if w is None else ok * w
        lh = (ce(hl, h) * okw).sum() / okw.sum().clamp_min(1.0)
        ls_raw = bce(sl, s)
        if w is None:
            ls = ls_raw.mean()
        else:
            ls = (ls_raw * w).sum() / w.sum().clamp_min(1.0)
        return lh + ls, lh, ls

    metrics: Dict[str, float] = {}
    best_adj = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()
        rng.shuffle(train_starts)
        policy.train()
        ep_loss = n_b = 0
        for i in range(0, len(train_starts), batch_windows):
            batch = train_starts[i:i + batch_windows]
            hls, sls = _run_window(policy, obs, is_reset, batch, window, dev)
            loss = hloss = sloss = 0.0
            for t in range(window):
                rows_t = torch.as_tensor(batch + t)
                l, lh, ls = _loss_on(hls[t], sls[t], rows_t)
                loss = loss + l
            loss = loss / window
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item())
            n_b += 1

        # held-out heading accuracy (the probe number). ``adj`` counts a
        # prediction within ±1 bin (15°) as correct — behaviorally, steering one
        # bin off the expert still closes on the food, so adj is the number
        # that predicts eval mass; exact top-1 is the stricter fit signal.
        # Split-head precision/recall at logit 0 is the split-resurrection
        # signal: a head that never fires reads recall 0 no matter the BCE.
        policy.eval()
        correct = adjacent = total = 0
        s_tp = s_fp = s_fn = s_pos = 0
        with torch.no_grad():
            for i in range(0, len(val_starts), batch_windows):
                batch = val_starts[i:i + batch_windows]
                hls, sls = _run_window(policy, obs, is_reset, batch, window, dev)
                for t in range(window):
                    rows_t = batch + t
                    ok = data["label_ok"][rows_t]
                    pred = hls[t].argmax(dim=-1).cpu().numpy()
                    true = data["heading"][rows_t]
                    dist = np.abs((pred[ok] - true[ok] + 12) % 24 - 12)
                    correct += int((dist == 0).sum())
                    adjacent += int((dist <= 1).sum())
                    total += int(ok.sum())
                    sp = (sls[t] > 0.0).cpu().numpy()
                    st = data["split"][rows_t] > 0.5
                    s_tp += int((sp & st).sum())
                    s_fp += int((sp & ~st).sum())
                    s_fn += int((~sp & st).sum())
                    s_pos += int(st.sum())
        val_acc = correct / max(total, 1)
        val_adj = adjacent / max(total, 1)
        metrics = {"epoch": epoch, "train_loss": ep_loss / max(n_b, 1),
                   "val_acc": val_acc, "val_adj_acc": val_adj,
                   "val_split_recall": s_tp / max(s_pos, 1),
                   "val_split_precision": s_tp / max(s_tp + s_fp, 1),
                   "val_split_pos": s_pos}
        if keep_best and val_adj >= best_adj:
            best_adj = val_adj
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in policy.state_dict().items()}
        if log:
            print(f"[bc {epoch}/{epochs}] loss={metrics['train_loss']:.4f} "
                  f"val_heading_acc={val_acc:.1%} (±1 bin: {val_adj:.1%}) "
                  f"split_recall={metrics['val_split_recall']:.1%}"
                  f"/prec={metrics['val_split_precision']:.1%} "
                  f"(n={s_pos}) ({time.monotonic() - t0:.0f}s)", flush=True)
    if keep_best and best_state is not None:
        policy.load_state_dict(best_state)
        metrics["best_epoch"] = best_epoch
        metrics["best_val_adj_acc"] = best_adj
        if log:
            print(f"[bc] restored best epoch {best_epoch} "
                  f"(val_adj {best_adj:.1%})", flush=True)
    return policy, metrics


def save_bc_checkpoint(policy: nn.Module, metrics: Dict[str, float],
                       out_path: str, data_path: str = "") -> None:
    """Write an ``evaluate``/``init_weights``-compatible checkpoint blob."""
    cfg = TrainConfig()
    blob = {
        "version": 0,
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "arch": _arch_meta(cfg),
        "provenance": {
            **registry_row(cfg, run_name="bc_warmstart", stage="S0-bc"),
            "bc_data": data_path,
            "bc_val_acc": float(metrics.get("val_acc", 0.0)),
        },
    }
    _atomic(out_path, lambda p: torch.save(blob, p))


def _main() -> None:
    p = argparse.ArgumentParser(description="Behaviour-cloning warm-start (Track D)")
    p.add_argument("--data", default="data/demos_greedy_food.npz")
    p.add_argument("--out", default="data/bc_greedy_food.pt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--batch-windows", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--split-pos-weight", type=float, default=1.0,
                   help="up-weight split=1 rows in the BCE (replay BC: base "
                        "rate is 1-2%)")
    p.add_argument("--keep-best", action="store_true",
                   help="restore the best-val_adj epoch instead of the last")
    args = p.parse_args()

    data = dict(np.load(args.data))
    print(f"[bc] {len(data['obs'])} rows from {args.data} "
          f"(label_ok {data['label_ok'].mean():.1%})", flush=True)
    policy, metrics = train_bc(data, epochs=args.epochs, window=args.window,
                               batch_windows=args.batch_windows, lr=args.lr,
                               device=args.device,
                               split_pos_weight=args.split_pos_weight,
                               keep_best=args.keep_best)
    save_bc_checkpoint(policy, metrics, args.out, data_path=args.data)
    print(f"[bc] wrote {args.out} (val_heading_acc {metrics['val_acc']:.1%}) — "
          f"eval with: python -m train.harness.evaluate --weights {args.out} "
          f"--opponents dont_move --games 30 --vs-random", flush=True)


if __name__ == "__main__":
    _main()


__all__ = ["train_bc", "save_bc_checkpoint"]
