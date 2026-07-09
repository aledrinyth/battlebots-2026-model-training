"""P3.2 — PPO update wiring (``train.harness.learner``).

Two checks: action sampling produces correct joint log-probs, and the
buffer→GAE→PPO path actually *reduces loss* on a frozen buffer (the overfit
wiring sanity — a lite version of P3.3's acceptance, proving the update loop
closes over the real SkeletonPolicy forward, not that it trains a good agent).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # train-side test; skipped in the torch-free bot venv
import torch

from bot.adapters.act_v1 import HEADING_BINS
from train.harness.config import TrainConfig
from train.harness.learner import _evaluate, ppo_update, sample_actions
from train.harness.policy_stub import SkeletonPolicy
from train.harness.rollout import RolloutBuffer
from train.harness.flat_obs import OBS_WIDTH


def test_sample_actions_shapes_and_logp():
    torch.manual_seed(0)
    B = 32
    heading_logits = torch.randn(B, HEADING_BINS)
    split_logit = torch.randn(B)
    heading, split, logp = sample_actions(heading_logits, split_logit)

    assert heading.shape == (B,)
    assert split.shape == (B,)
    assert torch.all((heading >= 0) & (heading < HEADING_BINS))
    assert torch.all((split == 0) | (split == 1))
    # joint logp == categorical + bernoulli, and ≤ 0
    hd = torch.distributions.Categorical(logits=heading_logits)
    sd = torch.distributions.Bernoulli(logits=split_logit)
    expect = hd.log_prob(heading) + sd.log_prob(split)
    torch.testing.assert_close(logp, expect)
    assert torch.all(logp <= 0)


def test_sample_actions_follows_peaked_logits():
    torch.manual_seed(1)
    B = 64
    heading_logits = torch.full((B, HEADING_BINS), -10.0)
    heading_logits[:, 7] = 10.0
    split_logit = torch.full((B,), 10.0)
    heading, split, _ = sample_actions(heading_logits, split_logit)
    assert torch.all(heading == 7)
    assert torch.all(split == 1)


def _frozen_buffer(policy, T=3, R=8, seed=0):
    """A buffer with real obs + collection logp/value from ``policy``, and a
    fixed positive-return target so an update has something to fit."""
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer(rollout_len=T, rows=R, hidden_size=policy.hidden_size)
    # small finite obs (schema-shaped; values irrelevant to the wiring sanity)
    buf.obs[:] = rng.standard_normal((T, R, OBS_WIDTH)).astype(np.float32) * 0.1
    buf.is_reset[0] = True
    buf.active[:] = True

    # collection pass: fill actions, logp, value from the current policy
    hidden = torch.zeros(R, policy.hidden_size)
    with torch.no_grad():
        for t in range(T):
            from train.harness.collate import build_batched_obs
            batched = build_batched_obs(buf.obs[t], buf.is_reset[t])
            hl, sl, v, hidden = policy(batched, hidden)
            heading, split, logp = sample_actions(hl, sl)
            buf.heading[t] = heading.numpy()
            buf.split[t] = split.numpy()
            buf.logp[t] = logp.numpy()
            buf.value[t] = v.numpy()
    return buf


def test_ppo_update_reduces_loss_on_frozen_buffer():
    torch.manual_seed(0)
    np.random.seed(0)
    policy = SkeletonPolicy(entity_hidden=16, trunk_hidden=32)
    buf = _frozen_buffer(policy)
    cfg = TrainConfig(ppo_epochs=1, minibatch_rows=0, lr=1e-2, entropy_coef=0.0)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    # Fixed advantage/return targets: reward every taken action (returns > value).
    returns = np.ones((buf.T, buf.R), dtype=np.float32)
    advantages = (returns - buf.value).astype(np.float32)

    first = ppo_update(policy, opt, buf, advantages, returns, cfg, torch.device("cpu"))
    for _ in range(60):
        last = ppo_update(policy, opt, buf, advantages, returns, cfg, torch.device("cpu"))

    # Value head fits the (constant) return target ⇒ value loss collapses.
    assert last["value_loss"] < first["value_loss"] * 0.5
    assert np.isfinite(last["loss"])
    # Positive advantage on taken actions ⇒ their prob rises ⇒ entropy drops.
    assert last["entropy"] < first["entropy"]


def test_ppo_update_masks_inactive_rows():
    """Inactive rows must not contribute gradient — an all-inactive buffer is a
    no-op update (zero loss, no NaNs)."""
    torch.manual_seed(0)
    policy = SkeletonPolicy(entity_hidden=8, trunk_hidden=16)
    buf = _frozen_buffer(policy, T=2, R=4)
    buf.active[:] = False  # nobody acted
    cfg = TrainConfig(ppo_epochs=1)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    m = ppo_update(
        policy, opt, buf,
        np.zeros((buf.T, buf.R), np.float32), np.zeros((buf.T, buf.R), np.float32),
        cfg, torch.device("cpu"),
    )
    assert m["value_loss"] == 0.0
    assert np.isfinite(m["loss"])
