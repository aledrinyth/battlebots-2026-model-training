"""P3.3 — the real network (``train.model.AgarPolicy``) + PPO (``train.ppo``).

Acceptance (playbook P3.3):

* **Parameter count 300k–800k (actor path)** — ``test_actor_param_budget``.
* **Overfit smoke:** on a frozen 1-game replay buffer, the policy loss drives to
  near-zero (wiring sanity — gradients flow through the deep-set encoders, the
  trunk, the GRU-over-BPTT, and both heads) — ``test_overfit_smoke_frozen_game``.
* **Hidden-state reset on death** verified by a unit test: rows flagged
  ``is_reset`` ignore their incoming GRU state; other rows do not —
  ``test_hidden_reset_on_death``.

Plus the contract the P3.2 transport depends on (forward shapes, the
permutation-invariant deep-set property) and the **asymmetric critic**: the
privileged full-map state moves the *value* but never the actor heads.

Train-venv only (torch); skipped in the match-legal bot venv.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
import torch.nn.functional as F

from bot.adapters.act_v1 import HEADING_BINS
from train.harness.collate import build_batched_obs
from train.harness.config import TrainConfig
from train.harness.flat_obs import OBS_WIDTH
from train.harness.rollout import RolloutBuffer
from train.model import (
    ACTOR_PARAM_MAX,
    ACTOR_PARAM_MIN,
    PRIV_BLOB_DIM,
    PRIV_BLOB_MAX,
    PRIV_FOOD_DIM,
    PRIV_FOOD_MAX,
    PRIV_VIRUS_DIM,
    PRIV_VIRUS_MAX,
    AgarPolicy,
    PrivilegedObs,
    assert_actor_param_budget,
    build_policy,
    count_actor_parameters,
    count_total_parameters,
)
from train.ppo import evaluate_actions, ppo_update, sample_actions


# --- helpers ---------------------------------------------------------------

def _obs(b: int, rng, scale: float = 0.3) -> np.ndarray:
    return (rng.standard_normal((b, OBS_WIDTH)) * scale).astype(np.float32)


def _random_privileged(b: int, rng) -> PrivilegedObs:
    """A fully-populated (mask all 1) random privileged slab."""
    def arr(*s):
        return torch.tensor(rng.standard_normal(s), dtype=torch.float32)

    def ones(*s):
        return torch.ones(*s)

    return PrivilegedObs(
        blobs=arr(b, PRIV_BLOB_MAX, PRIV_BLOB_DIM), blobs_mask=ones(b, PRIV_BLOB_MAX),
        food=arr(b, PRIV_FOOD_MAX, PRIV_FOOD_DIM), food_mask=ones(b, PRIV_FOOD_MAX),
        virus=arr(b, PRIV_VIRUS_MAX, PRIV_VIRUS_DIM), virus_mask=ones(b, PRIV_VIRUS_MAX),
    )


# --- parameter budget (playbook P3.3) --------------------------------------

def test_actor_param_budget():
    """Default AgarPolicy actor path is within the 300k–800k budget; the
    train-only critic is excluded from that count."""
    policy = AgarPolicy()  # default entity_hidden=64, trunk_hidden=256
    actor = count_actor_parameters(policy)
    total = count_total_parameters(policy)
    assert ACTOR_PARAM_MIN <= actor <= ACTOR_PARAM_MAX, actor
    # the privileged critic (value head + priv encoders) is real, non-trivial,
    # and NOT part of the shipped actor count
    assert total > actor
    # the assertion helper agrees and returns the count
    assert assert_actor_param_budget(policy) == actor


def test_actor_param_count_excludes_critic():
    """Actor-path count must equal the sum over exactly the deployment modules."""
    policy = AgarPolicy()
    by_module = sum(
        p.numel()
        for name in ("food_enc", "enemy_enc", "own_enc", "virus_enc",
                     "trunk", "gru", "heading_head", "split_head")
        for p in getattr(policy, name).parameters()
    )
    assert count_actor_parameters(policy) == by_module
    # value_head + privileged encoders are the difference
    critic = sum(
        p.numel()
        for name in ("priv_blob_enc", "priv_food_enc", "priv_virus_enc",
                     "priv_mlp", "value_head")
        for p in getattr(policy, name).parameters()
    )
    assert count_total_parameters(policy) == by_module + critic


# --- forward contract ------------------------------------------------------

def test_forward_shapes():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    B = 6
    policy = AgarPolicy(entity_hidden=16, trunk_hidden=32)
    obs = build_batched_obs(_obs(B, rng), np.zeros(B, bool))
    hidden = policy.initial_hidden(B)
    hl, sl, v, nh = policy(obs, hidden)
    assert hl.shape == (B, HEADING_BINS)
    assert sl.shape == (B,)
    assert v.shape == (B,)
    assert nh.shape == (B, policy.hidden_size)
    for t in (hl, sl, v, nh):
        assert torch.all(torch.isfinite(t))


def test_deep_set_permutation_invariance():
    """Shuffling the entity order within each category leaves the outputs
    unchanged (the deep-set masked mean+max pool is permutation-invariant)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    B = 4
    policy = AgarPolicy(entity_hidden=16, trunk_hidden=32).eval()
    flat = _obs(B, rng)
    hidden = policy.initial_hidden(B)
    base = build_batched_obs(flat, np.zeros(B, bool))
    with torch.no_grad():
        hl0, sl0, v0, _ = policy(base, hidden)

    # permute each entity category along its K axis
    perm = build_batched_obs(flat, np.zeros(B, bool))
    for field, mask in (("food", "food_mask"), ("enemy", "enemy_mask"),
                        ("own", "own_mask"), ("virus", "virus_mask")):
        k = getattr(perm, field).shape[1]
        idx = torch.randperm(k)
        object.__setattr__(perm, field, getattr(perm, field)[:, idx])
        object.__setattr__(perm, mask, getattr(perm, mask)[:, idx])
    with torch.no_grad():
        hl1, sl1, v1, _ = policy(perm, hidden)

    torch.testing.assert_close(hl0, hl1, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(sl0, sl1, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(v0, v1, atol=1e-5, rtol=1e-4)


# --- hidden-state reset on death (playbook P3.3 acceptance) ----------------

def test_hidden_reset_on_death():
    """``is_reset`` rows zero the incoming hidden before the GRU step; other rows
    carry it. This is the death/respawn-gap contract: a seat's first frame after
    a respawn (round-number jump) must not inherit stale recurrent state."""
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    B = 4
    policy = AgarPolicy(entity_hidden=16, trunk_hidden=32).eval()
    is_reset = np.array([True, False, True, False])
    flat = _obs(B, rng)
    obs = build_batched_obs(flat, is_reset)

    hidden_a = torch.randn(B, policy.hidden_size)
    hidden_b = hidden_a + 7.0  # a very different incoming state

    with torch.no_grad():
        hl_a, sl_a, v_a, nh_a = policy(obs, hidden_a)
        hl_b, sl_b, v_b, nh_b = policy(obs, hidden_b)

    reset = torch.tensor(is_reset)
    # reset rows: output is independent of the incoming hidden (it was zeroed)
    torch.testing.assert_close(nh_a[reset], nh_b[reset])
    torch.testing.assert_close(hl_a[reset], hl_b[reset])
    torch.testing.assert_close(sl_a[reset], sl_b[reset])
    # non-reset rows: the incoming hidden actually feeds the GRU ⇒ output differs
    assert not torch.allclose(nh_a[~reset], nh_b[~reset])
    assert not torch.allclose(hl_a[~reset], hl_b[~reset])


def test_reset_row_matches_zero_hidden():
    """A reset row's output equals the same row run from an all-zero hidden."""
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    B = 3
    policy = AgarPolicy(entity_hidden=16, trunk_hidden=32).eval()
    flat = _obs(B, rng)
    with torch.no_grad():
        reset_out = policy(build_batched_obs(flat, np.ones(B, bool)), torch.randn(B, policy.hidden_size))
        zero_out = policy(build_batched_obs(flat, np.zeros(B, bool)), torch.zeros(B, policy.hidden_size))
    for a, b in zip(reset_out, zero_out):
        torch.testing.assert_close(a, b)


# --- asymmetric critic (plan v1 §4.3) --------------------------------------

def test_asymmetric_critic_uses_privileged_state():
    """Privileged full-map state moves the *value* but never the actor heads or
    the recurrent state — the whole point of the asymmetric critic."""
    torch.manual_seed(0)
    rng = np.random.default_rng(4)
    B = 5
    policy = AgarPolicy(entity_hidden=16, trunk_hidden=32).eval()
    obs = build_batched_obs(_obs(B, rng), np.zeros(B, bool))
    hidden = torch.randn(B, policy.hidden_size)
    priv = _random_privileged(B, rng)

    with torch.no_grad():
        hl0, sl0, v0, nh0 = policy(obs, hidden, privileged=None)
        hl1, sl1, v1, nh1 = policy(obs, hidden, privileged=priv)

    # actor path is byte-identical regardless of privileged info
    torch.testing.assert_close(hl0, hl1)
    torch.testing.assert_close(sl0, sl1)
    torch.testing.assert_close(nh0, nh1)
    # value is different (the critic consumed the privileged encoding)
    assert not torch.allclose(v0, v1)


def test_privileged_obs_zeros_shapes():
    z = PrivilegedObs.zeros(3)
    assert z.blobs.shape == (3, PRIV_BLOB_MAX, PRIV_BLOB_DIM)
    assert z.food.shape == (3, PRIV_FOOD_MAX, PRIV_FOOD_DIM)
    assert z.virus.shape == (3, PRIV_VIRUS_MAX, PRIV_VIRUS_DIM)
    assert z.batch_size == 3


# --- overfit smoke: policy loss → near-zero on a frozen 1-game buffer -------

def _frozen_game(policy, T=8, R=4, seed=0):
    """A frozen fixed mini-trajectory (one BPTT window per seat) with distinct
    per-(t, seat) inputs and fixed heading/split labels — the 'one game replay
    buffer' the overfit smoke memorizes."""
    rng = np.random.default_rng(seed)
    obs = (rng.standard_normal((T, R, OBS_WIDTH)) * 0.5).astype(np.float32)
    is_reset = np.zeros((T, R), bool)
    is_reset[0] = True  # first frame of the segment resets the GRU
    tgt_heading = torch.tensor(rng.integers(0, HEADING_BINS, size=(T, R)), dtype=torch.long)
    tgt_split = torch.tensor(rng.integers(0, 2, size=(T, R)), dtype=torch.float32)
    return obs, is_reset, tgt_heading, tgt_split


def _supervised_loss(policy, obs, is_reset, tgt_heading, tgt_split):
    """CE(heading) + BCE(split) summed over the recurrent BPTT window (mean per
    item), carrying the GRU state across time and resetting on ``is_reset``."""
    T, R = tgt_heading.shape
    hidden = policy.initial_hidden(R)
    total = 0.0
    for t in range(T):
        batched = build_batched_obs(obs[t], is_reset[t])
        hl, sl, _v, hidden = policy(batched, hidden)
        total = total + F.cross_entropy(hl, tgt_heading[t]) \
            + F.binary_cross_entropy_with_logits(sl, tgt_split[t])
    return total / T


def test_overfit_smoke_frozen_game():
    """Wiring sanity: the real network memorizes a frozen 1-game buffer, driving
    the policy loss to near-zero. Proves gradients flow end-to-end through the
    deep-set encoders → trunk → GRU(BPTT) → both heads."""
    torch.manual_seed(0)
    policy = AgarPolicy(entity_hidden=32, trunk_hidden=64)  # small-but-real, for speed
    obs, is_reset, tgt_heading, tgt_split = _frozen_game(policy)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

    with torch.no_grad():
        first = float(_supervised_loss(policy, obs, is_reset, tgt_heading, tgt_split))
    last = first
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        loss = _supervised_loss(policy, obs, is_reset, tgt_heading, tgt_split)
        loss.backward()
        opt.step()
        last = float(loss.detach())

    assert np.isfinite(last)
    assert last < first * 0.05, f"loss did not collapse: {first:.3f} -> {last:.3f}"
    assert last < 0.05, f"policy loss not near-zero: {last:.4f}"


# --- PPO closes the loop with the real network -----------------------------

def _frozen_ppo_buffer(policy, T=4, R=6, seed=1):
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer(rollout_len=T, rows=R, hidden_size=policy.hidden_size)
    buf.obs[:] = rng.standard_normal((T, R, OBS_WIDTH)).astype(np.float32) * 0.2
    buf.is_reset[0] = True
    buf.active[:] = True
    hidden = policy.initial_hidden(R)
    with torch.no_grad():
        for t in range(T):
            batched = build_batched_obs(buf.obs[t], buf.is_reset[t])
            hl, sl, v, hidden = policy(batched, hidden)
            heading, split, logp = sample_actions(hl, sl)
            buf.heading[t] = heading.numpy()
            buf.split[t] = split.numpy()
            buf.logp[t] = logp.numpy()
            buf.value[t] = v.numpy()
    return buf


def test_ppo_update_reduces_loss_with_real_model():
    """train.ppo.ppo_update closes the buffer→update loop over AgarPolicy: value
    loss collapses and entropy drops when every taken action is rewarded."""
    torch.manual_seed(0)
    np.random.seed(0)
    policy = AgarPolicy(entity_hidden=16, trunk_hidden=32)
    buf = _frozen_ppo_buffer(policy)
    cfg = TrainConfig(ppo_epochs=1, minibatch_rows=0, lr=1e-2, entropy_coef=0.0)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    returns = np.ones((buf.T, buf.R), dtype=np.float32)
    advantages = (returns - buf.value).astype(np.float32)

    first = ppo_update(policy, opt, buf, advantages, returns, cfg, torch.device("cpu"))
    for _ in range(50):
        last = ppo_update(policy, opt, buf, advantages, returns, cfg, torch.device("cpu"))

    assert last["value_loss"] < first["value_loss"] * 0.5
    assert last["entropy"] < first["entropy"]
    assert np.isfinite(last["loss"])


def test_evaluate_actions_grad_and_shapes():
    """BPTT re-eval returns grad-carrying (T, |rows|) tensors matching stored
    actions — the recompute PPO's ratio needs."""
    torch.manual_seed(0)
    policy = AgarPolicy(entity_hidden=8, trunk_hidden=16)
    buf = _frozen_ppo_buffer(policy, T=3, R=4)
    rows = np.array([0, 2, 3])
    logp, ent, val = evaluate_actions(policy, buf, rows, torch.device("cpu"))
    assert logp.shape == (buf.T, len(rows))
    assert ent.shape == (buf.T, len(rows))
    assert val.shape == (buf.T, len(rows))
    assert logp.requires_grad and val.requires_grad


# --- factory ---------------------------------------------------------------

def test_build_policy_factory():
    from train.harness.policy_stub import SkeletonPolicy

    real = build_policy(TrainConfig())
    assert isinstance(real, AgarPolicy)
    assert real.hidden_size == 256
    stub = build_policy(TrainConfig(use_skeleton=True))
    assert isinstance(stub, SkeletonPolicy)
