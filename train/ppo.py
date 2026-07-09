"""P3.3 — recurrent PPO update (formalized out of the P3.2 learner).

The P3.2 learner carried a "minimal correct" PPO update inline to prove the
buffer→update path closes. P3.3 formalizes it here (``train/ppo.py``, per the
playbook) next to ``train/model.py``, unchanged in behavior: the learner and the
two-box rollout worker now import :func:`sample_actions`, :func:`evaluate_actions`,
and :func:`ppo_update` from this module. GAE itself stays a method of
:class:`~train.harness.rollout.RolloutBuffer` (it is the buffer's own reduction);
this module owns the *policy-gradient* step that consumes ``(advantages,
returns)``.

Contract (unchanged from P3.2, so the transport/buffer/GAE are untouched):

* **Actions** are a categorical over the 24 heading bins × an independent
  Bernoulli split; the joint log-prob is their sum.
* **The update is recurrent.** For each minibatch of *rows* (BPTT segments kept
  intact — we split on the row axis, never the time axis), the GRU is replayed
  over the whole ``T``-step window from the stored ``hidden0`` to recompute
  ``logp / entropy / value`` under the *current* policy, forming the PPO
  importance ratio against the stored collection log-probs.
* **Dead-gap masking.** Every quantity is masked by ``buffer.active`` so the
  respawn gap (a seat is not queried while dead) contributes no gradient;
  advantages are normalized over active entries only.

The ``policy`` argument is any module honoring the
``policy(BatchedObs, hidden) -> (heading_logits, split_logit, value, new_hidden)``
contract — :class:`~train.model.AgarPolicy` in production, the
:class:`~train.harness.policy_stub.SkeletonPolicy` in the P3.2 wiring tests.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from train.harness.collate import build_batched_obs
from train.harness.config import TrainConfig
from train.harness.rollout import RolloutBuffer


# --- action sampling --------------------------------------------------------

def sample_actions(heading_logits: torch.Tensor, split_logit: torch.Tensor):
    """Sample ``(heading bin, split)`` and return ``(heading, split, logp)``.

    Heading is categorical over the 24 bins; split is Bernoulli. The joint
    log-prob is their sum (independent heads).
    """
    hd = torch.distributions.Categorical(logits=heading_logits)
    heading = hd.sample()
    logp_h = hd.log_prob(heading)

    sd = torch.distributions.Bernoulli(logits=split_logit)
    split = sd.sample()
    logp_s = sd.log_prob(split)

    return heading, split, logp_h + logp_s


def log_prob_of(heading_logits: torch.Tensor, split_logit: torch.Tensor,
                heading: torch.Tensor, split: torch.Tensor) -> torch.Tensor:
    """Joint log-prob of a *given* ``(heading, split)`` under the current logits.

    Used by temporally-extended exploration (``train.harness.explore``): when a
    stream is *holding* a previously-sampled action, the action taken is not a
    fresh sample of this step's distribution, so its collection-time log-prob must
    be recomputed here (it is what PPO stores as ``old`` logp for the ratio). For
    a fresh (non-held) action this returns exactly ``sample_actions``'s logp.
    """
    hd = torch.distributions.Categorical(logits=heading_logits)
    sd = torch.distributions.Bernoulli(logits=split_logit)
    return hd.log_prob(heading) + sd.log_prob(split.float())


# --- BPTT re-evaluation -----------------------------------------------------

def evaluate_actions(policy: nn.Module, buffer: RolloutBuffer, rows: np.ndarray, device):
    """Replay the GRU over the BPTT window for a subset of ``rows``.

    Returns ``(logp, entropy, value)`` each ``(T, len(rows))`` with grad, matching
    the collection-time actions stored in the buffer. This is the truncated-BPTT
    recompute PPO needs to form the importance ratio under the *current* policy.
    """
    T = buffer.T
    hidden = buffer.hidden0[rows].to(device)
    logps, ents, values = [], [], []
    for t in range(T):
        batched = build_batched_obs(buffer.obs[t, rows], buffer.is_reset[t, rows], device)
        hl, sl, v, hidden = policy(batched, hidden)
        hd = torch.distributions.Categorical(logits=hl)
        sd = torch.distributions.Bernoulli(logits=sl)
        heading = torch.as_tensor(buffer.heading[t, rows], device=device)
        split = torch.as_tensor(buffer.split[t, rows], dtype=torch.float32, device=device)
        logps.append(hd.log_prob(heading) + sd.log_prob(split))
        ents.append(hd.entropy() + sd.entropy())
        values.append(v)
    return torch.stack(logps), torch.stack(ents), torch.stack(values)


# --- the PPO update ---------------------------------------------------------

def ppo_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    advantages: np.ndarray,
    returns: np.ndarray,
    cfg: TrainConfig,
    device,
) -> Dict[str, float]:
    """One PPO update over the buffer. Returns loss/metrics for logging.

    Masks every quantity by ``active`` (dead-gap steps contribute nothing), and
    normalizes advantages over the active entries only.
    """
    T, R = buffer.T, buffer.R
    active_all = torch.as_tensor(buffer.active, device=device)  # (T,R) bool
    adv_all = torch.as_tensor(advantages, device=device)
    ret_all = torch.as_tensor(returns, device=device)
    oldlogp_all = torch.as_tensor(buffer.logp, device=device)

    # Advantage normalization over active entries (global, before minibatching).
    amask = active_all
    a_active = adv_all[amask]
    if a_active.numel() > 1:
        adv_all = (adv_all - a_active.mean()) / (a_active.std() + 1e-8)

    mb = cfg.minibatch_rows if cfg.minibatch_rows > 0 else R
    metrics: Dict[str, float] = {}
    n_batches = 0
    for _ in range(cfg.ppo_epochs):
        row_perm = np.random.permutation(R)
        for start in range(0, R, mb):
            rows = np.sort(row_perm[start:start + mb])
            new_logp, ent, value = evaluate_actions(policy, buffer, rows, device)  # (T, r)
            m = active_all[:, rows].float()
            n = m.sum().clamp_min(1.0)

            adv = adv_all[:, rows]
            ret = ret_all[:, rows]
            oldlogp = oldlogp_all[:, rows]

            ratio = torch.exp(new_logp - oldlogp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv
            policy_loss = -(torch.min(surr1, surr2) * m).sum() / n
            value_loss = (((ret - value) ** 2) * m).sum() / n
            entropy = (ent * m).sum() / n
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((oldlogp - new_logp) * m).sum() / n
                clipfrac = (((ratio - 1.0).abs() > cfg.clip).float() * m).sum() / n
            metrics = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "approx_kl": float(approx_kl.item()),
                "clipfrac": float(clipfrac.item()),
                "grad_norm": float(grad_norm),
            }
            n_batches += 1
    metrics["n_batches"] = n_batches
    return metrics


__all__ = ["sample_actions", "log_prob_of", "evaluate_actions", "ppo_update"]
