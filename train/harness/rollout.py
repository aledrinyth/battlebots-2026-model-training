"""Trajectory buffer + GAE (P3.2).

A fixed-capacity ``(T, R)`` buffer — ``T = rollout_len`` steps × ``R =
num_actors × seats`` fixed-slot rows — holding everything a recurrent PPO update
needs: the flattened obs and reset flags (to replay the GRU over the BPTT window),
the taken action + its collection log-prob and value, and the reward/done that
arrive one round late. It is **reset and refilled every update**, so it is a hard
bound on memory (a P3.2 acceptance concern: "without memory growth").

The subtlety P3.2 has to get right is that this is an 8-agent env with
**death/respawn gaps**: a row is *active* only while its seat is alive; on death
its last active step carries ``done=1`` and it goes inactive for the 30-round
respawn gap, then resumes with ``is_reset=1``. :meth:`compute_gae` handles that:

* ``done`` masks the bootstrap so no advantage crosses a death or a game
  boundary;
* an inactive step zeroes the running GAE accumulator, so two live segments of
  the *same* row never leak advantage across the dead gap between them;
* inactive steps get zero advantage/return and are masked out of the PPO loss.

Rewards and dones are filled one round after the step they belong to (they are
only known post-step — see ``vec_actor``); :meth:`set_prev_transition` writes
them into the previous step's row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from train.harness.flat_obs import OBS_WIDTH


@dataclass
class GAEResult:
    advantages: np.ndarray  # (T, R) float32, zero on inactive steps
    returns: np.ndarray     # (T, R) float32, zero on inactive steps


class RolloutBuffer:
    """Fixed ``(T, R)`` recurrent-PPO buffer over fixed-slot rows.

    All per-step arrays are numpy (filled from the shared-memory slab + detached
    policy outputs); only ``hidden0`` (the GRU state entering step 0, for BPTT
    replay in the update) is a torch tensor.
    """

    def __init__(self, rollout_len: int, rows: int, hidden_size: int) -> None:
        self.T = rollout_len
        self.R = rows
        self.H = hidden_size
        T, R = rollout_len, rows

        self.obs = np.zeros((T, R, OBS_WIDTH), dtype=np.float32)
        self.is_reset = np.zeros((T, R), dtype=bool)
        self.active = np.zeros((T, R), dtype=bool)
        self.heading = np.zeros((T, R), dtype=np.int64)
        self.split = np.zeros((T, R), dtype=np.float32)
        self.logp = np.zeros((T, R), dtype=np.float32)
        self.value = np.zeros((T, R), dtype=np.float32)
        self.reward = np.zeros((T, R), dtype=np.float32)
        self.done = np.zeros((T, R), dtype=np.float32)
        self.hidden0 = torch.zeros((R, hidden_size), dtype=torch.float32)
        self._filled = 0

    # -- filling -------------------------------------------------------------

    def reset(self) -> None:
        self._filled = 0
        # Arrays are fully overwritten before use each update; only masks that
        # gate reads need clearing defensively.
        self.active[:] = False
        self.reward[:] = 0.0
        self.done[:] = 0.0

    def set_hidden0(self, hidden: torch.Tensor) -> None:
        """Record the GRU state entering step 0 (detached; BPTT replay start)."""
        self.hidden0 = hidden.detach().clone()

    def add(
        self,
        t: int,
        obs: np.ndarray,       # (R, OBS_WIDTH)
        is_reset: np.ndarray,  # (R,)
        active: np.ndarray,    # (R,)
        heading: np.ndarray,   # (R,)
        split: np.ndarray,     # (R,)
        logp: np.ndarray,      # (R,)
        value: np.ndarray,     # (R,)
    ) -> None:
        self.obs[t] = obs
        self.is_reset[t] = is_reset
        self.active[t] = active
        self.heading[t] = heading
        self.split[t] = split
        self.logp[t] = logp
        self.value[t] = value
        self._filled = max(self._filled, t + 1)

    def set_prev_transition(
        self, t_prev: int, prev_reward: np.ndarray, prev_done: np.ndarray, prev_acted: np.ndarray
    ) -> None:
        """Write reward/done for step ``t_prev`` from round ``t_prev+1``'s meta.

        Only rows that actually acted at ``t_prev`` (``prev_acted``) get a
        transition; this should coincide with ``active[t_prev]`` and is asserted
        cheaply in tests.
        """
        acted = prev_acted > 0.5
        self.reward[t_prev] = np.where(acted, prev_reward, 0.0)
        self.done[t_prev] = np.where(acted, prev_done, 0.0)

    # -- advantage estimation ------------------------------------------------

    def compute_gae(
        self, last_values: np.ndarray, gamma: float, lam: float
    ) -> GAEResult:
        """GAE(λ) over the filled ``[0, T)`` steps, gap- and segment-aware.

        Args:
            last_values: ``(R,)`` bootstrap value ``V(s_T)`` for rows still active
                past the buffer end (masked out where the last step is ``done``).
        """
        T, R = self.T, self.R
        adv = np.zeros((T, R), dtype=np.float32)
        ret = np.zeros((T, R), dtype=np.float32)

        gae = np.zeros((R,), dtype=np.float32)
        for t in range(T - 1, -1, -1):
            act_t = self.active[t]                          # (R,) bool
            v_next = self.value[t + 1] if t + 1 < T else last_values.astype(np.float32)
            mask = np.where(self.done[t] > 0.5, 0.0, 1.0).astype(np.float32)  # 1-done
            delta = self.reward[t] + gamma * v_next * mask - self.value[t]
            gae = delta + gamma * lam * mask * gae
            # Inactive rows: zero the accumulator so segments never leak across a
            # dead gap, and contribute nothing to the loss.
            gae = np.where(act_t, gae, 0.0).astype(np.float32)
            adv[t] = np.where(act_t, gae, 0.0)
            ret[t] = np.where(act_t, gae + self.value[t], 0.0)
        return GAEResult(advantages=adv, returns=ret)

    # -- torch views ---------------------------------------------------------

    def active_count(self) -> int:
        return int(self.active.sum())


__all__ = ["RolloutBuffer", "GAEResult"]
