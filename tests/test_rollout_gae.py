"""P3.2 — GAE correctness under death/respawn gaps (``train.harness.rollout``).

GAE is the one place the 8-agent + dead-gap structure can silently corrupt the
training signal, so it gets hand-computed reference values for three cases: a
contiguous segment (standard GAE), a death (``done`` masks the bootstrap so no
advantage crosses it, even though the next stored value is garbage from an
inactive zero-row), and a respawn gap (the inactive step zeroes the accumulator
so two live segments of the same row never leak advantage across the gap).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # RolloutBuffer holds a torch hidden state

from train.harness.rollout import RolloutBuffer


def _buf(T, R, hidden=4):
    return RolloutBuffer(rollout_len=T, rows=R, hidden_size=hidden)


def test_gae_contiguous_matches_hand_computation():
    g, lam = 0.9, 0.5
    b = _buf(4, 1)
    b.value[:, 0] = [1.0, 1.0, 1.0, 1.0]
    b.reward[:, 0] = [0.0, 0.0, 0.0, 1.0]
    b.active[:, 0] = True
    b.done[:, 0] = 0.0
    res = b.compute_gae(last_values=np.array([0.0], dtype=np.float32), gamma=g, lam=lam)

    # backward: delta3=0, delta2=-0.1, delta1=-0.1, delta0=-0.1
    exp_adv = np.array([-0.16525, -0.145, -0.1, 0.0], dtype=np.float32)
    exp_ret = exp_adv + np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(res.advantages[:, 0], exp_adv, rtol=1e-5)
    np.testing.assert_allclose(res.returns[:, 0], exp_ret, rtol=1e-5)


def test_gae_death_and_respawn_gap():
    """t0,t1 live (dies at t1), t2 dead-gap, t3,t4 live (respawned)."""
    g, lam = 0.9, 0.5
    b = _buf(5, 1)
    b.value[:, 0] = [1.0, 1.0, 5.0, 1.0, 1.0]   # value[2] is garbage (inactive zero-row)
    b.reward[:, 0] = [0.0, -1.0, 0.0, 0.0, 1.0]  # -1 at t1 = death mass loss
    b.active[:, 0] = [True, True, False, True, True]
    b.done[:, 0] = [0.0, 1.0, 0.0, 0.0, 0.0]     # death terminates the segment at t1
    res = b.compute_gae(last_values=np.array([0.0], dtype=np.float32), gamma=g, lam=lam)

    # segment {t3,t4}: delta4=0 (gae4=0), delta3=-0.1 (gae3=-0.1)
    # t2 inactive: adv=0, accumulator reset
    # segment {t0,t1}: done[1]=1 masks the garbage value[2]; delta1=-2, gae1=-2
    #                  delta0=-0.1, gae0=-0.1 + g*lam*(-2) = -1.0
    exp_adv = np.array([-1.0, -2.0, 0.0, -0.1, 0.0], dtype=np.float32)
    exp_ret = np.array([0.0, -1.0, 0.0, 0.9, 1.0], dtype=np.float32)
    np.testing.assert_allclose(res.advantages[:, 0], exp_adv, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(res.returns[:, 0], exp_ret, rtol=1e-5, atol=1e-6)


def test_inactive_rows_get_zero_advantage():
    b = _buf(3, 2)
    b.value[:] = 1.0
    b.reward[:] = 0.5
    b.active[:, 0] = True
    b.active[:, 1] = False  # row 1 never acts
    b.done[:] = 0.0
    res = b.compute_gae(last_values=np.zeros(2, np.float32), gamma=0.99, lam=0.95)
    assert np.all(res.advantages[:, 1] == 0.0)
    assert np.all(res.returns[:, 1] == 0.0)
    assert np.any(res.advantages[:, 0] != 0.0)


def test_set_prev_transition_only_writes_acted_rows():
    b = _buf(2, 3)
    b.set_prev_transition(
        0,
        prev_reward=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        prev_done=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        prev_acted=np.array([1.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(b.reward[0], [1.0, 0.0, 3.0])
    np.testing.assert_array_equal(b.done[0], [1.0, 0.0, 1.0])
