"""P3.5 — temporally-extended exploration (action persistence).

Off by default (byte-identical collection); when on, a stream holds its action
for a random burst, and a held action's stored log-prob is recomputed under the
current policy so PPO's ratio is correct. Train-side (torch).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from train.harness.config import TrainConfig  # noqa: E402
from train.harness.explore import ActionPersistence  # noqa: E402
from train.ppo import log_prob_of  # noqa: E402


def _uniform_step(ap, fresh_heading, is_reset=False, n=1, logits=None):
    hl = torch.zeros(n, 24) if logits is None else logits[0]
    sl = torch.zeros(n, 1) if logits is None else logits[1]
    heading = torch.tensor([fresh_heading] * n if np.isscalar(fresh_heading) else fresh_heading)
    split = torch.zeros(n)
    logp = log_prob_of(hl, sl, heading, split)
    h, s, lp = ap.apply(hl, sl, heading, split, logp, np.array([is_reset] * n))
    return int(h[0]), float(s[0]), float(lp[0]), hl, sl


def test_off_by_default():
    cfg = TrainConfig()
    assert cfg.explore_persist_prob == 0.0
    # constructor guards the invalid range so "off" is never instantiated
    with pytest.raises(ValueError):
        ActionPersistence(4, 0.0, 8, base_seed=1)


def test_learner_builds_persistence_only_when_enabled():
    from train.harness.learner import VecLearner
    assert VecLearner(TrainConfig(num_actors=1, use_skeleton=True)).persistence is None
    on = VecLearner(TrainConfig(num_actors=1, use_skeleton=True,
                                explore_persist_prob=0.5, explore_persist_max=6))
    assert on.persistence is not None


def test_persistence_holds_the_action():
    # persist_prob=1 => every free step starts a hold, so the *next* step must
    # repeat the previous action rather than the new fresh one.
    ap = ActionPersistence(1, persist_prob=1.0, max_duration=3, base_seed=7)
    out0, *_ = _uniform_step(ap, fresh_heading=0)   # starts a hold on action 0
    assert out0 == 0                                 # start step returns the fresh action
    out1, *_ = _uniform_step(ap, fresh_heading=13)  # fresh is 13, but we're holding 0
    assert out1 == 0                                 # held, not 13


def test_reset_drops_the_hold():
    ap = ActionPersistence(1, persist_prob=1.0, max_duration=8, base_seed=3)
    _uniform_step(ap, fresh_heading=4)              # start hold on 4
    # a respawn on the next frame must clear the hold -> the fresh action wins
    out, *_ = _uniform_step(ap, fresh_heading=9, is_reset=True)
    assert out == 9


def test_held_logp_matches_recompute():
    # non-uniform logits so a held action's logp differs from the fresh sample's.
    torch.manual_seed(0)
    hl = torch.randn(1, 24)
    sl = torch.randn(1, 1)
    ap = ActionPersistence(1, persist_prob=1.0, max_duration=4, base_seed=11)
    # step 0 starts a hold on action 2
    h0 = torch.tensor([2]); s0 = torch.zeros(1)
    ap.apply(hl, sl, h0, s0, log_prob_of(hl, sl, h0, s0), np.array([False]))
    # step 1: fresh action 20, but we hold 2 -> returned logp must be logp of (2,0)
    h1 = torch.tensor([20]); s1 = torch.zeros(1)
    h, s, lp = ap.apply(hl, sl, h1, s1, log_prob_of(hl, sl, h1, s1), np.array([False]))
    assert int(h[0]) == 2
    expected = log_prob_of(hl, sl, torch.tensor([2]), torch.zeros(1))
    assert torch.allclose(lp, expected, atol=1e-6)


def test_explore_fields_in_config_hash():
    from train.harness.provenance import config_hash
    base = TrainConfig(policy_seats=(0,))
    assert config_hash(base) != config_hash(
        TrainConfig(policy_seats=(0,), explore_persist_prob=0.3))
    assert config_hash(base) != config_hash(
        TrainConfig(policy_seats=(0,), explore_persist_prob=0.3, explore_persist_max=12))


# --- oracle-guided bursts (Track C) ------------------------------------------

from train.harness.explore import (  # noqa: E402
    OracleGuidance, _FOOD0_DX, _FOOD0_DY, _FOOD_MASK0,
)


def _obs_with_food(n=1, dx=0.5, dy=0.0, present=True):
    from train.harness.flat_obs import OBS_WIDTH
    obs = np.zeros((n, OBS_WIDTH), dtype=np.float32)
    obs[:, _FOOD0_DX] = dx
    obs[:, _FOOD0_DY] = dy
    obs[:, _FOOD_MASK0] = 1.0 if present else 0.0
    return obs


def _oracle_step(og, obs, fresh_heading, fresh_split=0.0, is_reset=False, n=1):
    hl = torch.zeros(n, 24)
    sl = torch.zeros(n, 1)
    heading = torch.tensor([fresh_heading] * n)
    split = torch.full((n,), float(fresh_split))
    logp = log_prob_of(hl, sl, heading, split)
    h, s, lp = og.apply(hl, sl, heading, split, logp, np.array([is_reset] * n), obs)
    return int(h[0]), float(s[0]), lp


def test_oracle_offsets_match_flat_obs_layout():
    from train.harness import flat_obs
    assert _FOOD0_DX == flat_obs._OFFSETS["food"][0]
    assert _FOOD0_DY == flat_obs._OFFSETS["food"][0] + 1
    assert _FOOD_MASK0 == flat_obs._OFFSETS["food_mask"][0]


def test_oracle_off_by_default():
    cfg = TrainConfig()
    assert cfg.explore_oracle_prob == 0.0
    with pytest.raises(ValueError):
        OracleGuidance(4, 0.0, 8, base_seed=1)


def test_oracle_bins_point_at_food():
    # bin 0 = +x (east), CCW: +y -> bin 6 (90deg), -x -> bin 12, -y -> bin 18.
    for dx, dy, want in [(0.5, 0.0, 0), (0.0, 0.5, 6), (-0.5, 0.0, 12),
                         (0.0, -0.5, 18), (0.5, 0.5, 3)]:
        bins, has = OracleGuidance.oracle_bins(_obs_with_food(dx=dx, dy=dy))
        assert has[0]
        assert int(bins[0]) == want, (dx, dy, int(bins[0]), want)


def test_oracle_overrides_toward_food_and_zeroes_split():
    og = OracleGuidance(1, oracle_prob=1.0, max_duration=4, base_seed=5)
    # fresh action is bin 13 + split; food is at +x -> the oracle forces (0, no-split)
    h, s, _ = _oracle_step(og, _obs_with_food(dx=0.5, dy=0.0), fresh_heading=13,
                           fresh_split=1.0)
    assert h == 0 and s == 0.0


def test_oracle_reaims_every_round():
    og = OracleGuidance(1, oracle_prob=1.0, max_duration=8, base_seed=5)
    h0, *_ = _oracle_step(og, _obs_with_food(dx=0.5, dy=0.0), fresh_heading=13)
    h1, *_ = _oracle_step(og, _obs_with_food(dx=0.0, dy=0.5), fresh_heading=13)
    assert (h0, h1) == (0, 6)  # tracks the food as it moves in the obs


def test_oracle_skips_rows_without_visible_food():
    og = OracleGuidance(1, oracle_prob=1.0, max_duration=4, base_seed=5)
    h, s, _ = _oracle_step(og, _obs_with_food(present=False), fresh_heading=13,
                           fresh_split=1.0)
    assert h == 13 and s == 1.0  # fresh sample kept


def test_oracle_reset_ends_the_burst():
    og = OracleGuidance(1, oracle_prob=1.0, max_duration=8, base_seed=5)
    _oracle_step(og, _obs_with_food(), fresh_heading=13)  # burst running
    og.p = 0.0  # stop new bursts so the reset's effect is observable
    h, *_ = _oracle_step(og, _obs_with_food(), fresh_heading=13, is_reset=True)
    assert h == 13  # burst dropped on respawn -> fresh action wins


def test_oracle_logp_is_recomputed_for_overridden_action():
    torch.manual_seed(0)
    hl = torch.randn(1, 24)
    sl = torch.randn(1, 1)
    og = OracleGuidance(1, oracle_prob=1.0, max_duration=4, base_seed=5)
    heading = torch.tensor([13])
    split = torch.ones(1)
    logp = log_prob_of(hl, sl, heading, split)
    h, s, lp = og.apply(hl, sl, heading, split, logp, np.array([False]),
                        _obs_with_food(dx=0.5, dy=0.0))
    expected = log_prob_of(hl, sl, torch.tensor([0]), torch.zeros(1))
    assert int(h[0]) == 0
    assert torch.allclose(lp, expected, atol=1e-6)


def test_oracle_learner_gating():
    from train.harness.learner import VecLearner
    assert VecLearner(TrainConfig(num_actors=1, use_skeleton=True)).oracle is None
    on = VecLearner(TrainConfig(num_actors=1, use_skeleton=True,
                                explore_oracle_prob=0.5, explore_oracle_max=6))
    assert on.oracle is not None


def test_oracle_fields_hash_only_when_active():
    # Additive-knob discipline: at the inert default the new fields must NOT
    # change pre-existing experiment hashes (so g1_rig/g1_rig_ms4 keep resuming);
    # any active value is a distinct experiment.
    from train.harness.provenance import config_dict, config_hash
    base = TrainConfig(policy_seats=(0,))
    assert "explore_oracle_prob" not in config_dict(base, for_hash=True)
    assert config_hash(base) != config_hash(
        TrainConfig(policy_seats=(0,), explore_oracle_prob=0.1))
    assert config_hash(base) != config_hash(
        TrainConfig(policy_seats=(0,), explore_oracle_prob=0.1, explore_oracle_max=15))
