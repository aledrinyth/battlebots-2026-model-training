"""P3.5 — optional potential-based food shaping (Stage-0 bootstrap aid).

The shaping is **off by default** (``food_shaping_coef=0.0``), in which case the
reward must be *byte-identical* to the pure dense-Δmass reward the playbook
specifies. When enabled it adds a dense "move toward the nearest food" term.
These tests pin both properties. The shaping/reward tests are numpy-only (run in
either venv); the one config-hash test needs torch (via provenance) and skips in
the match-legal bot venv.
"""

from __future__ import annotations

import numpy as np
import pytest

from train.harness.config import TrainConfig
from train.harness.game_runner import GameRunner, nearest_food_dist


def _runner(coef):
    # One food-only game (seat 0 learns, rest sit still); deterministic seed.
    return GameRunner(
        stream_index=0, policy_seats=(0,), scripted_by_seat={p: "dont_move" for p in range(1, 8)},
        base_seed=4242, mass_scale=1.0, reward_clip=10.0, food_shaping_coef=coef,
    )


def _roll(runner, headings, seed_actions=0):
    """Drive `len(headings)` rounds with a fixed heading; return the rewards seen."""
    rewards = []
    for hb in headings:
        frame = runner.observe()
        runner.step(np.array([hb], dtype=np.float32), np.array([0.0], dtype=np.float32))
        # the reward for the action just taken surfaces on the *next* observe;
        # read it straight from the stash instead (same value).
        rewards.append(runner._stash[0][0])
    return rewards


def test_default_is_pure_delta_mass():
    cfg = TrainConfig()
    assert cfg.food_shaping_coef == 0.0
    r = _runner(0.0)
    # With coef 0 the nearest-food scan is skipped and _dist_before stays empty.
    r.observe()
    r.step(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32))
    assert r._dist_before == {}                 # no shaping bookkeeping at all
    # reward is exactly clipped Δmass (0 on a step that ate nothing at the floor)
    assert r._stash[0][0] == 0.0


def test_shaping_config_in_hash():
    pytest.importorskip("torch")  # config_hash pulls train.model (torch); train venv only
    from train.harness.provenance import config_hash
    base = TrainConfig(policy_seats=(0,))
    assert config_hash(base) != config_hash(TrainConfig(policy_seats=(0,), food_shaping_coef=0.05))


def test_shaping_rewards_approaching_over_fleeing():
    # Convention-agnostic: from the *same* initial state, take one step in each of
    # the 24 heading bins (fresh runner per bin). The shaping is symmetric, so:
    #   * heading at the nearest food (biggest distance reduction) yields the
    #     highest reward and it is positive, while
    #   * heading away yields the lowest and it is negative
    #   -> the spread proves the term actually tracks net approach (not jitter).
    from bot.adapters.act_v1 import HEADING_BINS

    rewards = []
    for hb in range(HEADING_BINS):
        rr = GameRunner(0, (0,), {p: "dont_move" for p in range(1, 8)}, 4242, 1.0, 10.0, 0.1)
        rewards.append(_roll(rr, [hb])[0])
    assert max(rewards) > 0.0        # approaching food is rewarded
    assert min(rewards) < 0.0        # fleeing is penalised (symmetric -> jitter-proof)

    # With shaping OFF, that same best-approach step is pure Δmass (0 here).
    off = GameRunner(0, (0,), {p: "dont_move" for p in range(1, 8)}, 4242, 1.0, 10.0, 0.0)
    best_bin = int(np.argmax(rewards))
    assert _roll(off, [best_bin])[0] == 0.0


def test_shaping_symmetric_cancels_under_jitter():
    # A symmetric term must (nearly) cancel when a blob goes out and comes back:
    # the exploit that broke the one-sided version was jitter farming positive
    # half-steps. Over an even out-and-back the shaped extra should be ~0, not the
    # large positive a max(0,·) bonus would accumulate.
    from bot.adapters.act_v1 import HEADING_BINS
    r = _runner(0.1)
    fwd, back = 3, (3 + HEADING_BINS // 2) % HEADING_BINS
    rr = _roll(r, [fwd, back, fwd, back, fwd, back])
    # net displacement ~0 over the out-and-back -> summed shaping small in magnitude
    assert abs(sum(rr)) < 0.1 * 3.0   # << what one-sided harvesting would give


def test_nearest_food_dist_finite_at_start():
    r = _runner(0.1)
    r.observe()
    d = nearest_food_dist(r.actor, 0)
    assert np.isfinite(d) and d >= 0.0
