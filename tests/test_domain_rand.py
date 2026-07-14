"""P6.1 domain randomization: food-density DR is real, seeded, and inert off.

The load-bearing property is the OFF case: with ``dr_food_density=0.0`` (every
eval / demo / parity / compliance path), trajectories must be byte-identical
to the pristine engine even when the patch is installed in the process — the
same discipline fast_vision proved at P1.5.
"""

from __future__ import annotations

import numpy as np
import pytest

from lib.config.arena import FOOD_COUNT
from train.env import domain_rand
from train.env.wrapper import WrapperEnv, make_move


@pytest.fixture(autouse=True)
def _clean_patch():
    yield
    domain_rand.uninstall()


def _food_positions(env: WrapperEnv) -> list:
    return sorted((f.pos[0], f.pos[1]) for f in env.state.map.foods)


def _run_rounds(env: WrapperEnv, obs, n: int):
    for _ in range(n):
        obs, done = env.step({pid: make_move(pid, 0.0, 0.0) for pid in obs})
        if done:
            break
    return obs


def test_off_is_bit_identical_even_when_installed():
    env_a = WrapperEnv(seed=7)          # pristine path
    obs_a = env_a.reset()
    foods_a = _food_positions(env_a)

    domain_rand.install()               # installed but unconfigured
    env_b = WrapperEnv(seed=7, dr_food_density=0.0)
    obs_b = env_b.reset()
    assert len(env_b.state.map.foods) == FOOD_COUNT
    assert _food_positions(env_b) == foods_a

    _run_rounds(env_a, obs_a, 20)
    _run_rounds(env_b, obs_b, 20)
    assert _food_positions(env_b) == _food_positions(env_a)  # streams stayed aligned


def test_dr_draws_are_seeded_and_bounded():
    counts = set()
    for seed in range(8):
        env = WrapperEnv(seed=seed, dr_food_density=0.2)
        env.reset()
        n = len(env.state.map.foods)
        lo, hi = round(FOOD_COUNT * 0.8), round(FOOD_COUNT * 1.2)
        assert lo <= n <= hi, (seed, n)
        counts.add(n)
    assert len(counts) > 1, "eight seeds all drew the same food count"

    # Determinism: same seed ⇒ same draw and same spawn layout.
    e1 = WrapperEnv(seed=3, dr_food_density=0.2)
    e1.reset()
    e2 = WrapperEnv(seed=3, dr_food_density=0.2)
    e2.reset()
    assert len(e1.state.map.foods) == len(e2.state.map.foods)
    assert _food_positions(e1) == _food_positions(e2)


def test_regrowth_tracks_the_drawn_target_all_game():
    env = WrapperEnv(seed=11, dr_food_density=0.2)
    obs = env.reset()
    target = len(env.state.map.foods)
    assert target != FOOD_COUNT or True  # target may equal 160 by chance; not the assert

    # Eat-equivalent: remove a few foods by hand, then step — the mutator's
    # per-round _ensure_food_count must refill to the DR target, not to 160.
    for f in list(env.state.map.foods[:5]):
        env.state.map.remove_food(f.food_id)
    assert len(env.state.map.foods) == target - 5
    _run_rounds(env, obs, 1)
    assert len(env.state.map.foods) == target


def test_concurrent_envs_keep_their_own_targets():
    # Two DR games in one process (the vec_actor shape): each game's target is
    # captured on its own GameState; interleaved stepping must not leak.
    e1 = WrapperEnv(seed=101, dr_food_density=0.2)
    o1 = e1.reset()
    t1 = len(e1.state.map.foods)
    e2 = WrapperEnv(seed=505, dr_food_density=0.2)
    o2 = e2.reset()
    t2 = len(e2.state.map.foods)

    for _ in range(3):  # interleave
        o1 = _run_rounds(e1, o1, 1)
        o2 = _run_rounds(e2, o2, 1)
    assert len(e1.state.map.foods) == t1
    assert len(e2.state.map.foods) == t2


def test_dr_default_is_hash_neutral():
    pytest.importorskip("torch")  # config/provenance are train-side; skipped in the bot venv
    from train.harness.config import TrainConfig
    from train.harness.provenance import config_dict

    base = config_dict(TrainConfig(), for_hash=True)
    assert "dr_food_density" not in base           # inert at default
    on = config_dict(TrainConfig(dr_food_density=0.2), for_hash=True)
    assert on["dr_food_density"] == 0.2            # a real experiment hashes in
