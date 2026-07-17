"""Deploy-time defensive rails (Track 2) — behavior + wiring.

Constructs CanonicalObs frames by hand and asserts the rail vetoes the
suicidal move classes while leaving safe actions untouched, then checks the
`railed:` make_bot seam that gates it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bot.adapters.act_v1 import (
    HEADING_BINS,
    CanonicalAction,
    bin_to_vector,
)
from bot.adapters.obs_v1 import (
    ENEMY_DIM,
    ENEMY_MAX,
    FOOD_DIM,
    FOOD_MAX,
    OWN_DIM,
    OWN_MAX,
    SELF_DIM,
    VIRUS_DIM,
    VIRUS_MAX,
    CanonicalObs,
)
from bot.rails import apply_rails


def _blank_obs(enemies=(), viruses=()):
    enemy = np.zeros((ENEMY_MAX, ENEMY_DIM), np.float32)
    emask = np.zeros(ENEMY_MAX, np.float32)
    for i, feats in enumerate(enemies):
        enemy[i, : len(feats)] = feats
        emask[i] = 1.0
    virus = np.zeros((VIRUS_MAX, VIRUS_DIM), np.float32)
    vmask = np.zeros(VIRUS_MAX, np.float32)
    for i, feats in enumerate(viruses):
        virus[i, : len(feats)] = feats
        vmask[i] = 1.0
    return CanonicalObs(
        self_vec=np.zeros(SELF_DIM, np.float32),
        food=np.zeros((FOOD_MAX, FOOD_DIM), np.float32),
        food_mask=np.zeros(FOOD_MAX, np.float32),
        enemy=enemy, enemy_mask=emask,
        own=np.zeros((OWN_MAX, OWN_DIM), np.float32),
        own_mask=np.zeros(OWN_MAX, np.float32),
        virus=virus, virus_mask=vmask,
        is_reset=False,
    )


def _bin_toward(dx, dy):
    step = 2 * math.pi / HEADING_BINS
    return int(round(math.atan2(dy, dx) / step)) % HEADING_BINS


# enemy feature vector: [dx, dy, radius, vx, vy, cd, can_eat_me, split_threat, i_can_eat]
def _enemy(dx, dy, can_eat_me=0.0, split_threat=0.0):
    return [dx, dy, 0.3, 0, 0, 0, can_eat_me, split_threat, 0.0]


# virus feature vector: [dx, dy, dist, radius, is_dangerous]
def _virus(dx, dy, dist, dangerous):
    return [dx, dy, dist, 0.05, dangerous]


def test_no_threat_passes_through_unchanged():
    obs = _blank_obs(enemies=[_enemy(0.5, 0.0, i_can_eat := 0.0)])
    act = CanonicalAction(heading_bin=0, split=True)
    out = apply_rails(obs, act)
    assert out is act  # identical object → policy action preserved


def test_flees_predator_in_heading_direction():
    # predator to the +x; policy heads +x (bin 0) straight into it
    obs = _blank_obs(enemies=[_enemy(0.4, 0.0, split_threat=0.8)])
    act = CanonicalAction(heading_bin=0, split=False)
    out = apply_rails(obs, act)
    fx, fy = bin_to_vector(out.heading_bin)
    assert fx < 0  # now heading away (−x)


def test_never_splits_next_to_predator():
    obs = _blank_obs(enemies=[_enemy(0.0, 0.4, can_eat_me=1.0)])
    # policy heads away (−y) but wants to split — the split must be vetoed
    act = CanonicalAction(heading_bin=_bin_toward(0.0, -1.0), split=True)
    out = apply_rails(obs, act)
    assert out.split is False
    assert out.heading_bin == act.heading_bin  # heading was already safe


def test_does_not_flee_when_heading_away_from_predator():
    # predator +x, policy already heading −x → keep heading (only split vetoed)
    obs = _blank_obs(enemies=[_enemy(0.5, 0.0, split_threat=0.9)])
    away = _bin_toward(-1.0, 0.0)
    act = CanonicalAction(heading_bin=away, split=False)
    out = apply_rails(obs, act)
    assert out.heading_bin == away


def test_deflects_from_dangerous_virus():
    # we are poppable; a dangerous virus dead ahead (+x), close
    obs = _blank_obs(viruses=[_virus(0.3, 0.0, dist=0.3, dangerous=1.0)])
    act = CanonicalAction(heading_bin=0, split=False)
    out = apply_rails(obs, act)
    # heading should no longer point straight at the virus
    hx, hy = bin_to_vector(out.heading_bin)
    assert hx < 0.99


def test_ignores_far_or_safe_virus():
    # dangerous but far
    obs = _blank_obs(viruses=[_virus(0.9, 0.0, dist=0.9, dangerous=1.0)])
    act = CanonicalAction(heading_bin=0, split=False)
    assert apply_rails(obs, act) is act
    # near but not dangerous (we're small)
    obs2 = _blank_obs(viruses=[_virus(0.3, 0.0, dist=0.3, dangerous=0.0)])
    assert apply_rails(obs2, act) is act


def test_railed_make_bot_seam(tmp_path):
    from train.opponents.frozen import RailedPolicyBot
    from train.opponents.scripted import make_bot

    fx = "tests/fixtures/adversarial/weights_random.npz"
    bot = make_bot(f"railed:{fx}")
    assert isinstance(bot, RailedPolicyBot)
    assert bot.name == "railed:weights_random"

    # it still acts on a real query (rail is a no-op most frames)
    from train.env.wrapper import WrapperEnv
    env = WrapperEnv(seed=2, emit_update=False)
    obs = env.reset()
    pid = next(iter(obs))
    move = bot.act(obs[pid])
    assert move.player_id == pid
