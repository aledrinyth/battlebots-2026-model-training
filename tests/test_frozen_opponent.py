"""`make_bot('frozen:<npz>')` — R1 meta-clone opponent seam (Track 1).

A trained clone becomes a named opponent for ``evaluate``/``promote`` purely
by prefixing its exported actor-weights path with ``frozen:``; both tools
resolve every opponent through :func:`make_bot`, so this one seam is enough.
These tests pin: torch-free resolution, path handling (relative→repo-root and
absolute), the shared-policy cache, a real acting seat, and the spawn-safe
name round-trip that ``evaluate --workers>1`` depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "adversarial" / "weights_random.npz"
FIXTURE_REL = "tests/fixtures/adversarial/weights_random.npz"


def test_make_bot_frozen_is_torch_free_and_named():
    # A fresh interpreter would be ideal, but at minimum resolving a frozen
    # opponent must not require torch to be importable in this process path.
    from train.opponents.frozen import FrozenPolicyBot
    from train.opponents.scripted import make_bot

    bot = make_bot(f"frozen:{FIXTURE_REL}")
    assert isinstance(bot, FrozenPolicyBot)
    assert bot.name == "frozen:weights_random"
    assert "torch" not in sys.modules  # the numpy inference path only


def test_relative_and_absolute_paths_resolve_same():
    from train.opponents.scripted import make_bot

    rel = make_bot(f"frozen:{FIXTURE_REL}")
    absol = make_bot(f"frozen:{FIXTURE}")
    assert rel.weights_path == absol.weights_path == str(FIXTURE)


def test_shared_policy_cache_across_seats():
    """Two seats of one clone share the immutable NumpyPolicy but keep
    distinct recurrent state (mirrors league_seats weight-sharing)."""
    from train.opponents.scripted import make_bot

    a = make_bot(f"frozen:{FIXTURE_REL}")
    b = make_bot(f"frozen:{FIXTURE_REL}")
    assert a._policy is b._policy
    assert a._hidden is not b._hidden


def test_missing_weights_raise_clearly():
    from train.opponents.scripted import make_bot

    with pytest.raises(FileNotFoundError, match="frozen opponent weights not found"):
        make_bot("frozen:data/clones/does_not_exist.npz")


def test_frozen_seat_acts_and_resets_in_a_real_game():
    from train.env.wrapper import WrapperEnv
    from train.opponents.scripted import make_bot

    bot = make_bot(f"frozen:{FIXTURE_REL}")
    env = WrapperEnv(seed=3, emit_update=False)
    obs = env.reset()
    pid = next(iter(obs))

    m1 = bot.act(obs[pid])
    bot.reset()
    m2 = bot.act(obs[pid])
    # greedy + zeroed hidden ⇒ identical first move after reset
    assert (m1.direction.x, m1.direction.y, m1.split) == (
        m2.direction.x, m2.direction.y, m2.split)


def test_frozen_name_survives_spawn_pickle():
    """evaluate --workers>1 ships opponent names through a spawn Pool as plain
    strings; the frozen name must reconstruct the same bot on the far side."""
    import pickle

    from train.opponents.scripted import make_bot

    name = f"frozen:{FIXTURE_REL}"
    assert pickle.loads(pickle.dumps(name)) == name  # trivially, but explicit
    bot = make_bot(pickle.loads(pickle.dumps(name)))
    assert bot.name == "frozen:weights_random"


def test_plain_names_still_resolve():
    from train.opponents.scripted import make_bot

    assert make_bot("greedy_food").__class__.__name__ == "GreedyFoodBot"
    with pytest.raises(KeyError, match="unknown scripted bot"):
        make_bot("no_such_bot")
