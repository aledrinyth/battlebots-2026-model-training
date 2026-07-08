"""Behavioral unit-scenarios for the greedy submission bot (``greedy 4``).

Each test builds a synthetic ``game`` (duck-typed ``SimpleNamespace`` state, the
same shape ``choose_move`` reads) and asserts on the returned ``(dx, dy, split)``.
Where a test targets one of the three behavior upgrades it also runs the frozen
unmodified ``greedy4_baseline.py`` on the *identical* scene and asserts the old
bot behaves differently — pinning the improvement, not just the new behavior.

Run: ``.venv-bot/bin/python -m pytest tests/test_greedy_bot.py -q``
"""

from __future__ import annotations

import importlib.util
import math
import types
from pathlib import Path

import pytest

BOTS = Path(__file__).parent / "fixtures" / "bots"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


greedy = _load("greedy_new", BOTS / "greedy 4.py")
baseline = _load("greedy_old", BOTS / "greedy4_baseline.py")


# --- scene builders ----------------------------------------------------------

def blob(player_id, x, y, radius, blob_id=0, merge_cooldown=0, team_id=None):
    return types.SimpleNamespace(
        player_id=player_id,
        team_id=player_id if team_id is None else team_id,
        blob_id=blob_id,
        pos=(x, y),
        radius=radius,
        merge_cooldown=merge_cooldown,
    )


def virus(x, y, radius, virus_id=0):
    return types.SimpleNamespace(virus_id=virus_id, pos=(x, y), radius=radius)


def food(x, y, food_id=0):
    return types.SimpleNamespace(food_id=food_id, pos=(x, y))


def make_game(me_x, me_y, me_radius, *, blobs=(), viruses=(), foods=(),
              rnd=0, my_blobs=None, player_id=0):
    if my_blobs is None:
        my_blobs = {0: types.SimpleNamespace(pos=(me_x, me_y), radius=me_radius,
                                             blob_id=0, merge_cooldown=0)}
    me = types.SimpleNamespace(x=me_x, y=me_y, radius=me_radius,
                               player_id=player_id, blobs=my_blobs)
    state = types.SimpleNamespace(
        me=me,
        visible_blobs=list(blobs),
        visible_viruses=list(viruses),
        visible_food=list(foods),
        rankings=[],
        round=rnd,
    )
    return types.SimpleNamespace(state=state)


def mass_to_r(m):
    return math.sqrt(m)


# --- 1. value-based targeting ------------------------------------------------

def test_targeting_prefers_bigger_boost_over_nearest():
    """A single mass-20 blob (farther) should beat two mass-2.5 blobs (nearer)."""
    me_mass = 30.0
    # Big juicy prey to the RIGHT, farther (dist 15).
    big = blob(player_id=1, x=45.0, y=30.0, radius=mass_to_r(20.0))
    # Two small blobs ABOVE, near (dist ~5) — a fragmented owner.
    small1 = blob(player_id=2, x=30.0, y=35.0, radius=mass_to_r(2.5), blob_id=0)
    small2 = blob(player_id=2, x=30.5, y=35.2, radius=mass_to_r(2.5), blob_id=1)
    g = make_game(30.0, 30.0, mass_to_r(me_mass), blobs=[big, small1, small2])

    dx, dy, _ = greedy.choose_move(g)
    # New bot heads RIGHT (toward the big blob): dominant +x.
    assert dx > 0 and abs(dx) > abs(dy), f"new bot should chase big blob, got ({dx},{dy})"

    # The old bot chose purely nearest → heads UP toward the small pair.
    bdx, bdy, _ = baseline.choose_move(g)
    assert bdy > 0 and abs(bdy) > abs(bdx), f"baseline should chase nearest, got ({bdx},{bdy})"


def test_targeting_still_takes_close_big_target():
    """Sanity: when the big blob is also nearest, both bots agree on it."""
    big = blob(player_id=1, x=38.0, y=30.0, radius=mass_to_r(20.0))
    g = make_game(30.0, 30.0, mass_to_r(30.0), blobs=[big])
    dx, dy, _ = greedy.choose_move(g)
    assert dx > 0 and abs(dx) > abs(dy)


# --- 2. virus baiting --------------------------------------------------------

def test_bait_bends_flee_toward_poppable_virus():
    """A heavy pursuer + an aligned virus we can't pop → flee bends to the virus."""
    me_mass = 2.0  # below virus pop threshold (~2.7) → safe to hug the virus
    enemy = blob(player_id=1, x=20.0, y=30.0, radius=mass_to_r(30.0))  # to our LEFT
    v = virus(40.0, 45.0, 1.5)  # up-and-right: aligned with flee (+x) but off-axis (+y)
    g = make_game(30.0, 30.0, mass_to_r(me_mass), blobs=[enemy], viruses=[v])

    dx, dy, split = greedy.choose_move(g)
    assert not split
    # Baited heading carries a clear +y (toward the virus), unlike a pure +x flee.
    assert dx > 0 and dy > 0, f"new bot should bait toward virus, got ({dx},{dy})"

    # Baseline just flees straight away from the enemy: +x, ~zero y.
    bdx, bdy, _ = baseline.choose_move(g)
    assert bdx > 0 and abs(bdy) < 1e-6, f"baseline should flee straight, got ({bdx},{bdy})"


def test_no_bait_when_we_would_pop_the_virus_too():
    """If we're big enough to be popped by the virus, we must NOT bait into it."""
    me_mass = 10.0  # > pop threshold ~2.7 → the virus would shatter us as well
    enemy = blob(player_id=1, x=20.0, y=30.0, radius=mass_to_r(60.0))
    v = virus(40.0, 45.0, 1.5)
    g = make_game(30.0, 30.0, mass_to_r(me_mass), blobs=[enemy], viruses=[v])
    dx, dy, _ = greedy.choose_move(g)
    # Should flee away from the enemy (+x) and steer AWAY from (not into) the virus,
    # so no upward pull toward it.
    assert dx > 0
    assert dy <= 1e-9, f"must not bait when self-poppable, got dy={dy}"


# --- 3. wall / corner escape -------------------------------------------------

def test_wall_escape_slides_along_wall_not_into_corner():
    """Cornered bottom-left with the enemy in the open → slide out, don't pin."""
    enemy = blob(player_id=1, x=12.0, y=12.0, radius=mass_to_r(40.0))  # up-right, open
    g = make_game(1.5, 1.5, 1.0, blobs=[enemy])  # jammed into the bottom-left corner

    dx, dy, _ = greedy.choose_move(g)
    # New bot must not push further into either wall (no negative components).
    assert dx >= -1e-9 and dy >= -1e-9, f"should not dive into corner, got ({dx},{dy})"
    assert dx * dx + dy * dy > 1e-9, "should still move (not freeze)"

    # Baseline dives straight into the corner (both components negative).
    bdx, bdy, _ = baseline.choose_move(g)
    assert bdx < 0 and bdy < 0, f"baseline should pin into corner, got ({bdx},{bdy})"


def test_wall_escape_noop_in_open_space():
    """Away from walls the flee heading is unchanged by the wall logic."""
    enemy = blob(player_id=1, x=20.0, y=30.0, radius=mass_to_r(40.0))  # left of centre
    g = make_game(30.0, 30.0, 1.0, blobs=[enemy])  # centre of a 60x60 arena
    dx, dy, _ = greedy.choose_move(g)
    assert dx > 0, "should flee right, away from the left-side enemy"


def test_wall_escape_single_wall_slides_tangentially():
    """Pinned against one wall with the enemy straight out → slide along it."""
    # Against the bottom wall (y small); enemy directly above pushing us down.
    enemy = blob(player_id=1, x=30.0, y=12.0, radius=mass_to_r(40.0))
    g = make_game(30.0, 1.5, 1.0, blobs=[enemy])
    dx, dy, _ = greedy.choose_move(g)
    # The into-wall (-y) push is cancelled; we slide horizontally along the wall.
    assert dy >= -1e-9, f"must not push into bottom wall, got dy={dy}"
    assert abs(dx) > 1e-9, "should slide tangentially along the wall"


# --- regression guards -------------------------------------------------------

def test_childhood_never_splits():
    """Below CHILDHOOD_MASS the bot must never emit split=True."""
    prey = blob(player_id=1, x=36.0, y=30.0, radius=mass_to_r(4.0))
    g = make_game(30.0, 30.0, mass_to_r(10.0), blobs=[prey])  # mass 10 < 15
    _, _, split = greedy.choose_move(g)
    assert split is False


def test_always_moves_when_idle():
    """With nothing visible the bot still returns a finite, nonzero heading."""
    g = make_game(30.0, 30.0, 1.0)
    dx, dy, split = greedy.choose_move(g)
    assert math.isfinite(dx) and math.isfinite(dy)
    assert dx * dx + dy * dy > 0.0
    assert split is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
