"""Potential-field controller (bot/heuristic.py) — behavior + safety.

Deterministic, training-free bot: each force (food/prey/threat/virus/wall) and
the split-to-kill gate is checked in isolation with hand-built single-frame
queries, plus a real-game smoke run and the ban-safety validator check the
existing scripted bots must also pass.
"""

from __future__ import annotations

import math

import pytest

from bot import spec
from bot.heuristic import HeuristicParams, heuristic_action
from engine.interface.io.input_validator import MoveValidator
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.blob_model import BlobModel, VisibleBlobModel
from lib.models.food_model import FoodModel
from lib.models.player_model import PlayerModel
from lib.models.virus_model import VirusModel

from train.opponents.scripted import PotentialFieldBot


def mk_query(*, own=((0, 30.0, 30.0, 1.0, 0),), enemies=(), foods=(), viruses=(), pid=0):
    """own: (blob_id,x,y,r,cd); enemies: (pid,blob_id,x,y,r); foods: (id,x,y);
    viruses: (id,x,y,r)."""
    blobs = tuple(BlobModel(blob_id=b[0], pos=(b[1], b[2]), radius=b[3],
                            merge_cooldown=b[4]) for b in own)
    masses = [b.radius ** 2 for b in blobs]
    tm = sum(masses) or 1.0
    cx = sum(b.pos[0] * m for b, m in zip(blobs, masses)) / tm
    cy = sum(b.pos[1] * m for b, m in zip(blobs, masses)) / tm
    you = PlayerModel(player_id=pid, team_id=pid, pos=(cx, cy),
                      radius=math.sqrt(tm), alive=True, blobs=blobs)
    return QueryMovePlayer(
        update={}, round=0, rankings=[pid], you=you, view_center=(cx, cy),
        vision_size=20.0,
        visible_food=[FoodModel(food_id=f[0], pos=(f[1], f[2])) for f in foods],
        visible_blobs=[VisibleBlobModel(player_id=e[0], team_id=e[0], blob_id=e[1],
                                        pos=(e[2], e[3]), radius=e[4]) for e in enemies],
        visible_viruses=[VirusModel(virus_id=v[0], pos=(v[1], v[2]), radius=v[3])
                         for v in viruses],
    )


def _dir(q, params=HeuristicParams()):
    dx, dy, split = heuristic_action(q, params)
    n = math.hypot(dx, dy)
    return (dx / n, dy / n, split) if n > 1e-9 else (0.0, 0.0, split)


def test_seeks_nearest_food():
    dx, dy, split = _dir(mk_query(foods=((0, 40.0, 30.0),)))  # food to the +x
    assert dx > 0.9 and abs(dy) < 0.1 and not split


def test_flees_a_bigger_threat():
    # enemy at +x big enough to eat us (mass ratio 1.2): our mass 1, need >1.2
    dx, dy, _ = _dir(mk_query(own=((0, 30.0, 30.0, 1.0, 0),),
                              enemies=((1, 0, 34.0, 30.0, 2.0),)))
    assert dx < -0.8  # flee in -x


def test_chases_prey_it_can_eat():
    # we are big (r=3 -> mass 9), enemy small (r=1 -> mass 1); 9 >= 1.2*1
    dx, dy, _ = _dir(mk_query(own=((0, 30.0, 30.0, 3.0, 0),),
                              enemies=((1, 0, 34.0, 30.0, 1.0),)))
    assert dx > 0.8  # pursue in +x


def test_threat_dominates_food_when_close():
    # food ahead (+x), lethal threat also ahead and close -> net flee (-x)
    q = mk_query(own=((0, 30.0, 30.0, 1.0, 0),),
                 foods=((0, 33.0, 30.0),),
                 enemies=((1, 0, 33.5, 30.0, 3.0),))
    dx, dy, _ = _dir(q)
    assert dx < 0  # survival beats the pellet


def test_virus_inert_until_poppable():
    # small: mass 1 < 2.7 -> virus is inert (cannot eat, will not pop), no steer
    small = mk_query(own=((0, 30.0, 30.0, 1.0, 0),),
                     viruses=((0, 33.0, 30.0, spec.VIRUS_SIZE),))
    dx2, dy2, _ = heuristic_action(small)
    assert not (dx2 < -0.5 and abs(dy2) < 0.5)  # neither strongly toward nor away


def test_eats_virus_when_safe_and_small_enough():
    # poppable (mass 4), no eater in range, below the eat cap -> attract & eat (+x)
    q = mk_query(own=((0, 30.0, 30.0, 2.0, 0),),
                 viruses=((0, 33.0, 30.0, spec.VIRUS_SIZE),))
    dx, _, _ = _dir(q)
    assert dx > 0.5  # steer *toward* the +x virus for the mass


def test_avoids_virus_when_an_eater_is_in_range():
    # same size, but an eater directly below (adds only a +y push) is in range ->
    # popping would scatter us next to a predator, so avoid the +x virus (-x).
    q = mk_query(own=((0, 30.0, 30.0, 2.0, 0),),
                 viruses=((0, 33.0, 30.0, spec.VIRUS_SIZE),),
                 enemies=((1, 0, 30.0, 24.0, 3.0),))
    dx, _, _ = _dir(q)
    assert dx < 0  # steer away from the virus


def test_avoids_virus_when_too_big():
    # above the eat cap (mass 12.25 > 12): the fixed +2.25 is not worth scattering
    # a big blob, so avoid even with no eater around.
    q = mk_query(own=((0, 30.0, 30.0, 3.5, 0),),
                 viruses=((0, 34.0, 30.0, spec.VIRUS_SIZE),))
    dx, _, _ = _dir(q)
    assert dx < 0  # steer away from the +x virus


def test_split_to_kill_fires_when_child_still_eats_and_safe():
    # big single blob (r=3, mass 9), prey (r=1, mass1) just outside body but in
    # split reach; child mass 4.5 >= 1.2*1. No other enemies -> safe.
    q = mk_query(own=((0, 30.0, 30.0, 3.0, 0),),
                 enemies=((1, 0, 30.0 + 3.0 + 4.0, 30.0, 1.0),))
    _, _, split = _dir(q)
    assert split is True


def test_no_split_into_danger():
    # same split-kill prey, but a bigger enemy nearby could eat a child
    # (child mass 4.5; enemy mass 9 >= 1.2*4.5) -> safety veto
    q = mk_query(own=((0, 30.0, 30.0, 3.0, 0),),
                 enemies=((1, 0, 30.0 + 3.0 + 4.0, 30.0, 1.0),
                          (2, 0, 30.0, 34.0, 3.0)))
    _, _, split = _dir(q)
    assert split is False


def test_no_split_when_child_too_small():
    # blob mass just above 2*SPLIT_MIN_MASS won't be split toward equal-ish prey
    # if a half can't out-mass it by the ratio
    q = mk_query(own=((0, 30.0, 30.0, 2.1, 0),),  # mass 4.41, child 2.205
                 enemies=((1, 0, 30.0 + 2.1 + 4.0, 30.0, 1.4),))  # prey mass 1.96
    # child 2.205 vs 1.2*1.96=2.35 -> child does NOT out-mass -> no split
    _, _, split = _dir(q)
    assert split is False


# --- corner-kill: relax the anti-corner push to finish a safe cornered kill ---

def _corner_scene(extra_enemies=()):
    """Eater at the top-left wall-margin ring; a small prey pinned in the corner
    (its center clamped ~1 unit off each wall). Add enemies via ``extra_enemies``
    as ``(pid, blob_id, x, y, r)`` tuples."""
    return mk_query(own=((0, 5.5, 54.5, 1.2, 0),),
                    enemies=((4, 0, 1.5, 58.0, 1.0),) + tuple(extra_enemies))


def test_dives_for_cornered_prey_when_safe():
    # no threat in view: the anti-corner push is relaxed so prey-pull wins and we
    # commit up-left into the corner toward the prey (dx<0, dy>0).
    dx, dy, _ = _dir(_corner_scene())
    assert dx < -0.4 and dy > 0.4


def test_original_wall_term_stalls_at_the_corner():
    # gate off (safe_wall_scale=1.0 — the shipped-v1 behavior): the anti-corner
    # push cancels prey-pull and the net heading points *away* from the cornered
    # prey. This is the stable local minimum (the "jitter in place" dither) the
    # relaxation removes.
    dx, dy, _ = _dir(_corner_scene(), HeuristicParams(safe_wall_scale=1.0))
    assert dy < 0.0  # not toward the up-corner prey -> would never close the eat


def test_keeps_wall_guard_when_threat_near_corner():
    # a predator within strike clearance (directly above) -> guard holds, we do
    # NOT dive; survival (flee down, away from the eater) dominates. Keeps the
    # "a corner is a death trap under threat" behavior intact.
    dx, dy, _ = _dir(_corner_scene(extra_enemies=((9, 0, 5.5, 59.0, 3.0),)))
    assert dy < 0.0


def test_cornered_prey_is_actually_eaten_over_time():
    # end-to-end: step the eater under the engine movement model against a pinned
    # prey; the fix must close the center-distance below the eat threshold
    # (prey center within the eater's radius) within a handful of ticks.
    ex, ey, er = 5.5, 54.5, 1.2
    px, py, pr = 1.5, 58.0, 1.0  # pinned in the corner
    eaten = False
    for _ in range(40):
        d = math.hypot(ex - px, ey - py)
        if er * er >= pr * pr * spec.EAT_SIZE_RATIO and d <= er:
            eaten = True
            break
        q = mk_query(own=((0, ex, ey, er, 0),), enemies=((4, 0, px, py, pr),))
        dx, dy, _ = heuristic_action(q)
        n = math.hypot(dx, dy) or 1e-9
        s = spec.movement_speed(er)
        ex = min(max(er, ex + dx / n * s), spec.ARENA_SIZE - er)
        ey = min(max(er, ey + dy / n * s), spec.ARENA_SIZE - er)
    assert eaten, f"never closed to eat range (final center-distance {d:.2f} > {er})"


def test_eater_in_range_predicate():
    from bot.heuristic import _eater_in_range

    # a nearby bigger enemy that can eat us -> in range
    assert _eater_in_range(mk_query(own=((0, 30.0, 30.0, 1.0, 0),),
                                    enemies=((1, 0, 33.0, 30.0, 2.0),)))
    # only smaller prey -> not in range
    assert not _eater_in_range(mk_query(own=((0, 30.0, 30.0, 3.0, 0),),
                                        enemies=((1, 0, 33.0, 30.0, 1.0),)))
    # nothing but food -> not in range
    assert not _eater_in_range(mk_query(own=((0, 30.0, 30.0, 1.0, 0),),
                                        foods=((0, 33.0, 30.0),)))
    # an eater too far to strike (non-splitting, beyond reach + pad) -> not in range
    assert not _eater_in_range(mk_query(own=((0, 30.0, 30.0, 1.0, 0),),
                                        enemies=((1, 0, 42.0, 30.0, 1.4),)))


def test_wall_push_gated_by_eater_in_range():
    # bot hugging the left wall (x=3 < margin), nothing else pulling. With no eater
    # in range the anti-corner push is all but off (safe_wall_scale); an eater in
    # range (placed on the y-axis so it adds only a y-push) restores full strength.
    fx_safe, _, _ = heuristic_action(mk_query(own=((0, 3.0, 30.0, 1.0, 0),)))
    fx_danger, _, _ = heuristic_action(
        mk_query(own=((0, 3.0, 30.0, 1.0, 0),),
                 enemies=((1, 0, 3.0, 24.0, 3.0),)))  # eater directly below
    assert fx_danger > fx_safe > 0  # both push inward (+x), danger much harder
    assert fx_safe == pytest.approx(0.15 * fx_danger, rel=1e-3)  # exactly the scale


def test_moves_are_ban_safe():
    """Every emitted move validates through the engine's MoveValidator (the
    same ban-safety bar the other scripted bots meet — same pattern as
    tests/test_opponents.py)."""
    from lib.interface.events.moves.move_player import MovePlayer
    from train.env.wrapper import WrapperEnv

    bot = PotentialFieldBot()
    env = WrapperEnv(seed=11, emit_update=False)
    obs = env.reset()
    validator = MoveValidator(env.state)
    for _ in range(40):
        moves = {}
        for pid, q in obs.items():
            mv = bot.act(q)
            parsed = MovePlayer.model_validate_json(mv.model_dump_json())
            validator.validate(parsed, q, pid)  # raises on any invalid move
            moves[pid] = mv
        obs, done = env.step(moves)
        if done:
            break


def test_deterministic_and_stateless():
    q = mk_query(foods=((0, 40.0, 30.0),), enemies=((1, 0, 20.0, 30.0, 5.0),))
    a = heuristic_action(q)
    b = heuristic_action(q)
    assert a == b


# --- two-phase controller (FEED -> HUNT latch) -------------------------------

def _q_round(rnd, **kw):
    q = mk_query(**kw)
    return q.model_copy(update={"round": rnd})


def test_feed_seeks_food_and_flees_eaters():
    from bot.heuristic import greedy_feed_action

    # small bot, food +x, no threat -> toward food, no split
    dx, dy, split = greedy_feed_action(mk_query(own=((0, 30.0, 30.0, 0.9, 0),),
                                                foods=((0, 40.0, 30.0),)))
    assert dx > 0 and not split
    # an eater close by -> flee it (‑x), ignore the food behind it
    dx, dy, _ = greedy_feed_action(mk_query(own=((0, 30.0, 30.0, 0.9, 0),),
                                            enemies=((1, 0, 32.0, 30.0, 2.0),),
                                            foods=((0, 40.0, 30.0),)))
    assert dx < 0


def test_controller_feeds_early_then_latches_to_hunt():
    from bot.heuristic import HeuristicController

    c = HeuristicController()
    big = ((0, 30.0, 30.0, 3.0, 0),)  # can eat a small enemy
    prey = ((1, 0, 34.0, 30.0, 1.0),)

    # round < 100: never hunts, even with eatable prey in view
    c.act(_q_round(50, own=big, enemies=prey))
    assert c._hunting is False

    # round >= 100 with eatable prey: latch to HUNT
    c.act(_q_round(120, own=big, enemies=prey))
    assert c._hunting is True

    # never reverts, even with no prey and back to small
    c.act(_q_round(130, own=((0, 30.0, 30.0, 0.9, 0),), foods=((0, 20.0, 30.0),)))
    assert c._hunting is True

    # reset returns to FEED (fresh game)
    c.reset()
    assert c._hunting is False


def test_controller_stays_in_feed_without_prey_after_threshold():
    from bot.heuristic import HeuristicController

    c = HeuristicController()
    # round > 100 but no eatable enemy (only a bigger one) -> still feeding
    c.act(_q_round(200, own=((0, 30.0, 30.0, 1.0, 0),),
                   enemies=((1, 0, 34.0, 30.0, 3.0),)))
    assert c._hunting is False


# --- virus detour (FEED -> pop a virus -> HUNT) ------------------------------

_VIRUS = spec.VIRUS_SIZE  # stock virus radius (1.5); a blob eats it above mass 2.7


def test_virus_seek_beelines_when_big_enough_and_early():
    from bot.heuristic import HeuristicController

    c = HeuristicController()
    # mass 4 (> 2.7) so we can eat a virus; one in view at +x; early, no threat
    dx, dy, split = c.act(_q_round(50, own=((0, 30.0, 30.0, 2.0, 0),),
                                   viruses=((0, 36.0, 30.0, _VIRUS),)))
    assert c._virus_seeking is True
    assert dx > 0 and abs(dy) < 1e-6 and not split  # straight at the virus


def test_no_virus_seek_while_too_small():
    from bot.heuristic import HeuristicController

    c = HeuristicController()
    # mass 1 (< 2.7): cannot eat the virus, so no detour
    c.act(_q_round(50, own=((0, 30.0, 30.0, 1.0, 0),),
                   viruses=((0, 36.0, 30.0, _VIRUS),), foods=((0, 20.0, 30.0),)))
    assert c._virus_seeking is False and c._hunting is False


def test_no_virus_seek_after_cutoff_round():
    from bot.heuristic import HeuristicController

    c = HeuristicController()
    # big enough and a virus in view, but past the cutoff -> no detour
    c.act(_q_round(450, own=((0, 30.0, 30.0, 2.0, 0),),
                   viruses=((0, 36.0, 30.0, _VIRUS),)))
    assert c._virus_seeking is False


def test_no_virus_detour_when_eater_in_range():
    from bot.heuristic import HeuristicController

    c = HeuristicController()
    # big enough + a virus in view, but an eater is within range -> gate the
    # detour off (the pop would scatter us right next to a predator); flee instead.
    dx, dy, _ = c.act(_q_round(50, own=((0, 30.0, 30.0, 2.0, 0),),
                               viruses=((0, 36.0, 30.0, _VIRUS),),
                               enemies=((1, 0, 33.0, 30.0, 3.0),)))
    assert c._virus_seeking is False
    assert dx < 0  # flee the predator, not toward the virus


def test_virus_pop_hands_off_to_hunt():
    from bot.heuristic import HeuristicController

    c = HeuristicController()
    # enter the detour...
    c.act(_q_round(50, own=((0, 30.0, 30.0, 2.0, 0),),
                   viruses=((0, 36.0, 30.0, _VIRUS),)))
    assert c._virus_seeking is True
    # ...next frame the virus has popped us into >1 blob -> latch HUNT, never revert
    c.act(_q_round(51, own=((0, 30.0, 30.0, 0.9, 0), (1, 31.0, 30.0, 0.9, 0))))
    assert c._hunting is True and c._virus_seeking is False
    c.act(_q_round(52, own=((0, 30.0, 30.0, 0.9, 0),)))  # back to one blob
    assert c._hunting is True  # does not fall back to FEED/detour


# --- momentum low-pass (jitter damping) --------------------------------------

def test_momentum_blends_previous_heading():
    from bot.heuristic import HeuristicController

    p = HeuristicParams(momentum=0.5)
    c = HeuristicController(p)
    c._hunting = True  # exercise the HUNT heading directly
    qa = mk_query(own=((0, 30.0, 30.0, 3.0, 0),), foods=((0, 40.0, 30.0),))  # +x pull
    qb = mk_query(own=((0, 30.0, 30.0, 3.0, 0),), foods=((0, 30.0, 40.0),))  # +y pull

    ax, ay, _ = c.act(qa)
    # first tick: no prior movement, so the output is the raw policy heading
    rax, ray, _ = heuristic_action(qa, p)
    assert (ax, ay) == pytest.approx((rax, ray))

    # second tick: output = raw policy(B) + momentum * previous output
    bx, by, _ = c.act(qb)
    rbx, rby, _ = heuristic_action(qb, p)
    assert bx == pytest.approx(rbx + 0.5 * ax)
    assert by == pytest.approx(rby + 0.5 * ay)


def test_reset_clears_momentum():
    from bot.heuristic import HeuristicController

    c = HeuristicController(HeuristicParams(momentum=0.5))
    c._hunting = True
    c.act(mk_query(own=((0, 30.0, 30.0, 3.0, 0),), foods=((0, 40.0, 30.0),)))
    assert c._prev_move != (0.0, 0.0)
    c.reset()
    assert c._prev_move == (0.0, 0.0)
