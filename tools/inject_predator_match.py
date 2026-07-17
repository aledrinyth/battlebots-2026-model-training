#!/usr/bin/env python3
"""Sudden-large-predator stress test in a NATURALLY-AGED arena.

Runs the real seeded GameEngine in-process (like record_reference_match) but
steps the round loop manually. At round T it overwrites the two "killer" seats
with a single LARGE blob each, placed at the vision-edge of a designated target
seat -- simulating rounding a corner into a big opponent that has been feeding
off-screen. The arena is aged by T rounds of real play (food eaten, viruses
consumed, players grown), satisfying representativeness. The killer seats run an
aggressive split-lunge bot so the injected mass actively hunts.

Two targets (seat 0, seat 1) each get their own symmetric predator, so v12 vs a
candidate are compared in the SAME aged arena against IDENTICALLY specced
predators (paired). Reports, per target: did it die in [T, T+K], and its mass at
T+K.

Usage:
  inject_predator_match.py --seed 0 --inject-round 400 --window 150 \
     --pred-mass 40 --offset 15 \
     --target0 dist/potential_field_v12.py --target1 dist/potential_field_v13h.py
"""
from __future__ import annotations
import argparse, json, math, os, sys, tempfile, time, shutil
from pathlib import Path
from signal import SIGKILL

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.record_reference_match import (  # reuse the proven setup
    _setup_workspace, _start_bots, START_DELAY_SECONDS,
)

KILLER = str(REPO / "dist/spar_killer.py")


def _alive(pl):
    return bool(pl.blobs)


def run(seed, inject_round, window, pred_mass, offset,
        target0, target1, rounds_cap):
    from lib.config.arena import NUM_PLAYERS
    # seat layout: 0=target0, 1=target1, 2&3=killers (injected), 4..7=foragers
    foragers = [str(REPO / p) for p in (
        "dist/potential_field_v8.py", "dist/potential_field_v10.py",
        "dist/potential_field_v11.py", "dist/potential_field_v9.py")]
    bot_paths = [Path(target0), Path(target1), Path(KILLER), Path(KILLER)] + \
                [Path(p) for p in foragers]
    assert len(bot_paths) == NUM_PLAYERS, (len(bot_paths), NUM_PLAYERS)

    ws = Path(tempfile.mkdtemp(prefix=f"agario_inj_seed{seed}_"))
    _setup_workspace(ws, NUM_PLAYERS)
    os.environ["GAME_ENGINE_CORE_DIRECTORY"] = str(ws)
    bots = _start_bots(ws, bot_paths)
    pred_r = math.sqrt(pred_mass)
    try:
        time.sleep(START_DELAY_SECONDS)
        import random
        import engine.game_engine as _ge
        import engine.config.io_config as _io
        import engine.state.game_state as _gs
        from engine.state.blob_state import BlobState
        from lib.config.arena import ARENA_SIZE
        _io.CORE_DIRECTORY = str(ws); _gs.CORE_DIRECTORY = str(ws)
        _ge.sleep = lambda *a, **k: None
        _gs.MAX_ROUNDS = rounds_cap

        random.seed(seed)
        game = _ge.GameEngine()
        game.state.max_rounds = rounds_cap
        st = game.state
        # --- replicate GameEngine.run_game(), with injection --------------
        from random import sample
        from lib.interface.events.event_game_started import EventGameStarted
        from lib.interface.events.event_food_spawned import EventFoodSpawned
        from lib.interface.events.event_virus_spawned import EventVirusSpawned
        from lib.interface.events.event_player_won import EventPlayerWon
        from lib.config.arena import VISION_SIZE, TURN_DURATION_SECONDS

        st._connect_players()
        turn_order = sample(list(st.players.keys()), k=NUM_PLAYERS)
        st.turn_order = turn_order
        game.mutator.commit(EventGameStarted(
            turn_order=st.turn_order, arena_size=st.map.size, vision_size=VISION_SIZE,
            turn_duration_seconds=TURN_DURATION_SECONDS, max_rounds=rounds_cap,
            players=[p._to_model() for p in st.players.values()],
            engine_version=game.engine_version))
        game.mutator.commitPrivate(EventFoodSpawned(foods=list(st.map.foods)))
        game.mutator.commitPrivate(EventVirusSpawned(viruses=list(st.map.viruses)))

        injected = False
        died = {0: False, 1: False}
        alive_prev = {0: True, 1: True}
        mass_at_end = {0: 0.0, 1: 0.0}
        while not st.is_game_over():
            st.round += 1
            r = st.round
            # inject at the chosen round, in the aged arena
            if not injected and r >= inject_round:
                injected = True
                for tgt_seat, kill_seat in ((0, 2), (1, 3)):
                    tp = st.players[tgt_seat]
                    if not _alive(tp):
                        continue
                    tx, ty = tp.x, tp.y
                    # place predator at vision-edge, offset toward arena centre
                    cx, cy = ARENA_SIZE / 2, ARENA_SIZE / 2
                    vx, vy = cx - tx, cy - ty
                    n = math.hypot(vx, vy) or 1.0
                    px = min(max(pred_r, tx + vx / n * offset), ARENA_SIZE - pred_r)
                    py = min(max(pred_r, ty + vy / n * offset), ARENA_SIZE - pred_r)
                    kp = st.players[kill_seat]
                    kp.blobs = {0: BlobState(blob_id=0, x=px, y=py, radius=pred_r,
                                             merge_cooldown=0)}
                    kp._next_blob_id = 1
            moves = []
            for pid in turn_order:
                pl = st.players[pid]
                if pl.alive:
                    try:  # a slow/dead bot must not abort the whole game
                        moves.append(pl.connection.query_move_player(
                            st, game.validator, game.censor))
                    except Exception:
                        continue
            game.mutator.commit_round(moves)
            # observe targets during the window
            if injected:
                for s in (0, 1):
                    al = st.players[s].alive
                    if alive_prev[s] and not al and r <= inject_round + window:
                        died[s] = True
                    alive_prev[s] = al
            if r >= inject_round + window:
                for s in (0, 1):
                    pl = st.players[s]
                    mass_at_end[s] = sum(b.mass for b in pl.blobs.values()) if pl.alive else 0.0
                break  # metrics only need the observation window
        return {"died0": died[0], "died1": died[1],
                "mass0": mass_at_end[0], "mass1": mass_at_end[1]}
    finally:
        for proc in bots:
            try: os.kill(proc.pid, SIGKILL)
            except ProcessLookupError: pass
        for proc in bots:
            try: proc.wait(timeout=5)
            except Exception: pass
        shutil.rmtree(ws, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--inject-round", type=int, default=400)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--pred-mass", type=float, default=40.0)
    ap.add_argument("--offset", type=float, default=15.0)
    ap.add_argument("--target0", default="dist/potential_field_v12.py")
    ap.add_argument("--target1", default="dist/potential_field_v13h.py")
    ap.add_argument("--rounds-cap", type=int, default=1400)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = run(a.seed, a.inject_round, a.window, a.pred_mass, a.offset,
              str((REPO / a.target0).resolve()), str((REPO / a.target1).resolve()),
              a.rounds_cap)
    print(json.dumps(res))
    if a.out:
        a.out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
