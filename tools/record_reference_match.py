#!/usr/bin/env python3
"""Record a **seeded** reference match through the REAL pipe runner (P1.3).

The competition engine runs as an unseeded ``python -m engine`` subprocess, so
its spawns/turn-order/respawns are non-reproducible. To get a *bit-for-bit*
ground truth the in-process wrapper can be proven equal to, we run the engine's
own ``GameEngine`` **in this (seeded) process** — ``random.seed(SEED)`` before
``GameEngine()`` — while the eight bots run as **real subprocesses over real
named pipes**, exactly as the grader wires them. The only thing bypassed is the
0.1 s/round wall-clock sleep (monkeypatched out) so recording is fast.

Because the wrapper drives the engine's *own* ``StateMutator.commit_round`` and
issues the identical sequence of ``random`` calls (``GameState()`` →
``sample(turn_order)`` → per-round respawn/regen), seeding both identically means
``WrapperEnv.reset(seed)`` + replaying these recorded moves must reproduce every
recorded snapshot exactly. That equality is what ``tests/test_wrapper_parity.py``
asserts.

The emitted **trace** (compact JSON) is subprocess-free to replay: ``{seed,
turn_order, rounds: [{moves: [[player_id, dir, split]...], snaps: {player_id:
{alive, blobs: [[blob_id, x, y, radius, merge_cooldown]...]}}}]}``.

Usage:
    tools/record_reference_match.py --seed 0 --out tests/fixtures/parity/trace_0.json \
        [--rounds N] [--bots 4:tests/fixtures/bots/cautious.py 4:tests/fixtures/bots/dont_move.py]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from signal import SIGKILL


REPO = Path(__file__).resolve().parent.parent
DEFAULT_BOTS = [
    "4:tests/fixtures/bots/cautious.py",
    "4:tests/fixtures/bots/dont_move.py",
]
START_DELAY_SECONDS = 3.0
PIPE_PERMISSIONS = 0o660
DIR_PERMISSIONS = 0o775


def _expand_bot_specs(specs: list[str], num_players: int) -> list[Path]:
    out: list[Path] = []
    for spec in specs:
        count_str, _, path_str = spec.partition(":")
        out.extend([(REPO / path_str).resolve()] * int(count_str))
    if len(out) != num_players:
        raise SystemExit(f"bot counts must sum to {num_players}, got {len(out)}")
    return out


def _setup_workspace(ws: Path, num_players: int) -> None:
    shutil.rmtree(ws, ignore_errors=True)
    (ws / "output").mkdir(parents=True)
    (ws / "input").mkdir()
    for p in range(num_players):
        io_dir = ws / f"submission{p}" / "io"
        io_dir.mkdir(parents=True, mode=DIR_PERMISSIONS)
        os.mkfifo(io_dir / "to_engine.pipe", mode=PIPE_PERMISSIONS)
        os.mkfifo(io_dir / "from_engine.pipe", mode=PIPE_PERMISSIONS)
    (ws / "input" / "catalog.json").write_text(
        json.dumps([{"team_id": i} for i in range(num_players)])
    )


def _start_bots(ws: Path, bot_paths: list[Path]) -> list[subprocess.Popen]:
    env = os.environ.copy()
    env["GAME_ENGINE_CORE_DIRECTORY"] = str(ws)
    procs: list[subprocess.Popen] = []
    for player, script in enumerate(bot_paths):
        root = ws / f"submission{player}"
        with open(root / "io" / "submission.log", "w") as out, open(
            root / "io" / "submission.err", "w"
        ) as err:
            procs.append(
                subprocess.Popen(
                    [sys.executable, str(script)],
                    cwd=root,
                    env=env,
                    stdout=out,
                    stderr=err,
                    text=True,
                )
            )
    return procs


def _direction_repr(direction) -> dict:
    """Serialize a DirectionModel to the exact triple that reconstructs it."""
    return {"x": direction.x, "y": direction.y, "degrees": direction.degrees}


def _parse_history_to_rounds(history, num_players: int) -> list[dict]:
    """Segment the engine's private event history into per-round moves+snapshots."""
    from lib.interface.events.event_player_moved import EventPlayerMoved
    from lib.interface.events.moves.move_player import MovePlayer

    rounds: list[dict] = []
    i = 0
    n = len(history)
    # skip the leading EventGameStarted + initial food/virus spawns (no MovePlayer yet)
    while i < n and not isinstance(history[i], MovePlayer):
        i += 1
    while i < n:
        moves = []
        while i < n and isinstance(history[i], MovePlayer):
            mv = history[i]
            moves.append([mv.player_id, _direction_repr(mv.direction), bool(mv.split)])
            i += 1
        snaps: dict[int, dict] = {}
        while i < n and not isinstance(history[i], MovePlayer):
            e = history[i]
            if isinstance(e, EventPlayerMoved):
                snaps[e.player_id] = {
                    "alive": bool(e.alive),
                    "blobs": [
                        [b.blob_id, b.pos[0], b.pos[1], b.radius, b.merge_cooldown]
                        for b in e.blobs
                    ],
                }
            i += 1
        rounds.append({"moves": moves, "snaps": snaps})
    return rounds


def record(seed: int, out_path: Path, bot_specs: list[str], max_rounds: int | None) -> dict:
    from lib.config.arena import NUM_PLAYERS

    bot_paths = _expand_bot_specs(bot_specs, NUM_PLAYERS)
    ws = Path(tempfile.mkdtemp(prefix=f"agario_ref_seed{seed}_"))
    _setup_workspace(ws, NUM_PLAYERS)

    # The engine (run in *this* process) must read this workspace for pipes+catalog.
    os.environ["GAME_ENGINE_CORE_DIRECTORY"] = str(ws)

    bots = _start_bots(ws, bot_paths)
    try:
        time.sleep(START_DELAY_SECONDS)  # let bots open their pipe ends (3 s alarm)

        import random

        # Import engine modules AFTER the env var is set so CORE_DIRECTORY captures
        # the workspace, then neutralize the per-round sleep and (optionally) cap rounds.
        import engine.game_engine as _ge
        import engine.config.io_config as _io
        import engine.state.game_state as _gs

        _io.CORE_DIRECTORY = str(ws)
        _gs.CORE_DIRECTORY = str(ws)
        _ge.sleep = lambda *a, **k: None  # no wall-clock throttle while recording
        if max_rounds is not None:
            _gs.MAX_ROUNDS = max_rounds  # GameState reads module-global MAX_ROUNDS

        random.seed(seed)
        game = _ge.GameEngine()
        if max_rounds is not None:
            game.state.max_rounds = max_rounds
        game.start()

        rounds = _parse_history_to_rounds(game.state.private_event_history, NUM_PLAYERS)
        trace = {
            "seed": seed,
            "num_players": NUM_PLAYERS,
            "turn_order": list(game.state.turn_order),
            "rankings": game.state.get_rankings(),
            "final_masses": {str(k): v for k, v in game.state.get_final_masses().items()},
            "rounds": rounds,
        }
    finally:
        for proc in bots:
            try:
                os.kill(proc.pid, SIGKILL)
            except ProcessLookupError:
                pass
        for proc in bots:
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        shutil.rmtree(ws, ignore_errors=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(trace, separators=(",", ":")).encode()
    if out_path.suffix == ".gz":
        with gzip.open(out_path, "wb") as f:
            f.write(payload)
    else:
        out_path.write_bytes(payload)
    return trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=None, help="cap rounds (debug); default full game")
    ap.add_argument("--bots", nargs="*", default=DEFAULT_BOTS)
    args = ap.parse_args()
    trace = record(args.seed, args.out, args.bots, args.rounds)
    print(
        f"[record] seed={args.seed} rounds={len(trace['rounds'])} "
        f"rankings={trace['rankings']} -> {args.out}"
    )


if __name__ == "__main__":
    main()
