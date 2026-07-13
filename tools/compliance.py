#!/usr/bin/env python3
"""P4.1 — compliance & budget harness shell.

Runs any bot script(s) through the **real pipe runner** (engine in this
process, bots as real subprocesses over real named pipes — the
`record_reference_match.py` harness) and measures *what the engine would
measure*, reporting two clocks per player:

(a) **bot-window per turn** — engine-side observable closest to
    "query-received → move-written": the wall time from the query being fully
    written to the pipe until the move string is fully read back
    (bot parse + compute + serialize, plus pipe IPC both ways).

(b) **the engine's own verdict** — the per-turn increment to
    ``PlayerConnection._cumulative_time`` (the exact quantity the ban decision
    uses; the same instrumentation as ``bench_env.bench_engine_tax``), the
    final per-player cumulative, and the engine's own ``results.json``
    (``PLAYER_BANNED`` / ``SUCCESS`` — ground truth, produced by the engine's
    unmodified ban path: SIGALRM single-move timeout, cumulative timeout,
    invalid message/move, broken pipe).

Plus: pipe-open wait (counts toward the 8 s clock — ``time_limited`` accrues
it), per-message sizes vs ``MAX_CHARACTERS_READ``, and bot-process RSS sampled
at start / mid / end of the game (no published memory limit exists — leak
guard only).

Flags raised (exit code 1 if any):

* ``BANNED``          — the engine banned this player (any ban type).
* ``CUM_BREACH``      — final ``_cumulative_time`` > margin × 8 s.
* ``CUM_PROJECTED``   — pipe-open + mean-turn-cost × MAX_ROUNDS > margin × 8 s
                        (catches slow-drip bots in short, round-capped runs;
                        a worst-case projection — assumes the player answers
                        every round and turn cost stays at the measured mean,
                        which late-game bigger frames make a *floor*).
* ``SINGLE_MOVE_NEAR``— worst single-turn clock delta > margin × 2 s
                        (near-miss even when the SIGALRM ban didn't fire —
                        the alarm has whole-second granularity).
* ``RSS_LEAK``        — RSS growth over the game > --rss-growth-mb (50 default).
* ``MSG_NEAR``        — longest emitted move > margin × MAX_CHARACTERS_READ.

Thresholds derive from the engine's own ``engine.config.io_config`` constants
and ``lib.config.arena.MAX_ROUNDS`` — never typed in by hand (adaptation
discipline; the 0.75 margin policy mirrors ``notes/benchmarks.md``).

Usage (bot venv):
    tools/compliance.py --bots 1:bot/main.py 7:tests/fixtures/bots/dont_move.py \
        [--games N] [--seed 0] [--rounds R] [--margin 0.75] \
        [--rss-growth-mb 50] [--start-delay 0.0] [--out report.json]

Round-capped runs (``--rounds``) answer in seconds and rely on the projection
flag; the pre-submission gate (P4.4 full) runs full games.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import record_reference_match as rr  # noqa: E402  (path set above)

DEFAULT_MARGIN = 0.75  # 25 % safety on every engine limit (grader-hardware unknown)
DEFAULT_RSS_GROWTH_MB = 50.0  # playbook P4.1 leak-guard line


# --- raw per-game capture ----------------------------------------------------

@dataclass
class TurnRecord:
    round: int
    clock_delta_s: float          # increment to _cumulative_time (ban clock)
    bot_window_s: float | None    # send-complete -> move-read-back
    query_len: int | None
    move_len: int | None


@dataclass
class PlayerCapture:
    script: str
    turns: list[TurnRecord] = field(default_factory=list)
    pipe_open_wait_s: float = 0.0
    spawn_to_open_s: float = float("nan")  # import+init upper bound
    final_cumulative_s: float = 0.0
    rss_kb: dict[str, int] = field(default_factory=dict)  # "start"/"mid"/"end"
    stderr_tail: str = ""  # last few KB of the bot's stderr (ledger, resets)


def _sample_rss_kb(pids: list[int]) -> dict[int, int]:
    """One batched `ps` call for every bot pid (the P3.4 lesson: never per-pid)."""
    out = subprocess.run(
        ["ps", "-o", "pid=,rss=", "-p", ",".join(str(p) for p in pids)],
        capture_output=True, text=True,
    ).stdout
    rss: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            rss[int(parts[0])] = int(parts[1])
    return rss


def _start_bots_timed(
    ws: Path, bot_paths: list[Path]
) -> tuple[list[subprocess.Popen], list[float]]:
    """`rr._start_bots` with a perf_counter stamp per spawn (import+init clock)."""
    env = os.environ.copy()
    env["GAME_ENGINE_CORE_DIRECTORY"] = str(ws)
    procs: list[subprocess.Popen] = []
    spawn_ts: list[float] = []
    for player, script in enumerate(bot_paths):
        root = ws / f"submission{player}"
        with open(root / "io" / "submission.log", "w") as out, open(
            root / "io" / "submission.err", "w"
        ) as err:
            spawn_ts.append(perf_counter())
            procs.append(
                subprocess.Popen(
                    [sys.executable, str(script)],
                    cwd=root, env=env, stdout=out, stderr=err, text=True,
                )
            )
    return procs, spawn_ts


def run_game(
    bot_specs: list[str],
    seed: int = 0,
    rounds: int | None = None,
    start_delay: float = 0.0,
    quiet: bool = True,
) -> dict:
    """One instrumented game. Returns the raw capture (no thresholds applied).

    The engine runs unmodified except: CORE_DIRECTORY pointed at a temp
    workspace, the 0.1 s/round wall-clock sleep neutralized, and (optionally)
    MAX_ROUNDS capped. Instrumentation only *observes* — every patched method
    calls the original, and everything is restored in ``finally`` (the
    bench_env discipline: this process may host wrapper/test code afterwards).
    """
    from lib.config.arena import NUM_PLAYERS

    # The measured clocks must reflect the grader's pristine engine.
    from train.env import fast_vision
    fast_vision.uninstall()

    bot_paths = rr._expand_bot_specs(bot_specs, NUM_PLAYERS)
    ws = Path(tempfile.mkdtemp(prefix=f"agario_compliance_seed{seed}_"))
    rr._setup_workspace(ws, NUM_PLAYERS)
    os.environ["GAME_ENGINE_CORE_DIRECTORY"] = str(ws)

    players: dict[int, PlayerCapture] = {
        pid: PlayerCapture(script=str(bot_paths[pid])) for pid in range(NUM_PLAYERS)
    }

    procs, spawn_ts = _start_bots_timed(ws, bot_paths)
    engine_start_ts = 0.0
    result_json: dict = {}
    try:
        if start_delay > 0:
            time.sleep(start_delay)

        import random

        import engine.config.io_config as _io
        import engine.game_engine as _ge
        import engine.state.game_state as _gs
        import engine.interface.io.player_connection as _pc

        PC = _pc.PlayerConnection
        orig_query = PC.query_move_player
        orig_send = PC._send
        orig_receive = PC._receive
        orig_open = PC._open_pipes
        saved = {
            "io": _io.CORE_DIRECTORY, "gs": _gs.CORE_DIRECTORY,
            "pc": _pc.CORE_DIRECTORY, "ge": _ge.CORE_DIRECTORY,
            "sleep": _ge.sleep, "max_rounds": _gs.MAX_ROUNDS,
        }

        # RSS samples: "start" on the first observed query, "mid" halfway
        # through the effective game, "end" after game.start() returns (works
        # regardless of the engine's round numbering, and on ban-aborted
        # games — the bot processes stay alive until the outer finally).
        effective_rounds = rounds if rounds is not None else _gs.MAX_ROUNDS
        pids = [p.pid for p in procs]
        pid_of_proc = {i: procs[i].pid for i in range(NUM_PLAYERS)}

        def _snap_rss(label: str) -> None:
            snap = _sample_rss_kb(pids)
            for seat, pid in pid_of_proc.items():
                if pid in snap:
                    players[seat].rss_kb[label] = snap[pid]

        rss_state: dict = {"first_round": None, "mid_done": False}

        def _maybe_sample_rss(round_: int) -> None:
            if rss_state["first_round"] is None:
                rss_state["first_round"] = round_
                _snap_rss("start")
            elif (not rss_state["mid_done"]
                  and round_ - rss_state["first_round"] >= effective_rounds // 2):
                rss_state["mid_done"] = True
                _snap_rss("mid")

        def open_patched(self):  # noqa: ANN001 - engine instance
            t0 = perf_counter()
            try:
                orig_open(self)
            finally:
                cap = players[self.player_id]
                cap.pipe_open_wait_s = perf_counter() - t0
                cap.spawn_to_open_s = perf_counter() - spawn_ts[self.player_id]

        def send_patched(self, data):  # noqa: ANN001
            orig_send(self, data)
            self._compliance_query_len = len(data)
            self._compliance_send_done = perf_counter()

        def receive_patched(self):  # noqa: ANN001
            data = orig_receive(self)
            done = getattr(self, "_compliance_send_done", None)
            if done is not None:
                self._compliance_bot_window = perf_counter() - done
                self._compliance_send_done = None
            self._compliance_move_len = len(data)
            return data

        def query_patched(self, state, validator, censor):  # noqa: ANN001
            _maybe_sample_rss(state.round)
            before = self._cumulative_time
            self._compliance_bot_window = None
            self._compliance_move_len = None
            self._compliance_query_len = None
            try:
                return orig_query(self, state, validator, censor)
            finally:
                # Runs on the ban path too: a CumulativeTimeout turn has
                # already accrued; a SIGALRM turn records delta 0 (the +=
                # never ran) — the ban result itself is that turn's evidence.
                players[self.player_id].turns.append(TurnRecord(
                    round=state.round,
                    clock_delta_s=self._cumulative_time - before,
                    bot_window_s=getattr(self, "_compliance_bot_window", None),
                    query_len=getattr(self, "_compliance_query_len", None),
                    move_len=getattr(self, "_compliance_move_len", None),
                ))

        try:
            _io.CORE_DIRECTORY = str(ws)
            _gs.CORE_DIRECTORY = str(ws)
            _pc.CORE_DIRECTORY = str(ws)
            _ge.CORE_DIRECTORY = str(ws)
            _ge.sleep = lambda *a, **k: None
            if rounds is not None:
                _gs.MAX_ROUNDS = rounds

            PC._open_pipes = open_patched
            PC._send = send_patched
            PC._receive = receive_patched
            PC.query_move_player = query_patched

            random.seed(seed)
            engine_start_ts = perf_counter()
            # The engine prints one line per round; swallow it (a 10-game full
            # run would spam 14k lines into CI logs) unless quiet=False.
            import contextlib, io
            sink = io.StringIO() if quiet else None
            with (contextlib.redirect_stdout(sink) if sink
                  else contextlib.nullcontext()):
                game = _ge.GameEngine()
                if rounds is not None:
                    game.state.max_rounds = rounds
                game.start()

            _snap_rss("end")
            for pid, p in game.state.players.items():
                if hasattr(p, "connection"):
                    players[pid].final_cumulative_s = p.connection._cumulative_time
            result_json = json.loads((ws / "output" / "results.json").read_text())
        finally:
            # The engine's time_limited cancels its SIGALRM only on the
            # success and cumulative-breach paths; a ban via BrokenPipe /
            # InvalidMessage would leave a pending alarm armed in *this*
            # process. Disarm defensively.
            from signal import alarm
            alarm(0)
            PC._open_pipes = orig_open
            PC._send = orig_send
            PC._receive = orig_receive
            PC.query_move_player = orig_query
            _io.CORE_DIRECTORY = saved["io"]
            _gs.CORE_DIRECTORY = saved["gs"]
            _pc.CORE_DIRECTORY = saved["pc"]
            _ge.CORE_DIRECTORY = saved["ge"]
            _ge.sleep = saved["sleep"]
            _gs.MAX_ROUNDS = saved["max_rounds"]
    finally:
        from signal import SIGKILL
        for proc in procs:
            try:
                os.kill(proc.pid, SIGKILL)
            except ProcessLookupError:
                pass
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        for pid in players:
            err = ws / f"submission{pid}" / "io" / "submission.err"
            try:
                players[pid].stderr_tail = err.read_text()[-4096:]
            except OSError:
                pass
        shutil.rmtree(ws, ignore_errors=True)

    return {
        "seed": seed,
        "rounds_cap": rounds,
        "engine_wall_s": perf_counter() - engine_start_ts if engine_start_ts else None,
        "result": result_json,
        "players": players,
    }


# --- analysis ----------------------------------------------------------------

def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(math.ceil(p / 100.0 * len(sorted_vals))) - 1)
    return sorted_vals[max(0, idx)]


def analyze(
    raw: dict,
    margin: float = DEFAULT_MARGIN,
    rss_growth_mb: float = DEFAULT_RSS_GROWTH_MB,
) -> dict:
    """Apply thresholds to a raw capture → per-player stats + flags + verdict."""
    from engine.config.io_config import (
        CUMULATIVE_TIMEOUT_SECONDS,
        MAX_CHARACTERS_READ,
        TIMEOUT_SECONDS,
    )
    from lib.config.arena import MAX_ROUNDS

    usable_s = CUMULATIVE_TIMEOUT_SECONDS * margin
    single_move_line_s = TIMEOUT_SECONDS * margin
    msg_line = MAX_CHARACTERS_READ * margin

    result = raw["result"]
    banned_player = result.get("player") if result.get("result_type") == "PLAYER_BANNED" else None

    per_player: dict[int, dict] = {}
    all_flags: list[str] = []
    for pid, cap in raw["players"].items():
        deltas = sorted(t.clock_delta_s for t in cap.turns)
        windows = sorted(t.bot_window_s for t in cap.turns if t.bot_window_s is not None)
        move_lens = [t.move_len for t in cap.turns if t.move_len is not None]
        query_lens = [t.query_len for t in cap.turns if t.query_len is not None]
        mean_delta = statistics.fmean(deltas) if deltas else float("nan")
        projected_s = (
            cap.pipe_open_wait_s + mean_delta * MAX_ROUNDS if deltas else float("nan")
        )
        rss_growth = (
            (cap.rss_kb["end"] - cap.rss_kb["start"]) / 1024.0
            if "end" in cap.rss_kb and "start" in cap.rss_kb else None
        )

        flags: list[str] = []
        if pid == banned_player:
            flags.append("BANNED")
        if cap.final_cumulative_s > usable_s:
            flags.append("CUM_BREACH")
        if deltas and projected_s > usable_s:
            flags.append("CUM_PROJECTED")
        if deltas and deltas[-1] > single_move_line_s:
            flags.append("SINGLE_MOVE_NEAR")
        if rss_growth is not None and rss_growth > rss_growth_mb:
            flags.append("RSS_LEAK")
        if move_lens and max(move_lens) > msg_line:
            flags.append("MSG_NEAR")

        per_player[pid] = {
            "script": cap.script,
            "turns": len(cap.turns),
            "pipe_open_wait_s": cap.pipe_open_wait_s,
            "spawn_to_open_s": cap.spawn_to_open_s,
            "cumulative_s": cap.final_cumulative_s,
            "projected_full_game_s": projected_s,
            "clock_ms": {
                "mean": mean_delta * 1e3,
                "p50": _pct(deltas, 50) * 1e3,
                "p99": _pct(deltas, 99) * 1e3,
                "max": (deltas[-1] if deltas else float("nan")) * 1e3,
            },
            "bot_window_ms": {
                "p50": _pct(windows, 50) * 1e3,
                "p99": _pct(windows, 99) * 1e3,
                "max": (windows[-1] if windows else float("nan")) * 1e3,
            },
            "max_query_len": max(query_lens) if query_lens else None,
            "max_move_len": max(move_lens) if move_lens else None,
            "rss_kb": dict(cap.rss_kb),
            "rss_growth_mb": rss_growth,
            "flags": flags,
        }
        all_flags.extend(flags)

    return {
        "seed": raw["seed"],
        "rounds_cap": raw["rounds_cap"],
        "engine_wall_s": raw["engine_wall_s"],
        "limits": {
            "cumulative_timeout_s": CUMULATIVE_TIMEOUT_SECONDS,
            "single_move_timeout_s": TIMEOUT_SECONDS,
            "max_characters_read": MAX_CHARACTERS_READ,
            "max_rounds": MAX_ROUNDS,
            "margin": margin,
            "usable_cumulative_s": usable_s,
            "rss_growth_limit_mb": rss_growth_mb,
        },
        "engine_result": result,
        "players": per_player,
        "flags": sorted(set(all_flags)),
        "ok": not all_flags,
    }


def run_compliance(
    bot_specs: list[str],
    games: int = 1,
    seed: int = 0,
    rounds: int | None = None,
    margin: float = DEFAULT_MARGIN,
    rss_growth_mb: float = DEFAULT_RSS_GROWTH_MB,
    start_delay: float = 0.0,
) -> dict:
    """Run N seeded games, analyze each, and aggregate the verdict."""
    reports = []
    for g in range(games):
        raw = run_game(bot_specs, seed=seed + g, rounds=rounds, start_delay=start_delay)
        reports.append(analyze(raw, margin=margin, rss_growth_mb=rss_growth_mb))
    return {
        "schema": "compliance-v0",
        "bots": bot_specs,
        "games": reports,
        "flags": sorted({f for r in reports for f in r["flags"]}),
        "ok": all(r["ok"] for r in reports),
    }


# --- rendering ---------------------------------------------------------------

def render(report: dict) -> str:
    lines: list[str] = []
    for i, g in enumerate(report["games"]):
        lim = g["limits"]
        res = g["engine_result"]
        verdict = res.get("result_type", "?")
        if verdict == "PLAYER_BANNED":
            verdict += f" (player {res.get('player')}: {res.get('ban_type')} — {res.get('reason')})"
        lines.append(
            f"game {i} seed={g['seed']} rounds_cap={g['rounds_cap']} "
            f"wall={g['engine_wall_s']:.1f}s → {verdict}"
        )
        lines.append(
            f"  limits: cum {lim['usable_cumulative_s']:.2f}s "
            f"(margin {lim['margin']} × {lim['cumulative_timeout_s']}s), "
            f"single-move near-line {lim['margin'] * lim['single_move_timeout_s']:.2f}s, "
            f"msg {lim['max_characters_read']} chars, "
            f"rss +{lim['rss_growth_limit_mb']:.0f} MB"
        )
        hdr = (f"  {'seat':>4} {'turns':>5} {'open(s)':>8} {'cum(s)':>7} {'proj(s)':>8} "
               f"{'clockΔ p50/p99/max (ms)':>26} {'botwin p50/p99/max (ms)':>26} "
               f"{'mlen':>5} {'rssΔMB':>7}  script / flags")
        lines.append(hdr)
        for pid, p in sorted(g["players"].items()):
            c, w = p["clock_ms"], p["bot_window_ms"]
            rssg = f"{p['rss_growth_mb']:.1f}" if p["rss_growth_mb"] is not None else "-"
            script = Path(p["script"]).name
            flags = (" [" + ",".join(p["flags"]) + "]") if p["flags"] else ""
            lines.append(
                f"  {pid:>4} {p['turns']:>5} {p['pipe_open_wait_s']:>8.3f} "
                f"{p['cumulative_s']:>7.3f} {p['projected_full_game_s']:>8.3f} "
                f"{c['p50']:>8.3f}/{c['p99']:>8.3f}/{c['max']:>7.1f} "
                f"{w['p50']:>8.3f}/{w['p99']:>8.3f}/{w['max']:>7.1f} "
                f"{p['max_move_len'] or 0:>5} {rssg:>7}  {script}{flags}"
            )
    lines.append(f"VERDICT: {'PASS' if report['ok'] else 'FAIL ' + str(report['flags'])}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bots", nargs="+", required=True,
                    help="count:path specs summing to NUM_PLAYERS, e.g. "
                         "'1:bot/main.py 7:tests/fixtures/bots/dont_move.py'")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=None,
                    help="cap rounds (fast, projection-based); omit for full games")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--rss-growth-mb", type=float, default=DEFAULT_RSS_GROWTH_MB)
    ap.add_argument("--start-delay", type=float, default=0.0,
                    help="seconds between bot spawn and engine start; 0 makes "
                         "pipe-open waits measure bot import+init honestly")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = ap.parse_args()

    report = run_compliance(
        args.bots, games=args.games, seed=args.seed, rounds=args.rounds,
        margin=args.margin, rss_growth_mb=args.rss_growth_mb,
        start_delay=args.start_delay,
    )
    print(render(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"[compliance] wrote {args.out}")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
