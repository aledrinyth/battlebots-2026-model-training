#!/usr/bin/env python3
"""P1.4 — throughput benchmark + match-clock decomposition.

Two numbers the whole plan rests on:

(a) **Training throughput** — single-core rounds/s for a full 8-player in-process
    game with trivial policies (via `WrapperEnv`), projected across cores. Target
    ≥300 rounds/s/core; <100 ⇒ P1.5 (vectorization) is mandatory before Phase 6.

(b) **Match-clock decomposition** — the engine-side share of the 8 s cumulative
    budget. We run the real pipe runner with a bot that responds *instantly*
    (`dont_move`) and record, per turn, the increment to the engine's own
    `PlayerConnection._cumulative_time` — the exact quantity the ban decision
    uses. For a zero-compute bot this isolates the engine-side tax (pydantic query
    serialization + move parse/validation) plus pipe IPC, i.e. the cost we pay
    every turn *before* the policy does any work. From it we derive the bot-side
    per-turn budget: `(8 s · 0.75 − 1400 · engine_tax) / 1400`.

Writes a Markdown report to `notes/benchmarks.md`.

Usage:
    tools/bench_env.py [--throughput-rounds N] [--tax-rounds N] [--out notes/benchmarks.md]

Run from the repo root (the script dir is added to sys.path so it can reuse the
subprocess harness in `record_reference_match.py`).
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # so `import train...` / `import bot...` resolve
sys.path.insert(0, str(REPO / "tools"))  # so `import record_reference_match` works


# --- (a) in-process throughput ---------------------------------------------

def bench_throughput(rounds: int) -> dict:
    """Single-core rounds/s for a full 8-player game with trivial (stay) moves.

    Reports physics-only (`commit_round`, the ceiling) plus `WrapperEnv.step`
    throughput for both training paths (`emit_update` off/on), measured **twice**:
    once on the pristine engine and once with the P1.5 NumPy-vectorized vision
    override installed. The pristine/fast pair makes the P1.5 acceptance
    (≥3× speedup) self-verifying in a single run."""
    import random

    from train.env import fast_vision
    from train.env.wrapper import WrapperEnv, make_move, _ensure_engine_workspace
    from lib.config.arena import NUM_PLAYERS

    # Materialize the engine catalog workspace before constructing GameState directly.
    _ensure_engine_workspace(NUM_PLAYERS)

    # physics-only ceiling: drive commit_round directly, no observation building.
    from engine.state.game_state import GameState
    from engine.state.state_mutator import StateMutator
    from lib.interface.events.moves.move_player import MovePlayer
    from lib.models.penguin_model import DirectionModel

    def stay_moves_for(state):
        return [
            MovePlayer(player_id=pid, direction=DirectionModel(x=0.0, y=0.0))
            for pid, p in state.players.items()
            if p.alive
        ]

    random.seed(0)
    gs = GameState()
    mut = StateMutator(gs)
    gs.round += 1
    # warm up one round (JIT-free Python, but stabilizes caches)
    mut.commit_round(stay_moves_for(gs))
    gs.round += 1
    t0 = time.perf_counter()
    n = 0
    while n < rounds:
        mut.commit_round(stay_moves_for(gs))
        gs.round += 1
        n += 1
    physics_rps = n / (time.perf_counter() - t0)

    def bench_wrapper(emit_update: bool, use_fast: bool) -> float:
        env = WrapperEnv(seed=0, emit_update=emit_update)
        obs = env.reset()  # reset() installs fast_vision; set the real mode *after*
        # so only the timed loop runs under the chosen vision path (the single
        # untimed round-0 obs above is bit-identical either way).
        if use_fast:
            fast_vision.install()
        else:
            fast_vision.uninstall()
        t0 = time.perf_counter()
        n = 0
        done = False
        while not done and n < rounds:
            moves = {pid: make_move(pid, 0.0, 0.0) for pid in obs}
            obs, done = env.step(moves)
            n += 1
        return n / (time.perf_counter() - t0)

    # Pristine engine (P1.5 override removed) — the pre-optimization baseline.
    base_fast_rps = bench_wrapper(emit_update=False, use_fast=False)
    base_full_rps = bench_wrapper(emit_update=True, use_fast=False)

    # P1.5 NumPy-vectorized vision installed.
    step_fast_rps = bench_wrapper(emit_update=False, use_fast=True)
    step_full_rps = bench_wrapper(emit_update=True, use_fast=True)

    cores = os.cpu_count() or 1
    return {
        "rounds": rounds,
        "cores_detected": cores,
        "physics_only_rps": physics_rps,
        "base_fast_rps": base_fast_rps,   # pristine engine, no update stream
        "base_full_rps": base_full_rps,   # pristine engine, faithful update stream
        "step_fast_rps": step_fast_rps,   # P1.5 vision, no update stream (training path)
        "step_full_rps": step_full_rps,   # P1.5 vision, faithful update stream
        "speedup_fast": step_fast_rps / base_fast_rps if base_fast_rps else float("nan"),
        "speedup_full": step_full_rps / base_full_rps if base_full_rps else float("nan"),
    }


# --- (b) engine-side match-clock tax ---------------------------------------

def bench_engine_tax(tax_rounds: int) -> dict:
    """Per-turn increment to the engine's `_cumulative_time` for an instant bot.

    Runs the engine in-process against real `dont_move` subprocesses over real
    pipes (sleep neutralized), wrapping `PlayerConnection.query_move_player` to
    capture each turn's timed-region delta — the same seconds that accrue toward
    the 8 s ban clock."""
    import record_reference_match as rr  # reuses the seeded subprocess harness

    from lib.config.arena import NUM_PLAYERS

    # The engine tax must reflect the *grader's* pristine engine (we don't control
    # its copy), so ensure the P1.5 vision override is not active for this
    # measurement even if a WrapperEnv ran earlier in the same process.
    from train.env import fast_vision
    fast_vision.uninstall()

    bot_paths = rr._expand_bot_specs(["8:tests/fixtures/bots/dont_move.py"], NUM_PLAYERS)
    import tempfile

    ws = Path(tempfile.mkdtemp(prefix="agario_bench_tax_"))
    rr._setup_workspace(ws, NUM_PLAYERS)
    os.environ["GAME_ENGINE_CORE_DIRECTORY"] = str(ws)

    bots = rr._start_bots(ws, bot_paths)
    deltas: list[float] = []
    try:
        time.sleep(rr.START_DELAY_SECONDS)

        import random

        import engine.config.io_config as _io
        import engine.game_engine as _ge
        import engine.state.game_state as _gs
        import engine.interface.io.player_connection as _pc

        PC = _pc.PlayerConnection
        # This function mutates process-global engine state; it runs in-process
        # alongside other users (the wrapper, the test suite). Save EVERYTHING we
        # touch and restore it in the finally so we never contaminate later code.
        original = PC.query_move_player
        saved = {
            "io": _io.CORE_DIRECTORY,
            "gs": _gs.CORE_DIRECTORY,
            "pc": _pc.CORE_DIRECTORY,
            "ge": _ge.CORE_DIRECTORY,
            "sleep": _ge.sleep,
            "max_rounds": _gs.MAX_ROUNDS,
        }
        try:
            # Point every CORE_DIRECTORY binding at the pipe workspace and kill
            # sleep. (Each module did `from io_config import CORE_DIRECTORY`,
            # capturing its own copy — patch all of them.)
            _io.CORE_DIRECTORY = str(ws)
            _gs.CORE_DIRECTORY = str(ws)
            _pc.CORE_DIRECTORY = str(ws)
            _ge.CORE_DIRECTORY = str(ws)
            _ge.sleep = lambda *a, **k: None
            _gs.MAX_ROUNDS = tax_rounds

            # Instrument: record the per-turn increase in _cumulative_time.
            def timed(self, state, validator, censor):
                before = self._cumulative_time
                result = original(self, state, validator, censor)
                deltas.append(self._cumulative_time - before)
                return result

            PC.query_move_player = timed

            random.seed(0)
            game = _ge.GameEngine()
            game.state.max_rounds = tax_rounds
            game.start()
            final_cumulative = {
                pid: p.connection._cumulative_time
                for pid, p in game.state.players.items()
                if hasattr(p, "connection")
            }
        finally:
            PC.query_move_player = original
            _io.CORE_DIRECTORY = saved["io"]
            _gs.CORE_DIRECTORY = saved["gs"]
            _pc.CORE_DIRECTORY = saved["pc"]
            _ge.CORE_DIRECTORY = saved["ge"]
            _ge.sleep = saved["sleep"]
            _gs.MAX_ROUNDS = saved["max_rounds"]
    finally:
        from signal import SIGKILL

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
        import shutil

        shutil.rmtree(ws, ignore_errors=True)

    deltas.sort()
    n = len(deltas)

    def pct(p: float) -> float:
        if not deltas:
            return float("nan")
        idx = min(n - 1, int(math.ceil(p / 100.0 * n)) - 1)
        return deltas[max(0, idx)]

    return {
        "turns_measured": n,
        "mean_s": statistics.fmean(deltas) if deltas else float("nan"),
        "p50_s": pct(50),
        "p99_s": pct(99),
        "max_s": max(deltas) if deltas else float("nan"),
        "worst_player_cumulative_s": max(final_cumulative.values()) if final_cumulative else float("nan"),
    }


# --- report -----------------------------------------------------------------

def derive_budget(engine_tax_s: float) -> dict:
    from engine.config.io_config import CUMULATIVE_TIMEOUT_SECONDS
    from lib.config.arena import MAX_ROUNDS

    margin = 0.75  # 25% safety margin on the 8 s cumulative clock
    usable = CUMULATIVE_TIMEOUT_SECONDS * margin
    bot_side_total = usable - MAX_ROUNDS * engine_tax_s
    bot_side_per_turn = bot_side_total / MAX_ROUNDS
    return {
        "cumulative_budget_s": CUMULATIVE_TIMEOUT_SECONDS,
        "margin": margin,
        "usable_s": usable,
        "max_rounds": MAX_ROUNDS,
        "bot_side_total_s": bot_side_total,
        "bot_side_per_turn_ms": bot_side_per_turn * 1e3,
    }


def render_report(tp: dict, tax: dict, budget: dict) -> str:
    cores = tp["cores_detected"]
    target_ok = tp["step_fast_rps"] >= 300
    floor_breach = tp["step_fast_rps"] < 100
    speedup = tp.get("speedup_fast", float("nan"))
    speedup_ok = speedup >= 3.0
    tax_ms = tax["mean_s"] * 1e3
    tax_flag = tax_ms > 3.0

    def rps_row(name, v):
        return f"| {name} | {v:,.0f} | {v * cores:,.0f} | {v * 100:,.0f} |"

    lines = [
        "# Benchmarks (P1.4 / P1.5)",
        "",
        "Generated by `tools/bench_env.py`. Thresholds elsewhere read *these*",
        "numbers, never hard-coded (adaptation-pipeline discipline).",
        "",
        "## (a) In-process training throughput",
        "",
        f"Full 8-player game, trivial (stay) moves, single core "
        f"({cores} cores detected on this machine). The two `step` rows are the",
        "training path with the **P1.5 NumPy-vectorized vision** override installed;",
        "the two `pristine` rows are the same path on the unmodified engine.",
        "",
        "| Path | rounds/s/core | ×detected cores | ×100 cores (plan target rig) |",
        "|---|---:|---:|---:|",
        rps_row("physics-only (`commit_round`, ceiling)", tp["physics_only_rps"]),
        rps_row("pristine step + obs (`emit_update=False`)", tp["base_fast_rps"]),
        rps_row("pristine step + obs + update stream", tp["base_full_rps"]),
        rps_row("**P1.5** step + obs (training path, `emit_update=False`)", tp["step_fast_rps"]),
        rps_row("**P1.5** step + obs + update stream (faithful)", tp["step_full_rps"]),
        "",
        f"- **P1.5 speedup (training path):** "
        f"{tp['base_fast_rps']:,.0f} → {tp['step_fast_rps']:,.0f} rounds/s/core = "
        f"**{speedup:.2f}×** → **{'MEETS ≥3× target' if speedup_ok else 'below 3× target'}**.",
        f"- **Target:** ≥300 rounds/s/core → **{'MET' if target_ok else 'NOT met'}** "
        f"(training path = {tp['step_fast_rps']:,.0f}).",
        f"- **Hard floor:** <100 rounds/s/core makes P1.5 mandatory → "
        f"**{'BREACHED' if floor_breach else 'CLEARED by P1.5'}** "
        f"(pristine {tp['base_fast_rps']:,.0f} was below floor).",
        "",
        "> The gap between physics-only and the pristine training path was the",
        "> engine recomputing each observer's view-center + vision-size *per entity*",
        "> (160 foods × 8 players × every round). P1.5 hoists that to once-per-player",
        "> and vectorizes the box test in NumPy — bit-identical output (see",
        "> `tests/test_fast_vision.py`), no `commit_round` physics touched. The",
        "> remaining gap to the 300 ceiling is pydantic model construction, which the",
        "> Phase-9 C core removes entirely.",
        "",
        "## (b) Match-clock decomposition (engine-side tax)",
        "",
        f"Real pipe runner, 8× `dont_move` (instant bot), {tax['turns_measured']:,} turns "
        "measured. Per-turn increment to the engine's `_cumulative_time` (the exact",
        "ban-clock quantity): engine query serialization + pipe IPC + move validation,",
        "with ~zero bot compute.",
        "",
        "| Statistic | per-turn |",
        "|---|---:|",
        f"| mean | {tax['mean_s'] * 1e3:.4f} ms |",
        f"| P50 | {tax['p50_s'] * 1e3:.4f} ms |",
        f"| P99 | {tax['p99_s'] * 1e3:.4f} ms |",
        f"| max | {tax['max_s'] * 1e3:.4f} ms |",
        "",
        f"- Engine tax (mean) **{tax_ms:.4f} ms/turn** → "
        f"**{'FLAG: exceeds ~3 ms' if tax_flag else 'under the ~3 ms flag line'}**.",
        "",
        "### Derived bot-side budget",
        "",
        f"`(8 s × {budget['margin']} − {budget['max_rounds']} × {tax_ms:.4f} ms) / "
        f"{budget['max_rounds']}`",
        "",
        f"- Usable cumulative (25% margin): **{budget['usable_s']:.2f} s**",
        f"- Engine tax over the game: **{budget['max_rounds'] * tax['mean_s']:.3f} s**",
        f"- **Bot-side per-turn average budget: {budget['bot_side_per_turn_ms']:.4f} ms**",
        f"  (total bot-side {budget['bot_side_total_s']:.3f} s across {budget['max_rounds']} turns).",
        "",
        "> Caveats: (1) `dont_move` stays near starting mass, so its query carries a",
        "> *small* visible-entity set — the tax is a **floor**; late-game / adversarial",
        "> frames (P4.2) serialize more entities and cost more. (2) The measured region",
        "> includes local pipe IPC; the grader's IPC and hardware differ, which is why",
        "> the margin is 25%. P4.2 tightens the bot-side ceiling from *this* number, read",
        "> from this file — never hard-coded.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--throughput-rounds", type=int, default=800)
    ap.add_argument("--tax-rounds", type=int, default=400)
    ap.add_argument("--out", type=Path, default=REPO / "notes" / "benchmarks.md")
    args = ap.parse_args()

    print(f"[bench] (a) throughput over {args.throughput_rounds} rounds ...")
    tp = bench_throughput(args.throughput_rounds)
    print(
        f"        physics={tp['physics_only_rps']:.0f}  "
        f"pristine_fast={tp['base_fast_rps']:.0f} -> P1.5_fast={tp['step_fast_rps']:.0f}  "
        f"({tp['speedup_fast']:.2f}x)  step_full={tp['step_full_rps']:.0f} rounds/s/core"
    )

    print(f"[bench] (b) engine tax over {args.tax_rounds} rounds (real pipes) ...")
    tax = bench_engine_tax(args.tax_rounds)
    print(
        f"        engine tax mean={tax['mean_s'] * 1e3:.4f}ms  "
        f"P99={tax['p99_s'] * 1e3:.4f}ms  max={tax['max_s'] * 1e3:.4f}ms"
    )

    budget = derive_budget(tax["mean_s"])
    print(f"        bot-side per-turn budget = {budget['bot_side_per_turn_ms']:.4f} ms")

    report = render_report(tp, tax, budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
