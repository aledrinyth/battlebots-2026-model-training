#!/usr/bin/env python3
"""P4.2 — adversarial scene bank: generator + bot-path replay harness.

Worst-case frames, not average-case. The generator engineers extreme
``GameState``s inside the P1 wrapper (whose ``_build_query`` is proven
byte-identical to the live ``PlayerConnection``'s) and records the **exact
wire payload** (``QueryMovePlayer.model_dump_json()``) the grader's engine
would write to our pipe:

* ``max_entities``  — all 7 enemies fully split (7 × MAX_BLOB_COUNT blobs),
                      ourselves fully split (MAX_BLOB_COUNT), every food and
                      virus teleported into our (enlarged) vision box; frames
                      captured *after* a committed round so the ``update``
                      stream carries the round's event burst too.
* ``empty_vision``  — alone in a corner, nothing visible (zero-entity masks,
                      the division-guard path).
* ``respawn``       — a real death (engineered predator, engine's own eat +
                      respawn machinery): first query after the 30-round gap
                      carries the full event backlog and must reset the GRU.
* ``final_round``   — frames at MAX_ROUNDS−3 … −1 (round-watermark edge).
* ``organic_*``     — early/mid/late frame triples from a real scripted-
                      opponent game (P2.4 anchors), the average-case control.

The replay harness (`replay_bank`) runs every frame through the bot's full
match-time path — **parse** (kit pydantic ``model_validate_json``) →
**featurize** (`bot.featurizer`) → **forward** (`bot.inference.NumpyPolicy`,
greedy decode) → **serialize** (``MovePlayer.model_dump_json()``) — with a
per-stage breakdown (the P5.4 "kit loop vs direct pipe reader" decision
reads the parse share from this report).

Thresholds (playbook P4.2, "never hard-coded" discipline):

* mean/turn ≤ the bot-side per-turn budget **read from notes/benchmarks.md**
  (derived there from the engine's own measured tax);
* P99 < ``PLAN_P99_MS`` (3 ms — the plan's tightened target: the engine-side
  serialization tax verifiably shares our 8 s clock);
* max < ``PLAN_MAX_MS`` (100 ms single-frame ceiling).

This bank also formalizes P5.2's "≤1 ms forward pass" gate: the report's
``forward`` stage is checked against ``FORWARD_BUDGET_MS``.

Usage:
    tools/scene_bank.py generate [--seed 0] [--out tests/fixtures/adversarial/scenes.json.gz]
    tools/scene_bank.py replay   [--bank ...] [--weights tests/fixtures/adversarial/weights_random.npz]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BANK_PATH = REPO / "tests" / "fixtures" / "adversarial" / "scenes.json.gz"
WEIGHTS_PATH = REPO / "tests" / "fixtures" / "adversarial" / "weights_random.npz"
BENCHMARKS_PATH = REPO / "notes" / "benchmarks.md"

#: Playbook P4.2 planning targets — tightened from 4 ms because the engine-side
#: serialization tax verifiably shares our 8 s clock (see notes/benchmarks.md §b).
PLAN_P99_MS = 3.0
PLAN_MAX_MS = 100.0
#: P5.2's forward-pass gate, formalized here (playbook: "≤1 ms/turn").
FORWARD_BUDGET_MS = 1.0

FRAMES_PER_SCENE = 3


def read_bot_side_budget_ms(path: Path = BENCHMARKS_PATH) -> float:
    """The derived per-turn average budget from notes/benchmarks.md §b."""
    m = re.search(r"Bot-side per-turn average budget: ([0-9.]+) ms", path.read_text())
    if not m:
        raise ValueError(f"no bot-side budget line in {path} — rerun tools/bench_env.py")
    return float(m.group(1))


# --- generator ----------------------------------------------------------------

def _stay_moves(obs) -> dict:
    from train.env.wrapper import make_move
    return {pid: make_move(pid, 0.0, 0.0) for pid in obs}


def _capture(env, obs, pid: int, n: int = FRAMES_PER_SCENE) -> List[str]:
    """Step ``n`` stay-move rounds, recording ``pid``'s wire payload each round."""
    frames: List[str] = []
    while len(frames) < n:
        obs, done = env.step(_stay_moves(obs))
        if done:
            raise RuntimeError("game ended mid-capture")
        if pid in obs:
            frames.append(obs[pid].model_dump_json())
    return frames


def _grid_positions(cx: float, cy: float, n: int, spacing: float,
                    cols: int = 4) -> List[tuple]:
    """A compact n-point grid centred on (cx, cy)."""
    rows = math.ceil(n / cols)
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        out.append((cx + (c - (cols - 1) / 2) * spacing,
                    cy + (r - (rows - 1) / 2) * spacing))
    return out


def gen_max_entities(seed: int) -> dict:
    """7 enemies fully split + self fully split + all food/viruses in vision."""
    from engine.state.blob_state import BlobState
    from lib.config.arena import ARENA_SIZE, MAX_BLOB_COUNT
    from lib.models.food_model import FoodModel
    from lib.models.virus_model import VirusModel
    from train.env.wrapper import WrapperEnv

    env = WrapperEnv(seed=seed)
    obs = env.reset()
    st = env.state
    cx = cy = ARENA_SIZE / 2

    # Ourselves: MAX_BLOB_COUNT blobs, radii sized so the vision box grows
    # (sum_of_radii 16×0.9 = 14.4 > the 12.0 reference → ~21.5 vision).
    st.players[0].blobs = {
        i: BlobState(blob_id=i, x=x, y=y, radius=0.9, merge_cooldown=500)
        for i, (x, y) in enumerate(_grid_positions(cx, cy, MAX_BLOB_COUNT, 0.75))
    }
    # 7 enemies × MAX_BLOB_COUNT blobs on a ring inside the box, small enough
    # that nothing eats anything on the captured rounds (equal radii don't eat;
    # keep them clear of our 0.9s).
    for k, pid in enumerate(p for p in st.players if p != 0):
        ang = 2 * math.pi * k / 7
        ex, ey = cx + 7.5 * math.cos(ang), cy + 7.5 * math.sin(ang)
        st.players[pid].blobs = {
            i: BlobState(blob_id=i, x=x, y=y, radius=0.45, merge_cooldown=500)
            for i, (x, y) in enumerate(
                _grid_positions(ex, ey, MAX_BLOB_COUNT, 0.5))
        }
    # Every food and virus into the box (spacing away from blobs so the food
    # mostly survives the captured rounds — density is the point).
    st.map.foods = [
        FoodModel(food_id=i, pos=pos)
        for i, pos in enumerate(_grid_positions(cx, cy, len(st.map.foods), 0.9, cols=13))
    ]
    st.map.viruses = [
        VirusModel(virus_id=i, pos=(cx - 6 + 2.4 * i, cy - 6), radius=v.radius)
        for i, v in enumerate(st.map.viruses)
    ]

    frames = _capture(env, obs, pid=0)
    # Self-check: the engineered density actually reached the wire.
    from lib.interface.queries.query_move import QueryMovePlayer
    q = QueryMovePlayer.model_validate_json(frames[0])
    assert len(q.visible_blobs) >= 80, f"only {len(q.visible_blobs)} enemy blobs visible"
    assert len(q.visible_food) >= 100, f"only {len(q.visible_food)} foods visible"
    assert len(q.you.blobs) == MAX_BLOB_COUNT
    return {"name": "max_entities", "player_id": 0, "frames": frames}


def gen_empty_vision(seed: int) -> dict:
    """Alone in a corner: zero visible food / enemies / viruses."""
    from engine.state.blob_state import BlobState
    from lib.config.arena import ARENA_SIZE
    from lib.models.food_model import FoodModel
    from lib.models.virus_model import VirusModel
    from train.env.wrapper import WrapperEnv

    env = WrapperEnv(seed=seed)
    obs = env.reset()
    st = env.state
    far = ARENA_SIZE - 2.0

    st.players[0].blobs = {0: BlobState(blob_id=0, x=2.0, y=2.0, radius=0.9)}
    # Three *separate* far dumps — if enemies sat on the food they would eat
    # it all and the engine's _ensure_food_count would respawn 160 foods
    # uniformly, some right back into our vision box.
    for pid in (p for p in st.players if p != 0):
        st.players[pid].blobs = {
            i: BlobState(blob_id=b.blob_id, x=far, y=far, radius=b.radius)
            for i, b in enumerate(st.players[pid].blobs.values())
        }
    st.map.foods = [FoodModel(food_id=f.food_id, pos=(far, 2.0 + 0.01 * i))
                    for i, f in enumerate(st.map.foods)]
    st.map.viruses = [
        VirusModel(virus_id=v.virus_id, pos=(2.0 + 3.0 * i, far), radius=v.radius)
        for i, v in enumerate(st.map.viruses)
    ]

    frames = _capture(env, obs, pid=0)
    from lib.interface.queries.query_move import QueryMovePlayer
    q = QueryMovePlayer.model_validate_json(frames[0])
    assert not q.visible_food and not q.visible_blobs and not q.visible_viruses
    return {"name": "empty_vision", "player_id": 0, "frames": frames}


def gen_respawn(seed: int) -> dict:
    """A real death and respawn through the engine's own machinery.

    An engineered predator (radius 3.0 ≫ 1.2 × our 0.9) is dropped onto our
    blob; the engine eats us, we sit out RESPAWN_DELAY_ROUNDS unqueried, and
    the first query after respawn — round gap + full update backlog — is the
    frame under test.
    """
    from engine.state.blob_state import BlobState
    from train.env.wrapper import WrapperEnv

    env = WrapperEnv(seed=seed)
    obs = env.reset()
    st = env.state
    me = st.players[0]
    st.players[1].blobs = {
        0: BlobState(blob_id=0, x=me.x, y=me.y, radius=3.0, merge_cooldown=500)
    }

    died_round = None
    frames: List[str] = []
    # The predator eats us on the very first commit, so the pre-death round
    # is the round-0 query from reset().
    last_seen_round = obs[0].round
    while len(frames) < FRAMES_PER_SCENE:
        obs, done = env.step(_stay_moves(obs))
        if done:
            raise RuntimeError("game ended before respawn capture")
        if 0 in obs:
            if died_round is not None:
                frames.append(obs[0].model_dump_json())
            else:
                last_seen_round = obs[0].round  # freezes at the pre-death round
        elif died_round is None:
            died_round = st.round
    assert died_round is not None, "the engineered predator never ate us"

    from lib.interface.queries.query_move import QueryMovePlayer
    q = QueryMovePlayer.model_validate_json(frames[0])
    assert len(q.update) > 0, "respawn frame lost its event backlog"
    return {"name": "respawn", "player_id": 0, "frames": frames,
            "died_round": died_round, "respawn_round": q.round,
            "pre_death_round": last_seen_round}


def gen_final_round(seed: int) -> dict:
    """The last queried rounds of a game (MAX_ROUNDS−3 … −1)."""
    from lib.config.arena import MAX_ROUNDS
    from train.env.wrapper import WrapperEnv

    env = WrapperEnv(seed=seed)
    obs = env.reset()
    for _ in range(3):  # a few real rounds first so histories exist
        obs, _ = env.step(_stay_moves(obs))
    env.state.round = MAX_ROUNDS - FRAMES_PER_SCENE  # jump the clock
    frames = [env._build_query(0).model_dump_json()]
    while len(frames) < FRAMES_PER_SCENE:
        obs, done = env.step(_stay_moves(obs))
        if 0 in obs:
            frames.append(obs[0].model_dump_json())
        if done:
            break
    return {"name": "final_round", "player_id": 0, "frames": frames}


def gen_organic(seed: int, rounds: int = 200,
                capture_at: tuple = (40, 120)) -> List[dict]:
    """Early/mid/late frame triples from a real scripted-opponent game.

    Early/mid capture at fixed rounds; "late" is the **last** consecutive
    alive-triple seat 0 produced (it may get eaten by the chasers at any
    fixed window — a rolling buffer sidesteps the death schedule).
    """
    from train.env.wrapper import WrapperEnv
    from train.opponents.scripted import (
        CautiousBot, ChaserBot, CowardBot, GreedyFoodBot,
    )

    bots = {0: GreedyFoodBot(), 1: GreedyFoodBot(), 2: GreedyFoodBot(),
            3: CautiousBot(), 4: CautiousBot(), 5: ChaserBot(),
            6: ChaserBot(), 7: CowardBot()}
    env = WrapperEnv(seed=seed)
    obs = env.reset()
    scenes = {r: [] for r in capture_at}
    rolling: List[tuple] = []  # (round, frame) — last consecutive alive run
    for _ in range(rounds):
        moves = {pid: bots[pid].act(q) for pid, q in obs.items()}
        obs, done = env.step(moves)
        if done:
            break
        if 0 in obs:
            rnd = env.state.round
            if rolling and rolling[-1][0] != rnd - 1:
                rolling.clear()  # gap (death) — restart the run
            rolling.append((rnd, obs[0].model_dump_json()))
            del rolling[:-FRAMES_PER_SCENE]
            for r in capture_at:
                if r <= rnd < r + FRAMES_PER_SCENE:
                    scenes[r].append(obs[0].model_dump_json())
    out = [
        {"name": label, "player_id": 0, "frames": scenes[r]}
        for label, r in zip(("organic_early", "organic_mid"), capture_at)
        if len(scenes[r]) == FRAMES_PER_SCENE  # seat 0 may be dead at a window
    ]
    if len(rolling) == FRAMES_PER_SCENE:
        out.append({"name": "organic_late", "player_id": 0,
                    "frames": [f for _, f in rolling],
                    "late_rounds": [r for r, _ in rolling]})
    return out


def generate_bank(seed: int = 0, out_path: Path = BANK_PATH) -> dict:
    from agario_kit import get_engine_version

    scenes = [
        gen_max_entities(seed),
        gen_empty_vision(seed),
        gen_respawn(seed),
        gen_final_round(seed),
        *gen_organic(seed),
    ]
    bank = {
        "schema": "scene-bank-v1",
        "engine_version": get_engine_version(),
        "seed": seed,
        "generator": "tools/scene_bank.py",
        "scenes": scenes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wb") as f:
        f.write(json.dumps(bank, separators=(",", ":")).encode())
    return bank


def load_bank(path: Path = BANK_PATH) -> dict:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode())


# --- replay harness ------------------------------------------------------------

STAGES = ("parse", "featurize", "forward", "serialize", "total")


def replay_bank(bank: dict, weights_path: Path = WEIGHTS_PATH,
                reps: int = 15, warmup_reps: int = 1) -> dict:
    """Run every frame through the bot's full match-time path; time each stage.

    Per scene and rep the featurizer and GRU hidden start fresh (each scene is
    an episode slice); timings pool across frames and post-warmup reps —
    enough samples that P99 isn't literally the max (one scheduler hiccup
    must not fail the gate). GC is parked during the timed loop: the gate
    measures the *algorithmic* per-turn cost (GC policy is a P5.4 main-loop
    decision; real-process jitter is the P4.1 harness's job).
    """
    import gc

    from bot.adapters.act_v1 import ActionAdapter
    from bot.featurizer import Featurizer
    from bot.inference import NumpyPolicy
    from lib.interface.queries.query_move import QueryMovePlayer

    policy = NumpyPolicy.load(str(weights_path))
    times: Dict[str, List[float]] = {s: [] for s in STAGES}
    per_scene_max_ms: Dict[str, float] = {}

    gc.collect()
    gc.disable()
    for rep in range(reps):
        record = rep >= warmup_reps
        for scene in bank["scenes"]:
            featurizer = Featurizer()
            hidden = policy.initial_hidden()
            pid = scene["player_id"]
            for raw in scene["frames"]:
                t0 = perf_counter()
                query = QueryMovePlayer.model_validate_json(raw)
                t1 = perf_counter()
                obs = featurizer.featurize(query)
                t2 = perf_counter()
                action, hidden = policy.act(obs, hidden)
                t3 = perf_counter()
                move = ActionAdapter.to_move(pid, action)
                payload = move.model_dump_json()
                t4 = perf_counter()
                assert payload  # keep the serialize stage un-elided
                if record:
                    times["parse"].append(t1 - t0)
                    times["featurize"].append(t2 - t1)
                    times["forward"].append(t3 - t2)
                    times["serialize"].append(t4 - t3)
                    times["total"].append(t4 - t0)
                    key = scene["name"]
                    per_scene_max_ms[key] = max(per_scene_max_ms.get(key, 0.0),
                                                (t4 - t0) * 1e3)
    gc.enable()

    def stats_ms(vals: List[float]) -> dict:
        v = sorted(x * 1e3 for x in vals)
        n = len(v)
        return {
            "mean": statistics.fmean(v),
            "p50": v[n // 2],
            "p99": v[min(n - 1, math.ceil(0.99 * n) - 1)],
            "max": v[-1],
        }

    budget_ms = read_bot_side_budget_ms()
    report = {
        "frames_timed": len(times["total"]),
        "stages_ms": {s: stats_ms(times[s]) for s in STAGES},
        "per_scene_max_ms": per_scene_max_ms,
        "thresholds": {
            "mean_budget_ms": budget_ms,       # from notes/benchmarks.md
            "p99_ms": PLAN_P99_MS,
            "max_ms": PLAN_MAX_MS,
            "forward_budget_ms": FORWARD_BUDGET_MS,
        },
    }
    total = report["stages_ms"]["total"]
    report["gates"] = {
        "mean_under_budget": total["mean"] <= budget_ms,
        "p99_under_plan": total["p99"] < PLAN_P99_MS,
        "max_under_plan": total["max"] < PLAN_MAX_MS,
        "forward_under_p52_gate": report["stages_ms"]["forward"]["p99"] < FORWARD_BUDGET_MS,
    }
    report["ok"] = all(report["gates"].values())
    return report


def render(report: dict) -> str:
    lines = [f"frames timed: {report['frames_timed']}   "
             f"(budget {report['thresholds']['mean_budget_ms']:.3f} ms/turn from notes/benchmarks.md)"]
    lines.append(f"  {'stage':>10} {'mean':>8} {'p50':>8} {'p99':>8} {'max':>8}  (ms)")
    for s in STAGES:
        st = report["stages_ms"][s]
        lines.append(f"  {s:>10} {st['mean']:>8.3f} {st['p50']:>8.3f} "
                     f"{st['p99']:>8.3f} {st['max']:>8.3f}")
    lines.append("  per-scene worst total (ms): " + "  ".join(
        f"{k}={v:.3f}" for k, v in sorted(report["per_scene_max_ms"].items())))
    for gate, ok in report["gates"].items():
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {gate}")
    lines.append(f"VERDICT: {'PASS' if report['ok'] else 'FAIL'}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="(re)generate the scene bank fixtures")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", type=Path, default=BANK_PATH)
    r = sub.add_parser("replay", help="time the bot path over the bank")
    r.add_argument("--bank", type=Path, default=BANK_PATH)
    r.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    r.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()

    if args.cmd == "generate":
        bank = generate_bank(args.seed, args.out)
        for s in bank["scenes"]:
            print(f"[bank] {s['name']}: {len(s['frames'])} frames, "
                  f"{sum(len(f) for f in s['frames']):,} bytes")
        print(f"[bank] wrote {args.out}")
    else:
        report = replay_bank(load_bank(args.bank), args.weights, reps=args.reps)
        print(render(report))
        sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
