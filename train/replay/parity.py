"""The bit-identical parity gate: reconstruction vs the live query stream
(impl doc §7.2).

The impl doc's recipe records a match through the real pipe runner. We close
the same loop **in-process** through :class:`~train.env.wrapper.WrapperEnv`,
which P1.3 already proved reproduces the real seeded pipe-runner engine
snapshot-for-snapshot (``tests/test_wrapper_parity.py``) and whose
``_build_query`` is the maintained mirror of the live
``PlayerConnection`` payload (v2026.1.14 public views). The guarantee is
transitive — wrapper queries *are* the real queries — and the gate becomes
deterministic, subprocess-free and CI-runnable.

Flow: play a seeded wrapper game with scripted bots; featurize every live
query per seat (the ground truth a real bot would have seen); dump the
engine's own ``private_event_history`` — **the replay, in the exact
tournament JSON encoding** (``EventInspector.get_recording_json`` is a
pydantic dump of that same list); reconvert it; assert every
``CanonicalObs`` is bit-identical, per seat per round.

The two failure modes §7.2 calls out — a mis-mirrored ``view_center`` clamp
and an action/observation off-by-one in block segmentation — both produce
loud array mismatches here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from pydantic import RootModel

from agario_kit import get_engine_version
from bot.featurizer import Featurizer
from train.env.wrapper import WrapperEnv
from train.opponents.scripted import make_bot
from train.replay.reconstruct import Walker, build_query

#: Mixed field with predators so deaths/respawns occur early — the reset
#: path is exactly where reconstruction can silently diverge.
DEFAULT_BOTS = (
    "chaser", "chaser", "greedy_food", "greedy_food",
    "cautious", "cautious", "coward", "dont_move",
)

_OBS_FIELDS = (
    "self_vec", "food", "food_mask", "enemy", "enemy_mask",
    "own", "own_mask", "virus", "virus_mask",
)


@dataclass
class ParityCapture:
    """One recorded wrapper game: replay JSON + ground-truth obs stream."""

    seed: int
    events: List[dict]
    #: chronological (round, seat, CanonicalObs) — engine query order.
    ground_truth: List[Tuple[int, int, object]]
    deaths: int
    rounds_played: int


@dataclass
class ParityReport:
    frames: int = 0
    mismatches: List[str] = field(default_factory=list)
    deaths: int = 0
    resets: int = 0

    @property
    def passed(self) -> bool:
        return self.frames > 0 and not self.mismatches

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"parity {status}: {self.frames} frames bit-compared, "
            f"{len(self.mismatches)} mismatches, {self.deaths} deaths, "
            f"{self.resets} reset frames"
        )


def record_wrapper_match(
    seed: int,
    rounds: Optional[int] = None,
    bot_names: Tuple[str, ...] = DEFAULT_BOTS,
) -> ParityCapture:
    """Play a seeded in-process game and capture queries + the replay."""
    env = WrapperEnv(seed=seed, emit_update=False)
    obs = env.reset()
    bots = {pid: make_bot(bot_names[pid % len(bot_names)]) for pid in env.state.players}
    feats = {pid: Featurizer() for pid in env.state.players}

    ground_truth: List[Tuple[int, int, object]] = []
    deaths = 0
    played = 0
    alive_before = {pid for pid, p in env.state.players.items() if p.alive}

    while obs and (rounds is None or played < rounds):
        moves = {}
        for pid, query in obs.items():  # dict preserves engine query order
            ground_truth.append((query.round, pid, feats[pid].featurize(query)))
            moves[pid] = bots[pid].act(query)
        obs, done = env.step(moves)
        played += 1
        alive_now = {pid for pid, p in env.state.players.items() if p.alive}
        deaths += len(alive_before - alive_now)
        alive_before = alive_now
        if done:
            break

    events = json.loads(RootModel(env.state.private_event_history).model_dump_json())
    # A truncated run has no event_player_won yet; terminate it the way the
    # engine would right now. Stamp the vendored engine version (the wrapper
    # builds EventGameStarted without one).
    if not any(e["event_type"] == "event_player_won" for e in events):
        events.append(
            {"event_type": "event_player_won",
             "player_id": env.state.get_rankings()[0]}
        )
    if events[0].get("engine_version") is None:
        events[0]["engine_version"] = get_engine_version()
    return ParityCapture(
        seed=seed,
        events=events,
        ground_truth=ground_truth,
        deaths=deaths,
        rounds_played=played,
    )


def iter_reconstructed_obs(events: List[dict]) -> Iterator[Tuple[int, int, object]]:
    """Reconstruction side of the gate: the exact convert-loop featurization
    (one stateful Featurizer per seat, chronological), yielding obs objects
    instead of flattened rows."""
    walker = Walker(events, source="<parity>")
    feats = {pid: Featurizer() for pid in walker.gs.players}
    for rnd, move in walker.decisions():
        pid = move["player_id"]
        yield rnd, pid, feats[pid].featurize(build_query(walker.gs, pid, rnd))


def verify_parity(capture: ParityCapture, max_detail: int = 5) -> ParityReport:
    """Bit-compare ground truth vs reconstruction, frame by frame."""
    report = ParityReport(deaths=capture.deaths)
    recon = list(iter_reconstructed_obs(capture.events))

    if len(recon) != len(capture.ground_truth):
        report.mismatches.append(
            f"frame count: live {len(capture.ground_truth)} vs "
            f"reconstructed {len(recon)}"
        )
        return report

    for (lr, lp, live), (rr, rp, rec) in zip(capture.ground_truth, recon):
        report.frames += 1
        where = f"round {lr} seat {lp}"
        if (lr, lp) != (rr, rp):
            report.mismatches.append(
                f"{where}: alignment — reconstruction has round {rr} seat {rp}"
            )
            break
        if live.is_reset != rec.is_reset:
            report.mismatches.append(
                f"{where}: is_reset live={live.is_reset} rec={rec.is_reset}"
            )
        if live.is_reset:
            report.resets += 1
        for name in _OBS_FIELDS:
            a, b = getattr(live, name), getattr(rec, name)
            if not np.array_equal(a, b):
                if len(report.mismatches) < max_detail:
                    diff = np.abs(a - b)
                    report.mismatches.append(
                        f"{where}: {name} differs — max|Δ|={diff.max():.3e} "
                        f"at {np.unravel_index(diff.argmax(), diff.shape)}"
                    )
                else:
                    report.mismatches.append(f"{where}: {name} differs")
                    return report
    return report


def run_parity_gate(
    seed: int = 0,
    rounds: Optional[int] = None,
    bot_names: Tuple[str, ...] = DEFAULT_BOTS,
) -> ParityReport:
    return verify_parity(record_wrapper_match(seed, rounds, bot_names))
