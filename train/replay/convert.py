"""``GameState`` + ``Featurizer`` → dataset rows (impl doc §5).

One persistent :class:`~bot.featurizer.Featurizer` per ``(replay, seat)``,
fed chronologically — a fresh instance per frame would zero every enemy
velocity and silently corrupt the dataset (impl doc §4.4, mandatory).
Respawn gaps are detected by the featurizer itself (round jump ⇒
``is_reset``); the converter *independently* bookkeeps expected resets and
hard-fails on any disagreement, which turns the §7.1 validation into a
permanent invariant instead of a one-off check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from bot.featurizer import Featurizer
from train.harness import flat_obs
from train.replay.labels import heading_to_bin
from train.replay.reconstruct import ReplayError, Walker, build_query
from train.replay.schema import ROW_DTYPES


@dataclass
class SeatMeta:
    player_id: int
    team_id: int
    final_rank: int
    n_rows: int = 0
    n_resets: int = 0


@dataclass
class ConvertResult:
    """All rows of one replay + per-seat metadata + the self-check report.

    ``arrays`` carries the per-row columns produced here (``obs``,
    ``heading_bin``, ``has_heading``, ``split``, ``is_reset``, ``seat``,
    ``round``); the corpus-level columns (``rank_weight``, ``bot_id``,
    ``episode``) are stamped by the dataset/ingest layer, which owns
    identity interning and episode numbering.
    """

    source: str
    engine_version: Optional[str]
    rounds: int
    winner_pid: int
    arrays: Dict[str, np.ndarray]
    seats: Dict[int, SeatMeta]
    report: Dict[str, object] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.arrays["obs"].shape[0])


def convert_replay(
    events: List[dict],
    source: str = "<memory>",
    validate_obs: bool = True,
) -> ConvertResult:
    """Convert one replay's events into dataset rows.

    ``validate_obs`` runs ``CanonicalObs.validate()`` on every frame
    (shapes/dtypes/finiteness/mask-bounds). It is cheap relative to
    featurization; leave it on except in throughput-critical bulk runs.
    """
    walker = Walker(events, source=source)
    feats = {pid: Featurizer() for pid in walker.gs.players}
    last_round: Dict[int, int] = {}

    obs_rows: List[np.ndarray] = []
    heading_bins: List[int] = []
    has_headings: List[int] = []
    splits: List[int] = []
    is_resets: List[int] = []
    seats_col: List[int] = []
    rounds_col: List[int] = []
    bins_seen = set()

    for rnd, move in walker.decisions():
        pid = move["player_id"]
        obs = feats[pid].featurize(build_query(walker.gs, pid, rnd))
        if validate_obs:
            obs.validate()

        # Independent reset bookkeeping — must agree with the featurizer.
        expected_reset = pid not in last_round or rnd != last_round[pid] + 1
        if obs.is_reset != expected_reset:
            raise ReplayError(
                f"{source}: is_reset mismatch at round {rnd} seat {pid}: "
                f"featurizer={obs.is_reset} bookkeeping={expected_reset}"
            )
        last_round[pid] = rnd

        d = move["direction"]
        hb = heading_to_bin(d["x"], d["y"])
        obs_rows.append(flat_obs.flatten(obs))
        heading_bins.append(-1 if hb is None else hb)
        has_headings.append(0 if hb is None else 1)
        if hb is not None:
            bins_seen.add(hb)
        splits.append(1 if move["split"] else 0)
        is_resets.append(1 if obs.is_reset else 0)
        seats_col.append(pid)
        rounds_col.append(rnd)

    ranks = walker.final_ranks()
    seats = {
        pid: SeatMeta(
            player_id=pid,
            team_id=walker.team_of[pid],
            final_rank=ranks[pid],
        )
        for pid in walker.gs.players
    }
    for pid, reset in zip(seats_col, is_resets):
        seats[pid].n_rows += 1
        seats[pid].n_resets += reset

    n = len(obs_rows)
    arrays = {
        "obs": (
            np.stack(obs_rows).astype(ROW_DTYPES["obs"])
            if n
            else np.zeros((0, flat_obs.OBS_WIDTH), dtype=ROW_DTYPES["obs"])
        ),
        "heading_bin": np.asarray(heading_bins, dtype=ROW_DTYPES["heading_bin"]),
        "has_heading": np.asarray(has_headings, dtype=ROW_DTYPES["has_heading"]),
        "split": np.asarray(splits, dtype=ROW_DTYPES["split"]),
        "is_reset": np.asarray(is_resets, dtype=ROW_DTYPES["is_reset"]),
        "seat": np.asarray(seats_col, dtype=ROW_DTYPES["seat"]),
        "round": np.asarray(rounds_col, dtype=ROW_DTYPES["round"]),
    }

    n_resets = int(arrays["is_reset"].sum())
    report = {
        "engine_version": walker.engine_version,
        "rounds": walker.rounds,
        "rows": n,
        "move_events": walker.n_moves,
        "obs_validated": bool(validate_obs),
        "heading_bins_covered": len(bins_seen),
        "split_rows": int(arrays["split"].sum()),
        "stay_rows": int((arrays["has_heading"] == 0).sum()),
        "reset_rows": n_resets,
        "first_frames": walker.n_players,
        "respawns": n_resets - walker.n_players,
        "eaten_in_vision": walker.eaten_in_vision,
        "eaten_in_reach": walker.eaten_in_reach,
        "eaten_total": walker.eaten_total,
    }
    return ConvertResult(
        source=source,
        engine_version=walker.engine_version,
        rounds=walker.rounds,
        winner_pid=walker.winner_pid,
        arrays=arrays,
        seats=seats,
        report=report,
    )


def check_report(result: ConvertResult) -> List[str]:
    """§7.1-style acceptance on one converted replay → list of failures.

    Empty list ⇒ PASS. The checks are exactly the prototype-validation
    block from the impl doc, generalized off the one hard-coded replay:

    * every alive ``move_player`` produced exactly one row;
    * every obs passed ``CanonicalObs.validate()`` at width ``OBS_WIDTH``;
    * reset frames == one first-frame per seat + one per respawn (the
      converter already hard-fails on any featurizer/bookkeeping mismatch);
    * **100 %** of eaten pellets within ``2·radius + 3.0`` of the eater's
      nearest pre-round blob — the physical possibility bound (split-spawn
      offset ≤ 2r, eject + speed + stabilise slop < 3.0). The plain
      in-window fraction is telemetry only: the impl doc assumed "you can
      only eat what you see", but eating is per-blob while vision is
      centroid-anchored — giants outgrow their vision box, split children
      spawn a parent-radius away, and a 16-blob player spans the arena, so
      tournament giant matches legitimately eat 2–5 % of pellets outside
      the plain window (measured across the corpus). The walker separately
      hard-fails if an eaten pellet id is absent from the reconstructed
      map — the strict stream-consistency signal. Bit-level sufficiency
      is the parity gate's job.
    """
    failures: List[str] = []
    r = result.report
    if r["rows"] != r["move_events"]:
        failures.append(f"rows {r['rows']} != alive move events {r['move_events']}")
    if not r["obs_validated"]:
        failures.append("obs validation was disabled for this run")
    if result.arrays["obs"].shape[1] != flat_obs.OBS_WIDTH:
        failures.append(
            f"obs width {result.arrays['obs'].shape[1]} != {flat_obs.OBS_WIDTH}"
        )
    if r["respawns"] < 0:
        failures.append(
            f"reset frames {r['reset_rows']} < seat count {r['first_frames']}"
        )
    if r["eaten_in_reach"] != r["eaten_total"]:
        failures.append(
            f"eaten-food-in-reach {r['eaten_in_reach']}/{r['eaten_total']}"
            " — pellets eaten beyond any physically reachable window"
        )
    return failures
