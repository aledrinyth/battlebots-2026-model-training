"""P6.4 — learner-side league coordination: snapshots, result folding, metrics.

The counterpart of :mod:`train.harness.league_seats`. One
:class:`LeagueCoordinator` lives in the learner/trainer process and owns the
*writable* :class:`~train.league.LeaguePool`. Once per PPO update it:

1. **maybe_snapshot** — freezes the live policy into the pool on the
   configured wall-clock cadence (epoch time, so a killed-and-resumed run
   keeps its cadence — ``league.json`` persists the clock);
2. **folds results** — incrementally reads the per-actor
   ``<league_dir>/results/*.jsonl`` files from the last byte offset
   (offsets persist in ``fold_state.json``, so a resume never double-folds an
   EMA) and records one PFSP outcome per (policy seat, pool seat) pair:
   *did that policy seat outrank that frozen entry?* Per-pair — not
   best-policy-seat-per-game — so an entry that beats 3 of 4 learner seats
   keeps a mid-range ``p_current`` (informative) instead of being marked a
   pushover by the one seat that got lucky;
3. returns a flat **metrics dict** (pool size, p_current spread, policy
   rank/mass over the games folded this step) that the RunManager merges into
   ``metrics.jsonl`` — the dashboard's league feed.

Partial-line tolerance: an actor may be mid-``write`` when we read; any tail
without a newline is left un-consumed and re-read complete on the next step.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from train.harness.league_seats import GameResult
from train.league import LeaguePool


class LeagueCoordinator:
    """Owns the writable pool inside the learner/trainer process (P6.4)."""

    def __init__(
        self,
        league_dir: str | Path,
        anchors: Tuple[str, ...],
        snapshot_interval_s: float = 1800.0,
        anchor_min_frac: float = 0.10,
        ema: float = 0.1,
    ) -> None:
        self.league_dir = Path(league_dir)
        self.pool = LeaguePool(
            pool_dir=self.league_dir,
            anchors=tuple(anchors),
            snapshot_interval_s=float(snapshot_interval_s),
            anchor_min_frac=float(anchor_min_frac),
            ema=float(ema),
        )
        self.results_dir = self.league_dir / "results"
        self._fold_state_path = self.league_dir / "fold_state.json"
        self._offsets: Dict[str, int] = {}
        self.snapshots_taken = 0
        self.games_folded_total = 0
        if self._fold_state_path.exists():
            state = json.loads(self._fold_state_path.read_text())
            self._offsets = {str(k): int(v) for k, v in state["offsets"].items()}
            # Cumulative across resumes — metrics' league_games_total must be
            # monotone over a kill+resume, like the update counter itself.
            self.games_folded_total = int(state.get("games_total", 0))

    # -- persistence -----------------------------------------------------------

    def _save_fold_state(self) -> None:
        tmp = self._fold_state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"offsets": self._offsets, "games_total": self.games_folded_total},
            indent=2, sort_keys=True))
        tmp.replace(self._fold_state_path)

    # -- result folding ----------------------------------------------------------

    def _read_new_results(self) -> List[GameResult]:
        """New complete lines from every results file since the last fold."""
        out: List[GameResult] = []
        if not self.results_dir.is_dir():
            return out
        for path in sorted(self.results_dir.glob("*.jsonl")):
            key = path.name
            start = self._offsets.get(key, 0)
            try:
                size = path.stat().st_size
                if size <= start:
                    continue
                with open(path, "rb") as fh:
                    fh.seek(start)
                    blob = fh.read(size - start)
            except OSError:
                continue
            # Only consume through the last newline; a partial tail waits.
            end = blob.rfind(b"\n")
            if end < 0:
                continue
            for line in blob[: end + 1].splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(GameResult.from_json(line.decode()))
                except Exception:
                    continue  # a torn/foreign line never stalls the fold
            self._offsets[key] = start + end + 1
        return out

    def _fold(self, results: List[GameResult]) -> Dict[str, float]:
        """Record PFSP outcomes; return the fold's monitoring aggregates."""
        policy_ranks: List[float] = []
        policy_masses: List[float] = []
        pair_wins = pair_total = 0
        for res in results:
            rank_of = res.rank_of()
            p_seats = res.policy_seats()
            policy_ranks.extend(float(rank_of[p]) for p in p_seats)
            policy_masses.extend(float(res.final_masses.get(p, 0.0)) for p in p_seats)
            for pid_e, sid in res.pool_seats():
                if sid not in self.pool.entries:
                    continue  # pruned/foreign snapshot — nothing to attribute
                for pid_p in p_seats:
                    won = rank_of[pid_p] < rank_of[pid_e]
                    self.pool.record_result(sid, current_won=won)
                    pair_wins += int(won)
                    pair_total += 1
        agg: Dict[str, float] = {}
        if policy_ranks:
            agg["league_policy_rank"] = sum(policy_ranks) / len(policy_ranks)
            agg["league_policy_mass"] = sum(policy_masses) / len(policy_masses)
        if pair_total:
            agg["league_pair_winrate"] = pair_wins / pair_total
        return agg

    # -- the per-update step --------------------------------------------------------

    def step(self, policy, version: int, now: Optional[float] = None) -> Dict[str, float]:
        """Snapshot on cadence, fold new results, persist; return league metrics."""
        now = time.time() if now is None else now
        snap = self.pool.maybe_snapshot(policy, version, now)
        if snap is not None:
            self.snapshots_taken += 1
            print(f"[league] snapshot {snap.snapshot_id} -> pool "
                  f"(size={len(self.pool.entries)})", flush=True)

        results = self._read_new_results()
        metrics = self._fold(results)
        if results:
            self.games_folded_total += len(results)
            self.pool.save()
            self._save_fold_state()

        metrics["league_pool_size"] = float(len(self.pool.entries))
        metrics["league_games_folded"] = float(len(results))
        metrics["league_games_total"] = float(self.games_folded_total)
        probed = [e.p_current for e in self.pool.entries.values() if e.games > 0]
        if probed:
            metrics["league_p_mean"] = sum(probed) / len(probed)
            metrics["league_p_min"] = min(probed)
            metrics["league_p_max"] = max(probed)
        return metrics


def coordinator_from_config(cfg) -> "LeagueCoordinator":
    """Build a coordinator from a :class:`~train.harness.config.TrainConfig`."""
    if not cfg.league:
        raise ValueError("coordinator_from_config called with cfg.league=False")
    if not cfg.league_dir:
        raise ValueError("cfg.league_dir must be set (RunManager defaults it)")
    return LeagueCoordinator(
        league_dir=cfg.league_dir,
        anchors=tuple(cfg.scripted_names),
        snapshot_interval_s=cfg.league_snapshot_interval_s,
        anchor_min_frac=cfg.league_anchor_min_frac,
        ema=cfg.league_ema,
    )


__all__ = ["LeagueCoordinator", "coordinator_from_config"]
