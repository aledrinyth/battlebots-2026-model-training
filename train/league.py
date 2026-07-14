"""P6.2 — league manager: snapshot pool + PFSP opponent sampling.

Opponent diversity is the defense against self-play collapse. The league
keeps a **pool** of frozen past snapshots (exported actor-path ``.npz``,
played through :class:`train.opponents.frozen.FrozenPolicyBot`) and assigns
per-game opponent seats by **PFSP** — prioritized fictitious self-play:

    weight(entry) ∝ p·(1−p) + floor,   p = EMA win-rate of the CURRENT
                                           learner against that entry

so opponents the learner beats ~50 % of the time (the informative ones) are
sampled most, entries it always beats decay to the exploration ``floor``
(the playbook's "always-loses entry gets down-weighted automatically"), and
brand-new entries start at p=0.5 — maximum weight, so they get probed
immediately. **Scripted anchors always hold ≥ ``anchor_min_frac`` of the
game's seats** — the league can never drift away from reality's fixed
reference points.

State is a directory: ``snap_v*.npz`` + ``league.json`` (entries, win-rate
EMAs, snapshot clock). Everything is injectable-clock and injectable-RNG so
the P6.4 run loop and the tests drive it identically. Integration into the
training loop (who calls ``maybe_snapshot``/``record_result`` and builds the
``FrozenPolicyBot`` seats) is P6.4's wiring; this module is the mechanism.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Exploration floor: no pool entry's weight ever reaches exactly zero, and a
#: floor ≪ the p=0.5 weight (0.25) keeps dead entries rare but not extinct.
PFSP_FLOOR = 0.02

#: Seat spec kinds returned by :meth:`LeaguePool.sample_opponents`.
ANCHOR = "anchor"
POOL = "pool"


@dataclass
class PoolEntry:
    snapshot_id: str          # e.g. "snap_v000123"
    path: str                 # npz under the pool dir
    version: int              # trainer weights version at snapshot time
    created_s: float          # league clock at snapshot time
    games: int = 0            # games recorded vs the current learner
    p_current: float = 0.5    # EMA of "current learner beat this entry"


@dataclass
class LeaguePool:
    """The pool + sampler. One instance per training run (P6.4 owns it)."""

    pool_dir: Path
    anchors: Sequence[str] = ("greedy_food", "cautious", "chaser", "coward")
    snapshot_interval_s: float = 1800.0   # playbook: every 30 min
    anchor_min_frac: float = 0.10         # playbook: anchors ≥10% of seats
    ema: float = 0.1                      # win-rate EMA step
    entries: Dict[str, PoolEntry] = field(default_factory=dict)
    _last_snapshot_s: Optional[float] = None

    def __post_init__(self) -> None:
        self.pool_dir = Path(self.pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        reg = self.pool_dir / "league.json"
        if reg.exists():
            blob = json.loads(reg.read_text())
            self.entries = {k: PoolEntry(**v) for k, v in blob["entries"].items()}
            self._last_snapshot_s = blob.get("last_snapshot_s")

    # -- persistence ----------------------------------------------------------

    def save(self) -> None:
        blob = {
            "entries": {k: asdict(e) for k, e in self.entries.items()},
            "last_snapshot_s": self._last_snapshot_s,
        }
        tmp = self.pool_dir / "league.json.tmp"
        tmp.write_text(json.dumps(blob, indent=2, sort_keys=True))
        tmp.replace(self.pool_dir / "league.json")  # atomic (the shard.py rule)

    # -- snapshots --------------------------------------------------------------

    def maybe_snapshot(self, policy, version: int, now: float) -> Optional[PoolEntry]:
        """Freeze ``policy`` into the pool if the interval elapsed (or pool empty).

        ``policy`` is a live torch ``AgarPolicy``; only the actor path is
        exported (`train.export` discipline — the critic never leaves the
        trainer). ``now`` is the league clock (seconds; injectable for tests).
        """
        if self._last_snapshot_s is not None and self.entries \
                and now - self._last_snapshot_s < self.snapshot_interval_s:
            return None

        # Torch only on the snapshot path — sampling/record stay numpy-free.
        import numpy as np
        from train.export import MANIFEST_KEY, actor_arrays, build_manifest

        snapshot_id = f"snap_v{version:06d}"
        if snapshot_id in self.entries:  # same version twice ⇒ nothing new to freeze
            self._last_snapshot_s = now
            return None
        path = self.pool_dir / f"{snapshot_id}.npz"
        arrays = actor_arrays(policy.state_dict())
        manifest = build_manifest({}, {"version": version}, source=f"league:{snapshot_id}")
        np.savez(path, **arrays,
                 **{MANIFEST_KEY: np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)})

        entry = PoolEntry(snapshot_id=snapshot_id, path=str(path),
                          version=version, created_s=now)
        self.entries[snapshot_id] = entry
        self._last_snapshot_s = now
        self.save()
        return entry

    # -- PFSP -------------------------------------------------------------------

    def pfsp_weights(self) -> Dict[str, float]:
        """Sampling weight per entry: ``p·(1−p) + PFSP_FLOOR``."""
        return {
            sid: e.p_current * (1.0 - e.p_current) + PFSP_FLOOR
            for sid, e in self.entries.items()
        }

    def record_result(self, snapshot_id: str, current_won: bool) -> None:
        """Fold one game's outcome (did the CURRENT learner beat this entry?)."""
        e = self.entries[snapshot_id]
        e.games += 1
        e.p_current = (1.0 - self.ema) * e.p_current + self.ema * (1.0 if current_won else 0.0)

    # -- seat assignment ----------------------------------------------------------

    def sample_opponents(
        self, n_seats: int, rng: random.Random, total_seats: int = 8,
    ) -> List[Tuple[str, str]]:
        """Fill ``n_seats`` opponent seats: ``(ANCHOR, name)`` / ``(POOL, path)``.

        Anchors take at least ``ceil(anchor_min_frac × total_seats)`` of them
        (and everything, when the pool is empty); the rest draw from the pool
        by PFSP weight, with replacement (a strong snapshot may hold several
        seats — that's fictitious self-play, not a bug).
        """
        if n_seats <= 0:
            return []
        n_anchor = min(n_seats, max(1, math.ceil(self.anchor_min_frac * total_seats)))
        if not self.entries:
            n_anchor = n_seats

        seats: List[Tuple[str, str]] = [
            (ANCHOR, self.anchors[rng.randrange(len(self.anchors))])
            for _ in range(n_anchor)
        ]
        if n_seats > n_anchor:
            weights = self.pfsp_weights()
            ids = sorted(weights)  # deterministic order under a seeded rng
            w = [weights[i] for i in ids]
            picks = rng.choices(ids, weights=w, k=n_seats - n_anchor)
            seats.extend((POOL, self.entries[sid].path) for sid in picks)
        rng.shuffle(seats)  # no positional bias between anchors and pool
        return seats


__all__ = ["LeaguePool", "PoolEntry", "PFSP_FLOOR", "ANCHOR", "POOL"]
