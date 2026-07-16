"""P6.4 — actor-side league seams: per-game seat sampling + game-result records.

The league's *state* (the pool registry, win-rate EMAs, snapshots) is owned by
the learner process (:mod:`train.harness.league_wire`); actor processes must
stay torch-free and must never write ``league.json``. So the actor side is two
small, filesystem-coupled pieces, mirroring the existing weights hot-reload
discipline:

* :class:`LeagueSeatSampler` — at every game rollover, re-reads ``league.json``
  (mtime-cached), samples the non-policy seats by PFSP via
  :meth:`train.league.LeaguePool.sample_opponents` (read-only), and builds the
  bots: anchors through :func:`~train.opponents.scripted.make_bot`, pool
  entries through :class:`~train.opponents.frozen.FrozenPolicyBot` with an LRU
  cache of loaded :class:`~bot.inference.NumpyPolicy` weights (PFSP samples
  with replacement, so one snapshot may hold several seats — they share
  weights, never hidden state). Sampling is **deterministic per
  (stream, game_counter)** given the registry content — the seed-per-game
  shape every game-playing tool here keeps.

* :class:`ResultsWriter` — appends one JSON line per finished game to
  ``<league_dir>/results/<name>.jsonl`` (one file per actor/worker: no
  interleaving, and the learner folds them incrementally by byte offset).

Failure discipline: an actor lives inside a barrier protocol — it must never
crash the run because the registry was mid-rename or a snapshot file vanished.
Every filesystem read here degrades to "anchors only" (with one stderr note),
never raises.
"""

from __future__ import annotations

import json
import random
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from train.harness.game_runner import game_seed
from train.league import ANCHOR, LeaguePool, PoolEntry
from train.opponents.frozen import FrozenPolicyBot
from train.opponents.scripted import ScriptedBot, make_bot

#: XOR'd into the per-game engine seed so the league RNG is a distinct,
#: deterministic stream (pure int arithmetic — no hash randomization).
_LEAGUE_SEED_SALT = 0x5EA75

#: Seat-spec strings recorded in :class:`GameResult.plan`.
POLICY_SPEC = "policy"


@dataclass
class GameResult:
    """Terminal record of one finished training game (one JSON line)."""

    stream_index: int
    game_counter: int
    seed: int
    rankings: List[int]                 # player ids, best first
    final_masses: Dict[int, float]      # pid -> final mass
    plan: Dict[int, str]                # pid -> "policy" | "anchor:<name>" | "pool:<snapshot_id>"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "GameResult":
        d = json.loads(line)
        # JSON object keys are strings; pids are ints.
        d["final_masses"] = {int(k): float(v) for k, v in d["final_masses"].items()}
        d["plan"] = {int(k): str(v) for k, v in d["plan"].items()}
        return cls(**d)

    # -- derived views (the folding/monitoring vocabulary) --------------------

    def rank_of(self) -> Dict[int, int]:
        """pid -> rank (1 = winner)."""
        return {pid: i + 1 for i, pid in enumerate(self.rankings)}

    def policy_seats(self) -> List[int]:
        return sorted(p for p, s in self.plan.items() if s == POLICY_SPEC)

    def pool_seats(self) -> List[Tuple[int, str]]:
        """[(pid, snapshot_id)] for seats held by frozen pool entries."""
        return sorted((p, s.split(":", 1)[1]) for p, s in self.plan.items()
                      if s.startswith("pool:"))

    def anchor_seats(self) -> List[Tuple[int, str]]:
        return sorted((p, s.split(":", 1)[1]) for p, s in self.plan.items()
                      if s.startswith("anchor:") or s.startswith("scripted:"))


class LeagueSeatSampler:
    """Read-only PFSP seat sampling inside an actor process.

    ``base_seed`` must match the :class:`~train.harness.game_runner.GameRunner`'s
    (per-worker-offset included) so each game's league draw is a deterministic
    function of ``(stream_index, game_counter)`` and the registry content.
    """

    def __init__(
        self,
        league_dir: str | Path,
        anchors: Tuple[str, ...],
        anchor_min_frac: float,
        base_seed: int,
        policy_cache_size: int = 4,
    ) -> None:
        self.league_dir = Path(league_dir)
        self.anchors = tuple(anchors) or ("greedy_food",)
        self.anchor_min_frac = float(anchor_min_frac)
        self.base_seed = int(base_seed)
        self.policy_cache_size = int(policy_cache_size)
        self._registry = self.league_dir / "league.json"
        self._entries: Dict[str, PoolEntry] = {}
        self._mtime: Optional[float] = None
        self._policies: "OrderedDict[str, object]" = OrderedDict()  # path -> NumpyPolicy (LRU)
        self._warned = False
        # A read-only sampling view; entries are refreshed into it per draw.
        # (sample_opponents is pure given entries — save/maybe_snapshot never
        # run here, so this instance can never write league.json.)
        self._pool_view = LeaguePool(
            pool_dir=self.league_dir, anchors=self.anchors,
            anchor_min_frac=self.anchor_min_frac,
        )

    # -- registry hot-reload ---------------------------------------------------

    def _refresh(self) -> None:
        try:
            st = self._registry.stat()
        except OSError:            # no registry yet: pool is empty, anchors only
            self._entries = {}
            self._mtime = None
            return
        if self._mtime is not None and st.st_mtime == self._mtime:
            return
        try:
            blob = json.loads(self._registry.read_text())
            self._entries = {k: PoolEntry(**v) for k, v in blob["entries"].items()}
            self._mtime = st.st_mtime
        except Exception as exc:   # mid-write/corrupt: keep last-known entries
            self._note(f"league.json read failed ({exc}); keeping last-known pool")

    def _note(self, msg: str) -> None:
        if not self._warned:
            print(f"[league-seats] {msg}", file=sys.stderr, flush=True)
            self._warned = True

    # -- bot construction --------------------------------------------------------

    def _frozen_bot(self, path: str) -> Optional[ScriptedBot]:
        policy = self._policies.get(path)
        if policy is None:
            try:
                from bot.inference import NumpyPolicy
                policy = NumpyPolicy.load(path)
            except Exception as exc:
                self._note(f"snapshot load failed for {path} ({exc}); seating an anchor instead")
                return None
            self._policies[path] = policy
            while len(self._policies) > self.policy_cache_size:
                self._policies.popitem(last=False)
        else:
            self._policies.move_to_end(path)
        return FrozenPolicyBot(path, policy=policy)

    # -- the per-game draw --------------------------------------------------------

    def sample(
        self, stream_index: int, game_counter: int, seats: Tuple[int, ...],
    ) -> Tuple[Dict[int, ScriptedBot], Dict[int, str]]:
        """Fill ``seats`` (the non-policy pids) for one new game.

        Returns ``(bots, plan)`` where ``plan[pid]`` is the seat-spec string
        recorded in the game's :class:`GameResult`.
        """
        self._refresh()
        rng = random.Random(
            game_seed(self.base_seed, stream_index, game_counter) ^ _LEAGUE_SEED_SALT
        )
        self._pool_view.entries = dict(self._entries)
        picks = self._pool_view.sample_opponents(len(seats), rng, total_seats=8)

        path_to_sid = {e.path: sid for sid, e in self._entries.items()}
        bots: Dict[int, ScriptedBot] = {}
        plan: Dict[int, str] = {}
        for pid, (kind, ref) in zip(sorted(seats), picks):
            if kind == ANCHOR:
                bots[pid] = make_bot(ref)
                plan[pid] = f"anchor:{ref}"
            else:
                bot = self._frozen_bot(ref)
                if bot is None:  # unreadable snapshot: degrade to an anchor seat
                    name = self.anchors[rng.randrange(len(self.anchors))]
                    bots[pid] = make_bot(name)
                    plan[pid] = f"anchor:{name}"
                else:
                    bots[pid] = bot
                    plan[pid] = f"pool:{path_to_sid.get(ref, Path(ref).stem)}"
        return bots, plan


class ResultsWriter:
    """Append :class:`GameResult` lines to ``<league_dir>/results/<name>.jsonl``."""

    def __init__(self, league_dir: str | Path, name: str) -> None:
        self.path = Path(league_dir) / "results" / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def __call__(self, result: GameResult) -> None:
        if self._fh is None:
            self._fh = open(self.path, "a")
        self._fh.write(result.to_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


__all__ = ["GameResult", "LeagueSeatSampler", "ResultsWriter", "POLICY_SPEC"]
