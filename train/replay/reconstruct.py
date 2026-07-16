"""Event stream → per-round authoritative ``GameState`` (impl doc §4).

The replay stores **global** truth per round; the bot only ever saw a local
crop. We rebuild the exact global state the engine held at each player's
decision point and let the engine's **own getters** produce the crop — no
physics re-simulation (spawn RNG makes that unreliable), just recorded
positions.

Two things changed since the impl doc was written (engine v2026.1.13) and are
handled here:

* **v2026.1.14 public views.** The live query sends ``get_public_visible_*``,
  which re-index ``food_id``/``virus_id``/``blob_id`` to per-frame ordinals in
  **container iteration order**. So container order is no longer irrelevant:
  ``build_query`` uses the public getters, and the walker maintains
  ``map.foods``/``map.viruses`` in the engine's true order (eaten → removed
  in place, spawned → appended; an insertion-ordered dict reproduces this
  exactly), and ``players`` in pid order (the engine's dict is
  ``{0..N-1}``).
* **Structural cross-checks from the replay-stats audit** are baked in: every
  round emits exactly ``n_players`` snapshots; move blocks and snapshot
  blocks must interleave 1:1; unknown event types are fatal. A malformed file
  raises :class:`ReplayError` and can never leak half-parsed rows.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:  # pragma: no cover - direct-script convenience
    sys.path.insert(0, str(REPO))

from engine.state.blob_state import BlobState
from engine.state.game_state import GameState
from engine.state.player_state import PlayerState
from lib.interact.map import Map
from lib.models.food_model import FoodModel
from lib.models.virus_model import VirusModel

from bot import spec
from train.replay.schema import ABORT_EVENT_TYPES, KNOWN_EVENT_TYPES


class ReplayError(ValueError):
    """The file does not match the engine's replay structure."""


def build_state() -> GameState:
    """A minimal ``GameState`` — no catalog.json read, no RNG spawns.

    Sets only what the vision getters and ``get_rankings`` read:
    ``players``, ``map``, ``round``, ``max_rounds``.
    """
    gs = GameState.__new__(GameState)
    gs.round = 0
    gs.max_rounds = spec.MAX_ROUNDS
    gs.map = Map()
    gs.players = {}
    return gs


def set_player_blobs(player: PlayerState, blob_dicts: List[dict]) -> None:
    """Overwrite a player's blobs from snapshot/header dicts.

    Empty list ⇒ ``PlayerState.alive == False`` (death) automatically. Blob
    ids here are the engine's **raw stable ids** (the private history is
    uncensored); public re-indexing happens later, inside the getters.
    """
    player.blobs = {
        b["blob_id"]: BlobState(
            blob_id=b["blob_id"],
            x=b["pos"][0],
            y=b["pos"][1],
            radius=b["radius"],
            merge_cooldown=b.get("merge_cooldown", 0),
        )
        for b in blob_dicts
    }


def rebuild_map(
    gs: GameState,
    food: Dict[int, Tuple[float, float]],
    viruses: Dict[int, Tuple[Tuple[float, float], float]],
) -> None:
    """Materialize the tracked food/virus dicts into ``gs.map``, preserving
    the dicts' insertion order (== the engine's list order — public ids
    depend on it). ``model_construct`` skips pydantic validation: the fields
    are already exact-typed, and this runs 1,400× per replay."""
    gs.map.foods = [
        FoodModel.model_construct(food_id=fid, pos=pos)
        for fid, pos in food.items()
    ]
    gs.map.viruses = [
        VirusModel.model_construct(virus_id=vid, pos=pos, radius=radius)
        for vid, (pos, radius) in viruses.items()
    ]


class Query:
    """The seven fields ``Featurizer.featurize`` reads (duck-typed)."""

    __slots__ = (
        "round",
        "view_center",
        "vision_size",
        "you",
        "visible_food",
        "visible_blobs",
        "visible_viruses",
    )


def build_query(gs: GameState, pid: int, rnd: int) -> Query:
    """Assemble a decision-point query using the engine's own crop.

    Mirrors ``WrapperEnv._build_query`` / the live ``PlayerConnection``
    (v2026.1.14+): **public** re-indexed food/virus/blob views, raw
    ``_to_model`` for ``you``. The ``view_center`` clamp subtlety
    (``min(vision_size, map.size)`` while membership uses the uncapped
    size) comes free by reusing the getters.
    """
    q = Query()
    q.round = rnd
    q.view_center = gs.get_player_view_center(pid)
    q.vision_size = gs.get_player_vision_size(pid)
    q.you = gs.players[pid]._to_model()
    q.visible_food = gs.get_public_visible_food(pid)
    q.visible_blobs = gs.get_public_visible_blobs(pid)
    q.visible_viruses = gs.get_public_visible_viruses(pid)
    return q


class Walker:
    """Single forward pass over one replay's events, maintaining the
    authoritative state and yielding each decision point.

    Usage::

        w = Walker(events, source="match-x.json")
        for rnd, move_event in w.decisions():
            # w.gs is the exact global state those players observed
            q = build_query(w.gs, move_event["player_id"], rnd)
        # after exhaustion: w.rounds, w.winner_pid, w.final_ranks(), ...

    The yielded state is **live** (mutated as the walk advances) — consume
    each decision before pulling the next.
    """

    def __init__(self, events: List[dict], source: str = "<memory>") -> None:
        if not events:
            raise ReplayError(f"{source}: empty event list")
        header = events[0]
        if header.get("event_type") != "event_game_started":
            raise ReplayError(
                f"{source}: first event is {header.get('event_type')!r}, "
                "expected event_game_started"
            )
        self.source = source
        self.events = events
        self.header = header
        self.engine_version: Optional[str] = header.get("engine_version")
        self.n_players = len(header["players"])
        if self.n_players < 2:
            raise ReplayError(f"{source}: fewer than 2 players")

        self.gs = build_state()
        # Engine order: pid-ascending dict — public blob views iterate it.
        for p in sorted(header["players"], key=lambda p: p["player_id"]):
            ps = PlayerState(p["player_id"], p["team_id"])
            set_player_blobs(ps, p["blobs"])
            self.gs.players[p["player_id"]] = ps
        if len(self.gs.players) != self.n_players:
            raise ReplayError(f"{source}: duplicate player_id in header")
        self.team_of = {
            p["player_id"]: p["team_id"] for p in header["players"]
        }

        # Tracked in engine order (insertion-ordered dicts).
        self.food: Dict[int, Tuple[float, float]] = {}
        self.viruses: Dict[int, Tuple[Tuple[float, float], float]] = {}

        self.rounds = 0                     # completed rounds (snapshot blocks)
        self.winner_pid: Optional[int] = None
        self.n_moves = 0
        #: pid -> final radius (0.0 when dead) — for engine-exact ranking.
        self._final_radius: Dict[int, float] = {
            p["player_id"]: p["radius"] for p in header["players"]
        }
        #: Eaten-pellet diagnostics (impl doc §7.1, corrected for the real
        #: tournament meta). ``eaten_in_vision`` — pellet inside the eater's
        #: plain pre-move window; **informational only**: eating is
        #: per-blob while vision is centroid-anchored, so giants (blob
        #: radius ~ √mass vs window growth ~ ^0.4 above Σr = 12), split
        #: children (spawned a parent-radius away), and 16-blob players
        #: spread across the arena all legitimately eat "unseen" pellets.
        #: ``eaten_in_reach`` — pellet within the eater's *physical
        #: possibility bound*, from the audited intra-round kinematics
        #: (movement → decay → stabilise → viruses → stabilise → food):
        #: (a) ``2r + 4`` of a pre-round blob (split-child spawn ≤ 2r;
        #: speed 1.1 + eject 1.6 + attraction/separation + same-round food
        #: growth < 4); (b) that margin of the segment between two blobs
        #: (a merge survivor is their mass-weighted average — cascades stay
        #: in the hull); (c) after a same-round virus shatter, the
        #: replacement grid around the virus: ``1.8·√(r_max²+VIRUS_SIZE²)
        #: + 4`` of the virus position. Must be 100 %: a pellet no
        #: reconstructed blob could have reached ⇒ misaligned state.
        self.eaten_in_vision = 0
        self.eaten_in_reach = 0
        self.eaten_total = 0

    # -- the walk -----------------------------------------------------------

    @staticmethod
    def _pellet_in_reach(
        pos: Tuple[float, float],
        blobs: List,
        anchors: List[Tuple[float, float, float]],
    ) -> bool:
        px, py = pos
        for b in blobs:
            m = 2.0 * b.radius + 4.0
            if (b.x - px) ** 2 + (b.y - py) ** 2 <= m * m:
                return True
        for vx, vy, reach in anchors:  # same-round shatter grids
            if (vx - px) ** 2 + (vy - py) ** 2 <= reach * reach:
                return True
        for i, a in enumerate(blobs):  # merge segments (rare fallback)
            for b in blobs[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                L2 = dx * dx + dy * dy
                t = 0.0 if L2 == 0.0 else max(
                    0.0, min(1.0, ((px - a.x) * dx + (py - a.y) * dy) / L2)
                )
                m = 2.0 * max(a.radius, b.radius) + 4.0
                if (px - a.x - t * dx) ** 2 + (py - a.y - t * dy) ** 2 <= m * m:
                    return True
        return False

    def decisions(self) -> Iterator[Tuple[int, dict]]:
        gs, food, viruses = self.gs, self.food, self.viruses
        n = self.n_players
        rnd = -1
        snaps = 0            # total snapshots seen
        snaps_in_block = 0
        prev_was_move = False
        #: pid -> [(virus_x, virus_y, reach)] for viruses consumed this round
        shatter_anchors: Dict[int, List[Tuple[float, float, float]]] = {}

        for e in self.events[1:]:
            et = e.get("event_type")
            if et == "move_player":
                if snaps_in_block:
                    raise ReplayError(
                        f"{self.source}: move_player inside a snapshot block "
                        f"(round {rnd})"
                    )
                if not prev_was_move:
                    rnd += 1
                    if snaps != rnd * n:
                        raise ReplayError(
                            f"{self.source}: round {rnd} opens after {snaps} "
                            f"snapshots, expected {rnd * n}"
                        )
                    rebuild_map(gs, food, viruses)
                    shatter_anchors = {}
                prev_was_move = True
                pid = e["player_id"]
                if not gs.players[pid].alive:
                    raise ReplayError(
                        f"{self.source}: move from dead player {pid} "
                        f"at round {rnd}"
                    )
                self.n_moves += 1
                yield rnd, e
                continue

            prev_was_move = False
            if et == "event_player_moved":
                pid = e["player_id"]
                set_player_blobs(gs.players[pid], e["blobs"])
                if not e["alive"]:
                    gs.players[pid].blobs = {}
                self._final_radius[pid] = e["radius"] if e["alive"] else 0.0
                snaps += 1
                snaps_in_block += 1
                if snaps_in_block == n:
                    snaps_in_block = 0
                    self.rounds = rnd + 1
            elif et == "event_food_spawned":
                for f in e["foods"]:
                    food[f["food_id"]] = tuple(f["pos"])
            elif et == "event_food_eaten":
                pid = e["player_id"]
                # Food resolves before player-eating in commit_round, so the
                # eater's blobs here are still the pre-round decision state.
                eater_blobs = list(gs.players[pid].blobs.values())
                for fid in e["food_ids"]:
                    pos = food.pop(fid, None)
                    if pos is None:
                        raise ReplayError(
                            f"{self.source}: round {rnd} eats food {fid} "
                            "that is not on the reconstructed map"
                        )
                    self.eaten_total += 1
                    if gs.player_can_see_point(pid, pos[0], pos[1]):
                        self.eaten_in_vision += 1
                    if self._pellet_in_reach(
                        pos, eater_blobs, shatter_anchors.get(pid, ())
                    ):
                        self.eaten_in_reach += 1
            elif et == "event_virus_spawned":
                for v in e["viruses"]:
                    viruses[v["virus_id"]] = (tuple(v["pos"]), v["radius"])
            elif et == "event_virus_consumed":
                viruses.pop(e["virus_id"], None)
                # The shatter grid materializes around the virus-hit blob
                # position (≈ the virus, which sat inside the blob); pieces
                # there eat food this same round, before any snapshot.
                pid = e["player_id"]
                r_max = max(
                    (b.radius for b in gs.players[pid].blobs.values()),
                    default=0.0,
                )
                vx, vy = e["virus_pos"]
                reach = 1.8 * (r_max**2 + spec.VIRUS_SIZE**2) ** 0.5 + 4.0
                shatter_anchors.setdefault(pid, []).append((vx, vy, reach))
            elif et == "event_player_eaten":
                # Redundant with the snapshots that close every round, but
                # keeps state exact mid-round (and costs nothing).
                target = gs.players[e["eaten_player_id"]]
                if not e["eaten_player_alive"]:
                    target.blobs = {}
                else:
                    target.blobs.pop(e["eaten_blob_id"], None)
            elif et == "event_player_won":
                self.winner_pid = e["player_id"]
            elif et in ABORT_EVENT_TYPES:
                raise ReplayError(f"{self.source}: aborted game ({et})")
            elif et not in KNOWN_EVENT_TYPES:
                raise ReplayError(f"{self.source}: unknown event type {et!r}")

        if snaps_in_block:
            raise ReplayError(
                f"{self.source}: trailing partial snapshot block "
                f"({snaps_in_block}/{n})"
            )
        if self.winner_pid is None:
            raise ReplayError(f"{self.source}: no event_player_won")

    # -- terminal state -----------------------------------------------------

    def final_ranks(self) -> Dict[int, int]:
        """Engine-exact final ranking: radius desc, ties → lower pid
        (``GameState.get_rankings`` sorts a pid-ordered dict stably; dead
        players have radius 0). Identical rule in v2026.1.13–.15; reproduces
        ``event_player_won`` on 20/20 audited tournament replays."""
        order = sorted(
            self._final_radius,
            key=lambda pid: (-self._final_radius[pid], pid),
        )
        ranks = {pid: i + 1 for i, pid in enumerate(order)}
        if self.winner_pid is not None and ranks[self.winner_pid] != 1:
            raise ReplayError(
                f"{self.source}: rank reconstruction disagrees with "
                f"event_player_won (computed winner "
                f"{order[0]}, engine says {self.winner_pid})"
            )
        return ranks
