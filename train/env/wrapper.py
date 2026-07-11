"""In-process env wrapper (P1.2) — the authoritative Python physics, callable
in-process at maximum speed, with **no** subprocesses, pipes, pydantic wire
(de)serialization, or the engine's 0.1 s/round sleep.

Design (see `notes/mechanics-spec.md` for the rules this exposes):

* We instantiate the vendored engine's own `GameState` + `StateMutator` +
  `CensorEvent` **directly** and drive `commit_round` ourselves. All physics is
  therefore *literally the competition engine* — the wrapper adds only the
  scheduling the live `GameEngine.run_game` would do (turn order, round counter,
  game-start events) and the query construction `PlayerConnection` would do.
  This is what P1.3 proves equivalent to the real pipe runner.

* `reset()` seeds Python's global `random` (the single RNG the engine draws spawn
  / food / virus / respawn positions and turn order from), builds a fresh game,
  and returns per-player observations for round 0.

* `step(moves)` applies one simultaneous round via `commit_round` and returns the
  next round's observations plus a `done` flag. Only **alive** players get an
  observation (dead ⇒ no observation), exactly as the engine only queries alive
  players.

* Observations are real `QueryMovePlayer` objects — byte-for-byte the same object
  the engine serializes onto the wire — so the shared featurizer (P2.2) consumes
  identical input in training and at match time.

The engine reads `input/catalog.json` from `CORE_DIRECTORY` at `GameState`
construction; since we run headless with no grader workspace, we materialize a
tiny catalog in a temp dir once and point the engine's module-global at it.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from typing import Dict, Mapping, Tuple

# The engine captures CORE_DIRECTORY as a module global at import; patch *that*
# name (not io_config) because game_state.py did `from ... import CORE_DIRECTORY`.
import engine.state.game_state as _gs_mod

from engine.interface.io.censor_event import CensorEvent
from engine.state.game_state import GameState
from engine.state.state_mutator import StateMutator

from train.env import fast_vision

from lib.config.arena import (
    MAX_ROUNDS,
    NUM_PLAYERS,
    TURN_DURATION_SECONDS,
    VISION_SIZE,
)
from lib.interface.events.event_food_spawned import EventFoodSpawned
from lib.interface.events.event_game_started import EventGameStarted
from lib.interface.events.event_player_won import EventPlayerWon
from lib.interface.events.event_virus_spawned import EventVirusSpawned
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel


# --- engine workspace (catalog.json) ---------------------------------------

_WORKSPACE: str | None = None


def _ensure_engine_workspace(num_players: int) -> None:
    """Create (once) a temp `input/catalog.json` and point the engine at it.

    The catalog is an FFA roster: player i on its own team i (`team_id` stays a
    per-player feature; see playbook). Idempotent within a process.
    """
    global _WORKSPACE
    if _WORKSPACE is None:
        ws = tempfile.mkdtemp(prefix="agario_wrapper_ws_")
        os.makedirs(os.path.join(ws, "input"), exist_ok=True)
        with open(os.path.join(ws, "input", "catalog.json"), "w") as f:
            json.dump([{"team_id": i} for i in range(num_players)], f)
        _WORKSPACE = ws
    # Re-assert on every call: other in-process engine users (e.g. tools/bench_env
    # measuring the real pipe runner) legitimately repoint this global, so the
    # wrapper self-heals it back to its own catalog rather than trusting it once.
    _gs_mod.CORE_DIRECTORY = _WORKSPACE


# --- move helpers -----------------------------------------------------------

def make_move(player_id: int, x: float, y: float, split: bool = False) -> MovePlayer:
    """Build a vector-heading `MovePlayer`. `(0, 0)` is a legal 'stay' heading."""
    return MovePlayer(
        player_id=player_id,
        direction=DirectionModel(x=float(x), y=float(y)),
        split=bool(split),
    )


def make_move_degrees(player_id: int, degrees: float, split: bool = False) -> MovePlayer:
    """Build a degrees-heading `MovePlayer`."""
    return MovePlayer(
        player_id=player_id,
        direction=DirectionModel(degrees=float(degrees)),
        split=bool(split),
    )


# --- the wrapper ------------------------------------------------------------

class WrapperEnv:
    """A deterministic, in-process, headless agar game.

    Lifecycle: ``obs = env.reset()`` → repeatedly ``obs, done = env.step(moves)``.
    ``moves`` is a mapping ``{player_id: MovePlayer}`` covering **exactly** the
    players alive in the last returned ``obs`` (an alive player missing a move is
    an error; a move for a dead player is ignored — matching the engine, which
    never queries dead players).

    Args:
        seed: seeds Python's global ``random`` at ``reset`` — identical seed +
            identical move sequence ⇒ identical trajectory.
        num_players: roster size (defaults to the engine's ``NUM_PLAYERS``).
        emit_update: when ``True`` (default, faithful) each observation carries the
            censored ``update`` event stream a live query would, via per-player
            watermarks + the engine's ``CensorEvent``. Set ``False`` for a
            throughput fast-path in training (the featurizer derives velocities
            from frame-diffs of ``visible_blobs``, not the ``update`` stream —
            P2.2); observations are then valid ``QueryMovePlayer`` objects with an
            empty ``update``.
    """

    def __init__(
        self,
        seed: int,
        num_players: int = NUM_PLAYERS,
        emit_update: bool = True,
    ) -> None:
        if num_players != NUM_PLAYERS:
            # GameState hard-codes NUM_PLAYERS internally; the catalog must match.
            raise ValueError(
                f"engine is built for NUM_PLAYERS={NUM_PLAYERS}; got {num_players}"
            )
        self.seed = seed
        self.num_players = num_players
        self.emit_update = emit_update

        self.state: GameState
        self.mutator: StateMutator
        self.censor: CensorEvent
        self.turn_order: list[int] = []
        self._watermarks: Dict[int, int] = {}
        self._done: bool = False

    # -- public API ---------------------------------------------------------

    def reset(self) -> Dict[int, QueryMovePlayer]:
        """Start a fresh game and return round-0 observations for all players."""
        _ensure_engine_workspace(self.num_players)
        fast_vision.install()  # P1.5: NumPy-vectorized vision (idempotent, bit-exact)

        random.seed(self.seed)
        self.state = GameState()  # round == -1; spawns players/food/viruses via RNG
        self.mutator = StateMutator(self.state)
        self.censor = CensorEvent(self.state)
        self._watermarks = {pid: 0 for pid in self.state.players}
        self._done = False

        # Mirror GameEngine.run_game's game-start bookkeeping so the `update`
        # streams (and event history) match the real runner exactly.
        self.turn_order = random.sample(list(self.state.players.keys()), k=self.num_players)
        self.state.turn_order = self.turn_order
        self.mutator.commit(
            EventGameStarted(
                turn_order=self.state.turn_order,
                arena_size=self.state.map.size,
                vision_size=VISION_SIZE,
                turn_duration_seconds=TURN_DURATION_SECONDS,
                max_rounds=MAX_ROUNDS,
                players=[p._to_model() for p in self.state.players.values()],
            )
        )
        self.mutator.commitPrivate(EventFoodSpawned(foods=list(self.state.map.foods)))
        self.mutator.commitPrivate(EventVirusSpawned(viruses=list(self.state.map.viruses)))

        self.state.round += 1  # -1 -> 0
        return self._observations()

    def step(
        self, moves: Mapping[int, MovePlayer]
    ) -> Tuple[Dict[int, QueryMovePlayer], bool]:
        """Apply one simultaneous round.

        Returns ``(observations, done)``. When ``done`` is ``True`` the game has
        ended (round 1399 committed) and ``observations`` is empty — no player is
        queried after the final round, matching the engine.
        """
        if self._done:
            raise RuntimeError("step() called after the game ended; call reset().")

        # Order moves by turn_order over the currently-alive players, exactly as
        # the engine collects them. Dead players are skipped (never queried).
        events: list[MovePlayer] = []
        for pid in self.turn_order:
            player = self.state.players[pid]
            if not player.alive:
                continue
            move = moves.get(pid)
            if move is None:
                raise ValueError(f"missing move for alive player {pid} at round {self.state.round}")
            if move.player_id != pid:
                raise ValueError(
                    f"move.player_id {move.player_id} != player {pid} (would be an invalid move)"
                )
            events.append(move)

        self.mutator.commit_round(events)

        if self.state.is_game_over():
            self._done = True
            rankings = self.state.get_rankings()
            self.mutator.commit(EventPlayerWon(player_id=rankings[0]))
            return {}, True

        self.state.round += 1
        return self._observations(), False

    @property
    def done(self) -> bool:
        return self._done

    def rankings(self) -> list[int]:
        """Player ids, best (largest radius) first — the terminal ranking."""
        return self.state.get_rankings()

    def final_masses(self) -> Dict[int, float]:
        """Per-player final mass (radius²) — the terminal score input."""
        return self.state.get_final_masses()

    # -- observation construction ------------------------------------------

    def _observations(self) -> Dict[int, QueryMovePlayer]:
        obs: Dict[int, QueryMovePlayer] = {}
        for pid in self.turn_order:  # engine query order; content is order-independent
            if self.state.players[pid].alive:
                obs[pid] = self._build_query(pid)
        return obs

    def _build_query(self, pid: int) -> QueryMovePlayer:
        # v2026.1.14: the live ``PlayerConnection`` sends the **public** views —
        # ``get_public_visible_*`` re-index food/virus/blob ids to per-frame
        # ordinals (the "leaked info" fix). Mirror that here so the trainer's
        # observation is byte-identical to the match's. (fast_vision patches the
        # public-blobs getter to keep it fast; the public food/virus getters reuse
        # the already-patched raw getters.)
        return QueryMovePlayer(
            update=self._update_dict(pid),
            round=self.state.round,
            rankings=self.state.get_rankings(),
            you=self.state.players[pid]._to_model(),
            view_center=self.state.get_player_view_center(pid),
            vision_size=self.state.get_player_vision_size(pid),
            visible_food=self.state.get_public_visible_food(pid),
            visible_blobs=self.state.get_public_visible_blobs(pid),
            visible_viruses=self.state.get_public_visible_viruses(pid),
        )

    def _update_dict(self, pid: int):
        """Censored slice of public event history since this player's watermark.

        Mirrors ``PlayerConnection._get_record_update_dict``: advances the
        watermark **only** for the queried (alive) player (so a player that was
        dead receives the whole backlog on respawn), and — as of v2026.1.14 —
        keys the surviving censored events by a **sequential public index**
        (``0, 1, 2…`` over the non-``None`` results) rather than the raw
        event-history index.
        """
        if not self.emit_update:
            return {}
        result = {}
        history = self.state.event_history
        public_index = 0
        for i in range(self._watermarks[pid], len(history)):
            censored = self.censor.censor(history[i], pid)
            if censored is not None:
                result[public_index] = censored
                public_index += 1
        self._watermarks[pid] = len(history)
        return result
