"""Actor — one in-process game, numpy/CPU only (P3.1).

An :class:`Actor` owns a single :class:`~train.env.wrapper.WrapperEnv`, a shared
:class:`~bot.featurizer.Featurizer` per policy-driven seat, and any scripted bots
filling the remaining seats. It is the unit the driver fans out over: N actors =
N concurrent games, and the learner batches every actor's alive policy seats into
one forward pass.

**No torch here — by design.** The actor is exactly the match-legal surface
(wrapper physics + shared featurizer + shared action adapter, all numpy), so in
P3.2 an actor can be pushed into a worker process that never imports torch and
streams featurized arrays to the central learner over shared memory. This module
proves that seam works in-process first.

Two facts the custom harness must handle, both handled here:

* **8-agent simultaneous FFA.** Every alive seat is queried and moved each round;
  the actor returns a *dict* of per-seat observations, not a single agent's.
* **Dead-player gaps.** A dead seat receives no query for 30 rounds
  (``RESPAWN_DELAY_ROUNDS``) and then respawns. The actor simply omits dead seats
  from :meth:`observe`; on respawn the featurizer sees the round-number jump,
  flags ``is_reset``, and zeros its velocity tracker (and the driver zeros that
  seat's GRU hidden state).
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

from bot.adapters.act_v1 import ActionAdapter, CanonicalAction
from bot.adapters.obs_v1 import CanonicalObs
from bot.featurizer import Featurizer
from lib.interface.events.moves.move_player import MovePlayer
from train.env.wrapper import WrapperEnv
from train.opponents.scripted import ScriptedBot


class Actor:
    """A single game with a mix of policy-driven and scripted seats.

    Args:
        seed: wrapper seed (identical seed + identical moves ⇒ identical game).
        policy_seats: player ids driven by the central policy.
        scripted: player id → scripted bot for every non-policy seat. Together
            ``policy_seats`` and ``scripted.keys()`` must partition the roster.
        emit_update: passed through to the wrapper (``False`` = throughput
            fast-path; the featurizer derives velocities from frame diffs, not the
            update stream, so this is safe for training — see wrapper docstring).
    """

    def __init__(
        self,
        seed: int,
        policy_seats: Iterable[int],
        scripted: Optional[Mapping[int, ScriptedBot]] = None,
        emit_update: bool = False,
    ) -> None:
        self.seed = seed
        self.policy_seats = set(policy_seats)
        self.scripted: Dict[int, ScriptedBot] = dict(scripted or {})
        self.emit_update = emit_update

        overlap = self.policy_seats & set(self.scripted)
        if overlap:
            raise ValueError(f"seats both policy and scripted: {sorted(overlap)}")

        self.env = WrapperEnv(seed=seed, emit_update=emit_update)
        self._featurizers: Dict[int, Featurizer] = {
            pid: Featurizer() for pid in self.policy_seats
        }
        self._adapter = ActionAdapter()
        self._queries: Dict[int, object] = {}   # current alive-seat observations
        self._pending: Dict[int, CanonicalObs] = {}  # featurized policy obs this round
        self._done = False

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Start a fresh game; validate the seat partition against the roster."""
        for f in self._featurizers.values():
            f.reset()
        for bot in self.scripted.values():
            bot.reset()

        self._queries = self.env.reset()
        roster = set(self.env.state.players.keys())
        assigned = self.policy_seats | set(self.scripted)
        if assigned != roster:
            raise ValueError(
                f"seat assignment {sorted(assigned)} does not partition roster "
                f"{sorted(roster)}"
            )
        self._pending = {}
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    # -- the actor↔learner exchange ----------------------------------------

    def observe(self) -> Dict[int, CanonicalObs]:
        """Featurize the current frame for every **alive policy seat**.

        Idempotent within a round (results are cached until the next
        :meth:`step`). Dead policy seats are simply absent from the result — that
        is the dead-player gap, handled by omission.
        """
        if self._done:
            return {}
        if not self._pending:
            self._pending = {
                pid: self._featurizers[pid].featurize(self._queries[pid])
                for pid in self._queries
                if pid in self.policy_seats
            }
        return self._pending

    def alive_policy_seats(self) -> set[int]:
        """Alive seats driven by the policy this round (the batch's contribution)."""
        return {pid for pid in self._queries if pid in self.policy_seats}

    def query(self, pid: int):
        """The raw ``QueryMovePlayer`` for an alive seat this round (or ``None``).

        Read-only view of the current frame, for callers that need the wrapper's
        observation alongside :meth:`observe`'s featurized one — e.g. an expert
        demonstrator scoring the same frame the featurizer just encoded
        (P3.5 Track D / the v3 replay converter's parity gate).
        """
        return self._queries.get(pid)

    def step(self, policy_actions: Mapping[int, CanonicalAction]) -> bool:
        """Apply one round and advance to the next frame.

        Args:
            policy_actions: chosen action per alive policy seat (exactly the keys
                :meth:`observe` returned). Scripted seats act autonomously.

        Returns:
            ``done`` — ``True`` once the game has ended.
        """
        if self._done:
            raise RuntimeError("step() on a finished actor; call reset().")

        moves: Dict[int, MovePlayer] = {}
        for pid, query in self._queries.items():
            if pid in self.policy_seats:
                action = policy_actions.get(pid)
                if action is None:
                    raise ValueError(
                        f"missing action for alive policy seat {pid} "
                        f"at round {self.env.state.round}"
                    )
                moves[pid] = self._adapter.to_move(pid, action)
            else:
                moves[pid] = self.scripted[pid].act(query)

        self._queries, self._done = self.env.step(moves)
        self._pending = {}
        return self._done

    # -- terminal info ------------------------------------------------------

    def rankings(self) -> list[int]:
        return self.env.rankings()

    def final_masses(self) -> Dict[int, float]:
        return self.env.final_masses()


__all__ = ["Actor"]
