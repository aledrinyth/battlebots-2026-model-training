"""Shared-memory transport slabs (P3.2).

The custom actor–learner substrate chosen at P3.1 moves observations "over shared
memory" from ``num_actors`` CPU actor processes to the single learner process.
This module is that shared memory: three fixed-shape ``float32`` slabs indexed by
``[actor, seat_slot]`` (the fixed-slot layout — one stable row per
(actor, policy-seat) *stream*, constant every round), plus the tiny lifecycle
code to create them in the learner and *attach* to them by name in a spawned
actor.

Slabs (R rows = ``num_actors × seats_per_actor`` overall):

* ``obs``   ``(num_actors, seats, OBS_WIDTH)`` — the flattened featurized frame
  each actor writes for each of its policy seats (zeros for a dead seat).
* ``meta``  ``(num_actors, seats, META_W)`` — per-row scalars: liveness, reset
  flag, and the *previous* action's reward/done (completed after the actor
  stepped — see :data:`META_FIELDS`).
* ``act``   ``(num_actors, seats, 2)`` — the learner writes back
  ``(heading_bin, split)`` for each row.

Because macOS uses the *spawn* start method, children cannot inherit the
:class:`SharedMemory` objects; the learner passes the slab **names** (+ shapes)
to each actor, which calls :meth:`SlabSet.attach`. The creator
(:meth:`SlabSet.create`) is the only one that ``unlink``\\s.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, Tuple

import numpy as np

from train.harness.flat_obs import OBS_WIDTH

# --- meta row layout --------------------------------------------------------
# Per-row scalars exchanged alongside the obs vector. "prev_*" describe the
# transition the row's *last* action produced (known only after the actor
# stepped), so the learner completes step t-1's buffer entry when it reads them
# at round t. "active" is 1.0 when this policy seat is alive *this* round (it
# produced an obs and needs an action); "is_reset" is its first frame after a
# game start or a respawn gap.
META_FIELDS = ("active", "is_reset", "prev_reward", "prev_done", "prev_acted")
META_W = len(META_FIELDS)
_MI: Dict[str, int] = {name: i for i, name in enumerate(META_FIELDS)}

ACT_W = 2  # (heading_bin, split)


def _idx(name: str) -> int:
    """Column index of a meta field (so callers never hard-code positions)."""
    return _MI[name]


@dataclass
class _Slab:
    """One named shared-memory block viewed as a numpy array."""

    shm: shared_memory.SharedMemory
    arr: np.ndarray
    name: str
    created: bool

    def close(self) -> None:
        # Drop the numpy view first so the buffer has no exports, then close.
        self.arr = None  # type: ignore[assignment]
        try:
            self.shm.close()
        except Exception:
            pass
        if self.created:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


def _make(shape: Tuple[int, ...], name: str | None) -> _Slab:
    nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    if name is None:
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        created = True
    else:
        shm = shared_memory.SharedMemory(name=name)
        created = False
    arr = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
    if created:
        arr[:] = 0.0
    return _Slab(shm=shm, arr=arr, name=shm.name, created=created)


@dataclass
class SlabNames:
    """The three slab names + shape params an actor needs to attach."""

    obs: str
    meta: str
    act: str
    num_actors: int
    seats: int


class SlabSet:
    """The obs/meta/act shared-memory triplet for the whole run.

    Create it once in the learner (``SlabSet.create(cfg)``), pass ``.names()`` to
    each actor process, and each actor does ``SlabSet.attach(names)``. Every party
    calls :meth:`close` at the end; only the creator unlinks.
    """

    def __init__(self, obs: _Slab, meta: _Slab, act: _Slab, num_actors: int, seats: int) -> None:
        self._obs = obs
        self._meta = meta
        self._act = act
        self.num_actors = num_actors
        self.seats = seats

    # -- construction --------------------------------------------------------

    @classmethod
    def create(cls, num_actors: int, seats: int) -> "SlabSet":
        obs = _make((num_actors, seats, OBS_WIDTH), None)
        meta = _make((num_actors, seats, META_W), None)
        act = _make((num_actors, seats, ACT_W), None)
        return cls(obs, meta, act, num_actors, seats)

    @classmethod
    def attach(cls, names: SlabNames) -> "SlabSet":
        n, s = names.num_actors, names.seats
        obs = _make((n, s, OBS_WIDTH), names.obs)
        meta = _make((n, s, META_W), names.meta)
        act = _make((n, s, ACT_W), names.act)
        return cls(obs, meta, act, n, s)

    def names(self) -> SlabNames:
        return SlabNames(
            obs=self._obs.name,
            meta=self._meta.name,
            act=self._act.name,
            num_actors=self.num_actors,
            seats=self.seats,
        )

    # -- views ---------------------------------------------------------------

    @property
    def obs(self) -> np.ndarray:
        """``(num_actors, seats, OBS_WIDTH)`` flattened observations."""
        return self._obs.arr

    @property
    def meta(self) -> np.ndarray:
        """``(num_actors, seats, META_W)`` per-row scalars (see META_FIELDS)."""
        return self._meta.arr

    @property
    def act(self) -> np.ndarray:
        """``(num_actors, seats, 2)`` ``(heading_bin, split)`` written by learner."""
        return self._act.arr

    def meta_col(self, name: str) -> np.ndarray:
        """A ``(num_actors, seats)`` view of one meta field across all rows."""
        return self._meta.arr[:, :, _idx(name)]

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._obs.close()
        self._meta.close()
        self._act.close()


__all__ = ["SlabSet", "SlabNames", "META_FIELDS", "META_W", "ACT_W"]
