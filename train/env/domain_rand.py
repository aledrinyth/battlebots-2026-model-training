"""P6.1 — training-only domain randomization (food density).

The playbook's pipeline insurance for S1+ curriculum stages: vary the food
count per episode (±20 % by default when enabled) so the policy can't
overfit the exact 160-food carpet of the grader's arena. Spawn-*seed*
randomization needs no machinery — every episode already gets a distinct
seed (``game_runner.game_seed``); this module supplies the density knob.

Same discipline as :mod:`train.env.fast_vision`:

* **Training-only.** Installed by :class:`~train.env.wrapper.WrapperEnv` only
  when a nonzero ``dr_food_density`` is configured. Match, eval, demos,
  parity fixtures, and the compliance harness never touch it.
* **Bit-identical when off.** The patched ``_ensure_food_count`` delegates to
  the engine's original method unless the instance carries a DR target, so an
  installed-but-unconfigured process behaves exactly like the pristine
  engine (asserted in ``tests/test_domain_rand.py``).

Mechanics: ``GameState.__init__`` spawns the initial food itself, so the
per-episode target is handed over via a module-level *pending* value set by
``WrapperEnv.reset`` immediately before ``GameState()`` — the first
``_ensure_food_count`` call (inside the constructor, same thread) claims it
onto the instance, and every later call (the mutator's per-round regrowth)
reads the instance attribute. Concurrent games in one process therefore
can't cross-contaminate: the pending window is a single synchronous
constructor call.
"""

from __future__ import annotations

from typing import Optional

_pending_food_count: Optional[int] = None
_original = None  # engine's _ensure_food_count while installed


def set_next_food_count(target: Optional[int]) -> None:
    """Arm the food-count target for the next ``GameState`` constructed."""
    global _pending_food_count
    _pending_food_count = target


def install() -> None:
    """Patch ``GameState._ensure_food_count`` (idempotent)."""
    global _original
    if _original is not None:
        return
    from engine.state.game_state import GameState

    _original = GameState._ensure_food_count

    def _ensure_food_count_dr(self):  # noqa: ANN001 - engine instance
        global _pending_food_count
        if not hasattr(self, "_dr_food_count"):
            self._dr_food_count = _pending_food_count
            _pending_food_count = None
        target = self._dr_food_count
        if target is None:  # no DR for this game — engine's own path, bit-identical
            return _original(self)
        spawned = []
        while len(self.map.foods) < target:
            spawned.append(self.map.spawn_food())
        return spawned

    GameState._ensure_food_count = _ensure_food_count_dr


def uninstall() -> None:
    """Restore the engine's original method (tests / hygiene)."""
    global _original, _pending_food_count
    if _original is None:
        return
    from engine.state.game_state import GameState

    GameState._ensure_food_count = _original
    _original = None
    _pending_food_count = None


__all__ = ["install", "uninstall", "set_next_food_count"]
