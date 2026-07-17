"""P6.2 — frozen-snapshot opponents: past policies as league seats.

A pool opponent is a **frozen** past checkpoint playing exactly as it would at
match time: the exported actor-path ``weights.npz`` through the pure-NumPy
inference (`bot.inference.NumpyPolicy`, parity-proven at P5.3), greedy
decode, own `Featurizer`, own recurrent hidden. Wrapping that as a
:class:`~train.opponents.scripted.ScriptedBot` means a league seat needs *no*
changes to `Actor`/`GameRunner`/the slab transport — pool seats ride the
scripted seam, and actor processes stay torch-free.
"""

from __future__ import annotations

from pathlib import Path

from bot.adapters.act_v1 import ActionAdapter
from bot.featurizer import Featurizer
from bot.inference import NumpyPolicy

from train.opponents.scripted import ScriptedBot


class FrozenPolicyBot(ScriptedBot):
    """A past snapshot acting greedily via the match-time NumPy path."""

    def __init__(self, weights_path: str | Path, policy: NumpyPolicy | None = None) -> None:
        self.weights_path = str(weights_path)
        self.name = f"frozen:{Path(weights_path).stem}"
        # ``policy`` lets a caller share one loaded weight set across several
        # seats (PFSP samples with replacement, so the same snapshot can hold
        # multiple seats per game). NumpyPolicy.act is pure — the recurrent
        # hidden lives *here*, per seat — so sharing weights is safe.
        self._policy = policy if policy is not None else NumpyPolicy.load(self.weights_path)
        self._featurizer = Featurizer()
        self._hidden = self._policy.initial_hidden()

    def reset(self) -> None:
        """Fresh episode: clear velocity tracking and the recurrent state."""
        self._featurizer.reset()
        self._hidden = self._policy.initial_hidden()

    def act(self, query):  # noqa: ANN001 - QueryMovePlayer
        obs = self._featurizer.featurize(query)  # flags is_reset on round gaps
        action, self._hidden = self._policy.act(obs, self._hidden)
        return ActionAdapter.to_move(query.you.player_id, action)


class RailedPolicyBot(FrozenPolicyBot):
    """A frozen policy with the Track-2 deploy-time defensive rails applied.

    Same NumPy forward as :class:`FrozenPolicyBot`, then ``bot.rails.apply_rails``
    vetoes suicidal moves — the *exact* override the shipped bot uses. Seat it as
    an evaluate ``--scripted-subject`` to gate the rails against the un-railed
    policy on the meta field (identical seeds ⇒ paired).
    """

    def __init__(self, weights_path, policy: NumpyPolicy | None = None) -> None:
        super().__init__(weights_path, policy=policy)
        self.name = f"railed:{Path(weights_path).stem}"

    def act(self, query):  # noqa: ANN001 - QueryMovePlayer
        from bot.rails import apply_rails

        obs = self._featurizer.featurize(query)
        action, self._hidden = self._policy.act(obs, self._hidden)
        action = apply_rails(obs, action)
        return ActionAdapter.to_move(query.you.player_id, action)


__all__ = ["FrozenPolicyBot", "RailedPolicyBot"]
