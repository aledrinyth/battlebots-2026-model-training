"""Seat identity: ``team_id`` → bot name → interned ``bot_id`` (impl doc §6).

Tournament replays carry no bot names — only ``team_id``, which the
replay-stats audit showed is stable and behaviourally consistent across
matches. So the canonical identity is ``team_<team_id>``; an optional
catalog mapping (``{"53": "us_s1_champion", ...}``) overlays human names
when known. Interned ids are corpus-scoped and persisted in the dataset
index so shards written on different days agree.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional


class BotRegistry:
    """Stable ``name -> int`` interning, (de)serializable into index.json."""

    def __init__(
        self,
        names: Optional[Dict[str, int]] = None,
        catalog: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._name_to_id: Dict[str, int] = dict(names or {})
        self._catalog = {str(k): v for k, v in (catalog or {}).items()}

    def name_for_team(self, team_id: int) -> str:
        return self._catalog.get(str(team_id), f"team_{team_id}")

    def id_for_team(self, team_id: int) -> int:
        return self.intern(self.name_for_team(team_id))

    def intern(self, name: str) -> int:
        bot_id = self._name_to_id.get(name)
        if bot_id is None:
            bot_id = len(self._name_to_id)
            self._name_to_id[name] = bot_id
        return bot_id

    def to_table(self) -> Dict[str, int]:
        return dict(self._name_to_id)

    @property
    def names(self) -> Dict[int, str]:
        return {v: k for k, v in self._name_to_id.items()}
