from typing import Literal, Mapping, Sequence
from pydantic import BaseModel

from lib.interface.io.ban_type import BanType


class GameBanResult(BaseModel):
    result_type: Literal["PLAYER_BANNED"] = "PLAYER_BANNED"
    ban_type: BanType
    player: int
    reason: str


class GameSuccessResult(BaseModel):
    result_type: Literal["SUCCESS"] = "SUCCESS"
    ranking: Sequence[int]
    final_masses: Mapping[int, float]


class GameCancelledResult(BaseModel):
    result_type: Literal["CANCELLED"] = "CANCELLED"
    reason: str


class GameCrashedResult(BaseModel):
    result_type: Literal["CRASHED"] = "CRASHED"
    reason: str
