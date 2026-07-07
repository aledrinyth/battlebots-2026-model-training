# mypy: disable-error-code=return-value

from pydantic import TypeAdapter, ValidationError
from engine.config.io_config import (
    CORE_DIRECTORY,
    CUMULATIVE_TIMEOUT_SECONDS,
    OPEN_PIPE_TIMEOUT_SECONDS,
    PIPE_LEN_DELIM,
    MAX_CHARACTERS_READ,
    READ_CHUNK_SIZE,
    TIMEOUT_SECONDS,
)


from engine.interface.io.exceptions import (
    BrokenPipeException,
    CumulativeTimeoutException,
    InvalidMessageException,
    InvalidMoveException,
    TimeoutException,
)

from engine.interface.io.input_validator import MoveValidator
from engine.interface.io.censor_event import CensorEvent

from lib.interface.events.typing import EventType
from lib.interface.queries.query_move import QueryMovePlayer
from lib.interface.queries.typing import QueryType
from lib.interface.queries.base_query import BaseQuery
from lib.interface.events.moves.typing import MoveType
from lib.interface.events.moves.move_player import MovePlayer

from io import TextIOWrapper
import json
from math import log10, floor
from signal import SIGALRM, alarm, signal
from time import time
from itertools import islice
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NoReturn,
    Optional,
    ParamSpec,
    Type,
    TypeVar,
    Union,
    final,
)

# performance boost on deserializing unions.
cached_type_adapters: dict[frozenset[str], TypeAdapter[Any]] = {}

if TYPE_CHECKING:
    from engine.state.game_state import GameState


class InvalidMoveError(ValueError):
    def __init__(self, message: str, move: MoveType):
        super().__init__(message)
        self.invalid_move = move


P = ParamSpec("P")
T1 = TypeVar("T1")


def handle_sigpipe(fn: Callable[P, T1]) -> Callable[P, T1]:
    """Decorator to trigger ban if the player closes the from_engine pipe."""

    def dfn(*args: P.args, **kwargs: P.kwargs) -> T1:
        self: "PlayerConnection" = args[0]  # type: ignore
        query: Optional[QueryType] = None
        if len(args) >= 2 and isinstance(args[1], BaseQuery):
            query = args[1]  # type: ignore
        try:
            result = fn(*args, **kwargs)
        except BrokenPipeError:
            raise BrokenPipeException(
                self.player_id, "You closed 'from_engine.pipe'.", query
            )

        return result

    return dfn


def handle_invalid(fn: Callable[P, T1]) -> Callable[P, T1]:
    """Decorator to trigger ban if the player sends an invalid, but syntactically correct message."""

    def dfn(*args: P.args, **kwargs: P.kwargs) -> T1:
        self: "PlayerConnection" = args[0]  # type: ignore
        try:
            result = fn(*args, **kwargs)
        except ValidationError as e:
            raise InvalidMessageException(
                self.player_id,
                "You sent an invalid message to the game engine.",
                json.loads(e.json()),
            )
        except InvalidMoveError as e:
            raise InvalidMoveException(self.player_id, str(e), e.invalid_move)
        return result

    return dfn


def time_limited(
    error_message: str = "You took too long to respond.", initial: bool = False
) -> Callable[[Callable[P, T1]], Callable[P, T1]]:
    """Decorator to trigger ban if the player takes too long to respond."""

    def dfn1(fn: Callable[P, T1]) -> Callable[P, T1]:
        def dfn2(*args: P.args, **kwargs: P.kwargs) -> T1:
            self: "PlayerConnection" = args[0]  # type: ignore
            query: Optional[QueryType] = None
            if len(args) >= 2 and isinstance(args[1], BaseQuery):
                query = args[1]  # type: ignore

            def on_timeout_alarm(*_) -> NoReturn:  # type: ignore
                raise TimeoutException(self.player_id, error_message, query)

            signal(SIGALRM, on_timeout_alarm)

            alarm(OPEN_PIPE_TIMEOUT_SECONDS if initial else TIMEOUT_SECONDS)
            start = time()

            result = fn(*args, **kwargs)

            end = time()
            alarm(0)

            self._cumulative_time += end - start
            if self._cumulative_time > CUMULATIVE_TIMEOUT_SECONDS:
                raise CumulativeTimeoutException(self.player_id, error_message, query)

            return result

        return dfn2

    return dfn1


T2 = TypeVar("T2", bound=MoveType)
T3 = TypeVar("T3", bound=MoveType)


@final
class PlayerConnection:
    def __init__(self, player_id: int) -> None:
        self.player_id: int = player_id
        self._to_engine_pipe: TextIOWrapper
        self._from_engine_pipe: TextIOWrapper
        self._cumulative_time: float = 0
        self._record_update_watermark: int = 0

        self._open_pipes()

    @time_limited(
        "You didn't open 'to_engine' for writing or 'from_engine.pipe' for reading in time.",
        initial=True,
    )
    def _open_pipes(self) -> None:
        self._to_engine_pipe = open(
            f"{CORE_DIRECTORY}/submission{self.player_id}/io/to_engine.pipe", "r"
        )
        self._from_engine_pipe = open(
            f"{CORE_DIRECTORY}/submission{self.player_id}/io/from_engine.pipe", "w"
        )

    def _send(self, data: str) -> None:
        self._from_engine_pipe.write(str(len(data)) + PIPE_LEN_DELIM)
        self._from_engine_pipe.write(data)
        self._from_engine_pipe.flush()

    def _receive(self) -> str:
        # Read size of message.
        buffer = bytearray()
        while len(buffer) < floor(log10(MAX_CHARACTERS_READ)) + 1 and (
            len(buffer) == 0 or buffer[-1] != ord(",")
        ):
            buffer.extend(self._to_engine_pipe.read(1).encode())

        if buffer[-1] == ord(","):
            size = int(buffer[0:-1].decode())
        else:
            raise InvalidMessageException(
                player_id=self.player_id,
                error_message="You send a message with a malformed message size.",
            )

        if size > MAX_CHARACTERS_READ:
            raise InvalidMessageException(
                player_id=self.player_id,
                error_message=f"You send a message that was too long, {size} > {MAX_CHARACTERS_READ} maximum.",
            )

        # Read message.
        buffer = bytearray()
        while len(buffer) < size:
            buffer.extend(
                bytearray(
                    self._to_engine_pipe.read(
                        min((size - len(buffer)), READ_CHUNK_SIZE)
                    ).encode()
                )
            )

        return buffer.decode()

    @handle_invalid
    @handle_sigpipe
    @time_limited()
    def _query_move(
        self, query: QueryType, response_type: Type[T2], validator: MoveValidator
    ) -> T2:
        self._send(query.model_dump_json())

        move = response_type.model_validate_json(self._receive())
        try:
            validator.validate(move, query, self.player_id)
        except ValueError as e:
            raise InvalidMoveError(str(e), move)
        return move  # ignore: type

    @handle_invalid
    @handle_sigpipe
    @time_limited()
    def _query_move_union(
        self,
        query: QueryType,
        response_type_1: Type[T2],
        response_type_2: Type[T3],
        validator: MoveValidator,
    ) -> Union[T2, T3]:
        self._send(query.model_dump_json())

        types = frozenset([response_type_1.__name__, response_type_2.__name__])
        if types in cached_type_adapters:
            adapter = cached_type_adapters[types]
        else:
            cached_type_adapters[types] = TypeAdapter[Any](
                Union[response_type_1, response_type_2]
            )
            adapter = cached_type_adapters[types]

        move = adapter.validate_json(self._receive())
        try:
            validator.validate(move, query, self.player_id)
        except ValueError as e:
            raise InvalidMoveError(str(e), move)
        return move  # type: ignore[no-any-return]

    def _get_record_update_dict(
        self, state: "GameState", censor: CensorEvent
    ) -> dict[int, EventType]:
        result = {}
        public_index = 0
        for _, event in islice(
            enumerate(state.event_history),
            self._record_update_watermark,
            None,
        ):
            censored_event = censor.censor(event, self.player_id)
            if censored_event is not None:
                result[public_index] = censored_event
                public_index += 1
        self._record_update_watermark = len(state.event_history)
        return result

    def query_move_player(
        self, state: "GameState", validator: MoveValidator, censor: CensorEvent
    ) -> MovePlayer:
        query = QueryMovePlayer(
            update=self._get_record_update_dict(state, censor),
            round=state.round,
            rankings=state.get_rankings(),
            you=state.players[self.player_id]._to_model(),
            view_center=state.get_player_view_center(self.player_id),
            vision_size=state.get_player_vision_size(self.player_id),
            visible_food=state.get_public_visible_food(self.player_id),
            visible_blobs=state.get_public_visible_blobs(self.player_id),
            visible_viruses=state.get_public_visible_viruses(self.player_id),
        )
        return self._query_move(query, MovePlayer, validator)
