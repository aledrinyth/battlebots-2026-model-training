import math

from lib.interface.queries.typing import QueryType, QueryTypeAdapter
from lib.interface.events.moves.typing import MoveType

MAX_CHARACTERS_READ = 1000000
READ_CHUNK_SIZE = 1024


class Connection:
    def __init__(self) -> None:
        self._to_engine_pipe = open("./io/to_engine.pipe", "w")
        self._from_engine_pipe = open("./io/from_engine.pipe", "r")

    def _send(self, data: str) -> None:
        self._to_engine_pipe.write(str(len(data)) + ",")
        self._to_engine_pipe.write(data)
        self._to_engine_pipe.flush()

    def _receive(self) -> str:
        # Read size of message.
        buffer = bytearray()
        while len(buffer) < math.floor(math.log10(MAX_CHARACTERS_READ)) + 1 and (
            len(buffer) == 0 or buffer[-1] != ord(",")
        ):
            chunk = self._from_engine_pipe.read(1)
            if chunk == "":
                raise EOFError("Engine closed the pipe.")
            buffer.extend(chunk.encode())

        if buffer[-1] == ord(","):
            size = int(buffer[0:-1].decode())
        else:
            print(buffer)
            raise RuntimeError("Please send us a discord message with this error log.")

        if size > MAX_CHARACTERS_READ:
            raise RuntimeError("Please send us a discord message with this error log.")

        # Read message.
        buffer = bytearray()
        while len(buffer) < size:
            chunk = self._from_engine_pipe.read(
                min((size - len(buffer)), READ_CHUNK_SIZE)
            )
            if chunk == "":
                raise EOFError("Engine closed the pipe.")
            buffer.extend(bytearray(chunk.encode()))

        return buffer.decode()

    def get_next_query(self) -> QueryType:
        return QueryTypeAdapter.model_validate_json(self._receive()).root

    def send_move(self, move: MoveType) -> None:
        self._send(move.model_dump_json())
