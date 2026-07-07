from lib.interface.queries.typing import QueryType
from lib.interface.events.moves.typing import MoveType
from lib.interface.queries.query_move import QueryMovePlayer
from helper.client_state import ClientSate
from helper.state_mutator import StateMutator
from helper.interface import Connection


class Game:
    def __init__(self) -> None:
        self.state = ClientSate()
        self.mutator = StateMutator(self.state)
        self.connection = Connection()

    def get_next_query(self) -> QueryType:
        query = self.connection.get_next_query()

        new_events_mark = len(self.state.event_history)
        for i, record in query.update.items():
            self.mutator.commit(i, record)
        self.state.new_events = new_events_mark

        match query:
            case QueryMovePlayer() as q:
                self.state.round = q.round
                self.state.rankings = list(q.rankings)
                self.state.view_center = q.view_center
                self.state.vision_size = q.vision_size
                self.state.visible_food = list(q.visible_food)
                self.state.visible_blobs = list(q.visible_blobs)
                self.state.visible_viruses = list(q.visible_viruses)
                self.state.me.sync_from_model(q.you)

        return query

    def send_move(self, move: MoveType) -> None:
        self.connection.send_move(move)
