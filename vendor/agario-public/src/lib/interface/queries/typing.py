from lib.interface.queries.query_move import QueryMovePlayer

from pydantic import Field, RootModel
from typing import Annotated, TypeAlias, Union


QueryType: TypeAlias = Annotated[
    Union[
        QueryMovePlayer,
    ],
    Field(discriminator="query_type"),
]


class QueryTypeAdapter(RootModel[QueryType]):
    pass
