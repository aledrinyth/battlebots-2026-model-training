from lib.interface.queries.base_query import BaseQuery

from typing import Literal


class QueryPushPenguins(BaseQuery):
    query_type: Literal["move_penguins"] = "move_penguins"