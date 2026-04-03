import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from ..config import SQLITE_DB_PATH
from ._sqlite import normalize_sqlite_path, open_connection


@dataclass
class StorageTransaction:
    sqlite_path: Path = SQLITE_DB_PATH
    row_factory: object | None = None
    _connection: sqlite3.Connection | None = field(init=False, default=None)

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Storage transaction is not active.")
        return self._connection

    def __enter__(self) -> "StorageTransaction":
        from .schema import init_storage

        self.sqlite_path = normalize_sqlite_path(self.sqlite_path)
        init_storage(self.sqlite_path)
        self._connection = open_connection(
            self.sqlite_path,
            row_factory=self.row_factory,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._connection is None:
            return False
        try:
            if exc_type is None:
                self._connection.commit()
            else:
                self._connection.rollback()
        finally:
            self._connection.close()
            self._connection = None
        return False

    def execute(self, sql: str, parameters=None):
        return self.connection.execute(sql, parameters or [])

    def executemany(self, sql: str, seq_of_parameters):
        return self.connection.executemany(sql, seq_of_parameters)

    def executescript(self, sql_script: str):
        return self.connection.executescript(sql_script)

    def read_frame(self, query: str, params=None) -> pd.DataFrame:
        return pd.read_sql_query(query, self.connection, params=params)

    def write_frame(
        self,
        table_name: str,
        frame: pd.DataFrame,
        *,
        if_exists: Literal["fail", "replace", "append"] = "replace",
        index: bool = False,
    ) -> None:
        frame.to_sql(table_name, self.connection, if_exists=if_exists, index=index)


def transaction(sqlite_path=SQLITE_DB_PATH, *, row_factory=None) -> StorageTransaction:
    return StorageTransaction(sqlite_path=sqlite_path, row_factory=row_factory)
