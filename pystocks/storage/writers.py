import pandas as pd

from ..config import SQLITE_DB_PATH
from .txn import StorageTransaction, transaction


def replace_table(
    table_name: str,
    frame: pd.DataFrame,
    *,
    sqlite_path=SQLITE_DB_PATH,
    tx: StorageTransaction | None = None,
    index: bool = False,
) -> None:
    if tx is not None:
        tx.write_frame(table_name, frame, if_exists="replace", index=index)
        return

    with transaction(sqlite_path) as managed_tx:
        managed_tx.write_frame(table_name, frame, if_exists="replace", index=index)
