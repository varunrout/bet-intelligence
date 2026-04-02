"""
DuckDB connection helpers for Football Market Intelligence System.

All database access should go through get_connection() or run_query().
Never open duckdb.connect() directly in application code.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb
import pandas as pd

from src.utils.config import DB_PATH


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Context manager that yields a DuckDB connection and commits on clean exit.

    Usage
    -----
    with get_connection() as con:
        con.execute("SELECT 1")
    """
    con = duckdb.connect(str(db_path))
    try:
        yield con
        con.commit()
    except Exception:
        try:
            con.rollback()
        except Exception:
            pass
        raise
    finally:
        con.close()


def run_query(sql: str, params: list | None = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Execute a SELECT query and return results as a DataFrame.

    Parameters
    ----------
    sql    : SQL string (may contain ? placeholders)
    params : list of positional parameter values, or None
    """
    with get_connection(db_path) as con:
        if params:
            result = con.execute(sql, params).fetchdf()
        else:
            result = con.execute(sql).fetchdf()
    return result


def execute_statement(sql: str, params: list | None = None, db_path: Path = DB_PATH) -> None:
    """
    Execute a non-SELECT statement (INSERT, UPDATE, CREATE, etc.).
    """
    with get_connection(db_path) as con:
        if params:
            con.execute(sql, params)
        else:
            con.execute(sql)


def table_exists(table_name: str, db_path: Path = DB_PATH) -> bool:
    """Return True if table exists in the database."""
    with get_connection(db_path) as con:
        result = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
    return result[0] > 0


def get_row_count(table_name: str, db_path: Path = DB_PATH) -> int:
    """Return the number of rows in a table."""
    with get_connection(db_path) as con:
        result = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return result[0]


def upsert_dataframe(
    df: pd.DataFrame,
    table_name: str,
    conflict_columns: list[str],
    db_path: Path = DB_PATH,
) -> int:
    """
    Insert rows from df into table_name.
    Rows that violate UNIQUE constraints on conflict_columns are skipped (INSERT OR IGNORE).

    Returns the number of rows inserted.
    """
    if df.empty:
        return 0

    before = get_row_count(table_name, db_path)

    with get_connection(db_path) as con:
        # Register df as a temporary view
        con.register("_stage", df)
        cols = ", ".join(df.columns)
        con.execute(
            f"INSERT OR IGNORE INTO {table_name} ({cols}) SELECT {cols} FROM _stage"
        )

    after = get_row_count(table_name, db_path)
    return after - before
