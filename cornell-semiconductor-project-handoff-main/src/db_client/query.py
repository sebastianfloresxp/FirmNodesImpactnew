"""Query helpers built on top of pandas + SQLAlchemy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from .connection import get_engine


def run_query(
    sql: str,
    params: Mapping[str, Any] | None = None,
    *,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Execute *sql* and return a DataFrame of the full result set."""

    eng = engine or get_engine()
    query_params = params or None
    try:
        return pd.read_sql(sql, eng, params=query_params)
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Query failed: {exc}") from exc


def iter_query(
    sql: str,
    params: Mapping[str, Any] | None = None,
    *,
    engine: Engine | None = None,
    chunksize: int = 5000,
) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks for *sql* results, useful for large reads."""

    eng = engine or get_engine()
    query_params = params or None
    try:
        iterator = pd.read_sql(sql, eng, params=query_params, chunksize=chunksize)
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Query failed: {exc}") from exc
    if not hasattr(iterator, "__iter__"):
        # pandas returns a DataFrame when results are small even with chunksize.
        yield iterator  # type: ignore[misc]
        return
    yield from iterator
