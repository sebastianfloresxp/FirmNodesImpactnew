"""Database connection helpers for the FactSet SQL Server."""

from __future__ import annotations

import os
import urllib.parse
from collections.abc import Mapping, MutableMapping
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

_REQUIRED_VARS = ("DB_SERVER", "DB_DATABASE", "DB_USERNAME", "DB_PASSWORD")


class MissingEnvironmentError(RuntimeError):
    """Raised when required database environment variables are missing."""


def _load_env_vars() -> MutableMapping[str, str]:
    load_dotenv()
    values: MutableMapping[str, str] = {}
    for key in _REQUIRED_VARS:
        value = os.getenv(key)
        if not value:
            raise MissingEnvironmentError(
                f"Missing environment variable '{key}'. Expected: {', '.join(_REQUIRED_VARS)}"
            )
        values[key] = value
    return values


@lru_cache(maxsize=1)
def get_connection_string(overrides: Mapping[str, str] | None = None) -> str:
    """Return a SQLAlchemy connection string using env vars, optionally overridden."""

    values = _load_env_vars()
    if overrides:
        for key, value in overrides.items():
            if value:
                values[key] = value
    dsn = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={values['DB_SERVER']};"
        f"DATABASE={values['DB_DATABASE']};"
        f"UID={values['DB_USERNAME']};"
        f"PWD={values['DB_PASSWORD']};"
        "Trusted_Connection=no;Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;"
    )
    return f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(dsn)}"


def get_engine(*, echo: bool = False, overrides: Mapping[str, str] | None = None) -> Engine:
    """Create a SQLAlchemy engine for the FactSet SQL Server."""

    conn_str = get_connection_string(overrides=overrides)  # type: ignore[call-overload]
    return create_engine(conn_str, echo=echo, pool_pre_ping=True, pool_recycle=1800)
