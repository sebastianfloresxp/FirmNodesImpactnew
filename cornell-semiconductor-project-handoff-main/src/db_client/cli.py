"""CLI for running ad-hoc FactSet SQL queries."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .query import iter_query, run_query


def _load_sql(sql: str | None, sql_file: Path | None) -> str:
    if sql:
        return sql
    if not sql_file:
        raise ValueError("Either --sql or --sql-file must be provided")
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file}")
    return sql_file.read_text()


def _parse_params(param_items: Iterable[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for raw in param_items:
        key, sep, value = raw.partition("=")
        if not sep:
            raise ValueError(f"Could not parse param '{raw}'. Expected KEY=VALUE")
        value = value.strip()
        if not value:
            params[key] = value
            continue
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value
    return params


def _infer_format(output: Path) -> str:
    ext = output.suffix.lower()
    if ext in {".csv"}:
        return "csv"
    if ext in {".json"}:
        return "json"
    if ext in {".parquet", ".pq"}:
        return "parquet"
    return "table"


def _write_output(df: pd.DataFrame, output: Path | None, fmt: str) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        if output:
            df.to_csv(output, index=False)
        else:
            df.to_csv(sys.stdout, index=False)
    elif fmt == "json":
        indent = 2 if output else None
        data = df.to_dict(orient="records")
        text = json.dumps(data, indent=indent)
        if output:
            output.write_text(text)
        else:
            sys.stdout.write(text + "\n")
    elif fmt == "parquet":
        if not output:
            raise ValueError("Parquet output requires --output path")
        df.to_parquet(output, index=False)
    else:
        # Table output: truncate long tables for readability
        rows = len(df)
        preview = df.head(50)
        sys.stdout.write(preview.to_string(index=False))
        if rows > len(preview):
            sys.stdout.write(f"\n... truncated {rows - len(preview)} rows ...\n")
        sys.stdout.write("\n")
        sys.stdout.write("dtypes:\n")
        sys.stdout.write(df.dtypes.to_string())
        sys.stdout.write("\n")


def _stream_to_csv(chunks: Iterable[pd.DataFrame], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for idx, chunk in enumerate(chunks):
        chunk.to_csv(output, mode="w" if idx == 0 else "a", header=idx == 0, index=False)


def _stream_to_stdout(chunks: Iterable[pd.DataFrame]) -> None:
    for chunk in chunks:
        sys.stdout.write(chunk.to_csv(index=False))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ad-hoc SQL against the FactSet server")
    parser.add_argument("--sql", type=str, help="SQL statement to execute")
    parser.add_argument("--sql-file", type=Path, help="Path to a .sql file to execute")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Query parameter in KEY=VALUE form (VALUE may be JSON)",
    )
    parser.add_argument(
        "--chunksize", type=int, default=0, help="Stream results in chunks of this size (CSV only)"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional output path (format inferred from extension)"
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv", "json", "parquet"],
        default="table",
        help="Force output format",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved SQL and exit without running"
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    sql = _load_sql(args.sql, args.sql_file)
    params = _parse_params(args.param)
    query_params = params or None

    if args.dry_run:
        sys.stdout.write("-- SQL -----------------------------\n")
        sys.stdout.write(sql.strip() + "\n")
        if params:
            sys.stdout.write("-- Params ---------------------------\n")
            sys.stdout.write(json.dumps(params, indent=2) + "\n")
        return 0

    fmt = args.format
    if args.output and args.format == "table":
        fmt = _infer_format(args.output)

    if args.chunksize:
        if fmt not in {"csv"}:
            raise ValueError("Chunked streaming is only supported for CSV output")
        chunks = iter_query(sql, params=query_params, chunksize=args.chunksize)
        if args.output:
            _stream_to_csv(chunks, args.output)
        else:
            _stream_to_stdout(chunks)
        return 0

    df = run_query(sql, params=query_params)
    _write_output(df, args.output, fmt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
