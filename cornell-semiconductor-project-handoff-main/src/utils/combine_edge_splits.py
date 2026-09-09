"""Combine multiple edge split Parquet files into a single dataset."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _resolve_inputs(values: Iterable[str]) -> list[Path]:
    paths = [Path(v).expanduser().resolve() for v in values]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Input files not found: {', '.join(missing)}")
    return paths


def combine_parquet(inputs: list[Path], output: Path, *, coerce_schema: bool = True) -> None:
    tables: list[pa.Table] = []
    base_schema = None
    for _idx, path in enumerate(inputs):
        table = pq.read_table(path)
        if base_schema is None:
            base_schema = table.schema
        elif coerce_schema:
            table = table.cast(base_schema)
        tables.append(table)
    combined = pa.concat_tables(tables, mode="default")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, output)
    total_rows = combined.num_rows
    print(f"Wrote {total_rows:,} rows to {output}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Concatenate edge split Parquet files")
    parser.add_argument("--output", required=True, type=Path, help="Destination Parquet path")
    parser.add_argument("inputs", nargs="+", help="Input Parquet files (train/val/test)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    inputs = _resolve_inputs(args.inputs)
    combine_parquet(inputs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
