"""Split cleaned shipping edges into train/val/test windows."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import duckdb

try:
    from .common import (
        ensure_directory,
        load_temporal_boundaries,
    )
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from common import (  # type: ignore
        ensure_directory,
        load_temporal_boundaries,
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")

_BASE_CTE = """
WITH base AS (
    SELECT
        *,
        DATE_DIFF('day', DATE '1970-01-01', CAST(record_date AS DATE)) AS ts
    FROM read_parquet('{input_path}')
)
"""

_SELECT_COLUMNS = """
SELECT
    transaction_id,
    record_date,
    ts,
    source_factset_entity_id,
    target_factset_entity_id,
    carrier_factset_entity_id,
    estimated_arrival_date,
    actual_arrival_date,
    orig_port_factset_entity_id,
    dest_port_factset_entity_id,
    factset_insert_date
FROM base
WHERE {predicate}
ORDER BY record_date, transaction_id
"""


def _copy_split(
    con: duckdb.DuckDBPyConnection, input_path: Path, predicate: str, output_path: Path
) -> int:
    input_quoted = str(input_path).replace("'", "''")
    output_quoted = str(output_path).replace("'", "''")
    sql = f"COPY ({_BASE_CTE.format(input_path=input_quoted)}{_SELECT_COLUMNS.format(predicate=predicate)}) TO '{output_quoted}' (FORMAT PARQUET)"
    con.execute(sql)
    count = con.execute(
        f"{_BASE_CTE.format(input_path=input_quoted)}SELECT COUNT(*) FROM base WHERE {predicate}"
    ).fetchone()[0]  # type: ignore[index]
    logger.info("Wrote %s rows to %s", count, output_path)
    return int(count)


def split_transactions(
    *,
    input_path: Path,
    temporal_meta: Path,
    output_dir: Path,
    summary_path: Path | None,
) -> dict[str, int]:
    boundaries = load_temporal_boundaries(temporal_meta)

    con = duckdb.connect(database=":memory:")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_edges.parquet"
    val_path = output_dir / "val_edges.parquet"
    test_path = output_dir / "test_edges.parquet"

    counts = {
        "train": _copy_split(con, input_path, f"ts <= {boundaries.T0_end}", train_path),
        "val": _copy_split(
            con,
            input_path,
            f"ts >= {boundaries.val_start} AND ts <= {boundaries.val_end}",
            val_path,
        ),
        "test": _copy_split(con, input_path, f"ts >= {boundaries.test_start}", test_path),
    }

    if summary_path:
        ensure_directory(summary_path)
        summary_path.write_text(json.dumps(counts, indent=2))
        logger.info("Wrote split summary to %s", summary_path)

    return counts


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split cleaned shipping edges")
    parser.add_argument(
        "--input", required=True, type=Path, help="Cleaned parquet from clean_transactions"
    )
    parser.add_argument(
        "--temporal-meta",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/meta/temporal_splits.json"),
        help="Temporal metadata for determining split boundaries",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory for split parquet files"
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional JSON summary of split counts",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    summary_path = args.summary_out.expanduser().resolve() if args.summary_out else None

    split_transactions(
        input_path=input_path,
        temporal_meta=args.temporal_meta.expanduser().resolve(),
        output_dir=output_dir,
        summary_path=summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
