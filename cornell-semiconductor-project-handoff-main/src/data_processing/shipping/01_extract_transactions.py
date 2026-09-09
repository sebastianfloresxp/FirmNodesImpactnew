"""Extract raw shipping transactions from FactSet SQL."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from db_client import iter_query

try:
    from .common import (
        ensure_directory,
        load_temporal_boundaries,
        normalize_output_path,
        parse_date,
    )
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from common import (  # type: ignore
        ensure_directory,
        load_temporal_boundaries,
        normalize_output_path,
        parse_date,
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")


_COLUMNS = [
    "transaction_id",
    "record_date",
    "shipper_factset_entity_id",
    "consignee_factset_entity_id",
    "carrier_factset_entity_id",
    "record_status",
    "estimated_arrival_date",
    "actual_arrival_date",
    "orig_port_factset_entity_id",
    "dest_port_factset_entity_id",
    "factset_insert_date",
]


_SQL_TEMPLATE = """
SELECT
    transaction_id,
    record_date,
    shipper_factset_entity_id,
    consignee_factset_entity_id,
    carrier_factset_entity_id,
    record_status,
    estimated_arrival_date,
    actual_arrival_date,
    orig_port_factset_entity_id,
    dest_port_factset_entity_id,
    factset_insert_date
FROM sc_v1.sc_ship_trans_curr
WHERE record_status IN ({record_statuses})
  AND record_date <= '{as_of}'
  {min_record_date_clause}
"""


def _infer_cutoff(meta_path: Path, phase: str) -> date:
    boundaries = load_temporal_boundaries(meta_path)
    mapping = {
        "train_end": boundaries.train_end_date,
        "val_end": boundaries.val_end_date,
        "test_start": boundaries.test_start_date,
        "max": boundaries.max_date,
    }
    try:
        return mapping[phase]
    except KeyError as exc:
        raise ValueError(f"Unsupported cutoff phase '{phase}'") from exc


def _write_parquet(chunks: Iterable[pd.DataFrame], output: Path) -> int:
    ensure_directory(output)
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    for chunk in chunks:
        if chunk.empty:
            continue
        chunk = chunk[_COLUMNS]
        for col in [
            "record_date",
            "estimated_arrival_date",
            "actual_arrival_date",
            "factset_insert_date",
        ]:
            chunk[col] = pd.to_datetime(chunk[col], errors="coerce")
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output, table.schema)
        writer.write_table(table)
        total_rows += len(chunk)

    if writer is not None:
        writer.close()
    return total_rows


def extract_transactions(
    *,
    output: Path,
    cutoff_date: date,
    record_statuses: list[str],
    min_record_date: date | None,
    chunksize: int,
) -> int:
    allowed = {"N", "A"}
    bad = [s for s in record_statuses if s not in allowed]
    if bad:
        raise ValueError(f"Unsupported record_status codes: {bad}. Allowed: {sorted(allowed)}")
    quoted_statuses = ",".join(f"'{s}'" for s in sorted(set(record_statuses)))
    min_clause = ""
    if min_record_date is not None:
        min_clause = f"AND record_date >= '{min_record_date.strftime('%Y-%m-%d')}'"
    sql = _SQL_TEMPLATE.format(
        as_of=cutoff_date.strftime("%Y-%m-%d"),
        record_statuses=quoted_statuses,
        min_record_date_clause=min_clause,
    )
    logger.info(
        "Running shipping transaction extract (record_status in %s) with record_date <= %s%s",
        sorted(set(record_statuses)),
        cutoff_date,
        f" and record_date >= {min_record_date}" if min_record_date else "",
    )
    chunk_iter = iter_query(sql, chunksize=chunksize)
    total = _write_parquet(chunk_iter, output)
    logger.info("Wrote %s rows to %s", total, output)
    return total


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract raw shipping transactions")
    parser.add_argument("--output", required=True, type=Path, help="Destination parquet path")
    parser.add_argument(
        "--temporal-meta",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/meta/temporal_splits.json"),
        help="Path to temporal_splits.json for determining default cutoff",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        help="Override cutoff date (YYYY-MM-DD). Defaults to core train cutoff",
    )
    parser.add_argument(
        "--cutoff-phase",
        choices=["train_end", "val_end", "test_start", "max"],
        default="train_end",
        help=(
            "When --as-of-date is not provided, choose which temporal boundary to use: "
            "train_end (default), val_end, test_start, or max (entire dataset)."
        ),
    )
    parser.add_argument(
        "--record-status",
        nargs="+",
        default=["N", "A"],
        help="Record status codes to include (default: N A).",
    )
    parser.add_argument(
        "--record-date-min",
        type=str,
        help="Optional minimum record_date (YYYY-MM-DD) to bound the extract window.",
    )
    parser.add_argument("--chunksize", type=int, default=200000, help="SQL chunk size")

    args = parser.parse_args(list(argv) if argv is not None else None)

    output = normalize_output_path(args.output)
    cutoff = (
        parse_date(args.as_of_date)
        if args.as_of_date
        else _infer_cutoff(args.temporal_meta, args.cutoff_phase)
    )
    min_record_date = parse_date(args.record_date_min) if args.record_date_min else None

    extract_transactions(
        output=output,
        cutoff_date=cutoff,
        record_statuses=[str(s).strip().upper() for s in args.record_status],
        min_record_date=min_record_date,
        chunksize=args.chunksize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
