"""Deduplicate and sanitize raw shipping transactions."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

from db_client import run_query

try:
    from .common import ensure_directory, normalize_output_path
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from common import ensure_directory, normalize_output_path  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")

_PLACEHOLDER_SQL = """
SELECT factset_entity_id, entity_proper_name
FROM sym_v1.sym_entity
WHERE LOWER(entity_proper_name) LIKE '%unidentified%'
   OR LOWER(entity_proper_name) LIKE '%unknown%'
   OR LOWER(entity_proper_name) LIKE '%unspecified%'
   OR LOWER(entity_proper_name) LIKE '%various%'
   OR LOWER(entity_proper_name) LIKE '%not provided%'
   OR LOWER(entity_proper_name) LIKE '%not available%'
"""

_COVERAGE_SQL = """
SELECT factset_entity_id, shipper_flag, consignee_flag
FROM sc_v1.sc_ship_coverage_curr
"""

_BASE_CTE_TEMPLATE = """
WITH raw AS (
    SELECT * FROM read_parquet('{input_path}')
),
ranked AS (
    SELECT
        r.*,
        ROW_NUMBER() OVER (
            PARTITION BY transaction_id
            ORDER BY record_date DESC NULLS LAST,
                     factset_insert_date DESC NULLS LAST,
                     transaction_id
        ) AS rn
    FROM raw r
),
deduped AS (
    SELECT *
    FROM ranked
    WHERE rn = 1
      AND shipper_factset_entity_id IS NOT NULL AND shipper_factset_entity_id <> ''
      AND consignee_factset_entity_id IS NOT NULL AND consignee_factset_entity_id <> ''
),
placeholder_filtered AS (
    SELECT d.*
    FROM deduped d
    LEFT JOIN placeholder_ids ph_ship
        ON d.shipper_factset_entity_id = ph_ship.factset_entity_id
    LEFT JOIN placeholder_ids ph_cons
        ON d.consignee_factset_entity_id = ph_cons.factset_entity_id
    WHERE ph_ship.factset_entity_id IS NULL
      AND ph_cons.factset_entity_id IS NULL
),
no_self_loops AS (
    SELECT *
    FROM placeholder_filtered
    WHERE shipper_factset_entity_id <> consignee_factset_entity_id
),
final AS (
    SELECT n.*
    FROM no_self_loops n
    JOIN coverage cov_ship
        ON n.shipper_factset_entity_id = cov_ship.factset_entity_id
       AND cov_ship.shipper_flag = 1
    JOIN coverage cov_cons
        ON n.consignee_factset_entity_id = cov_cons.factset_entity_id
       AND cov_cons.consignee_flag = 1
)
"""

_SUMMARY_SQL = """
{base_cte}
SELECT
    (SELECT COUNT(*) FROM raw) AS input_rows,
    (SELECT COUNT(*) FROM deduped) AS post_dedup_rows,
    (SELECT COUNT(*) FROM placeholder_filtered) AS post_placeholder_rows,
    (SELECT COUNT(*) FROM no_self_loops) AS post_self_loop_rows,
    (SELECT COUNT(*) FROM final) AS final_rows
"""

_FINAL_SELECT = """
SELECT
    transaction_id,
    record_date,
    shipper_factset_entity_id AS source_factset_entity_id,
    consignee_factset_entity_id AS target_factset_entity_id,
    carrier_factset_entity_id,
    estimated_arrival_date,
    actual_arrival_date,
    orig_port_factset_entity_id,
    dest_port_factset_entity_id,
    factset_insert_date
FROM final
ORDER BY record_date, transaction_id
"""


def _load_placeholder_entities() -> pd.DataFrame:
    df = run_query(_PLACEHOLDER_SQL)
    logger.info("Loaded %s placeholder entity IDs", len(df))
    if df.empty:
        df = pd.DataFrame(
            {
                "factset_entity_id": pd.Series([], dtype="object"),
                "entity_proper_name": pd.Series([], dtype="object"),
            }
        )
    return df


def _load_coverage() -> pd.DataFrame:
    df = run_query(_COVERAGE_SQL)
    logger.info("Loaded %s coverage rows", len(df))
    return df


def clean_transactions(
    *,
    input_path: Path,
    output_path: Path,
    summary_path: Path | None,
) -> dict[str, int]:
    placeholder_df = _load_placeholder_entities()
    coverage_df = _load_coverage()

    con = duckdb.connect(database=":memory:")
    con.register("placeholder_ids", placeholder_df[["factset_entity_id"]])
    con.register("coverage", coverage_df)

    input_quoted = str(input_path).replace("'", "''")
    base_cte = _BASE_CTE_TEMPLATE.format(input_path=input_quoted)

    summary_sql = _SUMMARY_SQL.format(base_cte=base_cte)
    summary_row = con.execute(summary_sql).fetchone()
    assert summary_row is not None, "summary query returned no rows"
    summary = {
        "input_rows": int(summary_row[0]),
        "post_dedup_rows": int(summary_row[1]),
        "post_placeholder_rows": int(summary_row[2]),
        "post_self_loop_rows": int(summary_row[3]),
        "final_rows": int(summary_row[4]),
    }
    summary["duplicate_rows_removed"] = summary["input_rows"] - summary["post_dedup_rows"]
    summary["placeholder_rows_removed"] = (
        summary["post_dedup_rows"] - summary["post_placeholder_rows"]
    )
    summary["self_loops_removed"] = (
        summary["post_placeholder_rows"] - summary["post_self_loop_rows"]
    )
    summary["coverage_filtered_rows"] = summary["post_self_loop_rows"] - summary["final_rows"]

    output_quoted = str(output_path).replace("'", "''")
    final_sql = f"COPY ({base_cte}\n{_FINAL_SELECT}\n) TO '{output_quoted}' (FORMAT PARQUET)"
    con.execute(final_sql)
    logger.info("Wrote cleaned transactions to %s", output_path)

    if summary_path:
        ensure_directory(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("Wrote cleaning summary to %s", summary_path)

    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean raw shipping transactions")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to raw parquet produced by extract_transactions",
    )
    parser.add_argument("--output", required=True, type=Path, help="Cleaned parquet output path")
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional JSON path for cleaning statistics",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    input_path = args.input.expanduser().resolve()
    output_path = normalize_output_path(args.output)
    summary_path = args.summary_out.expanduser().resolve() if args.summary_out else None

    clean_transactions(input_path=input_path, output_path=output_path, summary_path=summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
