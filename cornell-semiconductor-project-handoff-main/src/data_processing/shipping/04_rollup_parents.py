"""Roll shipping edges up to ultimate parent entities."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from db_client import run_query

try:
    from .common import normalize_output_path
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from common import normalize_output_path  # type: ignore


PARENT_SQL = "SELECT factset_entity_id, factset_ult_parent_entity_id FROM sc_v1.sc_ship_parent"


def rollup(
    *,
    input_path: Path,
    output_path: Path,
) -> None:
    conn = duckdb.connect(database=":memory:")
    parent_df = run_query(PARENT_SQL)
    parent_df["factset_ult_parent_entity_id"].fillna(parent_df["factset_entity_id"], inplace=True)
    parent_df = parent_df.drop_duplicates(subset=["factset_entity_id"])
    conn.register("parent", parent_df)
    input_quoted = str(input_path).replace("'", "''")
    output_quoted = str(output_path).replace("'", "''")
    conn.execute(
        f"""
        COPY (
            SELECT
                transaction_id,
                record_date,
                DATE_DIFF('day', DATE '1970-01-01', CAST(record_date AS DATE)) AS ts,
                COALESCE(p_ship.factset_ult_parent_entity_id, source_factset_entity_id) AS source_factset_entity_id,
                COALESCE(p_cons.factset_ult_parent_entity_id, target_factset_entity_id) AS target_factset_entity_id,
                carrier_factset_entity_id,
                estimated_arrival_date,
                actual_arrival_date,
                orig_port_factset_entity_id,
                dest_port_factset_entity_id,
                factset_insert_date
            FROM read_parquet('{input_quoted}')
            LEFT JOIN parent AS p_ship
                ON source_factset_entity_id = p_ship.factset_entity_id
            LEFT JOIN parent AS p_cons
                ON target_factset_entity_id = p_cons.factset_entity_id
        ) TO '{output_quoted}' (FORMAT PARQUET)
        """
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Roll shipping edges to parent IDs")
    parser.add_argument(
        "--input", required=True, type=Path, help="Cleaned shipping edges (with ts column)"
    )
    parser.add_argument("--output", required=True, type=Path, help="Parent-level output parquet")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = normalize_output_path(args.output)
    rollup(input_path=input_path, output_path=output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
