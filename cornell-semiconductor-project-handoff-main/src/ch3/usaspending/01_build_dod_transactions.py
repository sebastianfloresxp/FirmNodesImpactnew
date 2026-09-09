#!/usr/bin/env python3
"""Build a filtered DoD transactions table (canonical + post-as-of splits).

Reads USAspending CSVs (FY start/end), keeps a compact column set, filters to
DoD awards, and splits by as-of date. Writes Parquet outputs to artifacts/ch3.
Uses DuckDB to stream and deduplicate globally by contract_transaction_unique_key.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")
logger = logging.getLogger("ch3.usaspending.tx")

# Columns to retain
BASE_COLS: list[str] = [
    # IDs/time
    "contract_transaction_unique_key",
    "contract_award_unique_key",
    "award_id_piid",
    "modification_number",
    "transaction_number",
    "action_date",
    "action_date_fiscal_year",
    "period_of_performance_start_date",
    "period_of_performance_current_end_date",
    # DoD side
    "awarding_agency_name",
    "awarding_sub_agency_name",
    "awarding_office_name",
    # Vendor side
    "recipient_uei",
    "recipient_duns",
    "recipient_name",
    "recipient_name_raw",
    "recipient_doing_business_as_name",
    "recipient_parent_uei",
    "recipient_parent_duns",
    "recipient_parent_name",
    "recipient_parent_name_raw",
    "recipient_country_code",
    "recipient_country_name",
    "recipient_state_code",
    "recipient_state_name",
    "recipient_city_name",
    "recipient_zip_4_code",
    # Money
    "federal_action_obligation",
    "total_dollars_obligated",
    # Classification
    "naics_code",
    "naics_description",
    "product_or_service_code",
    "product_or_service_code_description",
    "type_of_contract_pricing_code",
    "type_of_contract_pricing",
]


def find_files(raw_dir: Path, fy_start: int, fy_end: int) -> list[Path]:
    files: list[Path] = []
    for fy in range(fy_start, fy_end + 1):
        files += sorted(raw_dir.glob(f"FY{fy}_097_Contracts_Full_*.csv"))
    return files


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Build DoD transactions Parquet")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/usaspending"))
    ap.add_argument("--fy-start", type=int, default=2022)
    ap.add_argument("--fy-end", type=int, default=2025)
    ap.add_argument("--as-of-date", type=str, default="2025-06-09")
    ap.add_argument(
        "--out-prefix", type=Path, default=Path("artifacts/ch3/usaspending/dod_transactions")
    )
    args = ap.parse_args()

    as_of = args.as_of_date
    files = find_files(args.raw_dir, args.fy_start, args.fy_end)
    if not files:
        raise SystemExit("No USAspending files found for range")

    logger.info("Found %d files", len(files))
    out_canonical = args.out_prefix.with_name(
        f"{args.out_prefix.name}_fy{args.fy_start}-{args.fy_end}_asof.parquet"
    )
    out_post = args.out_prefix.with_name(
        f"{args.out_prefix.name}_fy{args.fy_start}-{args.fy_end}_post.parquet"
    )

    files_array = "[" + ", ".join(f"'{p!s}'" for p in files) + "]"
    select_cols = ",\n            ".join(BASE_COLS)

    # SQL template with dedup by contract_transaction_unique_key (keep latest action_date)
    def render_sql(where_clause: str) -> str:
        return f"""
WITH src AS (
    SELECT
        contract_transaction_unique_key,
        contract_award_unique_key,
        award_id_piid,
        modification_number,
        transaction_number,
        TRY_CAST(action_date AS DATE) AS action_date,
        action_date_fiscal_year,
        TRY_CAST(period_of_performance_start_date AS DATE) AS period_of_performance_start_date,
        TRY_CAST(period_of_performance_current_end_date AS DATE) AS period_of_performance_current_end_date,
        awarding_agency_name,
        awarding_sub_agency_name,
        awarding_office_name,
        recipient_uei,
        recipient_duns,
        recipient_name,
        recipient_name_raw,
        recipient_doing_business_as_name,
        recipient_parent_uei,
        recipient_parent_duns,
        recipient_parent_name,
        recipient_parent_name_raw,
        recipient_country_code,
        recipient_country_name,
        recipient_state_code,
        recipient_state_name,
        recipient_city_name,
        recipient_zip_4_code,
        CAST(federal_action_obligation AS DOUBLE) AS federal_action_obligation,
        CAST(total_dollars_obligated AS DOUBLE) AS total_dollars_obligated,
        naics_code,
        naics_description,
        product_or_service_code,
        product_or_service_code_description,
        type_of_contract_pricing_code,
        type_of_contract_pricing
    FROM read_csv_auto({files_array}, header=True, all_varchar=TRUE, sample_size=-1)
    WHERE awarding_agency_name = 'Department of Defense'
),
dedup AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY contract_transaction_unique_key ORDER BY action_date DESC NULLS LAST) AS rn
    FROM src
    WHERE action_date IS NOT NULL
          {where_clause}
)
SELECT
    {select_cols}
FROM dedup
WHERE rn = 1
"""

    con = duckdb.connect()

    logger.info("Writing canonical (<= %s) to %s", as_of, out_canonical)
    sql_canonical = render_sql(f"AND action_date <= DATE '{as_of}'")
    con.execute(f"COPY ({sql_canonical}) TO '{out_canonical!s}' (FORMAT PARQUET)")

    logger.info("Writing post-as-of (> %s) to %s", as_of, out_post)
    sql_post = render_sql(f"AND action_date > DATE '{as_of}'")
    con.execute(f"COPY ({sql_post}) TO '{out_post!s}' (FORMAT PARQUET)")
    con.close()


if __name__ == "__main__":
    main()
