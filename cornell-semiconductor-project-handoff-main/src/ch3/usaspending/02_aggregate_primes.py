#!/usr/bin/env python3
"""Aggregate DoD transactions into Tier-1 prime candidates (as-of and post).

Reads the Parquet slices from 01_build_dod_transactions.py and aggregates by
FY, DoD component, and vendor identifier. Outputs separate as-of and post-as-of
prime tables plus simple diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
from dotenv import load_dotenv

DEFAULT_OUT_DIR = Path("artifacts/ch3/usaspending")


def aggregate_primes(parquet_path: Path, out_path: Path, diag_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    con.execute(
        """
        CREATE OR REPLACE TABLE primes AS
        SELECT
            CAST(action_date_fiscal_year AS INTEGER) AS action_fy,
            awarding_sub_agency_name AS dod_component,
            COALESCE(NULLIF(recipient_uei, ''), NULLIF(recipient_duns, ''), recipient_name) AS vendor_key,
            recipient_uei,
            recipient_duns,
            recipient_name,
            recipient_country_code,
            recipient_country_name,
            recipient_state_code,
            recipient_state_name,
            recipient_city_name,
            recipient_zip_4_code,
            naics_code,
            naics_description,
            product_or_service_code,
            product_or_service_code_description,
            SUM(federal_action_obligation) AS sum_federal_action_obligation,
            SUM(total_dollars_obligated) AS sum_total_dollars_obligated,
            COUNT(*) AS txn_count
        FROM parquet_scan(?)
        WHERE action_fy >= 2022
        GROUP BY ALL
        """,
        [str(parquet_path)],
    )

    df = con.execute("SELECT * FROM primes").fetchdf()
    df.to_parquet(out_path, index=False)

    diag = con.execute(
        """
        SELECT dod_component,
               COUNT(DISTINCT vendor_key) AS vendors,
               SUM(sum_total_dollars_obligated) AS total_obligations
        FROM primes
        GROUP BY 1
        ORDER BY total_obligations DESC
        """
    ).fetchdf()
    diag.to_csv(diag_path, index=False)

    con.close()


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Aggregate DoD primes (as-of and post)")
    ap.add_argument(
        "--transactions-asof",
        type=Path,
        required=True,
        help="Parquet from 01_build_dod_transactions (as-of slice)",
    )
    ap.add_argument(
        "--transactions-post",
        type=Path,
        required=False,
        help="Parquet from 01_build_dod_transactions (post-as-of slice)",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_asof = args.out_dir / args.transactions_asof.name.replace("transactions", "primes").replace(
        ".parquet", ".parquet"
    )
    diag_asof = args.out_dir / args.transactions_asof.name.replace(
        "transactions", "primes_diag"
    ).replace(".parquet", ".csv")
    aggregate_primes(args.transactions_asof, out_asof, diag_asof)

    if args.transactions_post and args.transactions_post.exists():
        out_post = args.out_dir / args.transactions_post.name.replace(
            "transactions", "primes"
        ).replace(".parquet", ".parquet")
        diag_post = args.out_dir / args.transactions_post.name.replace(
            "transactions", "primes_diag"
        ).replace(".parquet", ".csv")
        aggregate_primes(args.transactions_post, out_post, diag_post)


if __name__ == "__main__":
    main()
