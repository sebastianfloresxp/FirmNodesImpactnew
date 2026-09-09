#!/usr/bin/env python3
"""Summarize USAspending ingests for Chapter 3.

Generates a small JSON/CSV with row counts, distinct vendor/component counts,
FY range, and obligation totals for the as-of and post-as-of slices (transactions
and primes). This lets the dissertation cite the transformation without rerunning
the heavy steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def summarize_transactions(path: Path, label: str) -> dict[str, object]:
    con = duckdb.connect()
    summary = con.execute(
        f"""
        SELECT
            '{label}' AS label,
            COUNT(*) AS rows,
            COUNT(DISTINCT awarding_sub_agency_name) AS components,
            COUNT(DISTINCT COALESCE(NULLIF(recipient_uei, ''), NULLIF(recipient_duns, ''), recipient_name)) AS vendors,
            MIN(action_date_fiscal_year) AS fy_min,
            MAX(action_date_fiscal_year) AS fy_max,
            SUM(federal_action_obligation) AS sum_federal_action_obligation,
            SUM(total_dollars_obligated) AS sum_total_dollars_obligated
        FROM '{path}'
        """
    ).fetchone()
    con.close()
    keys = [
        "label",
        "rows",
        "components",
        "vendors",
        "fy_min",
        "fy_max",
        "sum_federal_action_obligation",
        "sum_total_dollars_obligated",
    ]
    return dict(zip(keys, summary, strict=False))


def summarize_primes(path: Path, label: str) -> dict[str, object]:
    con = duckdb.connect()
    summary = con.execute(
        f"""
        SELECT
            '{label}' AS label,
            COUNT(*) AS rows,
            COUNT(DISTINCT dod_component) AS components,
            COUNT(DISTINCT vendor_key) AS vendors,
            MIN(action_fy) AS fy_min,
            MAX(action_fy) AS fy_max,
            SUM(sum_federal_action_obligation) AS sum_federal_action_obligation,
            SUM(sum_total_dollars_obligated) AS sum_total_dollars_obligated
        FROM '{path}'
        """
    ).fetchone()
    con.close()
    keys = [
        "label",
        "rows",
        "components",
        "vendors",
        "fy_min",
        "fy_max",
        "sum_federal_action_obligation",
        "sum_total_dollars_obligated",
    ]
    return dict(zip(keys, summary, strict=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize USAspending ingest outputs")
    ap.add_argument("--transactions-asof", type=Path, required=True)
    ap.add_argument("--transactions-post", type=Path, required=False)
    ap.add_argument("--primes-asof", type=Path, required=False)
    ap.add_argument("--primes-post", type=Path, required=False)
    ap.add_argument("--out-json", type=Path, default=Path("artifacts/ch3/usaspending/summary.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("artifacts/ch3/usaspending/summary.csv"))
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    rows.append(summarize_transactions(args.transactions_asof, "transactions_asof"))
    if args.transactions_post and args.transactions_post.exists():
        rows.append(summarize_transactions(args.transactions_post, "transactions_post"))
    if args.primes_asof and args.primes_asof.exists():
        rows.append(summarize_primes(args.primes_asof, "primes_asof"))
    if args.primes_post and args.primes_post.exists():
        rows.append(summarize_primes(args.primes_post, "primes_post"))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(rows, f, indent=2)
    # Write CSV
    if rows:
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
