#!/usr/bin/env python3
"""
Chapter 3: flip candidate-pool directionality (swap src_id and dst_id).

Why:
  - core_v1 uses supplier->customer orientation (src_id=supplier, dst_id=customer).
  - The original Chapter 3 candidate pool was built as (prime -> candidate), which
    aligns with "prime as supplier". For "predict upstream suppliers of primes",
    we want (candidate supplier -> prime customer).

This script creates a flipped candidate parquet suitable for re-scoring.

Notes:
  - Exporters only require columns: src_id, dst_id, label (+ optional ts).
  - The existing Chapter 3 candidate pool uses label=0 everywhere; this script
    preserves the label column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def main() -> None:
    p = argparse.ArgumentParser(description="Flip candidate pool (swap src_id/dst_id)")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/ch3/prediction_upstream/candidates_prime_to_candidate.parquet"),
        help="Input candidate pool parquet (must include src_id,dst_id,label)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ch3/prediction_upstream/candidates_with_label_sorted.parquet"),
        help="Output parquet (flipped)",
    )
    # Sorting is recommended (and fast) and avoids exporter edge-order quirks.
    p.add_argument(
        "--sort", dest="sort", action="store_true", help="Sort output by src_id, dst_id (default)"
    )
    p.add_argument("--no-sort", dest="sort", action="store_false", help="Disable sorting")
    p.add_argument(
        "--write-summary",
        action="store_true",
        help="Write a small JSON summary alongside the output",
    )
    p.set_defaults(sort=True)
    args = p.parse_args()

    inp = args.input
    out = args.output
    if not inp.exists():
        raise FileNotFoundError(inp)
    out.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(database=":memory:")
    conn.execute("SET preserve_insertion_order=false;")

    cols = (
        conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{inp.as_posix()}')")
        .fetchdf()["column_name"]
        .tolist()
    )
    required = {"src_id", "dst_id", "label"}
    missing = sorted(required - set(cols))
    if missing:
        raise ValueError(f"Missing required columns in {inp}: {missing}")

    has_ts = "ts" in set(cols)
    ts_expr = ", ts" if has_ts else ""

    order_clause = "ORDER BY src_id, dst_id" if args.sort else ""
    conn.execute(
        f"""
        COPY (
          SELECT
            dst_id AS src_id,
            src_id AS dst_id,
            label
            {ts_expr}
          FROM read_parquet('{inp.as_posix()}')
          {order_clause}
        ) TO '{out.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    if args.write_summary:
        n = int(
            conn.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
        )
        n_src = int(
            conn.execute(
                f"SELECT COUNT(DISTINCT src_id) FROM read_parquet('{out.as_posix()}')"
            ).fetchone()[0]
        )
        n_dst = int(
            conn.execute(
                f"SELECT COUNT(DISTINCT dst_id) FROM read_parquet('{out.as_posix()}')"
            ).fetchone()[0]
        )
        summary = {
            "input": str(inp),
            "output": str(out),
            "rows": n,
            "distinct_src": n_src,
            "distinct_dst": n_dst,
            "has_ts": bool(has_ts),
            "sorted": bool(args.sort),
        }
        out.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    conn.close()
    print(f"[done] wrote flipped candidates -> {out}")


if __name__ == "__main__":
    main()
