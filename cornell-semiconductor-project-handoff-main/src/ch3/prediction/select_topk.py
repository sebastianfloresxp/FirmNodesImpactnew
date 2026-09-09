#!/usr/bin/env python3
"""
Chapter 3: apply a per-group Top-K selection policy to meta-ranker scores.

This script writes:
  - top{K}_all.parquet : includes is_known flag (known SCR edge in train/val/test)
  - top{K}_pred.parquet: excludes known edges (predicted-only), drops is_known

Directionality note:
  - core_v1 edges are supplier->customer (src_id->dst_id).
  - If your candidates are oriented as (candidate supplier -> prime customer),
    you typically want Top-K per prime, i.e. group-by dst_id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top-K selection on Chapter 3 meta scores")
    p.add_argument(
        "--meta-scores",
        type=Path,
        required=True,
        help="meta_scores parquet with src_id,dst_id,meta_prob",
    )
    p.add_argument("--out-root", type=Path, required=True, help="Output directory for topK files")
    p.add_argument(
        "--group-by",
        default="src_id",
        choices=["src_id", "dst_id"],
        help="Partition axis for Top-K (src_id = per-supplier, dst_id = per-customer/prime)",
    )
    p.add_argument("--ks", nargs="+", type=int, default=[5, 10, 50], help="K values to emit")
    p.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/splits/train_edges.parquet"),
    )
    p.add_argument(
        "--val",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/splits/val_edges.parquet"),
    )
    p.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/splits/test_edges.parquet"),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta_scores = args.meta_scores
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    for p in [meta_scores, args.train, args.val, args.test]:
        if not p.exists():
            raise FileNotFoundError(p)

    conn = duckdb.connect(database=":memory:")
    conn.execute("SET preserve_insertion_order=false;")

    # Base rate (known edges in pool) is useful for sanity checks.
    base_rate = conn.execute(
        """
        WITH known AS (
          SELECT DISTINCT src_id, dst_id FROM read_parquet(?)
          UNION SELECT DISTINCT src_id, dst_id FROM read_parquet(?)
          UNION SELECT DISTINCT src_id, dst_id FROM read_parquet(?)
        ), hits AS (
          SELECT COUNT(*) AS c
          FROM read_parquet(?) s
          JOIN known k USING (src_id, dst_id)
        ), total AS (
          SELECT COUNT(*) AS c FROM read_parquet(?)
        )
        SELECT hits.c::DOUBLE / total.c FROM hits, total
        """,
        [
            args.train.as_posix(),
            args.val.as_posix(),
            args.test.as_posix(),
            meta_scores.as_posix(),
            meta_scores.as_posix(),
        ],
    ).fetchone()[0]

    summary = {
        "meta_scores": str(meta_scores),
        "group_by": args.group_by,
        "ks": [int(k) for k in args.ks],
        "base_rate_known_edges": float(base_rate) if base_rate is not None else None,
        "outputs": {},
    }

    for k in [int(x) for x in args.ks]:
        all_path = out_root / f"top{k}_all.parquet"
        pred_path = out_root / f"top{k}_pred.parquet"
        if not args.overwrite:
            for p in [all_path, pred_path]:
                if p.exists():
                    raise FileExistsError(p)

        # Top-K per chosen axis.
        conn.execute(
            f"""
            COPY (
              WITH known AS (
                SELECT DISTINCT src_id, dst_id FROM read_parquet('{args.train.as_posix()}')
                UNION SELECT DISTINCT src_id, dst_id FROM read_parquet('{args.val.as_posix()}')
                UNION SELECT DISTINCT src_id, dst_id FROM read_parquet('{args.test.as_posix()}')
              ),
              scored AS (
                SELECT
                  s.src_id,
                  s.dst_id,
                  s.meta_prob,
                  (k.src_id IS NOT NULL) AS is_known
                FROM read_parquet('{meta_scores.as_posix()}') s
                LEFT JOIN known k USING (src_id, dst_id)
              ),
              ranked AS (
                SELECT
                  *,
                  ROW_NUMBER() OVER (
                    PARTITION BY {args.group_by}
                    ORDER BY meta_prob DESC, src_id, dst_id
                  ) AS rk
                FROM scored
              )
              SELECT src_id, dst_id, meta_prob, is_known, rk
              FROM ranked
              WHERE rk <= {k}
            ) TO '{all_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
            """
        )

        # Predicted-only: remove known SCR edges; drop is_known to keep schema aligned to existing artifacts.
        conn.execute(
            f"""
            COPY (
              SELECT src_id, dst_id, meta_prob, rk
              FROM read_parquet('{all_path.as_posix()}')
              WHERE is_known = FALSE
            ) TO '{pred_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
            """
        )

        total, known = conn.execute(
            f"SELECT COUNT(*) AS total, SUM(is_known::INT) AS known FROM read_parquet('{all_path.as_posix()}')"
        ).fetchone()
        summary["outputs"][f"top{k}_all"] = str(all_path)
        summary["outputs"][f"top{k}_pred"] = str(pred_path)
        summary[f"top{k}_all_rows"] = int(total)
        summary[f"top{k}_known"] = int(known or 0)

    conn.close()
    (out_root / "topk_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
