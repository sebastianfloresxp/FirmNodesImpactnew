#!/usr/bin/env python3
"""
Chapter 3: build meta_inputs_zero_slices.parquet for a prediction run.

This is the deployment-style meta input used for Chapter 3 inference:
  - Includes the 6 calibrated prob_* features from base models.
  - Includes the slice_* features expected by the Chapter 2 meta-ranker, but sets
    them to zero (unavailable at inference in the Chapter 3 build).

Inputs:
  - --base: candidate pairs (src_id, dst_id, label) for scoring
  - --scores-root: staged per-model score files:
      graphsage_scores.parquet, node2vec_scores.parquet, heuristics_scores.parquet,
      twotower_scores.parquet, n2v_temporal_scores.parquet, tgnn_scores.parquet

Outputs (under --out-root):
  - meta_inputs_dir/part_*.parquet (bucketed joins; kept only if --keep-parts)
  - meta_inputs_zero_slices.parquet (single-file meta dataset for apply_meta.py)
  - meta_inputs_zero_slices.json (run summary)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import duckdb

SLICE_COLS = [
    "slice_WW",
    "slice_WC",
    "slice_CW",
    "slice_CC",
    "slice_WW3",
    "slice_WC3",
    "slice_CW3",
    "slice_CC3",
    "slice_twohop",
    "slice_gt2hop",
    "slice_deg_q1",
    "slice_deg_q2",
    "slice_deg_q3",
    "slice_deg_q4",
    "slice_horizon_0_6",
    "slice_horizon_6_12",
    "slice_horizon_12_24",
    "slice_horizon_24_36",
    "slice_horizon_36_48",
    "slice_horizon_48_60",
    "slice_horizon_60_72",
    "slice_horizon_72_plus",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build meta_inputs_zero_slices.parquet (bucketed join)")
    p.add_argument(
        "--base", type=Path, required=True, help="Candidates parquet (src_id,dst_id,label)"
    )
    p.add_argument(
        "--scores-root",
        type=Path,
        required=True,
        help="Directory with staged *_scores.parquet files",
    )
    p.add_argument("--out-root", type=Path, required=True, help="Output directory for meta inputs")
    p.add_argument("--buckets", type=int, default=64, help="Number of src_id buckets")
    p.add_argument("--threads", type=int, default=4, help="DuckDB threads")
    p.add_argument(
        "--keep-parts", action="store_true", help="Keep meta_inputs_dir parts after consolidation"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs under out-root"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_path = args.base
    scores_root = args.scores_root
    out_root = args.out_root

    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if not scores_root.exists():
        raise FileNotFoundError(scores_root)

    out_root.mkdir(parents=True, exist_ok=True)
    parts_dir = out_root / "meta_inputs_dir"
    out_file = out_root / "meta_inputs_zero_slices.parquet"
    summary_file = out_root / "meta_inputs_zero_slices.json"

    if args.overwrite:
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        if out_file.exists():
            out_file.unlink()
        if summary_file.exists():
            summary_file.unlink()
    else:
        if out_file.exists():
            raise FileExistsError(out_file)

    parts_dir.mkdir(parents=True, exist_ok=True)

    score_paths = {
        "graphsage": scores_root / "graphsage_scores.parquet",
        "node2vec": scores_root / "node2vec_scores.parquet",
        "heuristics": scores_root / "heuristics_scores.parquet",
        "twotower": scores_root / "twotower_scores.parquet",
        "n2v_temporal": scores_root / "n2v_temporal_scores.parquet",
        "tgnn": scores_root / "tgnn_scores.parquet",
    }
    for name, pth in score_paths.items():
        if not pth.exists():
            raise FileNotFoundError(f"Missing score file for {name}: {pth}")

    conn = duckdb.connect(database=":memory:")
    conn.execute(f"SET threads TO {int(args.threads)};")
    conn.execute("SET preserve_insertion_order=false;")

    start = time.perf_counter()
    print(f"[meta] base={base_path}")
    print(f"[meta] scores_root={scores_root}")
    print(f"[meta] out_root={out_root}")
    print(f"[meta] buckets={args.buckets} threads={args.threads}", flush=True)

    # Bucketed join on src_id to keep memory bounded.
    for b in range(int(args.buckets)):
        part = parts_dir / f"part_{b:03d}.parquet"
        sql = f"""
        COPY (
          SELECT
            base.src_id,
            base.dst_id,
            base.label,
            COALESCE(gs.calibrated, 0.0) AS prob_graphsage,
            COALESCE(n2v.calibrated, 0.0) AS prob_node2vec,
            COALESCE(h.calibrated, 0.0) AS prob_heuristics,
            COALESCE(tt.calibrated, 0.0) AS prob_twotower,
            COALESCE(tg.calibrated, 0.0) AS prob_tgnn,
            COALESCE(nt.calibrated, 0.0) AS prob_n2v_temporal
          FROM (
            SELECT DISTINCT src_id, dst_id, label
            FROM read_parquet('{base_path.as_posix()}')
            WHERE (src_id % {int(args.buckets)}) = {b}
          ) base
          LEFT JOIN (
            SELECT DISTINCT src_id, dst_id, calibrated
            FROM read_parquet('{score_paths["graphsage"].as_posix()}')
          ) gs USING (src_id, dst_id)
          LEFT JOIN (
            SELECT DISTINCT src_id, dst_id, calibrated
            FROM read_parquet('{score_paths["node2vec"].as_posix()}')
          ) n2v USING (src_id, dst_id)
          LEFT JOIN (
            SELECT DISTINCT src_id, dst_id, calibrated
            FROM read_parquet('{score_paths["heuristics"].as_posix()}')
          ) h USING (src_id, dst_id)
          LEFT JOIN (
            SELECT DISTINCT src_id, dst_id, calibrated
            FROM read_parquet('{score_paths["twotower"].as_posix()}')
          ) tt USING (src_id, dst_id)
          LEFT JOIN (
            SELECT DISTINCT src_id, dst_id, calibrated
            FROM read_parquet('{score_paths["tgnn"].as_posix()}')
          ) tg USING (src_id, dst_id)
          LEFT JOIN (
            SELECT DISTINCT src_id, dst_id, calibrated
            FROM read_parquet('{score_paths["n2v_temporal"].as_posix()}')
          ) nt USING (src_id, dst_id)
        ) TO '{part.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
        conn.execute(sql)
        if (b + 1) % 8 == 0:
            # Lightweight progress print.
            cnt = int(
                conn.execute(f"SELECT COUNT(*) FROM read_parquet('{part.as_posix()}')").fetchone()[
                    0
                ]
            )
            print(f"[meta] bucket {b + 1}/{args.buckets} wrote {cnt:,} rows", flush=True)

    # Consolidate + add zero slice flags.
    zero_slices = ",\n            ".join([f"0::INTEGER AS {c}" for c in SLICE_COLS])
    conn.execute(
        f"""
        COPY (
          SELECT
            src_id,
            dst_id,
            label,
            0::INTEGER AS fold,
            {zero_slices},
            prob_graphsage,
            prob_node2vec,
            prob_heuristics,
            prob_twotower,
            prob_tgnn,
            prob_n2v_temporal
          FROM read_parquet('{(parts_dir / "part_*.parquet").as_posix()}')
        ) TO '{out_file.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    rows = int(
        conn.execute(f"SELECT COUNT(*) FROM read_parquet('{out_file.as_posix()}')").fetchone()[0]
    )
    nulls = int(
        conn.execute(
            f"""
            SELECT
              SUM((prob_graphsage IS NULL)::INT
                + (prob_node2vec IS NULL)::INT
                + (prob_heuristics IS NULL)::INT
                + (prob_twotower IS NULL)::INT
                + (prob_tgnn IS NULL)::INT
                + (prob_n2v_temporal IS NULL)::INT) AS c
            FROM read_parquet('{out_file.as_posix()}')
            """
        ).fetchone()[0]
    )
    distinct_pairs = int(
        conn.execute(
            f"SELECT COUNT(*) FROM (SELECT DISTINCT src_id, dst_id FROM read_parquet('{out_file.as_posix()}'))"
        ).fetchone()[0]
    )
    elapsed = time.perf_counter() - start

    summary = {
        "base": str(base_path),
        "scores_root": str(scores_root),
        "buckets": int(args.buckets),
        "threads": int(args.threads),
        "rows": rows,
        "distinct_pairs": distinct_pairs,
        "prob_null_count": nulls,
        "outputs": {
            "meta_inputs_zero_slices": str(out_file),
            "meta_inputs_parts_dir": str(parts_dir),
        },
        "elapsed_seconds": elapsed,
    }
    summary_file.write_text(json.dumps(summary, indent=2))
    conn.close()

    if not args.keep_parts:
        shutil.rmtree(parts_dir, ignore_errors=True)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
