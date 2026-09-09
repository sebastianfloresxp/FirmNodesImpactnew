#!/usr/bin/env python3
"""
Chapter 3: Join base candidates with per-model scores (deduped, bucketed).

Steps:
- Deduplicate base and score tables on (src_id, dst_id).
- Process the full dataset in src_id buckets (default 64) to keep memory low.
- Writes partitioned outputs under artifacts/ch3/prediction/meta_inputs_dir/*.parquet.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb


def run_full(
    base_path: Path, scores_root: Path, out_dir: Path, buckets: int = 64, threads: int = 4
) -> None:
    """Process full dataset in src_id buckets with deduplication."""
    out_dir.mkdir(parents=True, exist_ok=True)

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

    start = time.perf_counter()
    conn = duckdb.connect(database=":memory:")
    conn.execute(f"SET threads TO {threads};")

    print(f"[full] Base: {base_path}")
    print(f"[full] Scores root: {scores_root}")
    print(f"[full] Output dir: {out_dir}")
    print(f"[full] Buckets: {buckets}, Threads: {threads}", flush=True)

    for b in range(buckets):
        out_file = out_dir / f"part_{b:03d}.parquet"
        print(f"[bucket {b + 1}/{buckets}] start -> {out_file}", flush=True)
        sql = f"""
        COPY (
          SELECT base.src_id, base.dst_id, base.label,
                 gs.calibrated AS prob_graphsage,
                 n2v.calibrated AS prob_node2vec,
                 h.calibrated AS prob_heuristics,
                 tt.calibrated AS prob_twotower,
                 nt.calibrated AS prob_n2v_temporal,
                 tg.calibrated AS prob_tgnn
          FROM (
            SELECT DISTINCT src_id, dst_id, label
            FROM read_parquet('{base_path.as_posix()}')
            WHERE (src_id % {buckets}) = {b}
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
            FROM read_parquet('{score_paths["n2v_temporal"].as_posix()}')
          ) nt USING (src_id, dst_id)
          LEFT JOIN (
            SELECT DISTINCT src_id, dst_id, calibrated
            FROM read_parquet('{score_paths["tgnn"].as_posix()}')
          ) tg USING (src_id, dst_id)
        ) TO '{out_file.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
        conn.execute(sql)
        # Optional: get row count for this bucket
        cnt = (
            duckdb.connect(database=":memory:")
            .execute(f"SELECT COUNT(*) FROM read_parquet('{out_file.as_posix()}')")
            .fetchone()[0]
        )
        print(f"[bucket {b + 1}/{buckets}] wrote {cnt:,} rows", flush=True)

    conn.close()
    elapsed = time.perf_counter() - start
    print(f"[full] Completed in {elapsed / 60:.1f} minutes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Join meta inputs (deduped, bucketed)")
    parser.add_argument(
        "--base", default="artifacts/ch3/prediction/meta_base_tmp/base_test.parquet"
    )
    parser.add_argument("--scores-root", default="artifacts/ch3/prediction/ch3_scores")
    parser.add_argument("--out-dir", default="artifacts/ch3/prediction/meta_inputs_dir")
    parser.add_argument("--buckets", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    run_full(
        base_path=Path(args.base),
        scores_root=Path(args.scores_root),
        out_dir=Path(args.out_dir),
        buckets=int(args.buckets),
        threads=int(args.threads),
    )


if __name__ == "__main__":
    main()
