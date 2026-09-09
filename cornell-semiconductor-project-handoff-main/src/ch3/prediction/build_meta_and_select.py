#!/usr/bin/env python3
"""
Chapter 3: build meta inputs from staged model scores and apply the meta-ranker + selection.

This version writes a partitioned meta_inputs directory to avoid single-file size limits,
then uses apply_meta.py with --meta-dataset to produce meta_scores, and finally applies
Core FHPE and DCS Top-5 selections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def main() -> None:
    p = argparse.ArgumentParser(description="Build meta inputs and apply meta-ranker for Chapter 3")
    p.add_argument("--base", default="artifacts/ch3/prediction/meta_base_tmp/base_test.parquet")
    p.add_argument("--scores-root", default="artifacts/ch3/prediction/ch3_scores")
    p.add_argument("--out-root", default="artifacts/ch3/prediction")
    p.add_argument(
        "--meta-model", default="artifacts/ensemble/meta_ranker/meta_ranker_v4/meta_model.json"
    )
    p.add_argument("--core-threshold", type=float, default=0.92)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--topk-floor", type=float, default=0.7)
    args = p.parse_args()

    base_path = Path(args.base)
    scores_root = Path(args.scores_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

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

    meta_dir = out_root / "meta_inputs_dir"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_parquet = meta_dir / "meta_inputs.parquet"

    sql = f"""
    COPY (
      SELECT
        base.*,
        (src_id / 10) AS src_id_div10,
        gs.calibrated AS prob_graphsage,
        n2v.calibrated AS prob_node2vec,
        h.calibrated AS prob_heuristics,
        tt.calibrated AS prob_twotower,
        nt.calibrated AS prob_n2v_temporal,
        tg.calibrated AS prob_tgnn
      FROM read_parquet('{base_path.as_posix()}') base
      LEFT JOIN read_parquet('{score_paths["graphsage"].as_posix()}') gs USING (src_id, dst_id)
      LEFT JOIN read_parquet('{score_paths["node2vec"].as_posix()}') n2v USING (src_id, dst_id)
      LEFT JOIN read_parquet('{score_paths["heuristics"].as_posix()}') h USING (src_id, dst_id)
      LEFT JOIN read_parquet('{score_paths["twotower"].as_posix()}') tt USING (src_id, dst_id)
      LEFT JOIN read_parquet('{score_paths["n2v_temporal"].as_posix()}') nt USING (src_id, dst_id)
      LEFT JOIN read_parquet('{score_paths["tgnn"].as_posix()}') tg USING (src_id, dst_id)
    )
    TO '{meta_parquet.as_posix()}' (FORMAT 'parquet', PARTITION_BY (src_id_div10), OVERWRITE_OR_IGNORE 1);
    """
    conn = duckdb.connect(database=":memory:")
    conn.execute("SET threads TO 4;")  # limit resource use
    conn.execute(sql)
    conn.close()
    print(f"Meta inputs written to partitioned dir: {meta_dir}")

    # Apply meta-ranker using apply_meta.py with --meta-dataset
    meta_scores = out_root / "meta_scores.parquet"
    cmd = [
        "python",
        "src/ensemble/apply_meta.py",
        "--model-config",
        str(Path(args.meta_model)),
        "--meta-dataset",
        str(meta_dir),
        "--split",
        "test",
        "--output",
        str(meta_scores),
    ]
    import subprocess

    subprocess.run(cmd, check=True)
    print(f"Meta scores -> {meta_scores}")

    # Selection policies
    pred_core = out_root / "pred_edges_core.parquet"
    pred_dcs5 = out_root / "pred_edges_dcs5.parquet"
    conn = duckdb.connect(database=":memory:")
    conn.execute(
        f"COPY (SELECT * FROM read_parquet('{meta_scores.as_posix()}') "
        f"WHERE meta_prob >= {args.core_threshold}) "
        f"TO '{pred_core.as_posix()}' (FORMAT 'parquet');"
    )
    conn.execute(
        f"""
        COPY (
          WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY src_id ORDER BY meta_prob DESC, dst_id) AS rk
            FROM read_parquet('{meta_scores.as_posix()}')
            WHERE meta_prob >= {args.topk_floor}
          )
          SELECT * FROM ranked WHERE rk <= {args.topk}
        ) TO '{pred_dcs5.as_posix()}' (FORMAT 'parquet');
        """
    )
    conn.close()
    print(f"Core FHPE edges -> {pred_core}")
    print(f"DCS Top-{args.topk} edges -> {pred_dcs5}")


if __name__ == "__main__":
    main()
