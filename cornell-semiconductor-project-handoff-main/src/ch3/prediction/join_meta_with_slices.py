#!/usr/bin/env python3
"""
Build meta_inputs with slice flags + six model probabilities (deduped) for Chapter 3.

Pipeline:
 1) Deduplicate candidate pairs (src_id, dst_id, label).
 2) Generate slice flags using build_base_features (same as Chapter 2 base features).
 3) Join slice features with calibrated probabilities from the six models.

Outputs:
  artifacts/ch3/prediction/meta_inputs.parquet

Schema matches the Chapter 2 meta-ranker (slice_* columns + prob_* columns + label/src/dst).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import duckdb

from src.ensemble.build_meta_dataset import build_base_features


def dedup_candidates(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(database=":memory:")
    conn.execute(
        f"COPY (SELECT DISTINCT src_id, dst_id, label FROM read_parquet('{src.as_posix()}')) "
        f"TO '{dst.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');"
    )
    conn.close()
    print(f"[dedup] wrote {dst}")


def join_meta(base_with_slices: Path, scores_root: Path, out_path: Path) -> None:
    score_paths = {
        "graphsage": scores_root / "graphsage_scores.parquet",
        "node2vec": scores_root / "node2vec_scores.parquet",
        "heuristics": scores_root / "heuristics_scores.parquet",
        "twotower": scores_root / "twotower_scores.parquet",
        "n2v_temporal": scores_root / "n2v_temporal_scores.parquet",
        "tgnn": scores_root / "tgnn_scores.parquet",
    }
    for _name, pth in score_paths.items():
        if not pth.exists():
            raise FileNotFoundError(f"Missing score file: {pth}")

    conn = duckdb.connect(database=":memory:")
    sql = f"""
    COPY (
      SELECT base.*, gs.calibrated AS prob_graphsage,
             n2v.calibrated AS prob_node2vec,
             h.calibrated AS prob_heuristics,
             tt.calibrated AS prob_twotower,
             nt.calibrated AS prob_n2v_temporal,
             tg.calibrated AS prob_tgnn
      FROM read_parquet('{base_with_slices.as_posix()}') base
      LEFT JOIN (SELECT DISTINCT src_id, dst_id, calibrated FROM read_parquet('{score_paths["graphsage"].as_posix()}')) gs USING (src_id, dst_id)
      LEFT JOIN (SELECT DISTINCT src_id, dst_id, calibrated FROM read_parquet('{score_paths["node2vec"].as_posix()}')) n2v USING (src_id, dst_id)
      LEFT JOIN (SELECT DISTINCT src_id, dst_id, calibrated FROM read_parquet('{score_paths["heuristics"].as_posix()}')) h USING (src_id, dst_id)
      LEFT JOIN (SELECT DISTINCT src_id, dst_id, calibrated FROM read_parquet('{score_paths["twotower"].as_posix()}')) tt USING (src_id, dst_id)
      LEFT JOIN (SELECT DISTINCT src_id, dst_id, calibrated FROM read_parquet('{score_paths["n2v_temporal"].as_posix()}')) nt USING (src_id, dst_id)
      LEFT JOIN (SELECT DISTINCT src_id, dst_id, calibrated FROM read_parquet('{score_paths["tgnn"].as_posix()}')) tg USING (src_id, dst_id)
    ) TO '{out_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
    """
    conn.execute(sql)
    cnt = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    conn.close()
    print(f"[join] wrote {cnt:,} rows -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build meta_inputs with slices + probs (deduped)")
    p.add_argument(
        "--candidate", default="artifacts/ch3/prediction/meta_base_tmp/base_test.parquet"
    )
    p.add_argument("--scores-root", default="artifacts/ch3/prediction/ch3_scores")
    p.add_argument(
        "--adj", default="data/processed/core/releases/core_v1/adjacency/train_adj_T0.npz"
    )
    p.add_argument("--splits-root", default="data/processed/core/releases/core_v1/splits")
    p.add_argument("--out", default="artifacts/ch3/prediction/meta_inputs.parquet")
    p.add_argument("--temp-dir", default="artifacts/ch3/prediction/meta_tmp")
    p.add_argument("--batch-size", type=int, default=1_000_000)
    p.add_argument(
        "--assume-ts",
        type=float,
        default=None,
        help="If set, attach this ts to all candidates before slicing",
    )
    args = p.parse_args()

    candidate = Path(args.candidate)
    scores_root = Path(args.scores_root)
    adj = Path(args.adj)
    splits = Path(args.splits_root)
    out_path = Path(args.out)
    temp_dir = Path(args.temp_dir)

    # Clean temp
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    dedup_path = temp_dir / "base_dedup.parquet"
    slices_dir = temp_dir / "base_slices"

    # 1) Dedup candidates
    dedup_candidates(candidate, dedup_path)

    # 2) Build slice flags using build_base_features
    print("[slices] building slice flags via build_base_features...", flush=True)
    assume_ts = args.assume_ts
    if assume_ts is not None:
        # Attach a constant ts if requested so horizon buckets are populated.
        import duckdb

        ts_path = temp_dir / "base_with_ts.parquet"
        conn = duckdb.connect(database=":memory:")
        conn.execute(
            f"""
            COPY (
              SELECT src_id, dst_id, label, {assume_ts}::DOUBLE AS ts
              FROM read_parquet('{dedup_path.as_posix()}')
            ) TO '{ts_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
            """
        )
        conn.close()
        cand_for_slices = ts_path
    else:
        cand_for_slices = dedup_path

    base_paths = build_base_features(
        splits_root=splits,
        adj_path=adj,
        cand_val=cand_for_slices,
        cand_test=cand_for_slices,
        out_dir=slices_dir,
        folds=1,
        batch_size=int(args.batch_size),
        assume_ts=False,
        horizon_all=True,
    )
    base_with_slices = Path(base_paths["test"])
    print(f"[slices] slice features at {base_with_slices}", flush=True)

    # 3) Join with probabilities
    join_meta(base_with_slices, scores_root, out_path)

    # Done
    print(f"[done] meta_inputs written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
