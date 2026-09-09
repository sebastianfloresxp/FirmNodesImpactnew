from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.ensemble.utils import (
    ParquetAppend,
    build_join_sql,
    discover_score_files,
    ensure_dir,
    load_json,
    now_iso,
    save_json,
)
from src.graphsage.run_graphsage_eval import KM


def build_logit_expression(intercept: float, coefficients: dict[str, float]) -> str:
    terms = [f"({intercept})"]
    for feature, coef in coefficients.items():
        terms.append(f"({coef}) * {feature}")
    return " + ".join(terms)


def write_meta_scores(
    conn: duckdb.DuckDBPyConnection,
    split: str,
    join_sql: str,
    feature_cols: list[str],
    logit_expr: str,
    retain_features: bool,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    base_cols = ["src_id", "dst_id", "label"]
    select_cols = base_cols.copy()
    if retain_features:
        select_cols.extend(feature_cols)
    select_cols.append(f"({logit_expr}) AS meta_logit")
    inner = f"SELECT {', '.join(select_cols)} FROM ({join_sql})"
    outer = "SELECT *, 1.0 / (1.0 + exp(-meta_logit)) AS meta_prob FROM (" + inner + ")"
    ordered = f"SELECT * FROM ({outer}) ORDER BY src_id, dst_id"
    copy_sql = f"COPY ({ordered}) TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION 'zstd')"
    conn.execute(copy_sql)


def compute_rank_metrics(path: Path, Ks: list[int], batch_size: int) -> dict[str, object]:
    pf = pq.ParquetFile(str(path))
    macro = KM(Ks=Ks)
    micro = KM(Ks=Ks)
    total_rows = 0
    total_sources = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        if df.empty:
            continue
        df = df.sort_values(["src_id", "dst_id"], kind="mergesort")
        for _src, sub in df.groupby("src_id", sort=False):
            probs = sub["meta_prob"].to_numpy(dtype=np.float64, copy=False)
            dst = sub["dst_id"].to_numpy(dtype=np.int64, copy=False)
            labels = sub["label"].to_numpy(dtype=np.int64, copy=False)
            order = np.lexsort((dst, -probs))
            y = labels[order]
            macro.upd(y)
            micro.upd(y)
            total_sources += 1
            total_rows += len(sub)
    metrics = {
        "macro": macro.row(True, "meta"),
        "micro": micro.row(False, "meta"),
        "counts": {"rows": int(total_rows), "sources": int(total_sources)},
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate logistic meta-ranker on full candidate sets"
    )
    parser.add_argument(
        "--model-config", required=True, help="Path to meta_model.json produced by train_meta.py"
    )
    parser.add_argument(
        "--score-root", action="append", default=[], help="Path to results/<model>/<tag> directory"
    )
    parser.add_argument(
        "--meta-dataset", default="", help="Directory containing meta_inputs_{val,test}.parquet"
    )
    parser.add_argument("--out-dir", default="results/ensemble/meta_ranker")
    parser.add_argument("--splits", nargs="*", default=["val", "test"])
    parser.add_argument("--retain-features", action="store_true")
    parser.add_argument("--Ks", default="1,10,50")
    parser.add_argument("--batch-size", type=int, default=2_000_000)
    parser.add_argument("--tag", default="meta_ranker_v1", help="Output tag under out-dir")

    args = parser.parse_args()

    config = load_json(Path(args.model_config))
    intercept = float(cast(float, config.get("intercept", 0.0)))
    coefficients = {
        k: float(cast(float, v))
        for k, v in cast(dict[str, object], config.get("coefficients", {})).items()
    }
    feature_order = cast(list[str], config.get("feature_order", list(coefficients.keys())))
    if not coefficients:
        raise RuntimeError("Meta model config missing coefficients")

    splits_results: dict[str, object] = {}
    metrics_payload: dict[str, object] = {
        "created_at": now_iso(),
        "tag": args.tag,
        "splits": splits_results,
        "features": feature_order,
    }

    out_root = Path(args.out_dir) / args.tag
    ensure_dir(out_root)

    if args.meta_dataset:
        meta_dir = Path(args.meta_dataset)
        for split in args.splits:
            dataset_path = meta_dir / f"meta_inputs_{split}.parquet"
            if not dataset_path.exists():
                raise FileNotFoundError(f"Meta dataset split missing: {dataset_path}")
            pf = pq.ParquetFile(dataset_path)
            available = set(pf.schema.names)
            missing = [feat for feat in feature_order if feat not in available]
            if missing:
                raise RuntimeError(f"Meta dataset for split {split} missing features: {missing}")
            out_path = out_root / f"scores_{split}.parquet"
            ensure_dir(out_path.parent)
            writer = ParquetAppend(out_path)
            for batch in pf.iter_batches(batch_size=args.batch_size):
                df = batch.to_pandas()
                logit = np.full(len(df), intercept, dtype=np.float64)
                for feat, coef in coefficients.items():
                    logit += coef * df[feat].to_numpy(dtype=np.float64, copy=False)
                prob = 1.0 / (1.0 + np.exp(-logit))
                out_df = pd.DataFrame(
                    {
                        "src_id": df["src_id"].to_numpy(dtype=np.int64, copy=False),
                        "dst_id": df["dst_id"].to_numpy(dtype=np.int64, copy=False),
                        "label": df.get(
                            "label", pd.Series(np.zeros(len(df), dtype=np.int8))
                        ).to_numpy(dtype=np.int8, copy=False),
                        "meta_logit": logit.astype(np.float32),
                        "meta_prob": prob.astype(np.float32),
                    }
                )
                if args.retain_features:
                    out_df = pd.concat([out_df, df[feature_order]], axis=1)
                writer.write(out_df)
            writer.close()
            Ks = [int(k.strip()) for k in str(args.Ks).split(",") if k.strip()]
            metrics = compute_rank_metrics(out_path, Ks, args.batch_size)
            splits_results[split] = metrics
    else:
        score_roots = [Path(p) for p in args.score_root]
        if not score_roots:
            raise RuntimeError("Provide either --meta-dataset or at least one --score-root")
        files = discover_score_files(score_roots)
        if not files:
            raise RuntimeError("No score parquets discovered")
        include_logits = any(feat.startswith("logit_") for feat in feature_order)
        conn = duckdb.connect(database=":memory:")
        for split in args.splits:
            join_sql, available_features = build_join_sql(
                files, split=split, include_logits=include_logits
            )
            missing = [feat for feat in feature_order if feat not in available_features]
            if missing:
                raise RuntimeError(f"Missing features for split {split}: {missing}")
            logit_expr = build_logit_expression(intercept, coefficients)
            out_path = out_root / f"scores_{split}.parquet"
            write_meta_scores(
                conn,
                split,
                join_sql,
                available_features,
                logit_expr,
                args.retain_features,
                out_path,
            )
            Ks = [int(k.strip()) for k in str(args.Ks).split(",") if k.strip()]
            metrics = compute_rank_metrics(out_path, Ks, args.batch_size)
            splits_results[split] = metrics
        conn.close()

    metrics_path = out_root / "metrics.json"
    save_json(metrics_path, metrics_payload)


if __name__ == "__main__":
    main()
