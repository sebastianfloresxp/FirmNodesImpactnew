"""Train the logistic meta-ranker (stacked ensemble) on calibrated base-model scores.

Reads per-model calibrated score Parquets, joins them into a feature matrix with
structural and temporal slice indicators, fits a logistic regression meta-ranker on
the validation split, and writes ensemble scores for validation and test candidate
pools. The output feeds selection-policy evaluation and Chapter 3 inference.

Usage:
    python -m ensemble.train_meta --score-roots artifacts/scores/ --out artifacts/ensemble/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.ensemble.utils import build_join_sql, discover_score_files, ensure_dir, now_iso, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit logistic meta-ranker on validation scores")
    parser.add_argument(
        "--score-root", action="append", default=[], help="Path to results/<model>/<tag> directory"
    )
    parser.add_argument(
        "--meta-dataset", default="", help="Directory containing meta_inputs_{val,test}.parquet"
    )
    parser.add_argument("--out-dir", default="artifacts/ensemble/meta_ranker")
    parser.add_argument("--tag", default="meta_ranker_v1", help="Name for saved meta model")
    parser.add_argument("--sample-rows", type=int, default=1_000_000)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--use-logits", action="store_true")
    parser.add_argument("--balanced", action="store_true", help="Use class_weight='balanced'")
    parser.add_argument("--penalty", default="l2", choices=["l2", "none"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", default="lbfgs")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    feature_cols: list[str]
    input_sources: list[str]

    if args.meta_dataset:
        meta_dir = Path(args.meta_dataset)
        val_path = meta_dir / "meta_inputs_val.parquet"
        if not val_path.exists():
            raise FileNotFoundError(f"Meta dataset missing: {val_path}")
        conn = duckdb.connect(database=":memory:")
        if args.sample_rows > 0:
            sample_query = f"SELECT * FROM read_parquet('{val_path.as_posix()}') USING SAMPLE {int(args.sample_rows)} ROWS"
        else:
            sample_query = f"SELECT * FROM read_parquet('{val_path.as_posix()}')"
        if args.verbose:
            print(f"[INFO] Sampling validation data: {sample_query}")
        df = conn.execute(sample_query).df()
        conn.close()
        df = df.dropna()
        if df.empty:
            raise RuntimeError("Sampled dataframe from meta dataset is empty after dropping NaNs")
        exclude = {"src_id", "dst_id", "label", "fold"}
        feature_cols = [c for c in df.columns if c not in exclude]
        if not feature_cols:
            raise RuntimeError("No feature columns found in meta dataset")
        input_sources = [str(meta_dir)]
    else:
        score_roots = [Path(p) for p in args.score_root]
        if not score_roots:
            raise RuntimeError("Provide either --meta-dataset or at least one --score-root")
        files = discover_score_files(score_roots)
        if not files:
            raise RuntimeError("No score parquet files discovered")
        sql, feature_cols = build_join_sql(files, split="val", include_logits=args.use_logits)
        conn = duckdb.connect(database=":memory:")
        conn.execute(f"CREATE OR REPLACE TEMP VIEW joined_val AS {sql}")
        if args.sample_rows > 0:
            sample_query = f"SELECT * FROM joined_val USING SAMPLE {int(args.sample_rows)} ROWS"
        else:
            sample_query = "SELECT * FROM joined_val"
        if args.verbose:
            print(f"[INFO] Sampling validation data: {sample_query}")
        df = conn.execute(sample_query).df()
        conn.close()
        df = df.dropna()
        if df.empty:
            raise RuntimeError("Sampled dataframe is empty after dropping NaNs")
        input_sources = [str(p) for p in score_roots]

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)

    class_weight = "balanced" if args.balanced else None
    clf = LogisticRegression(
        max_iter=args.max_iter,
        class_weight=class_weight,
        penalty="none" if args.penalty == "none" else args.penalty,
        C=args.C,
        solver=args.solver,
    )
    clf.fit(X, y)

    probs = clf.predict_proba(X)[:, 1]
    metrics = {
        "log_loss": float(log_loss(y, probs)),
        "brier": float(brier_score_loss(y, probs)),
        "auc": float(roc_auc_score(y, probs)),
        "positive_rate": float(probs.mean()),
    }

    if args.verbose:
        print(f"[INFO] Metrics: {metrics}")

    coeffs = clf.coef_[0]
    payload = {
        "created_at": now_iso(),
        "tag": args.tag,
        "model": "logistic_regression",
        "intercept": float(clf.intercept_[0]),  # type: ignore[index]
        "coefficients": {name: float(val) for name, val in zip(feature_cols, coeffs, strict=False)},
        "feature_order": feature_cols,
        "class_weight": class_weight,
        "sample_rows": len(df),
        "input_roots": input_sources,
        "options": {
            "use_logits": bool(args.use_logits),
            "penalty": args.penalty,
            "C": float(args.C),
            "solver": args.solver,
            "max_iter": int(args.max_iter),
        },
        "metrics": metrics,
    }

    out_dir = Path(args.out_dir) / args.tag
    ensure_dir(out_dir)
    model_path = out_dir / "meta_model.json"
    save_json(model_path, payload)
    if args.verbose:
        print(f"[INFO] Saved meta model to {model_path}")


if __name__ == "__main__":
    main()
