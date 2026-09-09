from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from src.ensemble.utils import ensure_dir


def build_expression(intercept: float, coefficients: dict[str, float]) -> str:
    terms: list[str] = [f"({intercept})"]
    for feat, coef in coefficients.items():
        terms.append(f"({coef}) * {feat}")
    return " + ".join(terms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a trained meta-ranker to meta_dataset features"
    )
    parser.add_argument("--model-config", required=True, help="Path to meta_model.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--meta-input", help="Path to meta_inputs parquet file")
    group.add_argument("--meta-dataset", help="Directory containing meta_inputs_<split>.parquet")
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Split to score when --meta-dataset is used",
    )
    parser.add_argument("--output", required=True, help="Destination parquet for scores")
    parser.add_argument(
        "--retain-features", action="store_true", help="Keep feature columns alongside meta_prob"
    )
    parser.add_argument(
        "--extra-columns",
        nargs="*",
        default=["label"],
        help="Additional columns to retain (default keeps label)",
    )
    parser.add_argument(
        "--prob-column", default="meta_prob", help="Name for probability column in output"
    )
    parser.add_argument(
        "--logit-column", default="meta_logit", help="Name for logit column in output"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.model_config).read_text())
    intercept = float(config.get("intercept", 0.0))
    coefficients = {k: float(v) for k, v in config.get("coefficients", {}).items()}
    feature_order = config.get("feature_order", list(coefficients.keys()))
    if not coefficients:
        raise RuntimeError("Model config missing coefficients")

    if args.meta_input:
        input_path = Path(args.meta_input)
    else:
        input_path = Path(args.meta_dataset) / f"meta_inputs_{args.split}.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Meta input parquet not found: {input_path}")

    out_path = Path(args.output)
    ensure_dir(out_path.parent)

    logit_expr = build_expression(intercept, coefficients)

    select_cols: list[str] = ["src_id", "dst_id"]
    extra_cols = list(dict.fromkeys(args.extra_columns))  # remove duplicates preserving order
    for col in extra_cols:
        if col not in select_cols:
            select_cols.append(col)
    if args.retain_features:
        for feat in feature_order:
            if feat not in select_cols:
                select_cols.append(feat)
    select_list = ", ".join(select_cols)

    prob_col = args.prob_column
    logit_col = args.logit_column

    sql = f"""
        SELECT {select_list},
               {logit_expr} AS {logit_col},
               1.0 / (1.0 + EXP(-({logit_col}))) AS {prob_col}
        FROM read_parquet('{input_path.as_posix()}')
    """

    conn = duckdb.connect(database=":memory:")
    conn.execute(f"COPY ({sql}) TO '{out_path.as_posix()}' (FORMAT 'parquet')")
    conn.close()


if __name__ == "__main__":
    main()
