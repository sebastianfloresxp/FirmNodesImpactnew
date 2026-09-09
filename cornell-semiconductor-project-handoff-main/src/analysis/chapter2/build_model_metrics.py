#!/usr/bin/env python3
"""Aggregate model performance metrics across slices and splits.

This script rewrites the "meta_evaluation" table in a reproducible way
by recomputing ranking metrics for each base model and the meta-ranker
using the consolidated meta_dataset parquet files. It supports both
validation and test splits and outputs a single CSV with global and
slice-level metrics (including temporal horizons).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

META_DATASET_DIR = Path("results/ensemble/meta_dataset_v4")
META_RANKER_DIR = Path("results/ensemble/meta_ranker/meta_ranker_v4")
OUTPUT_DEFAULT = Path("results/analysis/model_slice_metrics_v4.csv")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    score_path_template: str
    score_column: str
    color: str | None = None


BASE_MODEL_SPECS: Sequence[ModelSpec] = (
    ModelSpec("twotower", "Two-Tower", "twotower_{split}.parquet", "prob_twotower"),
    ModelSpec("node2vec", "Node2Vec", "node2vec_{split}.parquet", "prob_node2vec"),
    ModelSpec(
        "n2v_temporal",
        "Node2Vec-Temporal",
        "n2v_temporal_{split}.parquet",
        "prob_n2v_temporal",
    ),
    ModelSpec("graphsage", "GraphSAGE", "graphsage_{split}.parquet", "prob_graphsage"),
    ModelSpec("tgnn", "TGNN", "tgnn_{split}.parquet", "prob_tgnn"),
    ModelSpec(
        "heuristics", "Heuristic (PA/CN/AA)", "heuristics_{split}.parquet", "prob_heuristics"
    ),
)

META_SPEC = ModelSpec(
    "ensemble",
    "Meta-ranker",
    "scores_{split}.parquet",
    "meta_prob",
)

SLICE_GROUPS: dict[str, Sequence[tuple[str, str]]] = {
    "overall": [("overall", "overall")],
    "structure": [
        ("slice_WW", "WW flag"),
        ("slice_WW3", "WW3 flag"),
        ("slice_CC", "CC flag"),
        ("slice_CC3", "CC3 flag"),
        ("slice_twohop", "Two-hop flag"),
        ("slice_gt2hop", ">2-hop flag"),
        ("slice_deg_q1", "deg_q1 flag"),
        ("slice_deg_q2", "deg_q2 flag"),
        ("slice_deg_q3", "deg_q3 flag"),
        ("slice_deg_q4", "deg_q4 flag"),
    ],
    "temporal": [
        ("slice_horizon_0_6", "Horizon 0-6h"),
        ("slice_horizon_6_12", "Horizon 6-12h"),
        ("slice_horizon_12_24", "Horizon 12-24h"),
        ("slice_horizon_24_36", "Horizon 24-36h"),
        ("slice_horizon_36_48", "Horizon 36-48h"),
        ("slice_horizon_48_60", "Horizon 48-60h"),
        ("slice_horizon_60_72", "Horizon 60-72h"),
        ("slice_horizon_72_plus", "Horizon 72h+"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model slice metrics dataset")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["val", "test"],
        choices=["val", "test"],
        help="Which splits to process",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help="Path to output CSV",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="Optional row sampling for validation/debug (per split)",
    )
    return parser.parse_args()


def create_base_view(
    con: duckdb.DuckDBPyConnection,
    split: str,
    sample_rows: int,
) -> Path:
    base_path = META_DATASET_DIR / f"base_{split}.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base parquet for split {split}: {base_path}")

    sample_clause = f" USING SAMPLE {sample_rows} ROWS" if sample_rows > 0 else ""
    con.execute(
        f"""
        CREATE OR REPLACE VIEW base_data AS
        SELECT *
        FROM read_parquet('{base_path.as_posix()}'){sample_clause}
        """
    )
    return base_path


def collect_model_specs(split: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = list(BASE_MODEL_SPECS)
    specs.append(META_SPEC)
    return specs


def model_score_path(spec: ModelSpec, split: str) -> Path:
    if spec is META_SPEC:
        return META_RANKER_DIR / spec.score_path_template.format(split=split)
    return META_DATASET_DIR / spec.score_path_template.format(split=split)


def register_model_view(
    con: duckdb.DuckDBPyConnection,
    spec: ModelSpec,
    split: str,
    sample_rows: int,
) -> str:
    score_path = model_score_path(spec, split)
    if not score_path.exists():
        raise FileNotFoundError(f"Missing score parquet for {spec.label} ({split}): {score_path}")
    sample_clause = f" USING SAMPLE {sample_rows} ROWS" if sample_rows > 0 else ""
    view_name = f"model_data_{spec.key}"
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT base.*, preds.{spec.score_column} AS score
        FROM base_data AS base
        INNER JOIN read_parquet('{score_path.as_posix()}'){sample_clause} AS preds
            USING (src_id, dst_id)
        WHERE preds.{spec.score_column} IS NOT NULL
        """
    )
    return view_name


def compute_metrics(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    slice_filter: str | None = None,
) -> dict[str, float]:
    filter_clause = ""
    if slice_filter:
        filter_clause = f" AND {slice_filter}"
    query = f"""
        WITH data AS (
            SELECT src_id, label::INTEGER AS label, CAST(score AS DOUBLE) AS score
            FROM {view_name}
            WHERE score IS NOT NULL
            {filter_clause}
        ),
        ranked AS (
            SELECT
                src_id,
                label,
                score,
                ROW_NUMBER() OVER (PARTITION BY src_id ORDER BY score DESC, src_id) AS rn,
                SUM(label) OVER (PARTITION BY src_id) AS total_pos,
                SUM(label) OVER (
                    PARTITION BY src_id
                    ORDER BY score DESC, src_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cum_pos
            FROM data
        ),
        qualified AS (
            SELECT * FROM ranked WHERE total_pos > 0
        ),
        per_src AS (
            SELECT
                src_id,
                MAX(CASE WHEN label = 1 AND rn <= 1 THEN 1 ELSE 0 END) AS hit1,
                MAX(CASE WHEN label = 1 AND rn <= 10 THEN 1 ELSE 0 END) AS hit10,
                MAX(CASE WHEN label = 1 AND rn <= 50 THEN 1 ELSE 0 END) AS hit50,
                MIN(CASE WHEN label = 1 THEN rn END) AS first_pos,
                SUM(CASE WHEN label = 1 AND rn <= 100 THEN cum_pos::DOUBLE / rn END) AS sum_prec,
                SUM(CASE WHEN rn <= 100 THEN label * LN(2) / LN(rn + 1) ELSE 0 END) AS dcg,
                SUM(CASE WHEN rn <= 10 THEN label END) AS pos_in_top10,
                SUM(CASE WHEN rn <= 50 THEN label END) AS pos_in_top50,
                SUM(label) AS total_pos
            FROM qualified
            GROUP BY src_id
        )
        SELECT
            COUNT(*) AS n_sources,
            AVG(hit1) AS hit1,
            AVG(hit10) AS hit10,
            AVG(hit50) AS hit50,
            AVG(CASE WHEN first_pos IS NULL THEN 0 ELSE 1.0 / first_pos END) AS mrr,
            AVG(CASE WHEN total_pos > 0 THEN sum_prec / LEAST(total_pos, 100) ELSE 0 END) AS map100,
            AVG(
                CASE WHEN total_pos > 0 THEN dcg / (
                    SELECT SUM(LN(2) / LN(num + 1))
                    FROM generate_series(1::BIGINT, LEAST(total_pos, 100)::BIGINT) AS g(num)
                ) ELSE 0 END
            ) AS ndcg100,
            AVG(CASE WHEN total_pos > 0 THEN pos_in_top10::DOUBLE / 10 ELSE 0 END) AS precision10,
            AVG(CASE WHEN total_pos > 0 THEN pos_in_top50::DOUBLE / 50 ELSE 0 END) AS precision50,
            AVG(CASE WHEN total_pos > 0 THEN pos_in_top10::DOUBLE / total_pos ELSE 0 END) AS recall10,
            AVG(CASE WHEN total_pos > 0 THEN pos_in_top50::DOUBLE / total_pos ELSE 0 END) AS recall50
        FROM per_src
    """
    result = con.execute(query).fetchone()
    if result is None:
        return {}
    keys = [
        "n_sources",
        "hit1",
        "hit10",
        "hit50",
        "mrr",
        "map100",
        "ndcg100",
        "precision10",
        "precision50",
        "recall10",
        "recall50",
    ]
    return {
        k: float(v) if v is not None else float("nan") for k, v in zip(keys, result, strict=False)
    }


def evaluate_model(
    con: duckdb.DuckDBPyConnection,
    spec: ModelSpec,
    split: str,
    sample_rows: int,
) -> list[dict[str, object]]:
    view_name = register_model_view(con, spec, split, sample_rows)
    records: list[dict[str, object]] = []

    # Overall metrics
    overall_metrics = compute_metrics(con, view_name)
    overall_metrics.update(
        {
            "split": split,
            "model": spec.label,
            "model_key": spec.key,
            "slice_column": "overall",
            "slice_label": "overall",
            "slice_group": "overall",
        }
    )
    records.append(overall_metrics)

    for group, slices in SLICE_GROUPS.items():
        if group == "overall":
            continue
        for column, label in slices:
            metrics = compute_metrics(con, view_name, f"{column} = 1")
            metrics.update(
                {
                    "split": split,
                    "model": spec.label,
                    "model_key": spec.key,
                    "slice_column": column,
                    "slice_label": label,
                    "slice_group": group,
                }
            )
            records.append(metrics)

    con.execute(f"DROP VIEW {view_name}")
    return records


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    for split in args.splits:
        con = duckdb.connect()
        create_base_view(con, split, args.sample_rows)
        for spec in collect_model_specs(split):
            rows.extend(evaluate_model(con, spec, split, args.sample_rows))
        con.execute("DROP VIEW base_data")
        con.close()

    df = pd.DataFrame(rows)
    df.sort_values(by=["split", "slice_group", "slice_label", "model_key"], inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
