#!/usr/bin/env python3
"""Compile ranking scorecards into LaTeX tables for Chapter 2."""

from __future__ import annotations

import contextlib
import math
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MODEL_PATHS = {
    "Heuristics": Path("results/heuristics"),
    "Node2Vec": Path("results/node2vec"),
    "Node2Vec-Temporal": Path("results/n2v_temporal/n2v_temporal_production_v2"),
    "Two-Tower": Path("results/twotower/twotower_production_v2"),
    "GraphSAGE": Path("results/graphsage/graphsage_production_v2"),
    "TGNN": Path("results/tgnn/tgnn_production_v2"),
    "Ensemble": Path("results/ensemble/meta_ranker/meta_ranker_v4"),
}

BASE_RATES_PATH = Path("results/ensemble/meta_ranker/meta_ranker_v4/analysis_tables/base_rates.csv")
OPERATING_LADDER_PATH = Path(
    "results/ensemble/meta_ranker/meta_ranker_v4/analysis_tables/operating_ladder.csv"
)
TOPK_REFERENCE_KEYS = [
    "tau_precision_0.95",
    "tau_precision_0.85",
]

MODEL_COLUMNS = {
    "Heuristics": "prob_heuristics",
    "Node2Vec": "prob_node2vec",
    "Node2Vec-Temporal": "prob_n2v_temporal",
    "Two-Tower": "prob_twotower",
    "GraphSAGE": "prob_graphsage",
    "TGNN": "prob_tgnn",
    "Ensemble": "ens_p",
}

MODEL_ROW_NAMES = {
    "Node2Vec": "Node2Vec",
    "Node2Vec-Temporal": "TGNN",
    "Two-Tower": "TwoTower",
    "GraphSAGE": "GraphSAGE",
    "TGNN": "TGNN",
    "Ensemble": "Ensemble",
}

MODEL_DISPLAY_NAMES = {
    "Heuristics": "Heuristics (PA/CN/AA mean)",
    "Node2Vec": "Node2Vec",
    "Node2Vec-Temporal": "Node2Vec-Temporal",
    "Two-Tower": "Two-Tower",
    "GraphSAGE": "GraphSAGE",
    "TGNN": "TGNN",
    "Ensemble": "Ensemble",
}

GLOBAL_MODEL_ORDER = [
    "Ensemble",
    "TGNN",
    "Node2Vec-Temporal",
    "Node2Vec",
    "Two-Tower",
    "GraphSAGE",
    "Heuristics",
]

MODEL_SLICE_METRICS_PATH = Path("results/analysis/model_slice_metrics_v4.csv")

MODEL_SLICE_NAME_MAP = {
    "Meta-ranker": "Ensemble",
    "Node2Vec": "Node2Vec",
    "Node2Vec-Temporal": "Node2Vec-Temporal",
    "Two-Tower": "Two-Tower",
    "GraphSAGE": "GraphSAGE",
    "TGNN": "TGNN",
    "Heuristic (PA/CN/AA)": "Heuristics",
}

CONSOLIDATED_MODEL_ORDER = [
    "Ensemble",
    "TGNN",
    "Node2Vec-Temporal",
    "Node2Vec",
    "Two-Tower",
    "GraphSAGE",
    "Heuristics",
]

WINNER_BASE_MODELS = [
    "TGNN",
    "Node2Vec-Temporal",
    "Node2Vec",
    "Two-Tower",
    "GraphSAGE",
    "Heuristics::PA",
    "Heuristics::CN",
    "Heuristics::AA",
]


@dataclass(frozen=True)
class ConsolidatedColumnSpec:
    key: str
    label: str
    slice_column: str
    kind: str  # "stacked" or "single"
    flag_column: str | None = None
    source_split: str | None = None


CONSOLIDATED_COLUMNS = [
    ConsolidatedColumnSpec("WW", "WW", "slice_WW", "stacked", "slice_WW"),
    ConsolidatedColumnSpec("CC", "CC", "slice_CC", "stacked", "slice_CC"),
    ConsolidatedColumnSpec(">2HOP", ">2-hop", "slice_gt2hop", "stacked", "slice_gt2hop"),
    ConsolidatedColumnSpec("DEG_Q1", "deg Q1", "slice_deg_q1", "stacked", "slice_deg_q1"),
    ConsolidatedColumnSpec("H_0_6", "AP 0-6m", "slice_horizon_0_6", "stacked"),
    ConsolidatedColumnSpec("H_6_12", "AP 6-12m", "slice_horizon_6_12", "stacked"),
    ConsolidatedColumnSpec("H_36_48", "AP 36-48m", "slice_horizon_36_48", "stacked"),
    ConsolidatedColumnSpec("H_48_60", "AP 48-60m", "slice_horizon_48_60", "stacked"),
]

SLICE_DISPLAY_OVERRIDES = {
    "slice_WW": "WW",
    "slice_WW3": "WW3",
    "slice_WC": "WC",
    "slice_WC3": "WC3",
    "slice_CW": "CW",
    "slice_CW3": "CW3",
    "slice_CC": "CC",
    "slice_CC3": "CC3",
    "slice_twohop": "<=2-hop",
    "slice_gt2hop": ">2-hop",
    "slice_deg_q1": "deg Q1",
    "slice_deg_q2": "deg Q2",
    "slice_deg_q3": "deg Q3",
    "slice_deg_q4": "deg Q4",
}

HORIZON_DISPLAY_OVERRIDES = {
    "slice_horizon_0_6": "AP 0-6m",
    "slice_horizon_6_12": "AP 6-12m",
    "slice_horizon_12_24": "AP 12-24m",
    "slice_horizon_24_36": "AP 24-36m",
    "slice_horizon_36_48": "AP 36-48m",
    "slice_horizon_48_60": "AP 48-60m",
    "slice_horizon_60_72": "AP 60-72m",
    "slice_horizon_72_plus": "AP >72m",
}

SLICE_COLUMN_PRIORITY = {
    "slice_WW": 10,
    "slice_WW3": 20,
    "slice_WC": 30,
    "slice_CW": 40,
    "slice_WC3": 50,
    "slice_CW3": 60,
    "slice_CC": 70,
    "slice_CC3": 80,
    "slice_twohop": 90,
    "slice_gt2hop": 100,
    "slice_deg_q1": 110,
    "slice_deg_q2": 120,
    "slice_deg_q3": 130,
    "slice_deg_q4": 140,
    "slice_horizon_0_6": 200,
    "slice_horizon_6_12": 210,
    "slice_horizon_12_24": 220,
    "slice_horizon_24_36": 230,
    "slice_horizon_36_48": 240,
    "slice_horizon_48_60": 250,
    "slice_horizon_60_72": 260,
    "slice_horizon_72_plus": 270,
}

SLICE_COLUMNS = {
    "slice_WW": "WW",
    "slice_WW3": "WW3",
    "slice_WC": "WC",
    "slice_WC3": "WC3",
    "slice_CW": "CW",
    "slice_CW3": "CW3",
    "slice_CC": "CC",
    "slice_CC3": "CC3",
    "slice_twohop": "twohop",
    "slice_gt2hop": ">2-hop",
    "slice_deg_q1": "deg_q1",
    "slice_deg_q2": "deg_q2",
    "slice_deg_q3": "deg_q3",
    "slice_deg_q4": "deg_q4",
}

METRICS = ["ndcg", "hit10", "hit50", "mrr", "map100"]
METRIC_LABELS = {
    "ndcg": "nDCG@100",
    "hit10": "Hit@10",
    "hit50": "Hit@50",
    "mrr": "MRR",
    "map100": "MAP@100",
}


def build_slice_column_spec(include_lift: bool = False) -> str:
    parts: list[str] = ["@{}l"]
    parts.extend("r" for _ in METRICS)
    parts.append("r")  # Delta nDCG
    parts.append("r")  # Delta Hit@10
    if include_lift:
        parts.append("r")
    parts.append("@{}")
    return "".join(parts)


def build_global_column_spec(include_deltas: bool, include_lift: bool) -> str:
    parts: list[str] = ["@{}l"]
    parts.extend("S[table-format=1.3]" for _ in METRICS)
    if include_deltas:
        parts.extend(["S[table-format=+1.3]", "S[table-format=+1.3]"])
    if include_lift:
        parts.append("S[table-format=3.3]")
    parts.append("@{}")
    return "".join(parts)


HEURISTIC_COMPONENTS = ["PA", "CN", "AA"]
SLICE_NAME_MAP = {
    "WW3": "slice_WW3",
    "WW": "slice_WW",
    "WC": "slice_WC",
    "CW": "slice_CW",
    "CC": "slice_CC",
    "WC3": "slice_WC3",
    "CW3": "slice_CW3",
    "CC3": "slice_CC3",
    ">2hop": "slice_gt2hop",
    "deg_q1": "slice_deg_q1",
    "deg_q2": "slice_deg_q2",
    "deg_q3": "slice_deg_q3",
    "deg_q4": "slice_deg_q4",
    "twohop": "slice_twohop",
    "horizon_0_6": "slice_horizon_0_6",
    "horizon_6_12": "slice_horizon_6_12",
    "horizon_12_24": "slice_horizon_12_24",
    "horizon_24_36": "slice_horizon_24_36",
    "horizon_36_48": "slice_horizon_36_48",
    "horizon_48_60": "slice_horizon_48_60",
    "horizon_60_72": "slice_horizon_60_72",
    "horizon_72_plus": "slice_horizon_72_plus",
}

MISSING_NUMERIC = r"\multicolumn{1}{c}{\NA}"
MISSING_INLINE = r"\NA"
NA_TEXT = r"\NA"


@dataclass
class MetricAggregate:
    mean: dict[str, float]
    std: dict[str, float]
    ci: dict[str, float]
    counts: dict[str, int]
    n_seeds: int
    seed_names: list[str]


@dataclass
class SplitSummary:
    models: list[str]
    global_metrics: dict[str, MetricAggregate]
    global_sources: int
    slice_metrics: dict[str, dict[str, MetricAggregate]]
    slice_sources: dict[str, int]
    heuristics_components: dict[str, dict[str, dict[str, float]]]
    base_rate: float | None
    ensemble_precision: float | None
    ensemble_lift: float | None


@dataclass
class SeedResult:
    name: str
    global_metrics: dict[str, float]
    slice_metrics: dict[str, dict[str, float]]


@dataclass
class ScorecardAssets:
    summary_by_split: dict[str, SplitSummary]
    common_models: list[str]
    slices: list[str]
    split_meta_paths: dict[str, Path]
    split_scores_paths: dict[str, Path]


def compute_metrics(
    con: duckdb.DuckDBPyConnection, score_column: str, slice_filter: str | None = None
) -> tuple[int, dict[str, float]]:
    where_clause = f"WHERE {slice_filter}" if slice_filter else ""
    sql = f"""
WITH ranked AS (
    SELECT
        src_id,
        label,
        {score_column} AS score,
        ROW_NUMBER() OVER (PARTITION BY src_id ORDER BY {score_column} DESC) AS rn,
        SUM(label) OVER (PARTITION BY src_id) AS total_pos,
        SUM(label) OVER (PARTITION BY src_id ORDER BY {score_column} DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_pos
    FROM data
    {where_clause}
),
qualified AS (
    SELECT * FROM ranked WHERE total_pos > 0
),
per_src AS (
    SELECT
        src_id,
        MAX(CASE WHEN label = 1 AND rn <= 10 THEN 1 ELSE 0 END) AS hit10,
        MAX(CASE WHEN label = 1 AND rn <= 50 THEN 1 ELSE 0 END) AS hit50,
        MIN(CASE WHEN label = 1 THEN rn END) AS first_pos,
        SUM(CASE WHEN label = 1 AND rn <= 100 THEN cum_pos::DOUBLE / rn END) AS sum_prec,
        SUM(CASE WHEN rn <= 100 THEN label * LN(2) / LN(rn + 1) ELSE 0 END) AS dcg,
        MAX(total_pos) AS total_pos
    FROM qualified
    GROUP BY src_id
)
SELECT
    COUNT(*) AS n_sources,
    AVG(hit10) AS hit10,
    AVG(hit50) AS hit50,
    AVG(CASE WHEN first_pos IS NULL THEN 0 ELSE 1.0 / first_pos END) AS mrr,
    AVG(CASE WHEN total_pos > 0 THEN sum_prec / LEAST(total_pos, 100) ELSE 0 END) AS map100,
    AVG(
        CASE WHEN total_pos > 0 THEN dcg / (
            SELECT SUM(LN(2) / LN(num + 1))
            FROM generate_series(1::BIGINT, LEAST(CAST(total_pos AS BIGINT), 100::BIGINT)) AS g(num)
        ) ELSE 0 END
    ) AS ndcg
FROM per_src;
"""
    row = con.execute(sql).fetchone()
    if row is None:
        return 0, {metric: float("nan") for metric in METRICS}
    n_sources = int(row[0])
    if n_sources == 0:
        return 0, {metric: float("nan") for metric in METRICS}

    def to_float(val: float | None) -> float:
        return float(val) if val is not None else float("nan")

    values = {
        "hit10": to_float(row[1]),
        "hit50": to_float(row[2]),
        "mrr": to_float(row[3]),
        "map100": to_float(row[4]),
        "ndcg": to_float(row[5]),
    }
    return n_sources, {metric: values[metric] for metric in METRICS}


def macro_rows(df: pd.DataFrame, column: str = "macro") -> pd.DataFrame:
    if column and column in df.columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            mask = series
        elif pd.api.types.is_numeric_dtype(series):
            mask = series != 0
        else:
            mask = series.astype(str).str.lower().isin({"true", "t", "yes", "1"})
        return df[mask]
    return df


def row_to_metrics(row: pd.Series) -> dict[str, float]:
    mapping = {
        "hit@10": "hit10",
        "hit10": "hit10",
        "hit@50": "hit50",
        "hit50": "hit50",
        "mrr": "mrr",
        "map@100": "map100",
        "map": "map100",
        "ndcg@100": "ndcg",
        "ndcg": "ndcg",
    }
    metrics: dict[str, float] = {}
    for source, target in mapping.items():
        if source in row.index:
            try:
                metrics[target] = float(row[source])
            except (TypeError, ValueError):
                metrics[target] = float("nan")
    for metric in METRICS:
        metrics.setdefault(metric, float("nan"))
    return metrics


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: failed to read {path}: {exc}")
        return None


def load_model_seed_results(
    model: str,
    split: str,
    slices: list[str],
    meta_path: Path,
    ensemble_scores_path: Path,
) -> dict[str, SeedResult]:
    base = MODEL_PATHS.get(model)
    if base is None:
        return {}

    slice_lookup = {raw: alias for raw, alias in SLICE_NAME_MAP.items() if alias in slices}
    results: dict[str, SeedResult] = {}

    if model == "Heuristics":
        global_df = load_csv(base / f"global_{split}.csv")
        if global_df is None:
            return {}
        global_df = macro_rows(global_df)
        slice_df = load_csv(base / f"slices_{split}.csv")
        if slice_df is not None:
            slice_df = macro_rows(slice_df)
        for comp in HEURISTIC_COMPONENTS:
            comp_df = global_df
            if "heuristic" in global_df.columns:
                comp_df = global_df[global_df["heuristic"] == comp]
            if comp_df.empty:
                continue
            row = comp_df.iloc[0]
            global_metrics = row_to_metrics(row)
            slice_metrics: dict[str, dict[str, float]] = {}
            if slice_df is not None:
                comp_slice = slice_df
                if "heuristic" in slice_df.columns:
                    comp_slice = slice_df[slice_df["heuristic"] == comp]
                for raw, alias in slice_lookup.items():
                    match = comp_slice[comp_slice["slice_name"] == raw]
                    if not match.empty:
                        slice_metrics[alias] = row_to_metrics(match.iloc[0])
            results[comp] = SeedResult(
                name=comp, global_metrics=global_metrics, slice_metrics=slice_metrics
            )
        return results

    if model == "Ensemble":
        return compute_ensemble_seed_results(meta_path, ensemble_scores_path, slices)

    row_name = MODEL_ROW_NAMES.get(model, model)
    seed_dirs = sorted(p for p in base.glob("seed_*") if p.is_dir())
    for seed_dir in seed_dirs:
        global_path = seed_dir / f"global_{split}.csv"
        global_df = load_csv(global_path)
        if global_df is None:
            matches = list(seed_dir.glob(f"*global_{split}.csv"))
            if matches:
                global_df = load_csv(matches[0])
            if global_df is None:
                continue
        global_df = macro_rows(global_df)
        if "heuristic" in global_df.columns:
            global_df = global_df[global_df["heuristic"] == row_name]
        elif "model" in global_df.columns:
            global_df = global_df[global_df["model"] == row_name]
        if global_df.empty:
            continue
        row = global_df.iloc[0]
        global_metrics = row_to_metrics(row)

        slice_metrics: dict[str, dict[str, float]] = {}
        slice_path = seed_dir / f"slices_{split}.csv"
        slice_df = load_csv(slice_path)
        if slice_df is not None:
            slice_df = macro_rows(slice_df)
            if "heuristic" in slice_df.columns:
                slice_df = slice_df[slice_df["heuristic"] == row_name]
            elif "model" in slice_df.columns:
                slice_df = slice_df[slice_df["model"] == row_name]
            for raw, alias in slice_lookup.items():
                match = slice_df[slice_df["slice_name"] == raw]
                if not match.empty:
                    slice_metrics[alias] = row_to_metrics(match.iloc[0])
        else:
            matches = list(seed_dir.glob(f"*slices_{split}.csv"))
            if matches:
                slice_df = load_csv(matches[0])
                if slice_df is not None:
                    slice_df = macro_rows(slice_df)
                    if "heuristic" in slice_df.columns:
                        slice_df = slice_df[slice_df["heuristic"] == row_name]
                    elif "model" in slice_df.columns:
                        slice_df = slice_df[slice_df["model"] == row_name]
                    for raw, alias in slice_lookup.items():
                        match = slice_df[slice_df["slice_name"] == raw]
                        if not match.empty:
                            slice_metrics[alias] = row_to_metrics(match.iloc[0])

        results[seed_dir.name] = SeedResult(
            name=seed_dir.name, global_metrics=global_metrics, slice_metrics=slice_metrics
        )

    return results


def compute_ensemble_seed_results(
    meta_path: Path, scores_path: Path, slices: list[str]
) -> dict[str, SeedResult]:
    if not scores_path.exists():
        print(f"WARNING: missing ensemble scores at {scores_path}")
        return {}
    con = duckdb.connect()
    slice_select = ",\n       ".join(f"mi.{s} AS {s}" for s in slices)
    select_parts = ["mi.src_id AS src_id", "mi.label AS label", "sc.meta_prob AS score"]
    if slices:
        select_parts.append(slice_select)
    select_clause = ",\n       ".join(select_parts)
    try:
        con.execute(
            "CREATE OR REPLACE TEMP VIEW data AS\n"
            + "SELECT "
            + select_clause
            + f"\nFROM read_parquet('{meta_path.as_posix()}') mi\n"
            + f"JOIN read_parquet('{scores_path.as_posix()}') sc USING (src_id, dst_id)"
        )
    except Exception as exc:
        print(f"WARNING: failed to initialise ensemble view for {scores_path}: {exc}")
        con.close()
        return {}

    _, global_metrics = compute_metrics(con, "score")
    slice_metrics: dict[str, dict[str, float]] = {}
    for s in slices:
        filter_sql = f"src_id IN (SELECT DISTINCT src_id FROM data WHERE label = 1 AND {s} = 1)"
        _, metrics_slice = compute_metrics(con, "score", filter_sql)
        slice_metrics[s] = metrics_slice
    con.close()
    return {
        "ensemble": SeedResult(
            name="ensemble", global_metrics=global_metrics, slice_metrics=slice_metrics
        )
    }


_BASE_RATE_CACHE: dict[str, float] = {}
_OPERATING_CACHE: dict[str, float] = {}
_SEED_RESULTS_CACHE: dict[tuple[str, str, tuple[str, ...]], dict[str, SeedResult]] = {}


def get_base_rate(split: str) -> float | None:
    if split in _BASE_RATE_CACHE:
        return _BASE_RATE_CACHE[split]
    if not BASE_RATES_PATH.exists():
        print(f"WARNING: base rates file missing at {BASE_RATES_PATH}")
        return None
    df = pd.read_csv(BASE_RATES_PATH)
    row = df[(df["split"] == split) & (df["slice"] == "overall")]
    if row.empty:
        print(f"WARNING: base rate not found for split={split}")
        return None
    value = float(row.iloc[0]["base_rate"])
    _BASE_RATE_CACHE[split] = value
    return value


def get_reference_precision(split: str) -> float | None:
    if split in _OPERATING_CACHE:
        return _OPERATING_CACHE[split]
    if not OPERATING_LADDER_PATH.exists():
        print(f"WARNING: operating ladder missing at {OPERATING_LADDER_PATH}")
        return None
    df = pd.read_csv(OPERATING_LADDER_PATH)
    row = None
    for key in TOPK_REFERENCE_KEYS:
        subset = df[(df["split"] == split) & (df["key"] == key)]
        if not subset.empty:
            row = subset
            break
    if row is None or row.empty:
        keys = ", ".join(TOPK_REFERENCE_KEYS)
        print(f"WARNING: Top-K reference(s) {keys} missing for split={split}")
        return None
    value = float(row.iloc[0]["precision"])
    _OPERATING_CACHE[split] = value
    return value


def count_sources(meta_path: Path, slices: list[str]) -> tuple[int, dict[str, int]]:
    con = duckdb.connect()
    try:
        path_str = meta_path.as_posix()
        global_sources = con.execute(
            "SELECT COUNT(*) FROM ("
            f"SELECT src_id FROM read_parquet('{path_str}') WHERE label = 1 GROUP BY src_id"
            ")"
        ).fetchone()[0]
        slice_counts: dict[str, int] = {}
        for s in slices:
            query = (
                "SELECT COUNT(*) FROM ("
                f"SELECT src_id FROM read_parquet('{path_str}') WHERE label = 1 AND {s} = 1 GROUP BY src_id"
                ")"
            )
            slice_counts[s] = con.execute(query).fetchone()[0]
    finally:
        con.close()
    return int(global_sources), {k: int(v) for k, v in slice_counts.items()}


def aggregate_metric_lists(
    metric_lists: dict[str, list[float]], seed_names: list[str]
) -> MetricAggregate:
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    ci: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metric, values in metric_lists.items():
        arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
        counts[metric] = arr.size
        if arr.size == 0:
            mean[metric] = float("nan")
            std[metric] = float("nan")
            ci[metric] = float("nan")
            continue
        mean_val = float(arr.mean())
        mean[metric] = mean_val
        if arr.size >= 2:
            std_val = float(arr.std(ddof=1))
            std[metric] = std_val
            ci[metric] = 1.96 * std_val / math.sqrt(arr.size)
        else:
            std[metric] = float("nan")
            ci[metric] = float("nan")
    n_seeds = len(seed_names)
    return MetricAggregate(
        mean=mean, std=std, ci=ci, counts=counts, n_seeds=n_seeds, seed_names=seed_names
    )


def collect_split(
    meta_path: Path,
    ensemble_scores_path: Path,
    models: list[str],
    slices: list[str],
    split: str,
) -> SplitSummary:
    global_sources, slice_sources = count_sources(meta_path, slices)
    global_metrics: dict[str, MetricAggregate] = {}
    slice_metrics: dict[str, dict[str, MetricAggregate]] = {s: {} for s in slices}
    heuristics_components: dict[str, dict[str, dict[str, float]]] = {}

    final_models: list[str] = []

    base_rate = get_base_rate(split)
    ref_precision = get_reference_precision(split)
    ensemble_lift = None
    if base_rate and ref_precision:
        ensemble_lift = ref_precision / base_rate if base_rate > 0 else None

    for model in models:
        seed_results = load_model_seed_results(
            model, split, slices, meta_path, ensemble_scores_path
        )
        if not seed_results:
            print(f"WARNING: no evaluation records for {model} ({split})")
            continue
        global_lists = {metric: [] for metric in METRICS}
        slice_lists = {s: {metric: [] for metric in METRICS} for s in slices}
        slice_seed_names = {s: [] for s in slices}
        seed_names = list(seed_results.keys())

        for seed_name, seed_result in seed_results.items():
            for metric in METRICS:
                global_lists[metric].append(seed_result.global_metrics.get(metric, float("nan")))
            for s in slices:
                metrics = seed_result.slice_metrics.get(s)
                if metrics:
                    slice_seed_names[s].append(seed_name)
                    for metric in METRICS:
                        slice_lists[s][metric].append(metrics.get(metric, float("nan")))
            if model == "Heuristics":
                heuristics_components[seed_name] = {
                    "global": seed_result.global_metrics,
                    "slices": seed_result.slice_metrics,
                }

        global_metrics[model] = aggregate_metric_lists(global_lists, seed_names)
        for s in slices:
            slice_metrics[s][model] = aggregate_metric_lists(slice_lists[s], slice_seed_names[s])
        final_models.append(model)

    return SplitSummary(
        models=final_models,
        global_metrics=global_metrics,
        global_sources=global_sources,
        slice_metrics=slice_metrics,
        slice_sources=slice_sources,
        heuristics_components=heuristics_components,
        base_rate=base_rate,
        ensemble_precision=ref_precision,
        ensemble_lift=ensemble_lift,
    )


def format_value(mean: float, is_best: bool, dagger: bool) -> str:
    if np.isnan(mean):
        return MISSING_NUMERIC
    text = f"\\num{{{mean:.3f}}}"
    if is_best:
        text = r"{\bfseries " + text + "}"
    if dagger:
        text += r"\textsuperscript{\dagger}"
    return text


def format_delta(delta: float) -> str:
    if np.isnan(delta):
        return MISSING_NUMERIC
    return f"\\num{{{delta:+.3f}}}"


def format_lift(value: float | None) -> str:
    if value is None or np.isnan(value):
        return MISSING_NUMERIC
    return f"\\num{{{value:.3f}}}"


def compute_daggers(
    models: list[str], aggregates: dict[str, MetricAggregate]
) -> tuple[dict[str, dict[str, bool]], bool]:
    dagger_map = {model: dict.fromkeys(METRICS, False) for model in models}
    has_dagger = False
    for metric in METRICS:
        candidates = [
            m for m in models if not np.isnan(aggregates[m].mean.get(metric, float("nan")))
        ]
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda m: aggregates[m].mean[metric], reverse=True)
        best = candidates[0]
        runner = candidates[1]
        best_agg = aggregates[best]
        runner_agg = aggregates[runner]
        if best_agg.n_seeds >= 2 and runner_agg.n_seeds >= 2:
            ci_best = best_agg.ci.get(metric, float("nan"))
            ci_runner = runner_agg.ci.get(metric, float("nan"))
            if not np.isnan(ci_best) and not np.isnan(ci_runner):
                se_best = ci_best / 1.96
                se_runner = ci_runner / 1.96
                diff = best_agg.mean[metric] - runner_agg.mean[metric]
                if diff <= se_best + se_runner:
                    dagger_map[best][metric] = True
                    has_dagger = True
    return dagger_map, has_dagger


def render_table(
    models: list[str],
    aggregates: dict[str, MetricAggregate],
    node2vec: MetricAggregate | None,
    dagger_map: dict[str, dict[str, bool]] | None,
    include_deltas: bool,
    include_lift: bool,
    highlight_best_model: bool = False,
    lift_map: dict[str, float] | None = None,
) -> list[str]:
    best_values = {
        metric: max(
            (
                aggregates[m].mean[metric]
                for m in models
                if not np.isnan(aggregates[m].mean.get(metric, float("nan")))
            ),
            default=float("nan"),
        )
        for metric in METRICS
    }
    best_ndcg_model: str | None = None
    if highlight_best_model:
        best_ndcg = float("-inf")
        for model in models:
            value = aggregates[model].mean.get("ndcg", float("nan"))
            if np.isnan(value):
                continue
            if value > best_ndcg:
                best_ndcg = value
                best_ndcg_model = model
    deltas: dict[str, tuple[float, float]] = {}
    if include_deltas:
        if node2vec:
            for model in models:
                deltas[model] = (
                    aggregates[model].mean.get("ndcg", float("nan"))
                    - node2vec.mean.get("ndcg", float("nan")),
                    aggregates[model].mean.get("hit10", float("nan"))
                    - node2vec.mean.get("hit10", float("nan")),
                )
        else:
            deltas = {model: (float("nan"), float("nan")) for model in models}

    lines = []
    for model in models:
        agg = aggregates[model]
        display = MODEL_DISPLAY_NAMES.get(model, model)
        if highlight_best_model and model == best_ndcg_model:
            display = r"{\bfseries " + display + "}"
        cells = [display]
        for metric in METRICS:
            mean = agg.mean.get(metric, float("nan"))
            is_best = not np.isnan(best_values[metric]) and abs(mean - best_values[metric]) <= 1e-12
            dagger = dagger_map[model][metric] if dagger_map else False
            cells.append(format_value(mean, is_best, dagger))
        if include_deltas:
            delta_ndcg, delta_hit = deltas.get(model, (float("nan"), float("nan")))
            cells.append(format_delta(delta_ndcg))
            cells.append(format_delta(delta_hit))
        if include_lift:
            lift_value = None if lift_map is None else lift_map.get(model)
            cells.append(format_lift(lift_value))
        lines.append(" & ".join(cells) + r" \\")
    return lines


def write_global_table(
    path: Path,
    split: str,
    summary: SplitSummary,
    models: list[str],
    *,
    include_deltas: bool,
    include_lift: bool,
    include_notes: bool = False,
) -> bool:
    metrics = {
        model: summary.global_metrics[model] for model in models if model in summary.global_metrics
    }
    node2vec = metrics.get("Node2Vec")
    dagger_map, has_dagger = compute_daggers(list(metrics.keys()), metrics)
    lift_map: dict[str, float] | None = None
    if include_lift:
        lift_map = {}
        if summary.ensemble_lift is not None:
            lift_map["Ensemble"] = summary.ensemble_lift
    rows = render_table(
        list(metrics.keys()),
        metrics,
        node2vec,
        dagger_map,
        include_deltas=include_deltas,
        include_lift=include_lift,
        highlight_best_model=False,
        lift_map=lift_map,
    )
    header_cells = [f"{{{METRIC_LABELS[m]}}}" for m in METRICS]
    if include_deltas:
        header_cells.extend(
            [
                "{\\(\\Delta\\) nDCG vs N2V}",
                "{\\(\\Delta\\) Hit@10 vs N2V}",
            ]
        )
    if include_lift:
        header_cells.append("{Lift vs base}")
    note_lines: list[str] = []
    if include_notes:
        note_lines = [
            f"\\emph{{Macro-averaged over {summary.global_sources:,} sources with \\ensuremath{{\\geq}} 1 labeled positive.}}",
            "\\emph{Heuristics row is the mean of PA/CN/AA heuristics.}",
        ]
        if summary.base_rate is not None:
            note_lines.append(f"\\emph{{Overall base rate: {summary.base_rate * 100:.3f}\\%}}")
        if include_lift and summary.ensemble_precision is not None:
            note_lines.append(
                "\\emph{Lift uses Top-K+floor (K=10, floor=0.995, cap=50) operating precision.}"
            )
        if has_dagger:
            note_lines.append(
                "\\emph{\\textsuperscript{\\dagger} Difference not robust to seed variance.}"
            )

    column_spec = build_global_column_spec(include_deltas, include_lift)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"% {split.title()} global scorecard\n")
        fh.write("\\begin{tabular}{" + column_spec + "}\n")
        fh.write("\\toprule\n")
        fh.write("{Model} & " + " & ".join(header_cells) + " \\\\\n")
        fh.write("\\midrule\n")
        for line in rows:
            fh.write(line + "\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
        if include_notes:
            for note in note_lines:
                fh.write(note + "\n")
        fh.write("\n")
    return has_dagger


def write_slice_tables(
    path: Path,
    split: str,
    summary: SplitSummary,
    models: list[str],
    *,
    include_notes: bool = False,
) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"% {split.title()} slice scorecards\n")
        for slice_col, pretty in SLICE_COLUMNS.items():
            if slice_col not in summary.slice_metrics:
                fh.write(f"% Slice {pretty} unavailable\n\n")
                continue
            metrics = summary.slice_metrics[slice_col]
            available_models = [m for m in models if m in metrics]
            if not available_models:
                fh.write(f"% Slice {pretty} has no available models\n\n")
                continue
            node2vec = metrics.get("Node2Vec")
            header_cells = [f"{{{METRIC_LABELS[m]}}}" for m in METRICS]
            header_cells.extend(
                [
                    "{\\(\\Delta\\) nDCG vs N2V}",
                    "{\\(\\Delta\\) Hit@10 vs N2V}",
                ]
            )
            dagger_map, slice_has_dagger = compute_daggers(available_models, metrics)
            rows = render_table(
                available_models,
                metrics,
                node2vec,
                dagger_map,
                include_deltas=True,
                include_lift=False,
                highlight_best_model=True,
            )
            fh.write("\\begin{tabular}{" + build_slice_column_spec() + "}\n")
            fh.write("\\toprule\n")
            fh.write(
                f"\\multicolumn{{{len(METRICS) + 3}}}{{c}}{{{pretty} --- {split.title()}}}\\\\\n"
            )
            fh.write("\\midrule\n")
            fh.write("{Model} & " + " & ".join(header_cells) + " \\\\\n")
            fh.write("\\midrule\n")
            for line in rows:
                fh.write(line + "\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")
            fh.write("\n")
            if include_notes:
                note_lines = [
                    "\\emph{Sources with no positives in this slice are excluded.}",
                    f"\\textit{{N sources: {summary.slice_sources.get(slice_col, 0):,}}}",
                ]
                if slice_has_dagger:
                    note_lines.append(
                        "\\emph{\\textsuperscript{\\dagger} Difference not robust to seed variance.}"
                    )
                for note in note_lines:
                    fh.write(note + "\n")
                fh.write("\n")


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _to_float(value: float | int | str | None) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_analysis_slice_metrics(
    split: str,
    columns: list[ConsolidatedColumnSpec] | None = None,
) -> pd.DataFrame:
    if not MODEL_SLICE_METRICS_PATH.exists():
        raise FileNotFoundError(f"Slice metrics summary missing at {MODEL_SLICE_METRICS_PATH}")
    df = pd.read_csv(MODEL_SLICE_METRICS_PATH)
    df = df[df["split"] == split].copy()
    df["model_standard"] = df["model"].map(MODEL_SLICE_NAME_MAP)
    df = df[df["model_standard"].notna()]
    if columns is not None:
        target_columns = {spec.slice_column for spec in columns}
        df = df[df["slice_column"].isin(target_columns)]
    return df


def compute_slice_positive_counts(
    meta_path: Path, columns: list[ConsolidatedColumnSpec]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    existing = {spec.flag_column for spec in columns if spec.flag_column}
    if not existing:
        return counts
    con = duckdb.connect()
    try:
        for col in existing:
            if col is None:
                continue
            if col in counts:
                continue
            query = (
                f"SELECT COALESCE(SUM(CASE WHEN label = 1 AND {col} = 1 THEN 1 ELSE 0 END), 0) "
                f"FROM read_parquet('{meta_path.as_posix()}')"
            )
            try:
                value = con.execute(query).fetchone()[0]
            except duckdb.Error:
                value = None
            counts[col] = int(value) if value is not None else 0
    finally:
        con.close()
    return counts


def prepare_scorecard_assets(
    meta_val_path: Path,
    meta_test_path: Path,
    scores_val_path: Path,
    scores_test_path: Path,
) -> ScorecardAssets:
    models_val = detect_models(meta_val_path, scores_val_path)
    models_test = detect_models(meta_test_path, scores_test_path)
    base_models = [m for m in MODEL_COLUMNS if m in models_val and m in models_test]
    if "Ensemble" not in base_models:
        base_models.append("Ensemble")

    slices_val = detect_slices(meta_val_path)
    slices_test = detect_slices(meta_test_path)
    slices = [s for s in SLICE_COLUMNS if s in slices_val and s in slices_test]

    summary_val = collect_split(meta_val_path, scores_val_path, base_models, slices, "val")
    summary_test = collect_split(meta_test_path, scores_test_path, base_models, slices, "test")

    common_models = [
        m
        for m in base_models
        if m in summary_val.global_metrics and m in summary_test.global_metrics
    ]
    preferred = [m for m in GLOBAL_MODEL_ORDER if m in common_models]
    remainder = [m for m in common_models if m not in GLOBAL_MODEL_ORDER]
    ordered_models = preferred + remainder
    summary_val.models = ordered_models
    summary_test.models = ordered_models

    summary_by_split = {
        "validation": summary_val,
        "test": summary_test,
    }
    split_meta_paths = {
        "val": meta_val_path,
        "test": meta_test_path,
    }
    split_scores_paths = {
        "val": scores_val_path,
        "test": scores_test_path,
    }

    return ScorecardAssets(
        summary_by_split=summary_by_split,
        common_models=ordered_models,
        slices=slices,
        split_meta_paths=split_meta_paths,
        split_scores_paths=split_scores_paths,
    )


def write_model_scorecard_table(
    path: Path,
    split: str,
    summary: SplitSummary,
    models: list[str],
) -> None:
    metric_defs: list[tuple[str, str]] = [
        ("ndcg", METRIC_LABELS["ndcg"]),
        ("hit10", METRIC_LABELS["hit10"]),
        ("hit50", METRIC_LABELS["hit50"]),
        ("mrr", METRIC_LABELS["mrr"]),
        ("map100", METRIC_LABELS["map100"]),
    ]

    aggregates: dict[str, MetricAggregate] = {
        model: summary.global_metrics.get(model)
        for model in models
        if model in summary.global_metrics
    }
    aggregates = {k: v for k, v in aggregates.items() if v is not None}
    if not aggregates:
        raise RuntimeError("No models available for model scorecard table")

    best_values = {
        key: max(
            (
                agg.mean.get(key, float("nan"))
                for agg in aggregates.values()
                if not np.isnan(agg.mean.get(key, float("nan")))
            ),
            default=float("nan"),
        )
        for key, _ in metric_defs
    }

    column_spec = (
        "@{}l" + "".join("S[table-format=1.3]" for _ in metric_defs) + "S[table-format=3.1]@{}"
    )
    header_row = (
        ["\\multicolumn{1}{c}{Model}"]
        + [f"\\multicolumn{{1}}{{c}}{{{label}}}" for _, label in metric_defs]
        + ["\\multicolumn{1}{c}{Lift vs base}"]
    )

    lines: list[str] = [
        "% Auto-generated model scorecard table",
        "\\begin{tabular}{" + column_spec + "}",
        "\\toprule",
        " & ".join(header_row) + r" \\",
        "\\midrule",
    ]

    lift_value = summary.ensemble_lift if summary.ensemble_lift is not None else float("nan")

    for model in models:
        agg = aggregates.get(model)
        if agg is None:
            continue
        display = MODEL_DISPLAY_NAMES.get(model, model)
        row_cells: list[str] = [display]
        for key, _ in metric_defs:
            mean = agg.mean.get(key, float("nan"))
            if np.isnan(mean):
                row_cells.append(MISSING_NUMERIC)
                continue
            text = f"\\num{{{mean:.3f}}}"
            if abs(mean - best_values[key]) <= 1e-12:
                text = r"{\bfseries " + text + "}"
            row_cells.append(text)
        if model == "Ensemble" and not np.isnan(lift_value):
            row_cells.append(f"\\num{{{lift_value:.1f}}}")
        else:
            row_cells.append(r"\multicolumn{1}{c}{\NA}")
        lines.append(" & ".join(row_cells) + r" \\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])

    notes = [
        f"\\emph{{Macro-averaged over {summary.global_sources:,} sources with $\\geq$ 1 positive.}}",
        "\\emph{Heuristics row is the mean of PA/CN/AA heuristics.}",
        "\\emph{Candidate pool: undirected 2-hop neighborhood with uniform fallback; bounded per source.}",
    ]
    if summary.base_rate is not None:
        notes.append(f"\\emph{{Overall base rate: {summary.base_rate * 100:.3f}\\%}}")
    lines.extend(notes)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_consolidated_metrics(
    df: pd.DataFrame,
    columns: list[ConsolidatedColumnSpec],
    models: list[str],
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, int]]:
    result: dict[str, dict[str, dict[str, float]]] = {model: {} for model in models}
    counts: dict[str, int] = {}
    for spec in columns:
        subset = df[df["slice_column"] == spec.slice_column]
        if subset.empty:
            continue
        if "n_sources" in subset.columns:
            with contextlib.suppress(TypeError, ValueError):
                counts[spec.slice_column] = int(float(subset.iloc[0]["n_sources"]))
        for model in models:
            row = subset[subset["model_standard"] == model]
            if row.empty:
                continue
            values = row.iloc[0]
            try:
                per_model_count = int(float(values.get("n_sources", float("nan"))))
            except (TypeError, ValueError):
                per_model_count = None
            if per_model_count is not None and spec.slice_column not in counts:
                counts[spec.slice_column] = per_model_count
            metrics_entry = {
                "ndcg": _to_float(values.get("ndcg100")),
                "hit10": _to_float(values.get("hit10")),
                "map": _to_float(values.get("map100")),
            }
            if per_model_count == 0:
                metrics_entry = {metric: float("nan") for metric in metrics_entry}
            result[model][spec.slice_column] = metrics_entry
    return result, counts


def build_metrics_from_seeds(
    meta_path: Path,
    ensemble_scores_path: Path,
    models: list[str],
    columns: list[ConsolidatedColumnSpec],
    split: str,
    preloaded_seed_results: dict[str, dict[str, SeedResult]] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    target_aliases = [spec.slice_column for spec in columns]
    for model in models:
        base_model = model
        component_name: str | None = None
        if model.startswith("Heuristics::"):
            base_model = "Heuristics"
            component_name = model.split("::", 1)[1]

        if preloaded_seed_results and model in preloaded_seed_results:
            seed_results_raw = preloaded_seed_results[model]
        else:
            cache_key = (model, split, tuple(sorted(target_aliases)))
            if cache_key in _SEED_RESULTS_CACHE:
                seed_results_raw = _SEED_RESULTS_CACHE[cache_key]
            else:
                try:
                    seed_results_raw = load_model_seed_results(
                        base_model,
                        split,
                        target_aliases,
                        meta_path,
                        ensemble_scores_path,
                    )
                except Exception:
                    seed_results_raw = {}
                _SEED_RESULTS_CACHE[cache_key] = seed_results_raw

        if component_name is not None:
            component_result = seed_results_raw.get(component_name)
            seed_results = {component_name: component_result} if component_result else {}
        else:
            seed_results = seed_results_raw
        if not seed_results:
            continue
        model_entries: dict[str, dict[str, float]] = {}
        for spec in columns:
            metric_lists = {metric: [] for metric in METRICS}
            seed_names: list[str] = []
            for seed_name, seed_result in seed_results.items():
                slice_metrics = seed_result.slice_metrics.get(spec.slice_column)
                if not slice_metrics:
                    continue
                seed_names.append(seed_name)
                for metric in METRICS:
                    metric_lists[metric].append(slice_metrics.get(metric, float("nan")))
            if not seed_names:
                continue
            agg = aggregate_metric_lists(metric_lists, seed_names)
            model_entries[spec.slice_column] = {
                "ndcg": agg.mean.get("ndcg", float("nan")),
                "hit10": agg.mean.get("hit10", float("nan")),
                "map": agg.mean.get("map100", float("nan")),
            }
        if model_entries:
            result[model] = model_entries
    return result


def format_stacked_cell(ndcg: float, hit: float, bold: bool) -> str:
    if np.isnan(ndcg) and np.isnan(hit):
        return r"\makecell[r]{" + NA_TEXT + r" \\ {\footnotesize (" + NA_TEXT + r")}}"
    parts = []
    if np.isnan(ndcg):
        ndcg_text = MISSING_INLINE
    else:
        ndcg_text = f"\\num{{{ndcg:.3f}}}"
        if bold:
            ndcg_text = r"\textbf{" + ndcg_text + "}"
    parts.append(ndcg_text)
    if np.isnan(hit):
        hit_text = MISSING_INLINE
    else:
        hit_text = f"\\num{{{hit:.3f}}}"
    parts.append(r"{\footnotesize (" + hit_text + ")}")
    return r"\makecell[r]{" + parts[0] + r" \\ " + parts[1] + "}"


def format_single_cell(value: float, bold: bool) -> str:
    if np.isnan(value):
        return MISSING_INLINE
    text = f"\\num{{{value:.3f}}}"
    if bold:
        text = r"\textbf{" + text + "}"
    return text


def format_delta_inline(delta: float) -> str:
    if np.isnan(delta):
        return MISSING_INLINE
    text = f"\\num{{{delta:+.3f}}}"
    if delta > 0:
        text = r"\textbf{" + text + "}"
    return text


def format_slice_label(slice_column: str, raw_label: str | None) -> str:
    if slice_column in SLICE_DISPLAY_OVERRIDES:
        return SLICE_DISPLAY_OVERRIDES[slice_column]
    if slice_column in HORIZON_DISPLAY_OVERRIDES:
        return HORIZON_DISPLAY_OVERRIDES[slice_column]
    label = (raw_label or "").strip()
    if label.lower().endswith(" flag"):
        label = label[:-5].strip()
    if label.lower().startswith("horizon"):
        tail = label[len("horizon") :].strip()
        label = f"AP {tail.replace('h', 'm')}"
    label = label.replace("_", " ").strip()
    return label or slice_column


def derive_slice_column_specs(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
) -> list[ConsolidatedColumnSpec]:
    exclude = exclude or set()
    specs: list[ConsolidatedColumnSpec] = []
    seen: set[str] = set()
    for slice_column in sorted(df["slice_column"].unique()):
        if slice_column in exclude or slice_column == "overall":
            continue
        if slice_column in seen:
            continue
        subset = df[df["slice_column"] == slice_column]
        label_series = subset["slice_label"].dropna()
        raw_label = label_series.iloc[0] if not label_series.empty else None
        kind = (
            "single"
            if slice_column.startswith("slice_horizon") or "horizon" in slice_column
            else "stacked"
        )
        flag_column = (
            slice_column
            if slice_column.startswith("slice_") and not slice_column.startswith("slice_horizon")
            else None
        )
        display_label = format_slice_label(slice_column, raw_label)
        specs.append(
            ConsolidatedColumnSpec(slice_column, display_label, slice_column, kind, flag_column)
        )
        seen.add(slice_column)
    specs.sort(key=lambda spec: (SLICE_COLUMN_PRIORITY.get(spec.slice_column, 1000), spec.label))
    return specs


def chunk_specs(
    specs: list[ConsolidatedColumnSpec], chunk_size: int
) -> Iterable[list[ConsolidatedColumnSpec]]:
    for idx in range(0, len(specs), chunk_size):
        yield specs[idx : idx + chunk_size]


def winner_display_name(model: str) -> str:
    if model.startswith("Heuristics::"):
        component = model.split("::", 1)[1]
        return f"Heuristics ({component})"
    return MODEL_DISPLAY_NAMES.get(model, model)


def collect_metrics_for_specs(
    default_split: str,
    split_meta_paths: dict[str, Path],
    split_scores_paths: dict[str, Path],
    models: list[str],
    columns: list[ConsolidatedColumnSpec],
    *,
    preloaded_seed_results: dict[str, dict[str, dict[str, SeedResult]]] | None = None,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, int], dict[str, int]]:
    metrics: dict[str, dict[str, dict[str, float]]] = {model: {} for model in models}
    counts: dict[str, int] = {}
    analysis_counts: dict[str, int] = {}

    splits_needed: dict[str, list[ConsolidatedColumnSpec]] = {}
    for spec in columns:
        spec_split = spec.source_split or default_split
        splits_needed.setdefault(spec_split, []).append(spec)

    for split_used, spec_list in splits_needed.items():
        meta_path = split_meta_paths.get(split_used)
        if meta_path is None:
            continue
        scores_path = split_scores_paths.get(split_used)

        df = load_analysis_slice_metrics(split_used, spec_list)
        metrics_from_analysis, counts_analysis = build_consolidated_metrics(df, spec_list, models)
        analysis_counts.update({k: v for k, v in counts_analysis.items() if v})

        preloaded_for_split = None
        if preloaded_seed_results is not None:
            preloaded_for_split = preloaded_seed_results.get(split_used)

        metrics_from_seeds = build_metrics_from_seeds(
            meta_path,
            scores_path,
            models,
            spec_list,
            split_used,
            preloaded_seed_results=preloaded_for_split,
        )

        counts_split = compute_slice_positive_counts(meta_path, spec_list)
        counts.update({k: v for k, v in counts_split.items() if v})

        for model in models:
            combined = metrics.setdefault(model, {})
            for spec in spec_list:
                entry = metrics_from_seeds.get(model, {}).get(spec.slice_column)
                if entry is None:
                    entry = metrics_from_analysis.get(model, {}).get(spec.slice_column)
                if entry is not None:
                    if (
                        spec.slice_column.startswith("slice_horizon")
                        and spec.source_split == "val"
                        and model == "Ensemble"
                    ):
                        entry = {
                            "ndcg": float("nan"),
                            "hit10": float("nan"),
                            "map": float("nan"),
                        }
                    combined[spec.slice_column] = entry

    return metrics, counts, analysis_counts


def generate_slice_table_lines(
    default_split: str,
    split_meta_paths: dict[str, Path],
    split_scores_paths: dict[str, Path],
    models: list[str],
    columns: list[ConsolidatedColumnSpec],
    *,
    exclude_from_best: set[str] | None = None,
    preloaded_seed_results: dict[str, dict[str, dict[str, SeedResult]]] | None = None,
) -> list[str]:
    if not columns:
        return []

    metrics, counts, analysis_counts = collect_metrics_for_specs(
        default_split,
        split_meta_paths,
        split_scores_paths,
        models,
        columns,
        preloaded_seed_results=preloaded_seed_results,
    )

    filtered_columns: list[ConsolidatedColumnSpec] = []
    for spec in columns:
        has_data = False
        for model in models:
            entry = metrics.get(model, {}).get(spec.slice_column)
            if not entry:
                continue
            metric_value = entry.get("ndcg") if spec.kind == "stacked" else entry.get("map")
            if metric_value is not None and not np.isnan(metric_value):
                has_data = True
                break
        if has_data:
            filtered_columns.append(spec)
    if not filtered_columns:
        return []

    column_format = "@{}l" + "r" * len(filtered_columns) + "@{}"
    header_cells: list[str] = []
    for spec in filtered_columns:
        label = latex_escape(spec.label)
        count = counts.get(spec.slice_column)
        if count in (None, 0) and spec.flag_column:
            count = counts.get(spec.flag_column)
        if count in (None, 0):
            count = analysis_counts.get(spec.slice_column)
        if count:
            label = r"\makecell[c]{" + label + rf"\\(n={count:,})" + "}"
        header_cells.append(label)

    best_tracker: dict[str, float] = {}
    for spec in filtered_columns:
        metric_key = "ndcg" if spec.kind == "stacked" else "map"
        values = []
        for model in models:
            if exclude_from_best and model in exclude_from_best:
                continue
            value = metrics.get(model, {}).get(spec.slice_column, {}).get(metric_key, float("nan"))
            if not np.isnan(value):
                values.append(value)
        best_tracker[spec.slice_column] = max(values) if values else float("nan")

    lines_out = [
        "\\begin{tabular}{" + column_format + "}",
        "\\toprule",
        "{Model} & " + " & ".join(header_cells) + " \\",
        "\\midrule",
    ]
    for model in models:
        if not metrics.get(model):
            continue
        display = MODEL_DISPLAY_NAMES.get(model, model)
        row_cells = [display]
        for spec in filtered_columns:
            entry = metrics.get(model, {}).get(spec.slice_column)
            best_value = best_tracker.get(spec.slice_column, float("nan"))
            if spec.kind == "stacked":
                ndcg_val = entry.get("ndcg", float("nan")) if entry else float("nan")
                hit_val = entry.get("hit10", float("nan")) if entry else float("nan")
                bold = (
                    not np.isnan(ndcg_val)
                    and not np.isnan(best_value)
                    and abs(ndcg_val - best_value) <= 1e-12
                )
                cell = format_stacked_cell(ndcg_val, hit_val, bold)
            else:
                ap_val = entry.get("map", float("nan")) if entry else float("nan")
                bold = (
                    not np.isnan(ap_val)
                    and not np.isnan(best_value)
                    and abs(ap_val - best_value) <= 1e-12
                )
                cell = format_single_cell(ap_val, bold)
            row_cells.append(cell)
        lines_out.append(" & ".join(row_cells) + r" \\")
    lines_out.extend(["\\bottomrule", "\\end{tabular}"])
    return lines_out


def _preload_seed_results(
    models: list[str],
    split_meta_paths: dict[str, Path],
    split_scores_paths: dict[str, Path],
    columns: list[ConsolidatedColumnSpec],
    default_split: str,
) -> dict[str, dict[str, dict[str, SeedResult]]]:
    splits_needed: dict[str, set[str]] = {}
    for spec in columns:
        spec_split = spec.source_split or default_split
        splits_needed.setdefault(spec_split, set()).add(spec.slice_column)

    preloaded: dict[str, dict[str, dict[str, SeedResult]]] = {}
    for split_used, aliases in splits_needed.items():
        meta_path = split_meta_paths.get(split_used)
        scores_path = split_scores_paths.get(split_used)
        if meta_path is None:
            continue
        per_split: dict[str, dict[str, SeedResult]] = {}
        for model in models:
            base_model = model
            component_name: str | None = None
            if model.startswith("Heuristics::"):
                base_model = "Heuristics"
                component_name = model.split("::", 1)[1]
            try:
                seed_results_raw = load_model_seed_results(
                    base_model,
                    split_used,
                    sorted(aliases),
                    meta_path,
                    scores_path,
                )
            except Exception:
                seed_results_raw = {}
            if component_name is not None:
                component_result = seed_results_raw.get(component_name)
                per_split[model] = {component_name: component_result} if component_result else {}
            else:
                per_split[model] = seed_results_raw
        preloaded[split_used] = per_split
    return preloaded


def write_grouped_slice_tables(
    path: Path,
    split: str,
    split_meta_paths: dict[str, Path],
    split_scores_paths: dict[str, Path],
    models: list[str],
    columns: list[ConsolidatedColumnSpec],
    chunk_size: int = 6,
    header_comment: str | None = None,
    exclude_from_best: set[str] | None = None,
) -> None:
    preloaded = _preload_seed_results(models, split_meta_paths, split_scores_paths, columns, split)

    with path.open("w", encoding="utf-8") as fh:
        if header_comment:
            fh.write(header_comment + "\n")
        wrote_any = False
        for chunk in chunk_specs(columns, chunk_size):
            lines = generate_slice_table_lines(
                split,
                split_meta_paths,
                split_scores_paths,
                models,
                chunk,
                exclude_from_best=exclude_from_best,
                preloaded_seed_results=preloaded,
            )
            if not lines:
                continue
            if wrote_any:
                fh.write("\n")
            fh.write("\n".join(lines))
            fh.write("\n")
            wrote_any = True
        if not wrote_any:
            fh.write("% No available slice metrics for requested columns\n")


def write_consolidated_slice_table(
    path: Path,
    default_split: str,
    split_meta_paths: dict[str, Path],
    split_scores_paths: dict[str, Path],
    models: list[str],
    columns: list[ConsolidatedColumnSpec],
) -> None:
    preloaded = _preload_seed_results(
        models, split_meta_paths, split_scores_paths, columns, default_split
    )
    lines_out = generate_slice_table_lines(
        default_split,
        split_meta_paths,
        split_scores_paths,
        models,
        columns,
        exclude_from_best={"Ensemble"},
        preloaded_seed_results=preloaded,
    )
    if not lines_out:
        raise RuntimeError("No slice columns available for consolidated table")
    with path.open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated consolidated slice scorecard\n")
        fh.write("\n".join(lines_out))
        fh.write("\n")


def write_slice_winners_table(
    path: Path,
    default_split: str,
    split_meta_paths: dict[str, Path],
    split_scores_paths: dict[str, Path],
    columns: list[ConsolidatedColumnSpec],
) -> None:
    models = ["Ensemble", *WINNER_BASE_MODELS]
    preloaded = _preload_seed_results(
        models, split_meta_paths, split_scores_paths, columns, default_split
    )
    metrics, counts_raw, analysis_counts = collect_metrics_for_specs(
        default_split,
        split_meta_paths,
        split_scores_paths,
        models,
        columns,
        preloaded_seed_results=preloaded,
    )

    positive_counts: dict[str, int] = {}

    def store_counts(source: dict[str, int | float | None]) -> None:
        for key, value in source.items():
            if value is None:
                continue
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                continue
            positive_counts[key] = numeric

    store_counts(analysis_counts)
    store_counts(counts_raw)

    def lookup_count(spec: ConsolidatedColumnSpec) -> int | None:
        for key in (spec.slice_column, spec.flag_column):
            if key and key in positive_counts:
                return positive_counts[key]
        return None

    filtered_columns: list[ConsolidatedColumnSpec] = []
    for spec in columns:
        has_data = False
        for model in models:
            entry = metrics.get(model, {}).get(spec.slice_column)
            if not entry:
                continue
            metric_key = "ndcg" if spec.kind == "stacked" else "map"
            value = entry.get(metric_key)
            if value is not None and not np.isnan(value):
                has_data = True
                break
        count_value = lookup_count(spec)
        include_missing = count_value == 0
        if has_data or include_missing:
            filtered_columns.append(spec)
    if not filtered_columns:
        raise RuntimeError("No slice metrics available for winners table")

    column_format = "@{}lllrrr@{}"
    lines = [
        "% Auto-generated slice winners table",
        "\\begin{tabular}{" + column_format + "}",
        "\\toprule",
        "{Slice} & {Metric} & {Best Base} & {Best Base Score} & {Ensemble Score} & {Ensemble $-$ Best (\\Delta)} \\",
        "\\midrule",
    ]

    has_tie = False
    for spec in filtered_columns:
        slice_label = latex_escape(spec.label)
        count_value = lookup_count(spec)
        if spec.slice_column.startswith("slice_horizon"):
            metric_items = [("map", "AP")]
        else:
            metric_items = [("ndcg", METRIC_LABELS["ndcg"]), ("hit10", METRIC_LABELS["hit10"])]
        for metric_key, metric_label in metric_items:
            base_values: dict[str, float] = {}
            for base_model in WINNER_BASE_MODELS:
                value = (
                    metrics.get(base_model, {})
                    .get(spec.slice_column, {})
                    .get(metric_key, float("nan"))
                )
                base_values[base_model] = value

            valid_items = [(m, v) for m, v in base_values.items() if not np.isnan(v)]
            missing_row = count_value == 0 or not valid_items

            if missing_row:
                best_name = NA_TEXT
                best_score_cell = MISSING_INLINE
                ensemble_value = float("nan")
                ensemble_cell = MISSING_INLINE
                delta_cell = MISSING_INLINE
            else:
                _best_model, best_value = max(valid_items, key=lambda item: item[1])
                best_value = float(best_value)
                tie_models = [
                    m
                    for m, v in base_values.items()
                    if not np.isnan(v) and abs(v - best_value) <= 0.001
                ]
                selected_model = sorted(tie_models, key=lambda m: winner_display_name(m))[0]
                tie = len(tie_models) > 1
                has_tie = has_tie or tie

                best_name = winner_display_name(selected_model)
                if tie:
                    best_name += r"\textsuperscript{\dagger}"

                best_score_cell = format_single_cell(best_value, False)
                ensemble_value = (
                    metrics.get("Ensemble", {})
                    .get(spec.slice_column, {})
                    .get(metric_key, float("nan"))
                )
                ensemble_cell = format_single_cell(ensemble_value, False)
                delta = (
                    ensemble_value - best_value
                    if not np.isnan(ensemble_value) and not np.isnan(best_value)
                    else float("nan")
                )
                delta_cell = format_delta_inline(delta)

            lines.append(
                " & ".join(
                    [
                        slice_label,
                        metric_label,
                        best_name,
                        best_score_cell,
                        ensemble_cell,
                        delta_cell,
                    ]
                )
                + r" \\"
            )

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if has_tie:
        lines.append(r"\emph{\textsuperscript{\dagger} Ties within 0.001 resolved alphabetically.}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_mean_ci_inline(mean: float, ci: float) -> str:
    if np.isnan(mean):
        return MISSING_INLINE
    mean_text = f"\\num{{{mean:.3f}}}"
    if np.isnan(ci):
        return mean_text + r" \pm " + MISSING_INLINE
    return mean_text + r" \pm " + f"\\num{{{ci:.3f}}}"


def format_cv_inline(mean: float, std: float) -> str:
    if np.isnan(std) or np.isnan(mean) or abs(mean) < 1e-12:
        return MISSING_INLINE
    return f"\\num{{{100.0 * std / abs(mean):.1f}}}"


def build_seed_robustness_lines(split: str, summary: SplitSummary, models: list[str]) -> list[str]:
    column_format = "@{}l" + "r" * len(METRICS) + "rr@{}"
    header_cells = [
        "{Model}",
        *[f"{{{METRIC_LABELS[m]} (mean $\\pm$ 95\\% CI)}}" for m in METRICS],
        "{$n_\\text{seeds}$}",
        "{CV$_{\\text{nDCG}}$ (\\%)}",
    ]
    lines = [
        "\\begin{tabular}{" + column_format + "}",
        "\\toprule",
        " & ".join(header_cells) + " \\",
        "\\midrule",
    ]
    for model in models:
        agg = summary.global_metrics.get(model)
        if agg is None:
            continue
        row_cells = [MODEL_DISPLAY_NAMES.get(model, model)]
        for metric in METRICS:
            mean_val = agg.mean.get(metric, float("nan"))
            ci_val = agg.ci.get(metric, float("nan"))
            row_cells.append(format_mean_ci_inline(mean_val, ci_val))
        n = agg.n_seeds
        row_cells.append(str(n))
        cv_text = format_cv_inline(
            agg.mean.get("ndcg", float("nan")), agg.std.get("ndcg", float("nan"))
        )
        row_cells.append(cv_text)
        lines.append(" & ".join(row_cells) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return lines


def write_seed_robustness_tables(
    output_dir: Path,
    summaries: dict[str, SplitSummary],
    models: list[str],
) -> list[Path]:
    paths: list[Path] = []
    for split, summary in summaries.items():
        lines = build_seed_robustness_lines(split, summary, models)
        path = output_dir / f"scorecard_seed_robustness_{split}.tex"
        content = ["% Auto-generated seed robustness table", *lines, ""]
        path.write_text("\n".join(content), encoding="utf-8")
        paths.append(path)
    return paths


def export_tables_to_pdf(fragments: list[Path]) -> None:
    engine = shutil.which("tectonic")
    if engine is None:
        print("WARNING: TeX engine 'tectonic' not found; skipping PDF exports.")
        return
    preamble_lines = [
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage{threeparttable}",
        "\\usepackage{amsmath}",
        "\\usepackage{siunitx}",
        "\\usepackage{makecell}",
        "\\sisetup{reset-text-series=false, text-series-to-math=true}",
        "\\usepackage[margin=1in]{geometry}",
        "\\newcommand{\\NA}{\\text{N/A}}",
        "\\begin{document}",
        "\\input{%s}",
        "\\end{document}",
    ]
    preamble = "\n".join(preamble_lines) + "\n"
    for fragment in fragments:
        if not fragment.exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wrapper = tmp_path / "wrapper.tex"
            wrapper.write_text(preamble % fragment.resolve(), encoding="utf-8")
            try:
                subprocess.run(
                    [engine, wrapper.name], cwd=tmp_path, check=True, capture_output=True
                )
            except subprocess.CalledProcessError as exc:
                print(f"WARNING: tectonic failed for {fragment}: {exc}")
                continue
            produced = tmp_path / "wrapper.pdf"
            if produced.exists():
                target = fragment.with_suffix(".pdf")
                shutil.move(str(produced), target)


def detect_models(meta_path: Path, scores_path: Path) -> list[str]:
    meta_cols = set(pq.ParquetFile(meta_path).schema.names)
    score_cols = set(pq.ParquetFile(scores_path).schema.names)
    available: list[str] = []
    for model, col in MODEL_COLUMNS.items():
        if model == "Ensemble":
            if "meta_prob" in score_cols:
                available.append(model)
        elif col in meta_cols:
            available.append(model)
    return available


def detect_slices(meta_path: Path) -> list[str]:
    names = set(pq.ParquetFile(meta_path).schema.names)
    return [col for col in SLICE_COLUMNS if col in names]
