from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from src.ensemble.utils import ensure_dir, now_iso, save_json

DEFAULT_SLICES: Sequence[str] = (
    "slice_CC",
    "slice_CC3",
    "slice_deg_q1",
    "slice_gt2hop",
    "slice_WW",
    "slice_WW3",
)


@dataclass
class ThresholdResult:
    model: str
    prob_column: str
    threshold: float
    precision: float
    recall: float
    yield_rate: float
    predicted: int
    positives: int
    total: int
    precision_met: bool
    precision_target: float
    threshold_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate precision/recall tradeoffs for ranked edges"
    )
    parser.add_argument(
        "--scores", required=True, help="Parquet of scored edges (meta_prob + label)"
    )
    parser.add_argument(
        "--meta-input", required=True, help="Parquet from meta dataset containing slice flags"
    )
    parser.add_argument("--prob-column", default="meta_prob", help="Probability column in scores")
    parser.add_argument("--label-column", default="label", help="Ground-truth column")
    parser.add_argument(
        "--precision-target", type=float, default=0.90, help="Minimum precision to satisfy"
    )
    parser.add_argument("--threshold", type=float, help="Manually supplied threshold")
    parser.add_argument(
        "--hist-bin-width",
        type=float,
        default=1e-4,
        help="Histogram bin width for probability sweep",
    )
    parser.add_argument(
        "--baseline-prob-cols",
        nargs="*",
        default=[],
        help="Baseline probability columns to compare",
    )
    parser.add_argument(
        "--out-dir", default="results/ensemble/meta_ranker", help="Directory for threshold reports"
    )
    parser.add_argument(
        "--tag", help="Output tag (defaults to parent directory name of scores file)"
    )
    parser.add_argument("--split", choices=["val", "test"], help="Dataset split name for reporting")
    parser.add_argument(
        "--model-name", default="ensemble", help="Name for the primary model in reports"
    )
    return parser.parse_args()


def validate_identifier(name: str) -> str:
    if not name:
        raise ValueError("Identifier may not be empty")
    if name[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_":
        raise ValueError(f"Invalid identifier start: {name}")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if any(ch not in allowed for ch in name):
        raise ValueError(f"Identifier contains illegal characters: {name}")
    return name


def readable_model_name(column: str) -> str:
    if column.startswith("prob_"):
        return column[len("prob_") :]
    return column


def load_columns(path: Path) -> Iterable[str]:
    pf = pq.ParquetFile(str(path))
    return list(pf.schema.names)


def create_base_view(
    conn: duckdb.DuckDBPyConnection,
    scores_path: Path,
    meta_path: Path,
    prob_col: str,
    label_col: str,
    baseline_cols: Sequence[str],
    slice_cols: Sequence[str],
    label_in_meta: bool,
    baseline_in_scores: Iterable[str],
    baseline_in_meta: Iterable[str],
) -> None:
    select_items: list[str] = [
        "s.src_id AS src_id",
        "s.dst_id AS dst_id",
        f"s.{prob_col} AS {prob_col}",
    ]
    if label_in_meta:
        select_items.append(f"COALESCE(s.{label_col}, m.{label_col}) AS {label_col}")
    else:
        select_items.append(f"s.{label_col} AS {label_col}")
    for col in slice_cols:
        select_items.append(f"m.{col} AS {col}")
    for col in baseline_cols:
        col_in_scores = col in baseline_in_scores
        col_in_meta = col in baseline_in_meta
        if col_in_scores and col_in_meta:
            select_items.append(f"COALESCE(s.{col}, m.{col}) AS {col}")
        elif col_in_scores:
            select_items.append(f"s.{col} AS {col}")
        else:
            select_items.append(f"m.{col} AS {col}")
    select_clause = ", ".join(select_items)
    sql = (
        f"CREATE TEMP VIEW base AS "
        f"SELECT {select_clause} "
        f"FROM read_parquet('{scores_path.as_posix()}') s "
        f"JOIN read_parquet('{meta_path.as_posix()}') m USING (src_id, dst_id)"
    )
    conn.execute(sql)


def fetch_threshold(
    conn: duckdb.DuckDBPyConnection,
    prob_col: str,
    label_col: str,
    target_precision: float,
) -> float | None:
    query = f"""
        WITH data AS (
            SELECT CAST({prob_col} AS DOUBLE) AS prob,
                   CAST({label_col} AS BIGINT) AS label
            FROM base
            WHERE {prob_col} IS NOT NULL
        ),
        aggregated AS (
            SELECT prob,
                   SUM(label) AS pos_at_prob,
                   COUNT(*) AS count_at_prob
            FROM data
            GROUP BY prob
        ),
        ordered AS (
            SELECT
                prob,
                SUM(pos_at_prob) OVER (ORDER BY prob DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_pos,
                SUM(count_at_prob) OVER (ORDER BY prob DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_count
            FROM aggregated
        )
        SELECT MIN(prob) AS threshold
        FROM (
            SELECT prob,
                   CASE WHEN cum_count = 0 THEN NULL ELSE cum_pos::DOUBLE / cum_count END AS precision
            FROM ordered
        )
        WHERE precision IS NOT NULL AND precision >= ?
    """
    row = conn.execute(query, [float(target_precision)]).fetchone()
    threshold = row[0] if row and row[0] is not None else None
    return None if threshold is None else float(threshold)


def fallback_threshold(conn: duckdb.DuckDBPyConnection, prob_col: str) -> float:
    query = f"SELECT MAX({prob_col}) FROM base WHERE {prob_col} IS NOT NULL"
    row = conn.execute(query).fetchone()
    if not row or row[0] is None:
        return 1.0
    return float(row[0])


def compute_stats(
    conn: duckdb.DuckDBPyConnection,
    prob_col: str,
    label_col: str,
    threshold: float,
    where_clause: str = "",
    params: list[float] | None = None,
) -> dict[str, float]:
    condition = f"WHERE {prob_col} IS NOT NULL"
    if where_clause:
        condition += f" AND {where_clause}"
    sql = f"""
        SELECT
            SUM(CASE WHEN {prob_col} >= ? THEN 1 ELSE 0 END) AS predicted,
            SUM(CASE WHEN {prob_col} >= ? THEN CAST({label_col} AS BIGINT) ELSE 0 END) AS tp,
            SUM(CAST({label_col} AS BIGINT)) AS positives,
            COUNT(*) AS total
        FROM base
        {condition}
    """
    p = [threshold, threshold]
    if params:
        p.extend(params)
    row = conn.execute(sql, p).fetchone()
    assert row is not None, "aggregate query must return a row"
    predicted = int(row[0] or 0)
    tp = int(row[1] or 0)
    positives = int(row[2] or 0)
    total = int(row[3] or 0)
    precision = (tp / predicted) if predicted else 0.0
    recall = (tp / positives) if positives else 0.0
    yield_rate = (predicted / total) if total else 0.0
    return {
        "predicted": predicted,
        "tp": tp,
        "positives": positives,
        "total": total,
        "precision": precision,
        "recall": recall,
        "yield": yield_rate,
    }


def compute_histogram(
    conn: duckdb.DuckDBPyConnection,
    prob_col: str,
    label_col: str,
    bin_width: float,
) -> list[dict[str, float]]:
    if bin_width <= 0 or math.isnan(bin_width):
        raise ValueError("Histogram bin width must be positive")
    query = f"""
        WITH hist AS (
            SELECT
                FLOOR({prob_col} / ?) * ? AS bin_start,
                COUNT(*) AS count,
                SUM(CAST({label_col} AS BIGINT)) AS positives
            FROM base
            WHERE {prob_col} IS NOT NULL
            GROUP BY 1
        )
        SELECT CAST(bin_start AS DOUBLE) AS bin_start,
               CAST(bin_start + ? AS DOUBLE) AS bin_end,
               count,
               positives
        FROM hist
        ORDER BY bin_start
    """
    rows = conn.execute(query, [bin_width, bin_width, bin_width]).fetchall()
    hist: list[dict[str, float]] = []
    for row in rows:
        hist.append(
            {
                "bin_start": float(row[0]),
                "bin_end": float(row[1]),
                "count": int(row[2] or 0),
                "positives": int(row[3] or 0),
            }
        )
    return hist


def summarize_model(
    conn: duckdb.DuckDBPyConnection,
    model_name: str,
    prob_col: str,
    label_col: str,
    precision_target: float,
    provided_threshold: float | None,
) -> ThresholdResult:
    threshold_source = "provided" if provided_threshold is not None else "search"
    threshold = provided_threshold
    if threshold is None:
        threshold = fetch_threshold(conn, prob_col, label_col, precision_target)
        if threshold is None:
            threshold = fallback_threshold(conn, prob_col)
            threshold_source = "search_fallback"
    stats = compute_stats(conn, prob_col, label_col, threshold)
    precision_value = float(stats["precision"]) if math.isfinite(stats["precision"]) else 0.0
    precision_met = precision_value >= float(precision_target)
    return ThresholdResult(
        model=model_name,
        prob_column=prob_col,
        threshold=float(threshold),
        precision=precision_value,
        recall=float(stats["recall"]),
        yield_rate=float(stats["yield"]),
        predicted=int(stats["predicted"]),
        positives=int(stats["positives"]),
        total=int(stats["total"]),
        precision_met=precision_met,
        precision_target=float(precision_target),
        threshold_source=threshold_source,
    )


def slice_stats(
    conn: duckdb.DuckDBPyConnection,
    slices: Sequence[str],
    prob_col: str,
    label_col: str,
    threshold: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for slice_col in slices:
        clause = f"{slice_col} = 1"
        stats = compute_stats(conn, prob_col, label_col, threshold, where_clause=clause)
        out[slice_col] = {
            "predicted": stats["predicted"],
            "tp": stats["tp"],
            "positives": stats["positives"],
            "total": stats["total"],
            "precision": stats["precision"],
            "recall": stats["recall"],
            "yield": stats["yield"],
        }
    overall = compute_stats(conn, prob_col, label_col, threshold)
    out["overall"] = {
        "predicted": overall["predicted"],
        "tp": overall["tp"],
        "positives": overall["positives"],
        "total": overall["total"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "yield": overall["yield"],
    }
    return out


def build_output_paths(out_dir: Path, tag: str, split: str) -> dict[str, Path]:
    ensure_dir(out_dir)
    stem = f"threshold_report_{split}" if split else "threshold_report"
    return {
        "json": out_dir / f"{stem}.json",
        "csv": out_dir / f"{stem}.csv",
    }


def detect_split(path: Path) -> str:
    name = path.stem.lower()
    if "val" in name:
        return "val"
    if "test" in name:
        return "test"
    return "unknown"


def main() -> None:
    args = parse_args()

    scores_path = Path(args.scores)
    meta_path = Path(args.meta_input)
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores parquet not found: {scores_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta input parquet not found: {meta_path}")

    prob_col = validate_identifier(args.prob_column)
    label_col = validate_identifier(args.label_column)
    baseline_cols = [validate_identifier(col) for col in args.baseline_prob_cols]
    slice_cols = list(DEFAULT_SLICES)

    score_columns = set(load_columns(scores_path))
    meta_columns = set(load_columns(meta_path))

    if prob_col not in score_columns:
        raise ValueError(f"Probability column '{prob_col}' missing from scores parquet")
    if label_col not in score_columns and label_col not in meta_columns:
        raise ValueError(f"Label column '{label_col}' missing from inputs")
    missing_slices = [col for col in slice_cols if col not in meta_columns]
    if missing_slices:
        raise ValueError(f"Meta dataset missing required slice columns: {missing_slices}")
    missing_baselines = [
        col for col in baseline_cols if col not in score_columns and col not in meta_columns
    ]
    if missing_baselines:
        raise ValueError(f"Baseline columns not found: {missing_baselines}")

    target_precision = float(args.precision_target)
    if not 0 < target_precision < 1:
        raise ValueError("Precision target must be between 0 and 1")
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        raise ValueError("Threshold must lie within [0, 1]")

    tag = args.tag or scores_path.parent.name
    split = args.split or detect_split(scores_path)
    out_root = Path(args.out_dir) / tag
    outputs = build_output_paths(out_root, tag=tag, split=split)

    label_in_meta = label_col in meta_columns
    baseline_in_scores = {col for col in baseline_cols if col in score_columns}
    baseline_in_meta = {col for col in baseline_cols if col in meta_columns}

    conn = duckdb.connect(database=":memory:")
    try:
        create_base_view(
            conn,
            scores_path,
            meta_path,
            prob_col,
            label_col,
            baseline_cols,
            slice_cols,
            label_in_meta,
            baseline_in_scores,
            baseline_in_meta,
        )

        ensemble_result = summarize_model(
            conn,
            args.model_name,
            prob_col,
            label_col,
            target_precision,
            args.threshold,
        )

        baseline_results: list[ThresholdResult] = []
        baseline_threshold = args.threshold
        for col in baseline_cols:
            name = readable_model_name(col)
            result = summarize_model(
                conn,
                name,
                col,
                label_col,
                target_precision,
                baseline_threshold,
            )
            baseline_results.append(result)

        hist = compute_histogram(conn, prob_col, label_col, args.hist_bin_width)
        slices = slice_stats(conn, slice_cols, prob_col, label_col, ensemble_result.threshold)
    finally:
        conn.close()

    models_table = [ensemble_result, *baseline_results]
    summary_rows = []
    for item in models_table:
        summary_rows.append(
            {
                "model": item.model,
                "prob_column": item.prob_column,
                "threshold": item.threshold,
                "precision": item.precision,
                "recall": item.recall,
                "yield": item.yield_rate,
                "predicted": item.predicted,
                "positives": item.positives,
                "total": item.total,
                "precision_met": item.precision_met,
                "precision_target": item.precision_target,
                "threshold_source": item.threshold_source,
            }
        )

    report_payload = {
        "created_at": now_iso(),
        "tag": tag,
        "split": split,
        "scores_path": scores_path.as_posix(),
        "meta_input_path": meta_path.as_posix(),
        "precision_target": target_precision,
        "models": summary_rows,
        "histogram": hist,
        "slices": slices,
    }

    df_summary = pd.DataFrame(summary_rows)
    slice_records = []
    for slice_name, stats in slices.items():
        record: dict[str, object] = {"section": "slice", "name": slice_name}
        record.update(stats)  # type: ignore[arg-type]
        slice_records.append(record)
    df_slices = pd.DataFrame(slice_records)
    csv_frames = [df_summary.assign(section="model", name=df_summary["model"]), df_slices]
    csv_out = pd.concat(csv_frames, ignore_index=True, sort=False)

    ensure_dir(outputs["json"].parent)
    save_json(outputs["json"], report_payload)
    csv_out.to_csv(outputs["csv"], index=False)

    print(json.dumps(report_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
