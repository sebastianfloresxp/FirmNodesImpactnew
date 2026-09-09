#!/usr/bin/env python3
"""Render PR overlay for Chapter 2 figure set (defaults to test split)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, LogLocator, MultipleLocator


@dataclass
class ModelSpec:
    label: str
    parquet_path: Path
    prob_column: str
    join_required: bool

    def available(self) -> bool:
        return self.parquet_path.exists()


@dataclass
class PointAnnotation:
    identifier: str
    label: str
    precision: float
    recall: float
    yield_value: float
    tau: float | None = None
    show_tau: bool = False
    source_key: str | None = None
    text_position: tuple[float, float] | None = None  # (x_frac, y_frac) in axes coords
    val_precision: float | None = None
    val_recall: float | None = None
    color: str = "#111111"
    predicted: float | None = None
    true_positives: float | None = None


def build_source_sql(spec: ModelSpec, base_table: Path) -> str:
    if spec.join_required:
        return (
            f"SELECT CAST(model.{spec.prob_column} AS DOUBLE) AS prob, "
            f"CAST(base.label AS DOUBLE) AS label "
            f"FROM read_parquet('{spec.parquet_path.as_posix()}') AS model "
            f"INNER JOIN read_parquet('{base_table.as_posix()}') AS base USING (src_id, dst_id)"
        )
    return (
        f"SELECT CAST({spec.prob_column} AS DOUBLE) AS prob, "
        "CAST(label AS DOUBLE) AS label "
        f"FROM read_parquet('{spec.parquet_path.as_posix()}')"
    )


def fetch_curve_samples(
    con: duckdb.DuckDBPyConnection,
    source_sql: str,
    focus_recall: float = 0.25,
    buckets_focus: int = 800,
    buckets_tail: int = 300,
) -> pd.DataFrame:
    query = f"""
        WITH data AS (
            {source_sql}
        ),
        aggregated AS (
            SELECT prob,
                   SUM(label) AS pos_at_prob,
                   COUNT(*) AS count_at_prob
            FROM data
            WHERE prob IS NOT NULL
            GROUP BY prob
        ),
        curve AS (
            SELECT prob,
                   SUM(pos_at_prob) OVER (ORDER BY prob DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_pos,
                   SUM(count_at_prob) OVER (ORDER BY prob DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_count,
                   SUM(pos_at_prob) OVER () AS total_pos,
                   SUM(count_at_prob) OVER () AS total_count
            FROM aggregated
        ),
        metrics AS (
            SELECT prob,
                   CASE WHEN cum_count = 0 THEN 0 ELSE cum_pos::DOUBLE / cum_count END AS precision,
                   CASE WHEN total_pos = 0 THEN 0 ELSE cum_pos::DOUBLE / total_pos END AS recall
            FROM curve
        ),
        bucketed AS (
            SELECT *,
                   CASE
                       WHEN recall <= {focus_recall}
                           THEN CAST(FLOOR(recall / {focus_recall} * {buckets_focus}) AS INTEGER)
                       ELSE {buckets_focus} + CAST(FLOOR((recall - {focus_recall}) / (1 - {focus_recall}) * {buckets_tail}) AS INTEGER)
                   END AS bucket
            FROM metrics
        ),
        sampled AS (
            SELECT
                MAX(recall) AS recall,
                arg_max(precision, recall) AS precision,
                arg_max(prob, recall) AS prob,
                bucket
            FROM bucketed
            GROUP BY bucket
        )
        SELECT prob, precision, recall
        FROM sampled
        ORDER BY recall
    """
    return con.execute(query).fetch_df()


def load_selection_summary(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_operating_metrics(
    summary: dict[str, object], key: str, split: str = "val"
) -> dict[str, float] | None:
    if key.startswith("tau"):
        bucket = summary.get("thresholds", {}).get("metrics", {}).get(key)
    else:
        bucket = summary.get("topk", {}).get(key)
    if not bucket:
        return None
    overall = bucket.get(split, {}).get("overall")
    if not overall:
        return None
    return {
        "precision": float(overall.get("precision", 0.0)),
        "recall": float(overall.get("recall", 0.0)),
        "yield": float(overall.get("yield", 0.0)),
        "predicted": float(overall.get("predicted", 0.0)),
        "true_positives": float(overall.get("true_positives", 0.0)),
    }


def dedupe_preserve_order(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if not key or key in seen:
            continue
        ordered.append(key)
        seen.add(key)
    return ordered


def get_tau_value(summary: dict[str, object], key: str) -> float | None:
    thresholds = summary.get("thresholds", {})
    fixed = thresholds.get("fixed", {}) if isinstance(thresholds, dict) else {}
    precision_targets = (
        thresholds.get("precision_targets", {}) if isinstance(thresholds, dict) else {}
    )

    if isinstance(fixed, dict) and key in fixed:
        return float(fixed[key])

    if key.startswith("tau_precision_") and isinstance(precision_targets, dict):
        try:
            target = key.split("_")[-1]
            lookup_key = target
            value = precision_targets.get(lookup_key)
            if value is not None:
                return float(value)
        except IndexError:
            return None

    return None


def format_percent(
    value: float, decimals: int, tiny_threshold: float = 0.0, tiny_decimals: int = 0
) -> str:
    percent = value * 100.0
    decimals_to_use = decimals
    if tiny_threshold and abs(percent) > 0 and abs(percent) < tiny_threshold:
        decimals_to_use = max(decimals, tiny_decimals)
    return f"{percent:.{decimals_to_use}f}%"


def build_threshold_point(
    summary: dict[str, object],
    key: str,
    label: str,
    split: str,
    other_split: str,
    color: str,
) -> PointAnnotation | None:
    primary_metrics = get_operating_metrics(summary, key, split)
    if not primary_metrics:
        return None
    secondary_metrics = get_operating_metrics(summary, key, other_split) if other_split else None
    tau_value = get_tau_value(summary, key)
    return PointAnnotation(
        identifier=key,
        label=label,
        precision=float(primary_metrics["precision"]),
        recall=float(primary_metrics["recall"]),
        yield_value=float(primary_metrics["yield"]),
        tau=tau_value,
        show_tau=tau_value is not None,
        source_key=key,
        val_precision=float(secondary_metrics["precision"]) if secondary_metrics else None,
        val_recall=float(secondary_metrics["recall"]) if secondary_metrics else None,
        color=color,
        predicted=float(primary_metrics.get("predicted", 0.0)),
        true_positives=float(primary_metrics.get("true_positives", 0.0)),
    )


def build_dcs_point(
    parquet_path: Path,
    label: str,
    color: str,
    primary_counts: dict[str, object],
    secondary_counts: dict[str, object] | None,
) -> PointAnnotation | None:
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return None
    if df.empty:
        return None

    predicted = float(len(df))
    true_positives = float(df["label"].sum())
    total_rows = float(primary_counts.get("total_rows", 0.0) or 0.0)
    total_pos = float(primary_counts.get("positives", 0.0) or 0.0)
    if total_rows <= 0 or total_pos <= 0:
        return None

    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / total_pos if total_pos else 0.0
    yield_value = predicted / total_rows if total_rows else 0.0

    secondary_precision = precision
    secondary_recall = None
    if secondary_counts:
        sec_total_pos = float(secondary_counts.get("positives", 0.0) or 0.0)
        if sec_total_pos > 0:
            secondary_recall = true_positives / sec_total_pos

    return PointAnnotation(
        identifier="dcs_top5",
        label=label,
        precision=precision,
        recall=recall,
        yield_value=yield_value,
        show_tau=False,
        source_key="dcs_top5",
        val_precision=secondary_precision,
        val_recall=secondary_recall,
        color=color,
        predicted=predicted,
        true_positives=true_positives,
    )


def render_figure(
    curves: dict[str, pd.DataFrame],
    output_dir: Path,
    base_rate: float,
    operating_points: list[PointAnnotation],
    *,
    figure_stem: str,
    split_label: str,
    precision_target: float | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#4d4d4d")
        spine.set_linewidth(0.8)

    split_display = (
        "Validation"
        if split_label.lower() == "val"
        else ("Test" if split_label.lower() == "test" else split_label.capitalize())
    )

    positive_recalls = []
    for df in curves.values():
        positives = df["recall"][df["recall"] > 0]
        if not positives.empty:
            positive_recalls.append(float(positives.min()))
    min_recall_fraction = min(positive_recalls) if positive_recalls else 1e-4
    min_recall_fraction = max(min_recall_fraction, 1e-4)
    target_floor_pct = 0.03
    min_recall_pct = max(min_recall_fraction * 100.0, target_floor_pct)
    min_recall_fraction = min_recall_pct / 100.0

    # Plot all available model curves
    curve_colors: dict[str, str] = {}
    for label, df in curves.items():
        recall_pct = df["recall"].clip(lower=min_recall_fraction) * 100
        (line,) = ax.plot(
            recall_pct,
            df["precision"] * 100,
            label=label,
            linewidth=1.2 if label.lower() == "ensemble" else 1.0,
        )
        curve_colors[label] = line.get_color()

    ax.set_xlabel("Recall (% of labeled positives)", fontsize=11)
    ax.set_ylabel(f"Precision on {split_display} split (%)", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(min_recall_pct, 5.0)
    candidate_ticks = [min_recall_pct, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    major_xticks = sorted({tick for tick in candidate_ticks if tick >= min_recall_pct})
    ax.set_xticks(major_xticks)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=list(range(2, 10))))

    def recall_formatter(value: float, _pos: int) -> str:
        if value <= min_recall_pct * 1.05:
            return "0%"
        if value < 0.02:
            return f"{value:.3f}%"
        if value < 0.1:
            return f"{value:.2f}%"
        if value < 1:
            return f"{value:.1f}%"
        return f"{value:.0f}%"

    ax.xaxis.set_major_formatter(FuncFormatter(recall_formatter))

    ax.set_ylim(0, 60)
    ax.set_yticks(list(range(0, 61, 10)))
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    def precision_formatter(value: float, _pos: int) -> str:
        return f"{value:.0f}%"

    ax.yaxis.set_major_formatter(FuncFormatter(precision_formatter))
    ax.tick_params(axis="both", which="major", labelsize=9, length=4.2)
    ax.tick_params(axis="both", which="minor", length=2.6)
    ax.margins(x=0)

    # highlight band intentionally omitted for publication simplicity

    # skip annotations/markers during baseline layout tuning

    base_rate_pct = base_rate * 100.0 if base_rate else 0.0

    if base_rate_pct > 0:
        ax.axhline(
            base_rate_pct,
            color="#9a9a9a",
            linestyle="--",
            linewidth=0.8,
            alpha=0.65,
            zorder=1.1,
        )

    ax.grid(which="major", color="#e3e3e3", linewidth=0.6, alpha=0.9)
    ax.grid(which="minor", color="#f2f2f2", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)

    inset_bounds = [0.48, 0.38, 0.46, 0.5]
    shadow_offset = 0.018
    shadow = FancyBboxPatch(
        (inset_bounds[0] + shadow_offset, inset_bounds[1] + shadow_offset),
        inset_bounds[2],
        inset_bounds[3],
        boxstyle="round,pad=0.02",
        linewidth=0,
        facecolor="#d8d8d8",
        alpha=0.32,
        transform=ax.transAxes,
        zorder=1.5,
    )
    ax.add_patch(shadow)

    inset = ax.inset_axes(inset_bounds)
    inset.set_title(
        "Full Recall Graph - Ensemble", fontsize=9, pad=6.0, color="#2f2f2f", fontweight="bold"
    )
    inset.set_facecolor("#fbfbfb")
    for spine in inset.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.0)

    for label, df in curves.items():
        color = curve_colors.get(label)
        linewidth = 1.2 if label.lower() == "ensemble" else 0.9
        inset.plot(
            df["recall"] * 100,
            df["precision"] * 100,
            color=color,
            linewidth=linewidth,
            alpha=0.95 if label.lower() == "ensemble" else 0.85,
        )

    inset.set_xlim(0, 20.0)
    inset.set_xticks([0, 5, 10, 15, 20])
    inset.xaxis.set_minor_locator(MultipleLocator(2.5))

    def inset_recall_formatter(value: float, _pos: int) -> str:
        return f"{value:.0f}%"

    inset.xaxis.set_major_formatter(FuncFormatter(inset_recall_formatter))

    inset.set_ylim(0, 105)
    inset.set_yticks(list(range(0, 101, 20)))
    inset.yaxis.set_minor_locator(MultipleLocator(10))
    inset.yaxis.set_major_formatter(FuncFormatter(precision_formatter))

    inset.tick_params(axis="both", which="major", labelsize=7, length=3.2, pad=2.4)
    inset.tick_params(axis="both", which="minor", length=2.0)
    inset.grid(which="major", color="#e7e7e7", linewidth=0.5, alpha=0.7)
    inset.grid(which="minor", color="#f2f2f2", linewidth=0.35, alpha=0.5)
    inset.set_xlabel("Recall (%)", fontsize=8.0, labelpad=2.6)
    inset.set_ylabel("Precision (%)", fontsize=8.0, labelpad=3.2)
    inset.set_axisbelow(True)

    inset_labels: dict[str, str] = {
        "tau_precision_0.95": "Core (FHPE)",
        "tau_precision_0.85": "Discovery τ",
        "dcs_top5": "DCS Top-5",
    }

    for point in operating_points:
        if point.precision is None or point.recall is None:
            continue
        max(point.recall * 100.0, min_recall_pct)
        precision_pct_value = point.precision * 100.0
        inset_x = min(point.recall * 100.0, 20.0)
        marker_color = point.color or "#1f77b4"
        inset.scatter(
            [inset_x],
            [precision_pct_value],
            s=54,
            facecolors=marker_color,
            edgecolors="#ffffff",
            linewidths=0.9,
            zorder=6.2,
        )
        label_text = inset_labels.get(point.identifier, point.label)
        offset = (6, 0)
        if point.identifier == "dcs_top5":
            offset = (6, -6)
        inset.annotate(
            label_text,
            xy=(inset_x, precision_pct_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.0,
            ha="left",
            va="center",
            color="#1e1e1e",
            annotation_clip=False,
        )

    handles, labels = ax.get_legend_handles_labels()
    if base_rate_pct > 0:
        base_rate_handle = Line2D(
            [],
            [],
            color="#9a9a9a",
            linestyle="--",
            linewidth=0.9,
        )
        base_rate_label = f"Base rate ({base_rate_pct:.2f}%)"
        handles.append(base_rate_handle)
        labels.append(base_rate_label)

    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=4,
        fontsize=8.4,
        handlelength=1.2,
        handletextpad=0.6,
        columnspacing=1.4,
        borderaxespad=0.0,
    )

    ax.set_title("")

    output_base = output_dir / figure_stem
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 2 PR overlay figure")
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Dataset split to visualize (default: test)",
    )
    parser.add_argument(
        "--meta-inputs",
        default=None,
        help="Optional override for meta_inputs parquet",
    )
    parser.add_argument(
        "--scores",
        default=None,
        help="Optional override for scored edges parquet",
    )
    parser.add_argument(
        "--selection-summary",
        default="results/ensemble/meta_ranker/meta_ranker_v4/selection_summary.json",
        help="Path to selection_summary.json (default: v4 test)",
    )
    parser.add_argument(
        "--output-dir",
        default="figs/chapter2",
        help="Directory to write figure outputs",
    )
    parser.add_argument(
        "--threshold-report-val",
        default="results/ensemble/meta_ranker/meta_ranker_v4/threshold_report_val.json",
        help="Threshold report (validation) used to locate the locked τ",
    )
    parser.add_argument(
        "--threshold-report-test",
        default="results/ensemble/meta_ranker/meta_ranker_v4/threshold_report_test.json",
        help="Threshold report (test) used to evaluate the locked τ",
    )
    args = parser.parse_args()

    split = args.split
    default_meta = Path(f"results/ensemble/meta_dataset_v4/meta_inputs_{split}.parquet")
    default_scores = Path(f"results/ensemble/meta_ranker/meta_ranker_v4/scores_{split}.parquet")
    base_table = Path(args.meta_inputs) if args.meta_inputs else default_meta
    meta_scores = Path(args.scores) if args.scores else default_scores
    summary_path = Path(args.selection_summary)
    output_dir = Path(args.output_dir)
    val_report_path = Path(args.threshold_report_val)
    test_report_path = Path(args.threshold_report_test)

    specs: list[ModelSpec] = [
        ModelSpec("Ensemble", meta_scores, "meta_prob", False),
        ModelSpec(
            "GraphSAGE",
            Path(f"results/ensemble/meta_dataset_v4/graphsage_{split}.parquet"),
            "prob_graphsage",
            True,
        ),
        ModelSpec(
            "Node2Vec",
            Path(f"results/ensemble/meta_dataset_v4/node2vec_{split}.parquet"),
            "prob_node2vec",
            True,
        ),
        ModelSpec(
            "Node2Vec-Temporal",
            Path(f"results/ensemble/meta_dataset_v4/n2v_temporal_{split}.parquet"),
            "prob_n2v_temporal",
            True,
        ),
        ModelSpec(
            "Two-Tower",
            Path(f"results/ensemble/meta_dataset_v4/twotower_{split}.parquet"),
            "prob_twotower",
            True,
        ),
        ModelSpec(
            "TGNN",
            Path(f"results/ensemble/meta_dataset_v4/tgnn_{split}.parquet"),
            "prob_tgnn",
            True,
        ),
        ModelSpec(
            "Heuristics",
            Path(f"results/ensemble/meta_dataset_v4/heuristics_{split}.parquet"),
            "prob_heuristics",
            True,
        ),
    ]

    curves: dict[str, pd.DataFrame] = {}

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    for spec in specs:
        if not spec.available():
            continue
        source_sql = build_source_sql(spec, base_table)
        try:
            curves[spec.label] = fetch_curve_samples(con, source_sql)
        except Exception:  # nosec B112 -- best-effort curve fetch, continue is intentional
            continue

    con.close()

    if not curves:
        raise RuntimeError("No model curves available; cannot render figure")

    summary = load_selection_summary(summary_path)
    base_counts = summary.get("base_counts", {}).get(split, {}).get("overall", {})
    total_rows = base_counts.get("total_rows", 1)
    base_rate = (base_counts.get("positives", 0) / total_rows) if total_rows else 0.0

    def extract_threshold_metrics(report_path: Path) -> dict[str, float]:
        payload = json.loads(report_path.read_text())
        for model in payload.get("models", []):
            if model.get("model") == "ensemble":
                return {
                    "threshold": float(model.get("threshold", 0.0)),
                    "precision": float(model.get("precision", 0.0)),
                    "recall": float(model.get("recall", 0.0)),
                    "yield": float(model.get("yield", 0.0)),
                    "predicted": float(model.get("predicted", 0.0)),
                    "positives": float(model.get("positives", 0.0)),
                    "total": float(model.get("total", 0.0)),
                    "precision_target": float(model.get("precision_target", 0.0)),
                }
        raise KeyError(f"Ensemble metrics missing from {report_path}")

    val_metrics = extract_threshold_metrics(val_report_path)
    extract_threshold_metrics(test_report_path)

    target_precision = val_metrics.get("precision_target")

    other_split = "val" if split == "test" else "test"
    other_counts = (
        summary.get("base_counts", {}).get(other_split, {}).get("overall", {})
        if other_split in summary.get("base_counts", {})
        else {}
    )

    operating_points: list[PointAnnotation] = []
    core_point = build_threshold_point(
        summary,
        "tau_precision_0.95",
        "Core (FHPE)",
        split,
        other_split,
        "#264b96",
    )
    if core_point:
        operating_points.append(core_point)

    discovery_point = build_threshold_point(
        summary,
        "tau_precision_0.85",
        "Discovery τ (85% precision)",
        split,
        other_split,
        "#6f4c9b",
    )
    if discovery_point:
        operating_points.append(discovery_point)

    dcs_path = Path("predictions/meta_ranker_v4/dcs_top5_from_fhpe.parquet")
    dcs_point = build_dcs_point(
        dcs_path,
        "DCS (Top-5 within FHPE)",
        "#4b6272",
        base_counts,
        other_counts,
    )
    if dcs_point:
        operating_points.append(dcs_point)

    figure_stem = f"fig_02_pr_curve_{split}"
    render_figure(
        curves,
        output_dir,
        base_rate,
        operating_points,
        figure_stem=figure_stem,
        split_label=split,
        precision_target=target_precision,
    )

    total_points = sum(len(df) for df in curves.values())
    print(
        f"Rendered {figure_stem} with {len(curves)} models, {total_points} sampled points (split={split})."
    )
    print(f"Outputs written to {output_dir / (figure_stem + '.pdf')} and .png")


if __name__ == "__main__":
    main()
