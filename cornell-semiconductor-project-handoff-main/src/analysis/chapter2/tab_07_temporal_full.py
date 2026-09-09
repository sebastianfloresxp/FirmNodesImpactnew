#!/usr/bin/env python3
"""Appendix A.4 full temporal horizons."""

# Table CLI cheat sheet: python src/analysis/chapter2/tab_07_temporal_full.py \
#   --meta-val <val.parquet> --meta-test <test.parquet> \
#   --scores-val <val_scores.parquet> --scores-test <test_scores.parquet> \
#   --output tables/chapter2/tab_07_temporal_horizons_full.tex

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from scorecard_common import collect_split, prepare_scorecard_assets

VAL_SLICES = [
    "slice_horizon_0_6",
    "slice_horizon_6_12",
    "slice_horizon_12_24",
    "slice_horizon_24_36",
]
TEST_SLICES = [
    "slice_horizon_36_48",
    "slice_horizon_48_60",
    "slice_horizon_60_72",
    "slice_horizon_72_plus",
]

COLUMN_LABELS: list[tuple[str, str]] = [
    ("AP 0–6m (val)", "slice_horizon_0_6"),
    ("AP 6–12m (val)", "slice_horizon_6_12"),
    ("AP 12–24m (val)", "slice_horizon_12_24"),
    ("AP 24–36m (val)", "slice_horizon_24_36"),
    ("AP 36–48m (test)", "slice_horizon_36_48"),
    ("AP 48–60m (test)", "slice_horizon_48_60"),
    ("AP 60–72m (test)", "slice_horizon_60_72"),
    ("AP 72+m (test)", "slice_horizon_72_plus"),
]


def format_cell(value: float, best: float) -> str:
    if math.isnan(value):
        return r"\NA"
    cell = rf"\num{{{value:.3f}}}"
    if not math.isnan(best) and abs(value - best) <= 1e-12:
        cell = r"{\bfseries " + cell + "}"
    return cell


def main() -> None:
    parser = argparse.ArgumentParser(description="Appendix A.4 Temporal Horizons")
    parser.add_argument(
        "--meta-val",
        default="results/ensemble/meta_dataset_v4/meta_inputs_val.parquet",
        help="Validation meta dataset parquet",
    )
    parser.add_argument(
        "--meta-test",
        default="results/ensemble/meta_dataset_v4/meta_inputs_test.parquet",
        help="Test meta dataset parquet",
    )
    parser.add_argument(
        "--scores-val",
        default="results/ensemble/meta_ranker/meta_ranker_v4/scores_val.parquet",
        help="Validation ensemble scores",
    )
    parser.add_argument(
        "--scores-test",
        default="results/ensemble/meta_ranker/meta_ranker_v4/scores_test.parquet",
        help="Test ensemble scores",
    )
    parser.add_argument(
        "--output",
        default="tables/chapter2/tab_07_temporal_horizons_full.tex",
        help="Destination TeX fragment",
    )
    args = parser.parse_args()

    assets = prepare_scorecard_assets(
        Path(args.meta_val),
        Path(args.meta_test),
        Path(args.scores_val),
        Path(args.scores_test),
    )

    val_summary = collect_split(
        assets.split_meta_paths["val"],
        assets.split_scores_paths["val"],
        assets.common_models,
        VAL_SLICES,
        "val",
    )
    test_summary = collect_split(
        assets.split_meta_paths["test"],
        assets.split_scores_paths["test"],
        assets.common_models,
        TEST_SLICES,
        "test",
    )

    slice_lookup: dict[str, dict[str, object]] = {}
    slice_lookup.update({key: val_summary.slice_metrics.get(key, {}) for key in VAL_SLICES})
    slice_lookup.update({key: test_summary.slice_metrics.get(key, {}) for key in TEST_SLICES})
    slice_counts: dict[str, int] = {}
    slice_counts.update({key: val_summary.slice_sources.get(key, 0) for key in VAL_SLICES})
    slice_counts.update({key: test_summary.slice_sources.get(key, 0) for key in TEST_SLICES})

    best_per_column: dict[str, float] = {}
    for label, slice_key in COLUMN_LABELS:
        values = []
        for model in assets.common_models:
            agg = slice_lookup.get(slice_key, {}).get(model)
            if slice_counts.get(slice_key, 0) == 0:
                continue
            if agg is None:
                continue
            value = agg.mean.get("map100", float("nan"))
            if not math.isnan(value):
                values.append(value)
        best_per_column[label] = max(values) if values else float("nan")

    column_spec = "@{}l" + "S[table-format=1.3]" * len(COLUMN_LABELS) + "S[table-format=1.3]@{}"
    header = (
        ["\\multicolumn{1}{c}{Model}"]
        + [f"\\multicolumn{{1}}{{c}}{{{label}}}" for label, _ in COLUMN_LABELS]
        + ["\\multicolumn{1}{c}{Δ (48–60 − 0–6)}"]
    )

    lines: list[str] = [
        "% Auto-generated full temporal horizon table",
        "\\begin{tabular}{" + column_spec + "}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
    ]

    for model in assets.common_models:
        values_for_model: dict[str, float] = {}
        row_cells = [model]
        for label, slice_key in COLUMN_LABELS:
            agg = slice_lookup.get(slice_key, {}).get(model)
            if slice_counts.get(slice_key, 0) == 0:
                value = float("nan")
            else:
                value = agg.mean.get("map100", float("nan")) if agg else float("nan")
            values_for_model[label] = value
            row_cells.append(format_cell(value, best_per_column[label]))

        ap_long = values_for_model.get("AP 48–60m (test)", float("nan"))
        ap_short = values_for_model.get("AP 0–6m (val)", float("nan"))
        if math.isnan(ap_long) or math.isnan(ap_short):
            delta_cell = r"\NA"
        else:
            delta_cell = rf"\num{{{(ap_long - ap_short):.3f}}}"
        row_cells.append(delta_cell)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
