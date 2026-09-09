#!/usr/bin/env python3
"""Table 2.3 temporal horizons (mixed validation/test)."""

# Table CLI cheat sheet: python src/analysis/chapter2/tab_03_temporal_horizons.py \
#   --meta-val <val.parquet> --meta-test <test.parquet> \
#   --scores-val <val_scores.parquet> --scores-test <test_scores.parquet> \
#   --output tables/chapter2/tab_03_temporal_horizons.tex

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from scorecard_common import collect_split, prepare_scorecard_assets

HORIZON_LABELS: list[tuple[str, str, str]] = [
    ("AP 0–6m (val)", "val", "slice_horizon_0_6"),
    ("AP 6–12m (val)", "val", "slice_horizon_6_12"),
    ("AP 36–48m (test)", "test", "slice_horizon_36_48"),
    ("AP 48–60m (test)", "test", "slice_horizon_48_60"),
]


def format_cell(value: float, best: float) -> str:
    if math.isnan(value):
        return r"\NA"
    cell = f"\\num{{{value:.3f}}}"
    if not math.isnan(best) and abs(value - best) <= 1e-12:
        cell = r"{\bfseries " + cell + "}"
    return cell


def main() -> None:
    parser = argparse.ArgumentParser(description="Table 2.3 Temporal Horizons")
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
        default="tables/chapter2/tab_03_temporal_horizons.tex",
        help="Destination TeX fragment",
    )
    args = parser.parse_args()

    assets = prepare_scorecard_assets(
        Path(args.meta_val),
        Path(args.meta_test),
        Path(args.scores_val),
        Path(args.scores_test),
    )

    summary_val = collect_split(
        assets.split_meta_paths["val"],
        assets.split_scores_paths["val"],
        assets.common_models,
        ["slice_horizon_0_6", "slice_horizon_6_12"],
        "val",
    )
    summary_test = collect_split(
        assets.split_meta_paths["test"],
        assets.split_scores_paths["test"],
        assets.common_models,
        ["slice_horizon_36_48", "slice_horizon_48_60"],
        "test",
    )

    slice_lookup: dict[str, dict[str, object]] = {
        "slice_horizon_0_6": summary_val.slice_metrics.get("slice_horizon_0_6", {}),
        "slice_horizon_6_12": summary_val.slice_metrics.get("slice_horizon_6_12", {}),
        "slice_horizon_36_48": summary_test.slice_metrics.get("slice_horizon_36_48", {}),
        "slice_horizon_48_60": summary_test.slice_metrics.get("slice_horizon_48_60", {}),
    }

    best_per_column: dict[str, float] = {}
    for label, _, slice_key in HORIZON_LABELS:
        values = []
        for model in assets.common_models:
            agg = slice_lookup.get(slice_key, {}).get(model)
            if agg is None:
                continue
            value = agg.mean.get("map100", float("nan"))
            if not math.isnan(value):
                values.append(value)
        best_per_column[label] = max(values) if values else float("nan")

    column_spec = "@{}l" + "S[table-format=1.3]" * len(HORIZON_LABELS) + "S[table-format=1.3]@{}"
    header = (
        ["\\multicolumn{1}{c}{Model}"]
        + [f"\\multicolumn{{1}}{{c}}{{{label}}}" for label, _, _ in HORIZON_LABELS]
        + ["\\multicolumn{1}{c}{Retention}"]
    )

    lines = [
        "% Auto-generated temporal horizon table",
        "\\begin{tabular}{" + column_spec + "}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
    ]

    for model in assets.common_models:
        cells = [model]
        horizon_values: dict[str, float] = {}
        for label, _, slice_key in HORIZON_LABELS:
            agg = slice_lookup.get(slice_key, {}).get(model)
            value = agg.mean.get("map100", float("nan")) if agg else float("nan")
            horizon_values[label] = value
            cells.append(format_cell(value, best_per_column[label]))

        numerator = horizon_values.get("AP 48–60m (test)", float("nan"))
        denominator = horizon_values.get("AP 0–6m (val)", float("nan"))
        if math.isnan(numerator) or math.isnan(denominator) or abs(denominator) < 1e-12:
            retention = r"\NA"
        else:
            retention = f"\\num{{{(numerator / denominator):.3f}}}"
        cells.append(retention)
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
