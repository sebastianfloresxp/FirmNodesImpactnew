#!/usr/bin/env python3
"""Appendix A.3a full structural slices (Test)."""

# Table CLI cheat sheet: python src/analysis/chapter2/tab_05_structural_slices_test.py \
#   --meta-val <val.parquet> --meta-test <test.parquet> \
#   --scores-val <val_scores.parquet> --scores-test <test_scores.parquet> \
#   --output tables/chapter2/tab_05_structural_slices_test.tex

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from scorecard_common import ScorecardAssets, prepare_scorecard_assets

STRUCTURAL_SLICES: list[tuple[str, str]] = [
    ("nDCG@100 (WW)", "slice_WW"),
    ("nDCG@100 (WW3)", "slice_WW3"),
    ("nDCG@100 (CW)", "slice_CW"),
    ("nDCG@100 (WC)", "slice_WC"),
    ("nDCG@100 (CC)", "slice_CC"),
    ("nDCG@100 (CC3)", "slice_CC3"),
    ("nDCG@100 (Two-hop)", "slice_twohop"),
    ("nDCG@100 (>2-hop)", "slice_gt2hop"),
    ("nDCG@100 (deg Q1)", "slice_deg_q1"),
    ("nDCG@100 (deg Q2)", "slice_deg_q2"),
    ("nDCG@100 (deg Q3)", "slice_deg_q3"),
    ("nDCG@100 (deg Q4)", "slice_deg_q4"),
]


def format_cell(value: float, best: float) -> str:
    if math.isnan(value):
        return r"\NA"
    cell = f"\\num{{{value:.3f}}}"
    if not math.isnan(best) and abs(value - best) <= 1e-12:
        cell = r"{\bfseries " + cell + "}"
    return cell


def render_table(assets: ScorecardAssets, output_path: Path) -> None:
    summary = assets.summary_by_split.get("test")
    if summary is None:
        raise RuntimeError("Test summary unavailable for structural slices")

    best_per_column: dict[str, float] = {}
    for label, slice_key in STRUCTURAL_SLICES:
        values = []
        for model in assets.common_models:
            agg = summary.slice_metrics.get(slice_key, {}).get(model)
            if agg is None:
                continue
            value = agg.mean.get("ndcg", float("nan"))
            if not math.isnan(value):
                values.append(value)
        best_per_column[label] = max(values) if values else float("nan")

    column_spec = "@{}l" + "S[table-format=1.3]" * len(STRUCTURAL_SLICES) + "@{}"
    header = ["\\multicolumn{1}{c}{Model}"] + [
        f"\\multicolumn{{1}}{{c}}{{{label}}}" for label, _ in STRUCTURAL_SLICES
    ]

    lines = [
        "% Auto-generated full structural slice table (Test)",
        "\\begin{tabular}{" + column_spec + "}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
    ]

    for model in assets.common_models:
        row_values = []
        for label, slice_key in STRUCTURAL_SLICES:
            agg = summary.slice_metrics.get(slice_key, {}).get(model)
            value = agg.mean.get("ndcg", float("nan")) if agg else float("nan")
            row_values.append(format_cell(value, best_per_column[label]))
        row = " & ".join([model, *row_values]) + r" \\"
        lines.append(row)

    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Appendix A.3a structural slices (Test)")
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
        default="tables/chapter2/tab_05_structural_slices_test.tex",
        help="Destination TeX fragment",
    )
    args = parser.parse_args()

    assets = prepare_scorecard_assets(
        Path(args.meta_val),
        Path(args.meta_test),
        Path(args.scores_val),
        Path(args.scores_test),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_table(assets, output)


if __name__ == "__main__":
    main()
