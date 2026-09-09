#!/usr/bin/env python3
"""Structural & generalisation slices (Test split)."""

# Table CLI cheat sheet: python src/analysis/chapter2/tab_02_structural_generalization.py \
#   --meta-val <val.parquet> --meta-test <test.parquet> \
#   --scores-val <val_scores.parquet> --scores-test <test_scores.parquet> \
#   --output tables/chapter2/tab_02_structural_generalization.tex

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from scorecard_common import ScorecardAssets, prepare_scorecard_assets

SLICE_METRICS: list[tuple[str, str, str]] = [
    ("nDCG@100 (CC)", "slice_CC", "ndcg"),
    ("Hit@10 (CC)", "slice_CC", "hit10"),
    ("nDCG@100 (>2-hop)", "slice_gt2hop", "ndcg"),
    ("nDCG@100 (deg Q1)", "slice_deg_q1", "ndcg"),
    ("nDCG@100 (deg Q4)", "slice_deg_q4", "ndcg"),
]


def format_numeric(value: float, best: float) -> str:
    import math

    if math.isnan(value):
        return r"\NA"
    cell = f"\\num{{{value:.3f}}}"
    if not math.isnan(best) and abs(value - best) <= 1e-12:
        cell = r"{\bfseries " + cell + "}"
    return cell


def collect_best_values(assets: ScorecardAssets) -> dict[str, float]:
    import math

    summary = assets.summary_by_split["test"]
    best: dict[str, float] = {}
    for label, slice_key, metric_key in SLICE_METRICS:
        values = []
        for model in assets.common_models:
            agg = summary.slice_metrics.get(slice_key, {}).get(model)
            if agg is None:
                continue
            value = agg.mean.get(metric_key, float("nan"))
            if not math.isnan(value):
                values.append(value)
        best[label] = max(values) if values else float("nan")
    return best


def build_rows(assets: ScorecardAssets, best_map: dict[str, float]) -> list[str]:
    summary = assets.summary_by_split["test"]
    rows: list[str] = []
    for model in assets.common_models:
        agg_map = {
            label: summary.slice_metrics.get(slice_key, {}).get(model)
            for label, slice_key, _ in SLICE_METRICS
        }
        cells = [model]
        for label, _, metric_key in SLICE_METRICS:
            agg = agg_map[label]
            if agg is None:
                cells.append(r"\NA")
            else:
                value = agg.mean.get(metric_key, float("nan"))
                cells.append(format_numeric(value, best_map[label]))
        rows.append(" & ".join(cells) + r" \\")
    return rows


def render_table(assets: ScorecardAssets, output_path: Path) -> None:
    column_spec = "@{}l" + "".join("S[table-format=1.3]" for _ in SLICE_METRICS) + "@{}"
    header = ["\\multicolumn{1}{c}{Model}"] + [
        f"\\multicolumn{{1}}{{c}}{{{label}}}" for label, _, _ in SLICE_METRICS
    ]
    best_map = collect_best_values(assets)
    rows = build_rows(assets, best_map)

    lines = [
        "% Auto-generated structural/generalisation table (Test)",
        "\\begin{tabular}{" + column_spec + "}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Table 2.2 Structural & Generalization Slices (Test)"
    )
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
        help="Validation ensemble score parquet",
    )
    parser.add_argument(
        "--scores-test",
        default="results/ensemble/meta_ranker/meta_ranker_v4/scores_test.parquet",
        help="Test ensemble score parquet",
    )
    parser.add_argument(
        "--output",
        default="tables/chapter2/tab_02_structural_generalization.tex",
        help="Destination TeX fragment",
    )
    args = parser.parse_args()

    assets = prepare_scorecard_assets(
        Path(args.meta_val),
        Path(args.meta_test),
        Path(args.scores_val),
        Path(args.scores_test),
    )
    if "test" not in assets.summary_by_split:
        raise RuntimeError("Test split metrics unavailable")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_table(assets, output_path)


if __name__ == "__main__":
    main()
