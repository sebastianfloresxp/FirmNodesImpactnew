#!/usr/bin/env python3
"""Appendix A.2 seed robustness (Test split focus)."""

# Table CLI cheat sheet: python src/analysis/chapter2/tab_04_seed_robustness.py \
#   --meta-val <val.parquet> --meta-test <test.parquet> \
#   --scores-val <val_scores.parquet> --scores-test <test_scores.parquet> \
#   --output tables/chapter2/tab_04_seed_robustness.tex

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from scorecard_common import collect_split, prepare_scorecard_assets


def format_mean_ci(mean: float, ci: float) -> str:
    if math.isnan(mean):
        return r"\NA"
    if math.isnan(ci):
        return rf"\num{{{mean:.3f}}}"
    return rf"\num{{{mean:.3f}}} \pm \num{{{ci:.3f}}}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Appendix A.2 Seed Robustness")
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
        default="tables/chapter2/tab_04_seed_robustness.tex",
        help="Destination TeX fragment",
    )
    args = parser.parse_args()

    assets = prepare_scorecard_assets(
        Path(args.meta_val),
        Path(args.meta_test),
        Path(args.scores_val),
        Path(args.scores_test),
    )

    summary_test = assets.summary_by_split["test"]
    val_horiz = collect_split(
        assets.split_meta_paths["val"],
        assets.split_scores_paths["val"],
        assets.common_models,
        ["slice_horizon_0_6"],
        "val",
    )
    test_horiz = collect_split(
        assets.split_meta_paths["test"],
        assets.split_scores_paths["test"],
        assets.common_models,
        ["slice_horizon_48_60"],
        "test",
    )

    lines: list[str] = [
        "% Auto-generated seed robustness table (Test)",
        r"\begin{tabular}{@{}lS[table-format=1.0]cccc@{}}",
        r"\toprule",
    ]
    header = (
        r"\multicolumn{1}{c}{Model} & \multicolumn{1}{c}{n_{seeds}} & "
        r"\multicolumn{1}{c}{nDCG@100} & \multicolumn{1}{c}{Hit@10} & "
        r"\multicolumn{1}{c}{AP 0--6m} & \multicolumn{1}{c}{AP 48--60m}"
    )
    lines.append(header + r"\\")
    lines.append(r"\midrule")

    for model in assets.common_models:
        agg = summary_test.global_metrics.get(model)
        if agg is None:
            continue
        ndcg = format_mean_ci(agg.mean.get("ndcg", float("nan")), agg.ci.get("ndcg", float("nan")))
        hit10 = format_mean_ci(
            agg.mean.get("hit10", float("nan")), agg.ci.get("hit10", float("nan"))
        )

        short_agg = val_horiz.slice_metrics.get("slice_horizon_0_6", {}).get(model)
        long_agg = test_horiz.slice_metrics.get("slice_horizon_48_60", {}).get(model)

        ap_short = format_mean_ci(
            short_agg.mean.get("map100", float("nan")) if short_agg else float("nan"),
            short_agg.ci.get("map100", float("nan")) if short_agg else float("nan"),
        )
        ap_long = format_mean_ci(
            long_agg.mean.get("map100", float("nan")) if long_agg else float("nan"),
            long_agg.ci.get("map100", float("nan")) if long_agg else float("nan"),
        )

        row = (
            " & ".join(
                [
                    model,
                    rf"\num{{{agg.n_seeds:d}}}",
                    ndcg,
                    hit10,
                    ap_short,
                    ap_long,
                ]
            )
            + r" \\"
        )
        lines.append(row)

    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
