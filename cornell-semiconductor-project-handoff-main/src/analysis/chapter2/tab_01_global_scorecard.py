#!/usr/bin/env python3
"""Global scorecard table for Chapter 2."""

# Table CLI cheat sheet: python src/analysis/chapter2/tab_01_global_scorecard.py \
#   --meta-val <val.parquet> --meta-test <test.parquet> \
#   --scores-val <val_scores.parquet> --scores-test <test_scores.parquet> \
#   --output tables/chapter2

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from scorecard_common import ScorecardAssets, prepare_scorecard_assets, write_model_scorecard_table


def _render_global_tables(
    assets: ScorecardAssets, splits: Iterable[str], output_dir: Path
) -> list[Path]:
    split_to_suffix = {"validation": "val", "test": "test"}
    fragments: list[Path] = []
    for split in splits:
        if split not in assets.summary_by_split:
            raise ValueError(
                f"Split '{split}' not available; choose from {sorted(assets.summary_by_split)}"
            )
        summary = assets.summary_by_split[split]
        suffix = split_to_suffix[split]
        path = output_dir / f"tab_01_global_scorecard_{suffix}.tex"
        write_model_scorecard_table(path, split, summary, assets.common_models)
        fragments.append(path)
    return fragments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Chapter 2 global scorecard table")
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
        default="tables/chapter2",
        help="Directory for generated tables",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = prepare_scorecard_assets(
        Path(args.meta_val),
        Path(args.meta_test),
        Path(args.scores_val),
        Path(args.scores_test),
    )

    _render_global_tables(assets, ("validation", "test"), output_dir)


if __name__ == "__main__":
    main()
