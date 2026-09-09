#!/usr/bin/env python3
"""
Scoreboard Aggregator for Multi-Model Comparison

Reads aggregated results from multiple models and creates comparison tables
for dissertation-level analysis.
"""

import argparse
from pathlib import Path

import pandas as pd


def load_model_results(model_dir: Path, split: str = "val") -> dict | None:
    """Load aggregated results for a model."""
    agg_file = model_dir / f"aggregate_{split}.csv"
    if not agg_file.exists():
        return None

    df = pd.read_csv(agg_file)
    results = {}
    for _, row in df.iterrows():
        metric = row["metric"]
        results[metric] = {
            "mean": row["mean"],
            "std": row["std"],
            "ci_lower": row["ci_lower"],
            "ci_upper": row["ci_upper"],
            "n": row["n"],
        }
    return results


def create_scoreboard(models: list[str], results_dir: Path, split: str = "val") -> pd.DataFrame:
    """Create a scoreboard comparing multiple models."""
    scoreboard_data = []

    for model in models:
        model_dir = results_dir / model
        results = load_model_results(model_dir, split)

        if results is None:
            print(f"Warning: No results found for {model}")
            continue

        row: dict[str, object] = {"model": model}
        for metric in ["hit@1", "hit@10", "hit@50", "mrr", "map", "ndcg@100"]:
            if metric in results:
                stats = results[metric]
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
                row[f"{metric}_ci_lower"] = stats["ci_lower"]
                row[f"{metric}_ci_upper"] = stats["ci_upper"]
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_ci_lower"] = None
                row[f"{metric}_ci_upper"] = None

        scoreboard_data.append(row)

    return pd.DataFrame(scoreboard_data)


def format_scoreboard(df: pd.DataFrame) -> str:
    """Format scoreboard for display."""
    output = []
    output.append("=" * 80)
    output.append("MODEL COMPARISON SCOREBOARD")
    output.append("=" * 80)

    for _, row in df.iterrows():
        model = row["model"]
        output.append(f"\n{model}:")

        for metric in ["hit@1", "hit@10", "hit@50", "mrr", "map", "ndcg@100"]:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            ci_lower = row[f"{metric}_ci_lower"]
            ci_upper = row[f"{metric}_ci_upper"]

            if mean is not None:
                output.append(
                    f"  {metric}: {mean:.4f} ± {std:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
                )

    output.append("\n" + "=" * 80)
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Create model comparison scoreboard")
    parser.add_argument("--models", required=True, help="Comma-separated list of model names")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Results directory"
    )
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Split to compare")
    parser.add_argument("--output", type=Path, help="Output file for scoreboard")

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]

    # Create scoreboard
    scoreboard = create_scoreboard(models, args.results_dir, args.split)

    if scoreboard.empty:
        print("No results found for any models!")
        return

    # Display formatted scoreboard
    print(format_scoreboard(scoreboard))

    # Save scoreboard
    if args.output:
        scoreboard.to_csv(args.output, index=False)
        print(f"Scoreboard saved to {args.output}")
    else:
        output_file = args.results_dir / f"scoreboard_{args.split}.csv"
        scoreboard.to_csv(output_file, index=False)
        print(f"Scoreboard saved to {output_file}")


if __name__ == "__main__":
    main()
