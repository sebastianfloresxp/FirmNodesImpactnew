#!/usr/bin/env python3
"""
02_tgnn_finalize_hpo.py
-----------------------
Finalize TGNN HPO by running multi-seed robustness evaluation:

- Loads Optuna trials and selects the top-K configs by optimization metric
- Runs each top config across multiple seeds (default: 42,123,456,789,999)
- Aggregates metrics across seeds and selects the best config by mean validation metric
- Writes robust_best_params.json for use in production runs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TrialConfig:
    trial_number: int
    value: float
    hidden: int
    out_channels: int
    num_layers: int
    dropout: float
    lr: float
    temporal_mode: str
    epochs: int
    temporal_decay: float
    forecast_horizons: list[int] = field(default_factory=list)
    min_new_edges: int = 25

    def to_params_dict(self) -> dict[str, object]:
        return {
            "hidden": int(self.hidden),
            "out_channels": int(self.out_channels),
            "num_layers": int(self.num_layers),
            "dropout": float(self.dropout),
            "lr": float(self.lr),
            "temporal_mode": str(self.temporal_mode),
            "epochs": int(self.epochs),
            "temporal_decay": float(self.temporal_decay),
            "forecast_horizons": list(self.forecast_horizons),
            "min_new_edges": int(self.min_new_edges),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finalize TGNN HPO with multi-seed robustness runs")
    # Data paths
    p.add_argument("--adj", required=True, type=str)
    p.add_argument("--features", required=True, type=str)
    p.add_argument("--candidates-val", required=True, type=str)
    p.add_argument("--candidates-test", required=True, type=str)
    p.add_argument("--splits-root", required=True, type=str)
    p.add_argument(
        "--n2v-dir",
        required=True,
        type=str,
        help="Directory with per-snapshot Node2Vec embeddings (snapshot_000.npy, ...)",
    )
    # Artifacts & results
    p.add_argument("--artifacts-dir", type=str, default="artifacts/n2v_temporal/finalize")
    p.add_argument("--results-dir", type=str, default="results/n2v_temporal/finalize")
    p.add_argument("--logs-dir", type=str, default="logs/n2v_temporal")
    # HPO inputs
    p.add_argument(
        "--hpo-results",
        type=str,
        default="",
        help="Path to hpo_results.json (default: <artifacts-dir>/hpo_results.json)",
    )
    p.add_argument("--top-k", type=int, default=3, help="Number of top configs to evaluate")
    p.add_argument("--select-metric", type=str, default="ndcg@100")
    # Robustness evaluation
    p.add_argument("--seeds", type=str, default="42,123,456,789,999")
    p.add_argument("--epochs", type=int, default=60, help="Epochs per seed run")
    p.add_argument("--granularity", type=str, default="quarter", choices=["quarter", "annual"])
    p.add_argument("--max-snapshots", type=int, default=None, help="Max snapshots (None = all)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--forecast-horizons",
        type=str,
        default="1,4",
        help="Comma-separated list of snapshot horizons to predict during training",
    )
    p.add_argument(
        "--min-new-edges",
        type=int,
        default=25,
        help="Minimum new edges required to form a temporal training batch",
    )
    p.add_argument(
        "--gcn-checkpoint",
        action="store_true",
        help="Force gradient checkpointing during TGNN runs",
    )
    p.add_argument(
        "--checkpoint-threshold",
        type=int,
        default=16000,
        help="Auto-enable checkpointing when hidden*out ≥ threshold (≤0 disables)",
    )
    # Execution
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_top_configs(hpo_results_path: Path, top_k: int) -> list[TrialConfig]:
    """Load top-K trial configurations from HPO results (supports Phase B and Phase A)."""
    if not hpo_results_path.exists():
        raise FileNotFoundError(f"HPO results not found: {hpo_results_path}")

    # First, check if this is Phase B results (from 3-phase pipeline)
    if hpo_results_path.name == "phaseB_results.json":
        print(f"Loading top-{top_k} trials from Phase B results")
        data = json.loads(hpo_results_path.read_text())
        configs = []

        for rec in data["configs"][:top_k]:
            params = rec["params"]
            fh_param = params.get("forecast_horizons") or params.get("forecast_horizon", "1")
            if isinstance(fh_param, (list, tuple)):
                fh_values = [int(v) for v in fh_param]
            else:
                fh_values = [int(v.strip()) for v in str(fh_param).split(",") if v.strip()]
            min_new_edges = int(params.get("min_new_edges", 25))
            config = TrialConfig(
                trial_number=int(rec["trial"]),
                value=float(rec["value"]),
                hidden=int(params["hidden"]),
                out_channels=int(params["out_channels"]),
                num_layers=int(params["num_layers"]),
                dropout=float(params["dropout"]),
                lr=float(params["lr"]),
                temporal_mode=str(params.get("temporal_mode", "attention")),
                epochs=int(params["epochs"]),
                temporal_decay=float(params.get("temporal_decay", 0.05)),
                forecast_horizons=fh_values,
                min_new_edges=min_new_edges,
            )
            configs.append(config)

        print(f"Loaded {len(configs)} configs from Phase B")
        return configs

    # Otherwise, try Phase A optuna trials CSV
    trials_csv = hpo_results_path.parent / "optuna_trials.csv"

    if trials_csv.exists():
        print(f"Loading top-{top_k} trials from {trials_csv}")
        trials_df = pd.read_csv(trials_csv)

        # Filter completed trials with valid values
        completed = trials_df[
            (trials_df["state"] == "COMPLETE") & (trials_df["value"].notna())
        ].copy()

        # Sort by value (descending for maximization)
        completed = completed.sort_values("value", ascending=False)  # type: ignore[call-overload]

        # Take top-K
        top_trials = completed.head(top_k)

        configs = []
        for _, row in top_trials.iterrows():
            # Parse params from columns
            # temporal_mode is now fixed to "attention" and not in HPO search
            temporal_mode = row.get("params_temporal_mode", "attention")
            if pd.isna(temporal_mode):  # type: ignore[misc]
                temporal_mode = "attention"

            temporal_decay = row.get("params_temporal_decay", np.nan)
            if pd.isna(temporal_decay):  # type: ignore[misc]
                temporal_decay = 0.05

            fh_param = row.get("params_forecast_horizons") or row.get(
                "params_forecast_horizon", "1"
            )
            if isinstance(fh_param, (list, tuple)):
                fh_values = [int(v) for v in fh_param]
            else:
                fh_values = [int(v.strip()) for v in str(fh_param).split(",") if str(v).strip()]
            min_new_edges = row.get("params_min_new_edges", np.nan)
            if pd.isna(min_new_edges):  # type: ignore[misc]
                min_new_edges = 25

            config = TrialConfig(
                trial_number=int(row["number"]),
                value=float(row["value"]),
                hidden=int(row["params_hidden"]),
                out_channels=int(row["params_out_channels"]),
                num_layers=int(row["params_num_layers"]),
                dropout=float(row["params_dropout"]),
                lr=float(row["params_lr"]),
                temporal_mode=str(temporal_mode),
                epochs=int(row["params_epochs"]),
                temporal_decay=float(temporal_decay),  # type: ignore[arg-type]
                forecast_horizons=fh_values,
                min_new_edges=int(min_new_edges),  # type: ignore[arg-type]
            )
            configs.append(config)

        print(f"Loaded {len(configs)} configs for evaluation")
        return configs

    else:
        # Fallback: load single best from hpo_results.json
        print(f"WARNING: {trials_csv} not found, using single best config")
        data = json.loads(hpo_results_path.read_text())
        best_params = data["best_params"]

        fh_param = best_params.get("forecast_horizons") or best_params.get("forecast_horizon", "1")
        if isinstance(fh_param, (list, tuple)):
            fh_values = [int(v) for v in fh_param]
        else:
            fh_values = [int(v.strip()) for v in str(fh_param).split(",") if v.strip()]
        config = TrialConfig(
            trial_number=data["best_trial"],
            value=data["best_value"],
            hidden=int(best_params["hidden"]),
            out_channels=int(best_params["out_channels"]),
            num_layers=int(best_params["num_layers"]),
            dropout=float(best_params["dropout"]),
            lr=float(best_params["lr"]),
            temporal_mode=str(best_params.get("temporal_mode", "attention")),
            epochs=int(best_params.get("epochs", 12)),
            temporal_decay=float(best_params.get("temporal_decay", 0.05)),
            forecast_horizons=fh_values,
            min_new_edges=int(best_params.get("min_new_edges", 25)),
        )

        return [config]


def run_single_seed(
    config: TrialConfig,
    seed: int,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, float]:
    """Run TGNN training and evaluation for a single seed."""
    print(f"  Running seed {seed}...")

    # Create proper artifacts path (parallel to results, not nested inside)
    # out_dir is like: results/n2v_temporal/*/finalize/robust_t52/seed_42
    # We want artifacts to be: artifacts/tgnn/*/finalize/robust_t52/seed_42
    Path(args.results_dir)  # results/n2v_temporal/*/finalize
    artifacts_root = Path(args.artifacts_dir).parent  # artifacts/n2v_temporal/*
    trial_name = out_dir.parent.name  # robust_t52
    seed_artifacts_dir = artifacts_root / "finalize" / trial_name / f"seed_{seed}"
    seed_artifacts_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_bin,
        str(Path(__file__).parent / "03_n2v_temporal_train_eval.py"),
        "--adj",
        args.adj,
        "--features",
        args.features,
        "--candidates-val",
        args.candidates_val,
        "--candidates-test",
        args.candidates_test,
        "--splits-root",
        args.splits_root,
        "--hidden",
        str(config.hidden),
        "--out-channels",
        str(config.out_channels),
        "--num-layers",
        str(config.num_layers),
        "--dropout",
        str(config.dropout),
        "--lr",
        str(config.lr),
        "--temporal-mode",
        config.temporal_mode,
        "--granularity",
        args.granularity,
        "--epochs",
        str(config.epochs),  # Use config's optimized epochs, not override
        "--temporal-decay",
        f"{config.temporal_decay}",
        "--forecast-horizons",
        ",".join(str(h) for h in (config.forecast_horizons or args.forecast_horizons)),
        "--min-new-edges",
        str(config.min_new_edges),
        "--n2v-dir",
        args.n2v_dir,
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
        "--artifacts-dir",
        str(seed_artifacts_dir),
        "--device",
        args.device,
    ]

    if args.max_snapshots is not None:
        cmd.extend(["--max-snapshots", str(args.max_snapshots)])

    # Memory safety toggles
    auto_checkpoint = (
        args.checkpoint_threshold is not None
        and args.checkpoint_threshold > 0
        and config.hidden * config.out_channels >= args.checkpoint_threshold
    )
    if args.gcn_checkpoint or auto_checkpoint:
        cmd.append("--gcn-checkpoint")
    if args.checkpoint_threshold is not None:
        cmd.extend(["--checkpoint-threshold", str(args.checkpoint_threshold)])

    if args.dry_run:
        print(f"    [DRY RUN] Would execute: {' '.join(cmd)}")
        return {}

    # Execute
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ERROR: Seed {seed} failed!")
        print(result.stderr)
        return {}

    # Read metrics
    global_val_path = out_dir / "global_val.csv"
    global_test_path = out_dir / "global_test.csv"

    metrics = {}
    if global_val_path.exists():
        df = pd.read_csv(global_val_path)
        for col in df.columns:
            if col not in ["heuristic", "macro"]:
                metrics[f"val_{col}"] = float(df[col].iloc[0])

    if global_test_path.exists():
        df = pd.read_csv(global_test_path)
        for col in df.columns:
            if col not in ["heuristic", "macro"]:
                metrics[f"test_{col}"] = float(df[col].iloc[0])

    return metrics


def aggregate_seeds(seed_results: list[dict[str, float]]) -> dict[str, object]:
    """Aggregate metrics across seeds with mean and 95% CI."""
    if not seed_results:
        return {}

    # Collect all metric names
    metric_names = set()
    for result in seed_results:
        metric_names.update(result.keys())

    aggregated = {}
    for metric in metric_names:
        values = [r[metric] for r in seed_results if metric in r]
        if values:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci95 = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0

            aggregated[metric] = {
                "mean": mean,
                "std": std,
                "ci95": ci95,
                "values": values,
            }

    return aggregated


def main():
    args = parse_args()
    args.forecast_horizons = [
        int(h.strip()) for h in str(args.forecast_horizons).split(",") if h.strip()
    ] or [1]

    print("=" * 70)
    print("TGNN HPO Finalization - Multi-Seed Robustness Evaluation")
    print("=" * 70)

    # Setup directories
    artifacts_dir = Path(args.artifacts_dir)
    results_dir = Path(args.results_dir)
    logs_dir = Path(args.logs_dir)

    for d in [artifacts_dir, results_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load HPO results
    hpo_results_path = (
        Path(args.hpo_results) if args.hpo_results else (artifacts_dir / "hpo_results.json")
    )
    print(f"Loading HPO results from {hpo_results_path}")

    top_configs = load_top_configs(hpo_results_path, args.top_k)
    print(f"Found {len(top_configs)} config(s) to evaluate")

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"Seeds: {seeds}")
    if args.gcn_checkpoint:
        print("Gradient checkpointing: forced ON for all runs")
    elif args.checkpoint_threshold is not None and args.checkpoint_threshold > 0:
        print(f"Gradient checkpointing: auto when hidden*out ≥ {args.checkpoint_threshold}")
    print(f"Forecast horizons: {args.forecast_horizons}, min_new_edges: {args.min_new_edges}")
    print(f"Node2Vec dir: {args.n2v_dir}")

    # Evaluate each config across seeds
    all_results = []

    for config in top_configs:
        print(f"\nEvaluating Trial {config.trial_number} (value={config.value:.6f}):")
        print(f"  Params: {config.to_params_dict()}")

        trial_dir = results_dir / f"robust_t{config.trial_number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        seed_results = []
        for seed in seeds:
            seed_dir = trial_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            metrics = run_single_seed(config, seed, args, seed_dir)
            if metrics:
                seed_results.append(metrics)

        if not seed_results:
            print(f"  WARNING: No successful runs for trial {config.trial_number}")
            continue

        # Aggregate across seeds
        aggregated = aggregate_seeds(seed_results)

        # Store results
        result = {
            "trial": config.trial_number,
            "params": config.to_params_dict(),
            "seeds": seeds,
            "num_seeds": len(seed_results),
            "aggregated_metrics": aggregated,
        }
        all_results.append(result)

        # Save trial summary
        summary_path = trial_dir / "robust_summary.json"
        summary_path.write_text(json.dumps(result, indent=2))
        print(f"  Saved summary to {summary_path}")

        # Print key metrics
        val_metric = f"val_{args.select_metric}"
        if val_metric in aggregated:
            mean = aggregated[val_metric]["mean"]  # type: ignore[index]
            ci95 = aggregated[val_metric]["ci95"]  # type: ignore[index]
            print(f"  Val {args.select_metric}: {mean:.6f} ± {ci95:.6f}")

    # Select best config
    if not all_results:
        print("\nERROR: No successful evaluations!")
        return

    val_metric_key = f"val_{args.select_metric}"
    valid_results = [r for r in all_results if val_metric_key in r["aggregated_metrics"]]

    if not valid_results:
        print(f"\nERROR: No results with {val_metric_key}!")
        return

    best_result = max(valid_results, key=lambda r: r["aggregated_metrics"][val_metric_key]["mean"])

    print("\n" + "=" * 70)
    print("Best Configuration Selected!")
    print("=" * 70)
    print(f"Trial: {best_result['trial']}")
    print(f"Params: {json.dumps(best_result['params'], indent=2)}")

    val_mean = best_result["aggregated_metrics"][val_metric_key]["mean"]
    val_ci95 = best_result["aggregated_metrics"][val_metric_key]["ci95"]
    print(f"{args.select_metric}: {val_mean:.6f} ± {val_ci95:.6f}")

    # Save best params
    best_params_path = artifacts_dir / "robust_best_params.json"
    best_params_data = {
        "best_trial": best_result["trial"],
        "best_params": best_result["params"],
        "select_metric": args.select_metric,
        "val_metric_mean": val_mean,
        "val_metric_ci95": val_ci95,
        "num_seeds": best_result["num_seeds"],
        "seeds": best_result["seeds"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    best_params_path.write_text(json.dumps(best_params_data, indent=2))
    print(f"\nSaved best params to {best_params_path}")

    # Save all results summary
    summary_path = artifacts_dir / "robust_topk_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"Saved full summary to {summary_path}")


if __name__ == "__main__":
    main()
