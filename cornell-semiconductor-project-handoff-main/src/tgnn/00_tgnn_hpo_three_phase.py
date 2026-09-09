#!/usr/bin/env python3
"""
00_tgnn_hpo_three_phase.py
---------------------------
TGNN three-phase HPO orchestrator (matching GraphSAGE structure):

Phase A (screen): Run Optuna HPO with short epochs to find promising configs.
Phase B (refine): Re-evaluate top-M configs with higher epochs (single-seed) on
              full data, then select top-K for robustness finalization.
Finalize: Run top-K configs with multi-seed robustness and emit robust_best_params.json.

Artifacts layout under --artifacts-root/tgnn/<tag>:
- phaseA/ (Optuna study + results)
- phaseB_t<trial>/ (per-config refine runs)
- finalize/ (robust summaries + robust_best_params.json)

Usage example:
python src/tgnn/00_tgnn_hpo_three_phase.py \
  --adj data/processed/core/releases/core_v1/adjacency/train_adj_T0.npz \
  --features data/processed/core/releases/core_v1/features/node_features_T0.parquet \
  --candidates-val data/processed/core/releases/core_v1/candidates/val_candidates.parquet \
  --candidates-test data/processed/core/releases/core_v1/candidates/test_candidates.parquet \
  --splits-root data/processed/core/releases/core_v1/splits \
  --version-tag tgnn_hpo_v1_quarterly \
  --trials-a 60 --epochs-a 8 --top-m 10 --epochs-b-source best --top-k 3 --seeds "42,123,456,789,999"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Three-phase HPO orchestrator for TGNN (temporal forecasting)"
    )
    # Data
    p.add_argument("--adj", required=True, type=str)
    p.add_argument("--features", required=True, type=str)
    p.add_argument("--candidates-val", required=True, type=str)
    p.add_argument("--candidates-test", required=True, type=str)
    p.add_argument("--splits-root", required=True, type=str)
    # Layout / versioning
    p.add_argument("--version-tag", type=str, default="tgnn_v1")
    p.add_argument("--artifacts-root", type=str, default="artifacts/tgnn")
    p.add_argument("--results-root", type=str, default="results/tgnn")
    p.add_argument("--logs-dir", type=str, default="logs/tgnn")
    # Device
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    # Temporal settings
    p.add_argument("--granularity", type=str, default="quarter", choices=["quarter", "annual"])
    p.add_argument(
        "--max-snapshots-hpo",
        type=int,
        default=62,
        help="Max snapshots during Phase A HPO",
    )
    p.add_argument(
        "--max-snapshots-final",
        type=int,
        default=None,
        help="Max snapshots during Phase B/Finalize (None=all)",
    )
    # Phase A (Optuna screening)
    p.add_argument(
        "--trials-a",
        type=int,
        default=60,
        help="Total trials for Phase A (will resume from existing DB if present)",
    )
    p.add_argument("--epochs-a", type=int, default=8)
    p.add_argument("--max-sources-a", type=int, default=20000)
    p.add_argument("--n-jobs-a", type=int, default=1)
    # Phase B (refinement - single seed, all data)
    p.add_argument(
        "--top-m",
        type=int,
        default=10,
        help="Number of top configs to refine in Phase B",
    )
    p.add_argument(
        "--epochs-b-source",
        type=str,
        default="best",
        choices=["best", "fixed"],
        help="Use 'best' to inherit epochs from each trial, or 'fixed' for --epochs-b value",
    )
    p.add_argument(
        "--epochs-b",
        type=int,
        default=60,
        help="Fixed epochs for Phase B (only used if epochs-b-source=fixed)",
    )
    # Finalization (multi-seed robustness)
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top configs from Phase B to finalize",
    )
    p.add_argument("--seeds", type=str, default="42,123,456,789,999")
    # Negative sampling
    p.add_argument("--neg-ratio", type=float, default=2.0)
    return p.parse_args()


def run_cmd(cmd: list[str], log_file: Path | None = None) -> None:
    """Execute command and stream output to log file and stdout."""
    # Override with os.environ["SUPPLYCHAIN_ROOT"] if auto-detection fails
    project_root = os.environ.get("SUPPLYCHAIN_ROOT", str(Path(__file__).resolve().parents[2]))
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as f:
            f.write("[CMD] " + " ".join(cmd) + "\n")
            f.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=project_root,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Command failed: {' '.join(cmd)} (exit {proc.returncode})")
    else:
        subprocess.check_call(cmd, cwd=project_root)


def load_topm(trials_csv: Path, top_m: int) -> pd.DataFrame:
    """Load top-M trials from Phase A CSV."""
    df = pd.read_csv(trials_csv)
    df = df[df["state"] == "COMPLETE"].copy()
    df = df.sort_values("value", ascending=False).head(top_m)  # type: ignore[call-overload]
    return df


def main():
    args = parse_args()

    print("=" * 70)
    print("TGNN-Temporal Three-Phase HPO Orchestrator")
    print("=" * 70)
    print(f"Version tag: {args.version_tag}")
    print(f"Phase A: {args.trials_a} trials, {args.epochs_a} epochs, {args.max_sources_a} sources")
    print(f"Phase B: top-{args.top_m} configs, epochs from {args.epochs_b_source}, ALL sources")
    print(f"Finalize: top-{args.top_k} configs, seeds={args.seeds}")
    print("=" * 70)

    # Setup directories
    base_dir = Path(args.artifacts_root) / args.version_tag
    phase_a_dir = base_dir / "phaseA"
    finalize_dir = base_dir / "finalize"
    results_base = Path(args.results_root) / args.version_tag
    logs_dir = Path(args.logs_dir)

    for d in [phase_a_dir, finalize_dir, results_base, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ===================== Phase A: Optuna HPO =====================
    print("\n" + "=" * 70)
    print("Phase A: Optuna Hyperparameter Search")
    print("=" * 70)

    phase_a_log = logs_dir / f"{args.version_tag}_phaseA.log"

    cmd_a = [
        sys.executable,
        "src/tgnn/01_tgnn_hpo_optuna.py",
        "--adj",
        args.adj,
        "--features",
        args.features,
        "--candidates-val",
        args.candidates_val,
        "--splits-root",
        args.splits_root,
        "--artifacts-dir",
        str(phase_a_dir),
        "--log-dir",
        str(logs_dir),
        "--trials",
        str(args.trials_a),
        "--sweep-epochs",
        str(args.epochs_a),
        "--sweep-max-sources",
        str(args.max_sources_a),
        "--granularity",
        args.granularity,
        "--max-snapshots",
        str(args.max_snapshots_hpo),
        "--neg-ratio",
        str(args.neg_ratio),
        "--device",
        args.device,
        "--n-jobs",
        str(args.n_jobs_a),
    ]

    print(f"Logging to: {phase_a_log}")
    run_cmd(cmd_a, phase_a_log)

    # Check for results
    hpo_results_path = phase_a_dir / "hpo_results.json"
    if not hpo_results_path.exists():
        raise RuntimeError(f"Phase A did not produce hpo_results.json at {hpo_results_path}")

    data = json.loads(hpo_results_path.read_text())
    print("\nPhase A Complete!")
    print(f"  Best trial: {data['best_trial']}")
    print(f"  Best value: {data['best_value']:.6f}")
    print(f"  Best params: {json.dumps(data['best_params'], indent=2)}")

    # ===================== Phase B: Refinement (Single-Seed, Full Data) =====================
    print("\n" + "=" * 70)
    print(f"Phase B: Refinement (Top-{args.top_m}, Single Seed, Full Data)")
    print("=" * 70)

    # Load top-M configs from Phase A
    trials_csv = phase_a_dir / "optuna_trials.csv"
    if not trials_csv.exists():
        raise FileNotFoundError(f"Phase A trials CSV not found: {trials_csv}")
    topm = load_topm(trials_csv, args.top_m)

    print(f"Loaded {len(topm)} configs for Phase B")

    phaseB_records: list[dict] = []
    phaseB_art_root = base_dir
    phaseB_res_root = results_base

    for _idx, row in topm.iterrows():
        trial = int(row["number"])

        # Extract params from Phase A
        hidden = int(row.get("params_hidden", 128))  # type: ignore[arg-type]
        out_channels = int(row.get("params_out_channels", 64))  # type: ignore[arg-type]
        num_layers = int(row.get("params_num_layers", 2))  # type: ignore[arg-type]
        dropout = float(row.get("params_dropout", 0.1))  # type: ignore[arg-type]
        lr = float(row.get("params_lr", 1e-3))  # type: ignore[arg-type]
        temporal_mode = str(row.get("params_temporal_mode", "attention"))
        temporal_decay = float(row.get("params_temporal_decay", 0.08))  # type: ignore[arg-type]  # Use Phase A value

        # Determine epochs for this trial
        if args.epochs_b_source == "best":
            # Use the trial's optimized epoch count
            epochs = int(row.get("params_epochs", args.epochs_b))  # type: ignore[arg-type]
        else:
            # Use fixed epoch count
            epochs = args.epochs_b

        print(
            f"\nPhase B Trial {trial}: hidden={hidden}, out={out_channels}, layers={num_layers}, "
            f"dropout={dropout:.3f}, lr={lr:.6f}, mode={temporal_mode}, epochs={epochs}, "
            f"temporal_decay={temporal_decay:.3f}"
        )

        out_dir = phaseB_res_root / f"phaseB_t{trial}"
        art_dir = phaseB_art_root / f"phaseB_t{trial}"
        out_dir.mkdir(parents=True, exist_ok=True)
        art_dir.mkdir(parents=True, exist_ok=True)

        # Run single seed with full data
        cmd_b = [
            sys.executable,
            "src/tgnn/03_tgnn_train_eval.py",
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
            str(hidden),
            "--out-channels",
            str(out_channels),
            "--num-layers",
            str(num_layers),
            "--dropout",
            str(dropout),
            "--lr",
            str(lr),
            "--temporal-mode",
            temporal_mode,
            "--granularity",
            args.granularity,
            "--epochs",
            str(epochs),
            "--seed",
            "42",  # Single seed for Phase B
            "--out-dir",
            str(out_dir),
            "--artifacts-dir",
            str(art_dir),
            "--device",
            args.device,
        ]

        if args.max_snapshots_final is not None:
            cmd_b.extend(["--max-snapshots", str(args.max_snapshots_final)])

        # Add temporal_decay from Phase A
        cmd_b.extend(["--temporal-decay", str(temporal_decay)])

        log_b = logs_dir / f"{args.version_tag}_phaseB_t{trial}.log"
        print(f"  Logging to: {log_b}")
        run_cmd(cmd_b, log_b)

        # Read validation metric
        global_val_path = out_dir / "global_val.csv"
        if not global_val_path.exists():
            print(f"  WARNING: {global_val_path} not found, skipping trial {trial}")
            continue

        gdf = pd.read_csv(global_val_path)
        val_ndcg = float(gdf.iloc[0]["ndcg@100"]) if "ndcg@100" in gdf.columns else float("nan")

        print(f"  Phase B Trial {trial}: Val NDCG@100={val_ndcg:.6f}")

        phaseB_records.append(
            {
                "trial": trial,
                "value": val_ndcg,
                "params": {
                    "hidden": hidden,
                    "out_channels": out_channels,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "lr": lr,
                    "temporal_mode": temporal_mode,
                    "epochs": epochs,
                },
                "out_dir": str(out_dir),
            }
        )

    # Rank Phase B results and select top-K
    phaseB_records.sort(key=lambda x: x.get("value", 0.0), reverse=True)
    topk = phaseB_records[: args.top_k]

    print(f"\nPhase B Complete! Top-{args.top_k} configs:")
    for i, rec in enumerate(topk, 1):
        print(f"  {i}. Trial {rec['trial']}: NDCG@100={rec['value']:.6f}")

    # Save Phase B results
    phaseB_results_path = base_dir / "phaseB_results.json"
    phaseB_results_path.write_text(
        json.dumps(
            {
                "top_k": args.top_k,
                "configs": topk,
            },
            indent=2,
        )
    )
    print(f"\nPhase B results saved to: {phaseB_results_path}")

    # ===================== Finalize: Multi-Seed Robustness =====================
    print("\n" + "=" * 70)
    print(f"Finalize: Multi-Seed Robustness (Top-{args.top_k})")
    print("=" * 70)

    finalize_log = logs_dir / f"{args.version_tag}_finalize.log"

    cmd_finalize = [
        sys.executable,
        "src/tgnn/02_tgnn_finalize_hpo.py",
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
        "--artifacts-dir",
        str(finalize_dir),
        "--results-dir",
        str(results_base / "finalize"),
        "--logs-dir",
        str(logs_dir),
        "--hpo-results",
        str(phaseB_results_path),  # Use Phase B results
        "--top-k",
        str(args.top_k),
        "--epochs",
        "60",  # Will be overridden by config epochs
        "--seeds",
        args.seeds,
        "--granularity",
        args.granularity,
        "--device",
        args.device,
    ]

    if args.max_snapshots_final is not None:
        cmd_finalize.extend(["--max-snapshots", str(args.max_snapshots_final)])

    print(f"Logging to: {finalize_log}")
    run_cmd(cmd_finalize, finalize_log)

    # Check for final output
    best_params_path = finalize_dir / "robust_best_params.json"
    if not best_params_path.exists():
        raise RuntimeError(
            f"Finalization did not produce robust_best_params.json at {best_params_path}"
        )

    best_data = json.loads(best_params_path.read_text())
    print("\nFinalization Complete!")
    print(f"  Best trial: {best_data['best_trial']}")
    print(f"  Best params: {json.dumps(best_data['best_params'], indent=2)}")
    print(
        f"  Val metric (mean ± 95% CI): {best_data['val_metric_mean']:.6f} ± {best_data['val_metric_ci95']:.6f}"
    )

    # Final summary
    print("\n" + "=" * 70)
    print("Three-Phase HPO Complete!")
    print("=" * 70)
    print(f"Artifacts: {base_dir}")
    print(f"Best params: {best_params_path}")
    print("\nTo run production with best params:")
    print("python src/tgnn/03_tgnn_train_eval.py \\")
    print(f"  --adj {args.adj} \\")
    print(f"  --features {args.features} \\")
    print(f"  --candidates-val {args.candidates_val} \\")
    print(f"  --candidates-test {args.candidates_test} \\")
    print(f"  --splits-root {args.splits_root} \\")
    print(f"  --best-params-path {best_params_path} \\")
    print(f"  --multi-seed --seeds {args.seeds} \\")
    print(f"  --out-dir results/tgnn/{args.version_tag}_production \\")
    print(f"  --device {args.device}")


if __name__ == "__main__":
    main()
