#!/usr/bin/env python3
"""
00_tgnn_hpo_two_phase.py
------------------------
TGNN two-phase HPO orchestrator:

Phase A (screen): Run Optuna HPO with short sweep epochs to find promising configs.
Phase B (finalize): Re-evaluate top-K configs with multi-seed robustness and emit
robust_best_params.json.

Artifacts layout under --artifacts-root/tgnn/<tag>:
- phaseA/ (Optuna study + results)
- finalize/ (robust summaries + robust_best_params.json)

Usage example:
python src/tgnn/00_tgnn_hpo_two_phase.py \
  --adj data/processed/core/releases/core_v1/adjacency/train_adj_T0.npz \
  --features data/processed/core/releases/core_v1/features/node_features_T0.parquet \
  --candidates-val data/processed/core/releases/core_v1/candidates/val_candidates.parquet \
  --candidates-test data/processed/core/releases/core_v1/candidates/test_candidates.parquet \
  --splits-root data/processed/core/releases/core_v1/splits \
  --version-tag tgnn_v1 --trials-a 40 --epochs-a 8 --top-k 3 --epochs-finalize 60
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-phase HPO orchestrator for TGNN (temporal forecasting)"
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
    p.add_argument("--max-snapshots-hpo", type=int, default=62, help="Max snapshots during HPO")
    p.add_argument(
        "--max-snapshots-final",
        type=int,
        default=None,
        help="Max snapshots during finalization (None=all)",
    )
    # Phase A (Optuna)
    p.add_argument("--trials-a", type=int, default=40)
    p.add_argument("--epochs-a", type=int, default=8)
    p.add_argument("--max-sources-a", type=int, default=20000)
    p.add_argument("--n-jobs-a", type=int, default=1)
    # Finalization (multi-seed)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--epochs-finalize", type=int, default=60)
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


def main():
    args = parse_args()

    print("=" * 70)
    print("TGNN-Temporal Two-Phase HPO Orchestrator")
    print("=" * 70)
    print(f"Version tag: {args.version_tag}")
    print(f"Phase A: {args.trials_a} trials, {args.epochs_a} epochs")
    print(f"Finalize: top-{args.top_k} configs, {args.epochs_finalize} epochs, seeds={args.seeds}")
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
        "--forecast-horizons",
        "1,4",
        "--min-new-edges",
        "25",
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

    # ===================== Phase B: Multi-Seed Finalization =====================
    print("\n" + "=" * 70)
    print(f"Phase B: Multi-Seed Robustness (Top-{args.top_k})")
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
        str(hpo_results_path),
        "--top-k",
        str(args.top_k),
        "--epochs",
        str(args.epochs_finalize),
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
    print("Two-Phase HPO Complete!")
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
