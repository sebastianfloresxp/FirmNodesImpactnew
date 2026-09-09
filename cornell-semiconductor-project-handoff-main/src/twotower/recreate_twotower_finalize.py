#!/usr/bin/env python3
"""Rebuild Two-Tower finalize aggregates from completed seed outputs.

Use when `02_twotower_finalize_hpo.py` is interrupted after some seeds/trials
finish. The script scans the existing `results` / `artifacts` directories,
loads `global_{val,test}.csv` files for every available seed, and regenerates
`robust_selection.csv` plus `robust_best_params.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate Two-Tower finalize outputs from completed seeds"
    )
    parser.add_argument(
        "--explicit-configs",
        required=True,
        type=str,
        help="JSON file containing the explicit configs emitted by the orchestrator (list of {trial, params, ...})",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=str,
        help="Directory where `02_twotower_finalize_hpo.py` wrote per-seed evaluation outputs",
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        type=str,
        help="Directory where `02_twotower_finalize_hpo.py` wrote per-seed artifacts",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456,789,999",
        help="Comma-separated list of seeds that were scheduled during finalize",
    )
    parser.add_argument(
        "--write-missing-report",
        action="store_true",
        help="Emit a JSON file alongside the aggregates that lists missing seeds per trial",
    )
    return parser.parse_args()


def load_configs(path: Path) -> list[dict[str, object]]:
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("explicit configs JSON must contain a list of records")
        return data
    except Exception as exc:
        raise RuntimeError(f"Failed to load explicit configs from {path}: {exc}") from exc


def collect_trial_metrics(
    trial: int,
    params: dict[str, object],
    seeds: list[int],
    results_root: Path,
) -> dict[str, object]:
    available: list[int] = []
    val_scores: list[float] = []
    test_scores: list[float] = []
    for seed in seeds:
        seed_dir = results_root / f"finalize_t{trial}" / f"seed_{seed}"
        val_csv = seed_dir / "global_val.csv"
        test_csv = seed_dir / "global_test.csv"
        if val_csv.exists() and test_csv.exists():
            try:
                val_df = pd.read_csv(val_csv)
                test_df = pd.read_csv(test_csv)
                val_scores.append(float(val_df.iloc[0]["ndcg@100"]))
                test_scores.append(float(test_df.iloc[0]["ndcg@100"]))
                available.append(seed)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read metrics for trial {trial} seed {seed}: {exc}"
                ) from exc
    if not val_scores:
        raise RuntimeError(
            f"No completed seeds found for trial {trial}; expected one of {seeds}."
            " Ensure the finalize run produced per-seed outputs before rerunning this helper."
        )
    return {
        "trial": int(trial),
        "params": params,
        "val_mean_ndcg@100": float(pd.Series(val_scores).mean()),
        "test_mean_ndcg@100": float(pd.Series(test_scores).mean()),
        "val_std_ndcg@100": float(pd.Series(val_scores).std(ddof=0)),
        "test_std_ndcg@100": float(pd.Series(test_scores).std(ddof=0)),
        "seeds_completed": len(available),
        "available_seeds": available,
        "missing_seeds": [s for s in seeds if s not in available],
    }


def main() -> None:
    args = parse_args()
    explicit_path = Path(args.explicit_configs)
    results_root = Path(args.results_dir)
    artifacts_root = Path(args.artifacts_dir)
    if not explicit_path.exists():
        raise FileNotFoundError(f"Explicit configs JSON not found: {explicit_path}")
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory not found: {results_root}")
    artifacts_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    configs = load_configs(explicit_path)

    records: list[dict[str, object]] = []
    missing_report: dict[int, list[int]] = {}
    for rec in configs:
        trial = int(rec.get("trial", -1))  # type: ignore[arg-type]
        params = rec.get("params", {})
        metrics = collect_trial_metrics(trial, params, seeds, results_root)  # type: ignore[arg-type]
        records.append(metrics)
        missing_report[trial] = metrics["missing_seeds"]  # type: ignore[index]
        print(
            f"[RECREATE] Trial {trial}: val_mean_ndcg@100={metrics['val_mean_ndcg@100']:.6f}"
            f" (seeds={metrics['available_seeds']})"
        )

    robust_df = (
        pd.DataFrame(records)
        .sort_values("val_mean_ndcg@100", ascending=False)
        .reset_index(drop=True)
    )
    robust_path = artifacts_root / "robust_selection.csv"
    robust_df.to_csv(robust_path, index=False)
    print(f"[RECREATE] Wrote robust selection to {robust_path}")

    best = robust_df.iloc[0].to_dict()
    best_path = artifacts_root / "robust_best_params.json"
    best_payload = {
        "trial": int(best["trial"]),
        "val_mean_ndcg@100": float(best["val_mean_ndcg@100"]),
        "val_std_ndcg@100": float(best.get("val_std_ndcg@100", 0.0)),
        "test_mean_ndcg@100": float(best["test_mean_ndcg@100"]),
        "test_std_ndcg@100": float(best.get("test_std_ndcg@100", 0.0)),
        "seeds_completed": int(best.get("seeds_completed", 0)),
        "available_seeds": list(best.get("available_seeds", [])),
        "params": best.get("params", {}),
    }
    best_path.write_text(json.dumps(best_payload, indent=2))
    print(f"[RECREATE] Wrote best params to {best_path}")

    if args.write_missing_report:
        miss_path = artifacts_root / "robust_missing_seeds.json"
        miss_path.write_text(json.dumps(missing_report, indent=2))
        print(f"[RECREATE] Wrote missing-seed report to {miss_path}")


if __name__ == "__main__":
    main()
