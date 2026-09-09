#!/usr/bin/env python3
"""
02_graphsage_finalize_hpo.py
----------------------------
Finalize GraphSAGE HPO by running a robustness evaluation:

- Loads Optuna trials from artifacts (CSV) and selects the top-K configs by the
  optimization objective (value column; ndcg@100 in our setup).
- Runs src/graphsage/03_graphsage_train_eval.py for each top config across a fixed
  multi-seed set (default: 42,123,456,789,999) with the same short training
  budget used during HPO (default: 8 epochs).
- Aggregates validation (and test) metrics across seeds, computes mean and 95% CIs,
  and selects the best config by mean validation ndcg@100 (tie-breakers applied).
- Writes comprehensive artifacts to ensure full reproducibility:
  - Per-config multi-seed summaries under artifacts/graphsage/robust_t{TRIAL}
  - A project-level robust_topk_summary.json and robust_selection.json
  - robust_best_params.json containing the selected hyperparameters

Notes
- This script does not modify HPO outputs. It produces separate robust_* artifacts.
- Use 03_graphsage_train_eval.py's --best-params-path to load robust_best_params.json
  for your final training/evaluation runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ----------------------------- CLI Arguments ------------------------------ #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finalize GraphSAGE HPO with multi-seed robustness runs"
    )
    # Data paths (required)
    p.add_argument("--adj", required=True, type=str, help="Path to train adjacency CSR .npz")
    p.add_argument("--features", required=True, type=str, help="Path to T0 node features parquet")
    p.add_argument(
        "--struct-feats", type=str, default="", help="Optional path to node_structural_v1.parquet"
    )
    p.add_argument(
        "--candidates-val", required=True, type=str, help="Path to val candidates parquet"
    )
    p.add_argument(
        "--candidates-test", required=True, type=str, help="Path to test candidates parquet"
    )
    p.add_argument(
        "--splits-root",
        required=True,
        type=str,
        help="Path to splits dir containing train/val/test edges",
    )

    # Artifacts & results
    p.add_argument("--artifacts-dir", type=str, default="artifacts/graphsage")
    p.add_argument("--results-dir", type=str, default="results/graphsage")
    p.add_argument("--logs-dir", type=str, default="logs/graphsage")

    # HPO inputs
    p.add_argument(
        "--trials-csv",
        type=str,
        default="",
        help="Override path to optuna_trials.csv (default: <artifacts-dir>/optuna_trials.csv)",
    )
    p.add_argument(
        "--explicit-configs",
        type=str,
        default="",
        help="Optional JSON with explicit top configs to run instead of reading trials CSV.",
    )
    p.add_argument(
        "--top-k", type=int, default=3, help="Number of top configs to evaluate for robustness"
    )
    p.add_argument(
        "--select-metric",
        type=str,
        default="ndcg@100",
        help="Metric to select the final config (should match HPO objective)",
    )

    # Robustness evaluation controls
    p.add_argument(
        "--seeds",
        type=str,
        default="42,123,456,789,999",
        help="Comma-separated seeds for robustness evaluation",
    )
    p.add_argument(
        "--epochs", type=int, default=60, help="Epochs per seed run (match refine/finalize budget)"
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for evaluation runs",
    )
    p.add_argument(
        "--batch-size", type=int, default=2_000_000, help="Eval batch size in run_graphsage_eval"
    )
    p.add_argument(
        "--max-fanout-nodes",
        type=int,
        default=800_000,
        help="Fanout cap: batch*(1+n1+n1*n2) <= cap; auto-clamp when exceeded",
    )
    p.add_argument(
        "--auto-clamp",
        type=str,
        default="true",
        help="Auto-adjust neighbors/batch to honor fanout cap (true/false)",
    )
    p.add_argument(
        "--use-struct-feats",
        type=str,
        default="false",
        help="Whether to concatenate structural scalars during finalize runs (default: false)",
    )

    # Execution options
    p.add_argument(
        "--python-bin", type=str, default=sys.executable, help="Python binary to invoke child runs"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="List planned runs and exit without executing"
    )
    return p.parse_args()


# ------------------------------ Data Types -------------------------------- #


@dataclass
class TrialConfig:
    trial_number: int
    value: float
    hidden: int
    layers: int
    dropout: float
    lr: float
    batch: int
    neighbors: tuple[int, int]
    neg_ratio: float
    neg_strategy: str
    use_id_emb: bool = False
    id_emb_dim: int = 64
    normalize_emb: bool = True

    def to_params_dict(self) -> dict[str, object]:
        return {
            "hidden": int(self.hidden),
            "layers": int(self.layers),
            "dropout": float(self.dropout),
            "lr": float(self.lr),
            "batch": int(self.batch),
            "neighbors": [int(self.neighbors[0]), int(self.neighbors[1])],
            "neg_ratio": float(self.neg_ratio),
            "neg_strategy": str(self.neg_strategy),
            "use_id_emb": bool(self.use_id_emb),
            "id_emb_dim": int(self.id_emb_dim),
            "normalize_emb": bool(self.normalize_emb),
        }


# ------------------------------ Utilities --------------------------------- #


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def load_topk_trials(csv_path: Path, top_k: int) -> list[TrialConfig]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Trials CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Filter complete trials with a numeric objective value
    df = df[(df.get("state") == "COMPLETE") & (~df.get("value").isna())].copy()  # type: ignore[union-attr]
    if df.empty:
        raise RuntimeError("No COMPLETE trials with values found in trials CSV")
    df = df.sort_values("value", ascending=False).head(int(top_k)).reset_index(drop=True)  # type: ignore[call-overload]
    top: list[TrialConfig] = []
    for _, r in df.iterrows():
        nb = str(r.get("params_neighbors", "10,10")).strip()
        try:
            n1, n2 = [int(x) for x in nb.split(",")[:2]]
        except Exception:
            n1, n2 = 10, 10
        tc = TrialConfig(
            trial_number=int(r.get("number")),  # type: ignore[arg-type]
            value=float(r.get("value")),  # type: ignore[arg-type]
            hidden=int(r.get("params_hidden")),  # type: ignore[arg-type]
            layers=int(r.get("params_layers")),  # type: ignore[arg-type]
            dropout=float(r.get("params_dropout")),  # type: ignore[arg-type]
            lr=float(r.get("params_lr")),  # type: ignore[arg-type]
            batch=int(r.get("params_batch")),  # type: ignore[arg-type]
            neighbors=(n1, n2),
            neg_ratio=float(r.get("params_neg_ratio")),  # type: ignore[arg-type]
            neg_strategy=str(r.get("params_neg_strategy")),
        )
        top.append(tc)
    return top


def get_env_snapshot() -> dict[str, str]:
    """Collect environment info for reproducibility."""
    info: dict[str, str] = {
        "timestamp": _now_iso(),
        "python": sys.version.split(" ")[0],
    }
    # Git commit (best-effort)
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # nosec B607 -- git is a well-known system executable
        desc = subprocess.check_output(  # nosec B607 -- git is a well-known system executable
            ["git", "describe", "--always", "--dirty", "--tags"], text=True
        ).strip()
        info["git_rev"] = rev
        info["git_describe"] = desc
    except Exception:  # nosec B110 -- best-effort environment introspection, pass is intentional
        pass
    # Torch/CUDA
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
    except Exception:  # nosec B110 -- best-effort environment introspection, pass is intentional
        pass
    # Optuna
    try:
        import optuna  # type: ignore

        info["optuna"] = optuna.__version__
    except Exception:  # nosec B110 -- best-effort environment introspection, pass is intentional
        pass
    return info


def run_config_multi_seed(
    python_bin: str,
    run_script: Path,
    cfg: TrialConfig,
    seeds: list[int],
    epochs: int,
    device: str,
    batch_size: int,
    max_fanout_nodes: int,
    auto_clamp: bool,
    struct_feats: Path | None,
    use_struct_feats: bool,
    adj: Path,
    features: Path,
    cand_val: Path,
    cand_test: Path,
    splits_root: Path,
    out_dir: Path,
    art_dir: Path,
    log_file: Path,
) -> None:
    """Invoke 03_graphsage_train_eval.py for cfg across seeds; log to file."""
    cmd = [
        python_bin,
        str(run_script),
        "--adj",
        str(adj),
        "--features",
        str(features),
        "--candidates-val",
        str(cand_val),
        "--candidates-test",
        str(cand_test),
        "--splits-root",
        str(splits_root),
        "--out-dir",
        str(out_dir),
        "--artifacts-dir",
        str(art_dir),
        "--device",
        device,
        "--hidden",
        str(cfg.hidden),
        "--layers",
        str(cfg.layers),
        "--dropout",
        str(cfg.dropout),
        "--lr",
        str(cfg.lr),
        "--epochs",
        str(int(epochs)),
        "--batch",
        str(cfg.batch),
        "--neighbors",
        str(cfg.neighbors[0]),
        str(cfg.neighbors[1]),
        # Ranking-aligned negatives and loss across finalize
        "--neg-ratio",
        "2.0",
        "--neg-strategy",
        "mix_v3",
        "--deg-frac",
        "0.4",
        "--twohop-frac",
        "0.2",
        "--sem-frac",
        "0.2",
        "--deg-exp",
        "0.75",
        "--sem-keys",
        "primary_sic_code,gr_country",
        "--emit-strict-warm",
        "true",
        "--calibrate",
        "true",
        "--loss",
        "bpr",
        "--pairwise-negs",
        "2",
        "--multi-seed",
        "--seeds",
        ",".join(str(s) for s in seeds),
        "--batch-size",
        str(int(batch_size)),
        "--max-fanout-nodes",
        str(int(max_fanout_nodes)),
        "--auto-clamp",
        "true" if auto_clamp else "false",
    ]
    if struct_feats is not None:
        cmd += ["--struct-feats", str(struct_feats)]
    cmd += ["--use-struct-feats", "true" if use_struct_feats else "false"]
    # Pass booleans as explicit values (run_graphsage_eval expects type=_bool)
    cmd += ["--use-id-emb", ("true" if bool(cfg.use_id_emb) else "false")]
    cmd += ["--normalize-emb", ("true" if bool(cfg.normalize_emb) else "false")]
    cmd += ["--id-emb-dim", str(int(cfg.id_emb_dim))]
    # Stream output to file and stdout
    with log_file.open("w") as f:
        f.write(f"[FINALIZE] Command: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Robustness run failed for trial {cfg.trial_number} with exit code {proc.returncode}"
            )


def load_multi_seed_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"multi_seed_summary.json not found: {path}")
    return json.loads(path.read_text())


# ------------------------------- Main Flow -------------------------------- #


def main() -> None:
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    results_dir = Path(args.results_dir)
    logs_dir = Path(args.logs_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    struct_feats_path: Path | None = (
        Path(args.struct_feats).resolve() if args.struct_feats else None
    )
    use_struct_feats = _to_bool(args.use_struct_feats)

    # Resolve configs either from explicit JSON or trials CSV
    top_cfgs: list[TrialConfig]
    trials_csv = Path(args.trials_csv) if args.trials_csv else (artifacts_dir / "optuna_trials.csv")
    explicit_path = Path(args.explicit_configs) if args.explicit_configs else None
    if explicit_path and explicit_path.exists():
        obj = json.loads(explicit_path.read_text())
        top_cfgs = []
        for rec in obj:
            params = rec.get("params", {})
            nb = params.get("neighbors", [15, 10])
            n1, n2 = ([*nb, nb[-1]])[:2] if isinstance(nb, list) else (15, 10)
            tc = TrialConfig(
                trial_number=int(rec.get("trial", rec.get("trial_number", -1))),
                value=float(rec.get("value", 0.0)),
                hidden=int(params.get("hidden", 128)),
                layers=int(params.get("layers", 2)),
                dropout=float(params.get("dropout", 0.1)),
                lr=float(params.get("lr", 1e-3)),
                batch=int(params.get("batch", 2048)),
                neighbors=(int(n1), int(n2)),
                neg_ratio=float(params.get("neg_ratio", 1.0)),
                neg_strategy=str(params.get("neg_strategy", "uniform")),
                use_id_emb=bool(params.get("use_id_emb", False)),
                id_emb_dim=int(params.get("id_emb_dim", 64)),
                normalize_emb=bool(params.get("normalize_emb", True)),
            )
            top_cfgs.append(tc)
    else:
        top_cfgs = load_topk_trials(trials_csv, int(args.top_k))

    # Seeds
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if str(s).strip()]
    if len(seeds) < 2:
        raise ValueError("Provide at least 2 seeds for robustness evaluation")

    # Environment snapshot for reproducibility
    env = get_env_snapshot()
    env["finalize_script"] = str(Path(__file__).resolve())
    env["hpo_trials_csv"] = str(trials_csv.resolve()) if trials_csv else ""
    env["data_release"] = str(
        Path(args.splits_root).resolve().parents[1]
    )  # heuristic: .../releases/<name>
    env_path = artifacts_dir / "finalize_env_snapshot.json"
    env_path.write_text(json.dumps(env, indent=2))

    # Top-K run plan summary
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    master_log = logs_dir / f"finalize_graphsage_hpo_{ts}.out"
    (logs_dir).mkdir(parents=True, exist_ok=True)
    with master_log.open("w") as f:
        f.write(f"[FINALIZE] Start: {_now_iso()}\n")
        f.write(f"[FINALIZE] Seeds: {seeds}\n")
        f.write(f"[FINALIZE] Top-K: {len(top_cfgs)} from {trials_csv}\n")
        f.write(json.dumps({"env": env}, indent=2) + "\n")

    if args.dry_run:
        print("[DRY-RUN] Planned configs:")
        for i, cfg in enumerate(top_cfgs, 1):
            print(
                f"  {i}. trial={cfg.trial_number} value={cfg.value:.6f} params={cfg.to_params_dict()}"
            )
        return

    # Execute each config, collect summaries
    run_script = Path(__file__).resolve().parent / "03_graphsage_train_eval.py"
    per_cfg_records: list[dict[str, object]] = []
    for idx, cfg in enumerate(top_cfgs, 1):
        run_tag = f"robust_t{cfg.trial_number}"
        out_dir = results_dir / run_tag
        art_dir = artifacts_dir / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        art_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"{run_tag}.log"
        # Record planned run
        with master_log.open("a") as f:
            f.write(
                f"[FINALIZE] {idx}/{len(top_cfgs)} trial={cfg.trial_number} value={cfg.value:.6f} -> tag={run_tag}\n"
            )
            f.write(json.dumps(cfg.to_params_dict(), indent=2) + "\n")

        run_config_multi_seed(
            python_bin=str(args.python_bin),
            run_script=run_script,
            cfg=cfg,
            seeds=seeds,
            epochs=int(args.epochs),
            device=str(args.device),
            batch_size=int(args.batch_size),
            max_fanout_nodes=int(args.max_fanout_nodes),
            auto_clamp=_to_bool(args.auto_clamp),
            struct_feats=struct_feats_path,
            use_struct_feats=use_struct_feats,
            adj=Path(args.adj),
            features=Path(args.features),
            cand_val=Path(args.candidates_val),
            cand_test=Path(args.candidates_test),
            splits_root=Path(args.splits_root),
            out_dir=out_dir,
            art_dir=art_dir,
            log_file=log_file,
        )

        # Load result summary
        ms_path = art_dir / "multi_seed_summary.json"
        ms = load_multi_seed_summary(ms_path)
        val_ndcg = None
        with contextlib.suppress(Exception):
            val_ndcg = float(ms.get("val", {}).get("ndcg@100", {}).get("mean"))  # type: ignore[union-attr]
        rec = {
            "trial": cfg.trial_number,
            "objective_value": cfg.value,
            "params": cfg.to_params_dict(),
            "multi_seed_summary_path": str(ms_path),
            "val": ms.get("val", {}),
            "test": ms.get("test", {}),
            "val_ndcg@100_mean": val_ndcg,
            "seeds": ms.get("seeds", []),
        }
        per_cfg_records.append(rec)

    # Rank by mean validation ndcg@100
    def _key(rec: dict[str, object]) -> tuple[float, float, float]:
        val = rec.get("val", {})
        nd = val.get("ndcg@100", {}) if isinstance(val, dict) else {}
        mean = float(nd.get("mean", 0.0) or 0.0)
        lower = float(nd.get("ci_lower", -1e9) or -1e9)
        # Prefer higher mean; if tie, higher ci_lower; if tie, lower std
        std = float(nd.get("std", 1e9) or 1e9)
        return (mean, lower, -std)

    ranked = sorted(per_cfg_records, key=_key, reverse=True)
    winner = ranked[0]

    # Write robust summaries
    finalize_dir = artifacts_dir / "finalize"
    finalize_dir.mkdir(parents=True, exist_ok=True)
    (finalize_dir / "robust_topk_summary.json").write_text(
        json.dumps(
            {
                "created_at": _now_iso(),
                "seeds": seeds,
                "topk": ranked,
                "source_trials_csv": str(trials_csv),
                "note": "Selection uses mean validation ndcg@100; test is not used to choose hyperparameters.",
            },
            indent=2,
        )
    )

    # Resolve canonical best_params from the winner's robust summary to avoid drift
    winner_trial = int(winner.get("trial"))  # type: ignore[arg-type]
    robust_dir = artifacts_dir / f"robust_t{winner_trial}"
    best_params = winner["params"]
    try:
        robust_summary = json.loads((robust_dir / "summary.json").read_text())
        hps = robust_summary.get("hparams", {})
        # Prefer hparams recorded by the robust run if present
        if hps:
            best_params = hps
    except Exception:  # nosec B110 -- best-effort hparam recovery, pass is intentional
        pass

    # Produce robust_best_params.json for downstream usage (with canonical params)
    rbp = {
        "best_value": float(winner.get("val_ndcg@100_mean") or 0.0),  # type: ignore[arg-type]
        "best_params": best_params,
        "metric": str(args.select_metric),
        "trials": int(args.top_k),
        "sweep_epochs": int(args.epochs),
        "seed_policy": "multi-seed",
        "seeds": seeds,
        "timestamp": _now_iso(),
        "selection_rule": "rank by mean val ndcg@100; tie-break: higher ci_lower then lower std",
    }
    robust_best_path = artifacts_dir / "robust_best_params.json"
    robust_best_path.write_text(json.dumps(rbp, indent=2))

    # Selection record
    selection = {
        "created_at": _now_iso(),
        "selected_trial": int(winner.get("trial")),  # type: ignore[arg-type]
        "selected_params": winner["params"],
        "validation_summary": winner.get("val", {}),
        "test_summary": winner.get("test", {}),  # present for completeness; not used for selection
        "seeds": seeds,
        "selection_rule": "mean validation ndcg@100 with tie-breakers",
        "env": env,
    }
    (finalize_dir / "robust_selection.json").write_text(json.dumps(selection, indent=2))

    with master_log.open("a") as f:
        f.write(
            f"[FINALIZE] Winner trial={selection['selected_trial']} mean_val_ndcg={rbp['best_value']:.6f}\n"
        )
        f.write(f"[FINALIZE] robust_best_params.json -> {robust_best_path}\n")
        f.write(f"[FINALIZE] End: {_now_iso()}\n")

    print("[FINALIZE] Robustness evaluation complete.")
    print(
        f"[FINALIZE] Winner trial={selection['selected_trial']} (mean val ndcg@100={rbp['best_value']:.6f})"
    )
    print(
        f"[FINALIZE] Use --use-best-params --best-params-path {robust_best_path} for the final GraphSAGE run."
    )


if __name__ == "__main__":
    main()
