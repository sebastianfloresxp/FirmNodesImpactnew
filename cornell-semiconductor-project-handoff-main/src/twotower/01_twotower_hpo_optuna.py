#!/usr/bin/env python3
"""
01_twotower_hpo_optuna.py
-------------------------
Optuna HPO for Two-Tower on core_v1 using ASHA pruning.

Searches a compact space and evaluates on a capped number of validation
sources with slices disabled. Trains for few epochs and reports per-epoch
loss to enable early pruning.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging

# Import helpers from the run script
import sys
from pathlib import Path

import optuna
import torch
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from twotower.run_twotower_eval import (  # type: ignore
    MixV3Sampler,
    TwoTowerModel,
    compute_all_embeddings,
    evaluate_split,
    load_features,
    load_graph,
    train_twotower,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna HPO for Two-Tower on core_v1")
    # Data
    p.add_argument("--adj", required=True, type=str)
    p.add_argument("--features", required=True, type=str)
    p.add_argument("--struct-feats", required=True, type=str)
    p.add_argument("--candidates-val", required=True, type=str)
    p.add_argument("--splits-root", required=True, type=str)
    # Storage / logging
    p.add_argument("--artifacts-dir", type=str, default="artifacts/twotower/hpo")
    p.add_argument("--log-dir", type=str, default="logs/twotower")
    # HPO knobs
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--sweep-epochs", type=int, default=8)
    p.add_argument("--sweep-max-sources", type=int, default=5000)
    p.add_argument("--select-metric", type=str, default="ndcg@100", choices=["ndcg@100", "mrr"])
    p.add_argument("--n-jobs", type=int, default=1)
    # Device / reproducibility
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    # Fixed negatives policy
    p.add_argument("--pairwise-negs", type=int, default=2, help="K for BPR during Phase-A sweep")
    p.add_argument("--mix-deg-frac", type=float, default=0.4)
    p.add_argument("--mix-twohop-frac", type=float, default=0.2)
    p.add_argument("--mix-attr-frac", type=float, default=0.2)
    p.add_argument("--mix-deg-exp", type=float, default=0.75)
    # Attr keys used when loading features
    p.add_argument(
        "--attr-keys", type=str, default="country,region,continent,entity_type,primary_sic_code"
    )
    return p.parse_args()


def objective(trial: optuna.Trial, device: torch.device, args: argparse.Namespace, Ks: list[int]):
    # Sample hyperparameters
    embed_dim = trial.suggest_categorical("embed_dim", [64, 128])
    struct_hidden = trial.suggest_categorical("struct_hidden", ["64,64", "128,64"])
    attr_emb_dim = trial.suggest_categorical("attr_emb_dim", [16, 32])
    attr_hidden = trial.suggest_categorical("attr_hidden", ["64,64", "128,64"])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
    lr = trial.suggest_categorical("lr", [1e-3, 5e-4])
    batch_size = trial.suggest_categorical("batch_size", [2048, 4096])

    # Lazy-load global cached data to avoid reloading per trial
    g = objective._cache.get("g")
    feats = objective._cache.get("feats")
    if g is None or feats is None:
        g = load_graph(Path(args.adj), undirected=False)
        attr_keys = [k.strip() for k in str(args.attr_keys).split(",") if k.strip()]
        feats = load_features(Path(args.features), Path(args.struct_feats), attr_keys)
        objective._cache["g"] = g
        objective._cache["feats"] = feats

    # Build model
    model = TwoTowerModel(
        struct_in=int(feats.struct.shape[1]),
        struct_hidden=[int(x.strip()) for x in struct_hidden.split(",") if x.strip()],
        attr_num_cats=feats.num_cats,
        attr_emb_dim=int(attr_emb_dim),
        attr_hidden=[int(x.strip()) for x in attr_hidden.split(",") if x.strip()],
        tower_out_dim=int(embed_dim // 2),
        final_dim=int(embed_dim),
        dropout=float(dropout),
        normalize=True,
    ).to(device)

    sampler = MixV3Sampler(
        g=g,
        features=feats,
        deg_frac=float(args.mix_deg_frac),
        twohop_frac=float(args.mix_twohop_frac),
        attr_frac=float(args.mix_attr_frac),
        deg_exp=float(args.mix_deg_exp),
        seed=int(args.seed),
    )

    # Per-epoch pruning via progress callback
    def _progress_cb(epoch: int, epoch_loss: float) -> None:
        try:
            trial.report(float(-epoch_loss), step=int(epoch))
            if trial.should_prune():
                raise optuna.TrialPruned()
        except Exception:  # nosec B110 -- best-effort Optuna progress reporting, pass is intentional
            pass

    try:
        train_twotower.progress_callback = _progress_cb  # type: ignore[attr-defined]
        # Training toggles for performance
        train_twotower.group_by_src = True  # type: ignore[attr-defined]
        train_twotower.async_sampler = True  # type: ignore[attr-defined]
        train_twotower.use_amp = device.type == "cuda"  # type: ignore[attr-defined]
        # Train for few epochs
        train_twotower(
            model=model,
            g=g,
            feats=feats,
            train_edges_path=Path(args.splits_root) / "train_edges.parquet",
            device=device,
            epochs=int(args.sweep_epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            K=int(args.pairwise_negs),
            sampler=sampler,
        )
    finally:
        with contextlib.suppress(Exception):
            train_twotower.progress_callback = None  # type: ignore[attr-defined]

    # Compute embeddings and evaluate a capped number of sources
    with torch.no_grad():
        ZU, ZV = compute_all_embeddings(model, feats, device=device)
    gdf, _, _, _ = evaluate_split(
        name="val",
        ZU=ZU,
        ZV=ZV,
        device=device,
        g=g,
        Ks=Ks,
        cand_path=Path(args.candidates_val),
        out_dir=Path("/tmp"),  # nosec B108 -- ephemeral HPO trial scratch space
        batch_size=2_000_000,
        splits_root=Path(args.splits_root),
        collect_slices=False,
        max_sources=int(args.sweep_max_sources),
        emit_strict_warm=False,
    )
    score = (
        float(gdf.iloc[0][args.select_metric])
        if args.select_metric in gdf.columns
        else float(gdf.iloc[0]["mrr"])
    )
    trial.report(score, step=1)
    return score


def main() -> None:
    args = parse_args()
    # Logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "twotower_hpo.log"), logging.StreamHandler()],
    )
    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    Ks = [1, 10, 50]
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    # Shared cache for objective
    objective._cache = {}

    # Optuna study
    sampler = TPESampler(seed=int(args.seed))
    pruner = SuccessiveHalvingPruner(min_resource=3, reduction_factor=3, min_early_stopping_rate=0)
    study_name = "twotower_hpo_core_v1"
    storage_url = f"sqlite:///{art_dir}/twotower_optuna_study.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    print(f"[HPO] Using persistent study: {storage_url}")

    def obj(trial: optuna.Trial) -> float:
        return objective(trial, device, args, Ks)

    print(f"[HPO] Starting Optuna with {args.trials} trials on device={device}")
    study.optimize(obj, n_trials=int(args.trials), n_jobs=int(args.n_jobs), show_progress_bar=False)
    print("[HPO] Best value:", study.best_value)
    print("[HPO] Best params:", study.best_params)

    # Persist results
    best = {
        "best_value": study.best_value,
        "best_params": dict(study.best_params),
        "metric": args.select_metric,
        "trials": int(args.trials),
        "sweep_epochs": int(args.sweep_epochs),
        "seed": int(args.seed),
    }
    (art_dir / "best_params.json").write_text(json.dumps(best, indent=2))
    try:
        df = study.trials_dataframe()
        df.to_csv(art_dir / "optuna_trials.csv", index=False)
    except Exception:  # nosec B110 -- best-effort artifact save, pass is intentional
        pass
    (art_dir / "optuna_summary.json").write_text(json.dumps({"best": best}, indent=2))
    print(f"[HPO] Wrote results to {art_dir}")


if __name__ == "__main__":
    main()
