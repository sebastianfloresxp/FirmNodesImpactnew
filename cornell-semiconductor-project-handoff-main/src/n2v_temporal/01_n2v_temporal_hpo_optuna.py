#!/usr/bin/env python3
"""
01_tgnn_hpo_optuna.py
---------------------
Optuna hyperparameter optimization for TGNN (Temporal GNN) on core_v1.

Optimizes temporal GNN architecture against validation candidates using
a capped subset of sources with slices disabled for speed. Uses ASHA
pruning for efficient search.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler

# Import helpers from the run script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from n2v_temporal.run_n2v_temporal_eval import (  # type: ignore
    TemporalGNN,
    TwoHopCache,
    build_snapshots,
    evaluate_split,
    load_features,
    load_graph,
    load_n2v_embeddings,
    train_temporal,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna HPO for TGNN on core_v1")
    # Data
    p.add_argument("--adj", required=True, type=str)
    p.add_argument("--features", required=True, type=str)
    p.add_argument("--candidates-val", required=True, type=str)
    p.add_argument("--splits-root", required=True, type=str)
    p.add_argument(
        "--n2v-dir",
        required=True,
        type=str,
        help="Directory with per-snapshot Node2Vec embeddings (snapshot_000.npy, ...)",
    )
    # Storage / logging
    p.add_argument("--artifacts-dir", type=str, default="artifacts/n2v_temporal/hpo")
    p.add_argument("--log-dir", type=str, default="logs/n2v_temporal")
    # HPO knobs
    p.add_argument("--trials", type=int, default=40, help="Number of Optuna trials")
    p.add_argument(
        "--sweep-epochs", type=int, default=8, help="Training epochs per snapshot during HPO"
    )
    p.add_argument(
        "--sweep-max-sources",
        type=int,
        default=20000,
        help="Max validation sources to evaluate (for speed)",
    )
    p.add_argument("--select-metric", type=str, default="ndcg@100", choices=["ndcg@100", "mrr"])
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel trials (keep at 1 for GPU)")
    # Temporal settings
    p.add_argument("--granularity", type=str, default="quarter", choices=["quarter", "annual"])
    p.add_argument(
        "--max-snapshots", type=int, default=62, help="Max temporal snapshots to use during HPO"
    )
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
    # Device / reproducibility
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    # Negative sampling
    p.add_argument("--neg-ratio", type=float, default=2.0, help="Negative sampling ratio")
    return p.parse_args()


def objective(trial: optuna.Trial, device: torch.device, args: argparse.Namespace, Ks: list[int]):
    """Optuna objective function for TGNN hyperparameter search."""
    logger = logging.getLogger("tgnn.hpo")
    t_start = time.time()

    # Sample hyperparameters
    hidden = trial.suggest_categorical("hidden", [64, 128, 256])
    out_channels = trial.suggest_categorical("out_channels", [32, 64, 128])
    num_layers = trial.suggest_categorical("num_layers", [2, 3])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.3])
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    # FIXED: Only use attention mode for true temporal learning
    temporal_mode = "attention"
    # IMPORTANT: Higher epoch range for stable training - removed low epochs (8,10)
    epochs = trial.suggest_categorical("epochs", [12, 16, 20, 24, 28, 32, 36, 40])
    # Temporal decay: exponential recency bias (search to match full-history runs)
    temporal_decay = trial.suggest_float("temporal_decay", 0.0, 0.15)

    logger.info(
        f"Trial {trial.number}: hidden={hidden}, out={out_channels}, layers={num_layers}, "
        f"dropout={dropout:.3f}, lr={lr:.6f}, mode={temporal_mode}, epochs={epochs}, "
        f"temporal_decay={temporal_decay:.3f}"
    )

    # Lazy-load cached data (graph, features, snapshots)
    if not hasattr(objective, "_cache"):
        objective._cache = {}

    g = objective._cache.get("g")
    x = objective._cache.get("x")
    edge_lists = objective._cache.get("edge_lists")
    bucket_ids = objective._cache.get("bucket_ids")
    bucket_to_nodes = objective._cache.get("bucket_to_nodes")
    twohop_cache = objective._cache.get("twohop_cache")
    n2v_arrays = objective._cache.get("n2v_arrays")
    n2v_dim = objective._cache.get("n2v_dim")

    if g is None or x is None or edge_lists is None:
        logger.info("Loading data (first trial)...")
        g = load_graph(Path(args.adj), undirected=True)
        x_np, _ = load_features(Path(args.features))
        x = torch.from_numpy(x_np).float()

        # Build temporal snapshots
        _, edge_lists, _ = build_snapshots(
            Path(args.splits_root),
            args.granularity,
            args.max_snapshots,
        )

        if g.in_deg.size:
            quantiles = np.quantile(g.in_deg, [0.25, 0.5, 0.75])
            bucket_ids = np.digitize(g.in_deg, quantiles, right=True)
        else:
            bucket_ids = np.zeros(g.num_nodes, dtype=np.int64)
        bucket_to_nodes = {}
        for bucket in np.unique(bucket_ids):
            nodes = np.where(bucket_ids == bucket)[0]
            if nodes.size:
                bucket_to_nodes[int(bucket)] = nodes.astype(np.int64)
        twohop_cache = TwoHopCache(g.csr, g.csc)

        n2v_arrays = load_n2v_embeddings(Path(args.n2v_dir), len(edge_lists), g.num_nodes)
        n2v_dim = int(n2v_arrays[0].shape[1]) if n2v_arrays else 0

        objective._cache["g"] = g
        objective._cache["x"] = x
        objective._cache["edge_lists"] = edge_lists
        objective._cache["bucket_ids"] = bucket_ids
        objective._cache["bucket_to_nodes"] = bucket_to_nodes
        objective._cache["twohop_cache"] = twohop_cache
        objective._cache["n2v_arrays"] = n2v_arrays
        objective._cache["n2v_dim"] = n2v_dim
        logger.info(
            f"Cached: {len(edge_lists)} snapshots, {x.size(0):,} nodes, {x.size(1)} features"
        )

    # Build model
    model = TemporalGNN(
        in_channels=x.size(1),
        hidden_channels=hidden,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
        temporal_mode=temporal_mode,
        temporal_decay=temporal_decay,
        num_score_heads=len(args.forecast_horizons),
        n2v_dim=n2v_dim,  # type: ignore[arg-type]
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model: {num_params:,} parameters")

    # Train
    try:
        losses = train_temporal(
            model,
            x,
            g,
            n2v_arrays,  # type: ignore[arg-type]
            edge_lists,
            epochs=epochs,  # Use sampled epochs instead of fixed sweep_epochs
            lr=lr,
            neg_ratio=args.neg_ratio,
            device=device,
            seed=args.seed + trial.number,
            forecast_horizons=args.forecast_horizons,
            min_new_edges=args.min_new_edges,
            bucket_ids=bucket_ids,
            bucket_to_nodes=bucket_to_nodes,
            twohop_cache=twohop_cache,
        )

        # Note: Pruning disabled - let all trials complete due to high loss variance
        # for epoch, loss in enumerate(losses):
        #     trial.report(loss, epoch)
        #     if trial.should_prune():
        #         logger.info(f"  Trial {trial.number} pruned at epoch {epoch}")
        #         raise optuna.TrialPruned()

        logger.info(
            f"  Training: loss {losses[0]:.4f} -> {losses[-1]:.4f} ({time.time() - t_start:.1f}s)"
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"  Trial {trial.number} OOM: {e}")
            torch.cuda.empty_cache()
            return float("-inf")
        raise

    # Evaluate on validation set (capped sources)
    try:
        model.eval()
        x_dev = x.to(device)
        edge_indices = [torch.from_numpy(edges).long().to(device) for edges in edge_lists]
        n2v_tensors = [torch.from_numpy(arr).float().to(device) for arr in n2v_arrays]  # type: ignore[union-attr]
        with torch.no_grad():
            Z = model(x_dev, edge_indices, n2v_tensors)
            import torch.nn.functional as F

            Z = F.normalize(Z, p=2, dim=1)
            head_embs = model.project_heads(Z)
            head_embs = [F.normalize(emb, p=2, dim=1) for emb in head_embs]
            head_names = ["logit"]
            if len(head_embs) >= 2:
                head_names.append("logit_long")
            for idx in range(2, len(head_embs)):
                head_names.append(f"logit_head{idx + 1}")

        gdf, _mdf, _sdf, counts = evaluate_split(
            name="val",
            Z=Z,
            device=device,
            g=g,
            Ks=Ks,
            cand_path=Path(args.candidates_val),
            batch_size=2_000_000,
            splits_root=Path(args.splits_root),
            collect_slices=False,  # Disable slices for speed during HPO
            max_sources=args.sweep_max_sources,
            head_embeddings=head_embs,
            head_names=head_names,
        )

        metric_val = float(gdf[args.select_metric].iloc[0])
        ndcg = float(gdf["ndcg@100"].iloc[0])
        mrr = float(gdf["mrr"].iloc[0])

        elapsed = time.time() - t_start
        logger.info(
            f"  Val ({counts['sources']:,} sources): {args.select_metric}={metric_val:.6f}, "
            f"ndcg@100={ndcg:.6f}, mrr={mrr:.6f} ({elapsed:.1f}s)"
        )

        return metric_val

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"  Trial {trial.number} eval OOM: {e}")
            torch.cuda.empty_cache()
            return float("-inf")
        raise


def main():
    args = parse_args()
    args.forecast_horizons = [
        int(h.strip()) for h in str(args.forecast_horizons).split(",") if h.strip()
    ] or [1]

    # Setup logging
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_dir) / "tgnn_hpo_optuna.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("tgnn.hpo")

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("=" * 70)
    logger.info("TGNN Hyperparameter Optimization (Optuna)")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Trials: {args.trials}")
    logger.info(f"Sweep epochs: {args.sweep_epochs}")
    logger.info(f"Max validation sources: {args.sweep_max_sources}")
    logger.info(f"Selection metric: {args.select_metric}")
    logger.info(f"Temporal: {args.granularity}, max {args.max_snapshots} snapshots")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Forecast horizons: {args.forecast_horizons}, min_new_edges: {args.min_new_edges}")
    logger.info(f"Node2Vec dir: {args.n2v_dir}")

    # Setup output directory
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Create persistent Optuna database
    db_path = artifacts_dir / "optuna_study.db"
    storage_url = f"sqlite:///{db_path}"

    # Create Optuna study with persistent storage (pruner disabled due to high loss variance in TGNN)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        # pruner disabled - let all trials complete
        study_name="tgnn_hpo",
        storage=storage_url,
        load_if_exists=True,  # Resume existing study if it exists
    )

    # Log study status
    if len(study.trials) > 0:
        logger.info(f"Resumed existing study with {len(study.trials)} completed trials")
        logger.info(f"Best trial so far: {study.best_trial.number} (value={study.best_value:.6f})")
    else:
        logger.info("Created new study (no existing trials found)")

    # Define K values for metrics
    Ks = [1, 10, 50, 100]

    # Calculate remaining trials needed
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = max(0, args.trials - completed_trials)

    if remaining_trials > 0:
        logger.info(f"Starting Optuna optimization... ({remaining_trials} trials remaining)")
        study.optimize(
            lambda trial: objective(trial, device, args, Ks),
            n_trials=remaining_trials,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
        )
    else:
        logger.info(f"All {args.trials} trials already completed!")

    # Report results
    logger.info("=" * 70)
    logger.info("Optimization Complete!")
    logger.info("=" * 70)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best {args.select_metric}: {study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")

    # Save results
    best_params = dict(study.best_params)
    best_params.setdefault("forecast_horizons", args.forecast_horizons)
    best_params.setdefault("min_new_edges", args.min_new_edges)
    results = {
        "best_trial": study.best_trial.number,
        "best_value": float(study.best_value),
        "best_params": best_params,
        "select_metric": args.select_metric,
        "num_trials": len(study.trials),
        "sweep_max_sources": args.sweep_max_sources,
        "granularity": args.granularity,
        "max_snapshots": args.max_snapshots,
    }

    results_path = artifacts_dir / "hpo_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {results_path}")

    # Save study trials to CSV for easy access
    trials_df = study.trials_dataframe()
    trials_csv = artifacts_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    logger.info(f"Trials CSV saved to {trials_csv}")

    # Save study database using pickle (Optuna's native format)
    import joblib

    study_path = artifacts_dir / "optuna_study.pkl"
    joblib.dump(study, study_path)
    logger.info(f"Study object saved to {study_path}")


if __name__ == "__main__":
    main()
