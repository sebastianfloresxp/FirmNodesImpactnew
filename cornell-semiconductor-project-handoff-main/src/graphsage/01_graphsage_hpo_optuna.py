#!/usr/bin/env python3
"""
01_graphsage_hpo_optuna.py
----------------------
Optuna hyperparameter optimization for GraphSAGE (static) on core_v1.

Optimizes a compact search space against the shared scorecard on a capped
subset of validation sources, with slices disabled for speed. Selects by
ndcg@100 (default) or mrr.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import optuna
import torch
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

# Import minimal helpers from the GraphSAGE run script by path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graphsage.run_graphsage_eval import (  # type: ignore
    build_semantic_indices,
    csr_to_edge_index,
    evaluate_split,
    load_features,
    load_graph,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna HPO for GraphSAGE on core_v1")
    p.add_argument("--adj", required=True, type=str)
    p.add_argument("--features", required=True, type=str)
    # Added: accept struct-feats for compatibility with orchestrator; optional/no-op during HPO
    p.add_argument(
        "--struct-feats",
        type=str,
        default="",
        help="Optional path to node_structural_v1.parquet to concatenate (compat)",
    )
    p.add_argument("--candidates-val", required=True, type=str)
    p.add_argument("--splits-root", required=True, type=str)
    p.add_argument("--artifacts-dir", type=str, default="artifacts/graphsage")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--sweep-epochs", type=int, default=8)
    p.add_argument("--sweep-max-sources", type=int, default=5000)
    p.add_argument("--select-metric", type=str, default="ndcg@100", choices=["ndcg@100", "mrr"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", type=str, default="logs/graphsage")
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel trials within a process (Optuna n_jobs). Keep small for GPU stability.",
    )
    # Enhanced guardrails
    p.add_argument(
        "--max-fanout-nodes",
        type=int,
        default=800_000,
        help="Clamp batch/neighbors so batch*(1+n1+n1*n2) <= this cap",
    )
    p.add_argument(
        "--trial-timeout-sec",
        type=int,
        default=1200,
        help="If a trial exceeds this wall time, mark/prune it",
    )
    # New: allow fixing negative strategy for standardized sweeps
    p.add_argument(
        "--fixed-neg-strategy",
        type=str,
        default="",
        choices=["", "uniform", "mix_deg50", "mix", "mix_v3"],
        help="If set, force neg_strategy to this value for all trials",
    )
    # Optional concat of external embeddings (e.g., Node2Vec) for feature model
    p.add_argument(
        "--concat-emb",
        type=str,
        default="",
        help="Path to embeddings (.pt/.npy/.npz/.parquet) aligned to node ids",
    )
    p.add_argument(
        "--concat-emb-normalize",
        action="store_true",
        help="L2-normalize concatenated embeddings before fusion",
    )
    # Structural ID embeddings + normalization during HPO (fixed settings)
    p.add_argument(
        "--use-id-emb",
        action="store_true",
        help="Enable learnable node-ID embeddings (transductive) during HPO runs",
    )
    p.add_argument("--id-emb-dim", type=int, default=64, help="ID embedding dim when enabled")
    # Keep normalization consistent with run_graphsage_eval default (True)
    p.add_argument(
        "--normalize-emb",
        action="store_true",
        default=True,
        help="L2-normalize embeddings before ranking during HPO eval (and during training in this HPO)",
    )
    return p.parse_args()


def objective(
    trial: optuna.Trial,
    g,
    x,
    feat_df,
    device,
    Ks,
    cand_val_path: Path,
    splits_root: Path,
    max_sources: int,
    select_metric: str,
    max_fanout_nodes: int,
    trial_timeout_sec: int,
) -> float:
    import logging

    logger = logging.getLogger("graphsage.hpo")
    t_start = time.time()

    # Sample hyperparameters
    hidden = trial.suggest_categorical("hidden", [64, 128, 256])
    layers = trial.suggest_categorical("layers", [2, 3])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.3])
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    batch = int(trial.suggest_categorical("batch", [1024, 2048, 4096]))

    # IMPROVED: Remove problematic (25,15) configuration entirely
    neighbors_str = trial.suggest_categorical(
        "neighbors", ["10,10", "15,10", "20,10", "25,10", "10,15", "15,15", "20,15"]
    )
    neighbors = tuple(map(int, neighbors_str.split(",")))

    neg_ratio = trial.suggest_categorical("neg_ratio", [1.0, 2.0])
    # Fixed or sampled negative strategy
    if getattr(optuna.trial, "FixedSuggestionWarning", None):
        pass  # placeholder to satisfy linters if needed
    fixed = getattr(objective, "fixed_neg_strategy", None)
    if fixed is None:
        # Resolve from args via trial user_attrs
        fixed = trial.user_attrs.get("fixed_neg_strategy", "")
    if fixed:
        neg_strategy = str(fixed)
    else:
        neg_strategy = trial.suggest_categorical("neg_strategy", ["uniform", "mix_deg50", "mix_v3"])
    # Standardize negatives
    if neg_strategy == "mix_v3":
        deg_frac = 0.4
        twohop_frac = 0.2
        sem_frac = 0.2
    elif neg_strategy == "mix":
        deg_frac = 0.5
        twohop_frac = 0.1
        sem_frac = 0.0
    else:
        deg_frac = 0.5
        twohop_frac = 0.0
        sem_frac = 0.0

    # Attach mix params and structural settings for training function
    from graphsage import run_graphsage_eval as rge  # type: ignore

    rge.train_graphsage.deg_frac = deg_frac  # type: ignore[attr-defined]
    rge.train_graphsage.twohop_frac = twohop_frac  # type: ignore[attr-defined]
    rge.train_graphsage.sem_frac = sem_frac  # type: ignore[attr-defined]
    rge.train_graphsage.sem_keys = ["primary_sic_code", "gr_country"]  # type: ignore[attr-defined]
    # Use BPR with lighter K during HPO to speed screening
    rge.train_graphsage.loss = "bpr"  # type: ignore[attr-defined]
    rge.train_graphsage.pairwise_negs = 1  # type: ignore[attr-defined]
    # Improve loader throughput during HPO
    rge.train_graphsage.loader_workers = 4  # type: ignore[attr-defined]
    rge.train_graphsage.prefetch_factor = 2  # type: ignore[attr-defined]
    rge.train_graphsage.pin_memory = True  # type: ignore[attr-defined]
    rge.train_graphsage.persistent_workers = True  # type: ignore[attr-defined]
    rge.train_graphsage.use_amp = True  # type: ignore[attr-defined]
    # Structural settings propagated from args via trial.user_attrs
    use_id_emb = bool(trial.user_attrs.get("use_id_emb", False))
    id_emb_dim = int(trial.user_attrs.get("id_emb_dim", 64))
    normalize_emb = bool(trial.user_attrs.get("normalize_emb", False))
    rge.train_graphsage.use_id_emb = use_id_emb  # type: ignore[attr-defined]
    rge.train_graphsage.id_emb_dim = id_emb_dim  # type: ignore[attr-defined]
    # Align training objective with evaluation choice
    with contextlib.suppress(Exception):
        rge.train_graphsage.normalize_emb = bool(normalize_emb)  # type: ignore[attr-defined]
    if neg_strategy == "mix" and sem_frac > 0.0:
        rge.train_graphsage.feat_df = feat_df  # type: ignore[attr-defined]
        rge.train_graphsage.sem_indices = rge.build_semantic_indices(
            feat_df, rge.train_graphsage.sem_keys
        )  # type: ignore[attr-defined]

    # ENHANCED: Aggressive guardrails against subgraph explosion
    n1, n2 = neighbors
    fanout = 1 + n1 + (n1 * n2)
    eff_nodes = batch * fanout
    logger.info(
        f"Trial {trial.number} preflight: batch={batch}, neighbors=({n1},{n2}), fanout={fanout}, eff_nodes≈{eff_nodes}"
    )

    # NEW: Aggressive clamping for high-fanout configurations
    if neighbors == (25, 15):  # This should never happen now, but safety check
        n1, n2 = 15, 10
        batch = min(batch, 2048)
        trial.set_user_attr("neighbors_aggressive_clamp", (n1, n2))
        logger.warning(
            f"Trial {trial.number} aggressive clamp: (25,15) -> ({n1},{n2}), batch -> {batch}"
        )

    # NEW: Limit batch size for high-fanout neighbors
    if n1 >= 20 or n2 >= 15:
        batch = min(batch, 2048)
        trial.set_user_attr("batch_limited_for_high_fanout", batch)
        logger.info(f"Trial {trial.number} limited batch to {batch} for high-fanout neighbors")

    # Recalculate after clamping
    fanout = 1 + n1 + (n1 * n2)
    eff_nodes = batch * fanout

    # Clamp batch first; if still heavy, clamp neighbors
    if eff_nodes > max_fanout_nodes:
        batch = max(1024, int(max_fanout_nodes // max(1, fanout)))
        trial.set_user_attr("batch_clamped", batch)
        eff_nodes = batch * fanout
        logger.info(f"Trial {trial.number} clamped batch -> {batch}, eff_nodes≈{eff_nodes}")

    if eff_nodes > max_fanout_nodes:
        # Reduce neighbors progressively
        if n2 > 10:
            n2 = 10
        if n1 > 15:
            n1 = 15
        fanout = 1 + n1 + (n1 * n2)
        eff_nodes = batch * fanout
        trial.set_user_attr("neighbors_clamped", (n1, n2))
        logger.info(f"Trial {trial.number} clamped neighbors -> ({n1},{n2}), eff_nodes≈{eff_nodes}")

    # If still heavy, use conservative (10,10)
    if eff_nodes > max_fanout_nodes:
        n1, n2 = 10, 10
        fanout = 1 + n1 + (n1 * n2)
        eff_nodes = batch * fanout
        trial.set_user_attr("neighbors_forced_safe", (n1, n2))
        logger.info(
            f"Trial {trial.number} forced safe neighbors -> ({n1},{n2}), eff_nodes≈{eff_nodes}"
        )

    # Train for few epochs with OOM fallback
    sweep_epochs = int(trial.user_attrs.get("sweep_epochs", 8))
    # If still relatively heavy, reduce epochs to keep wall time bounded
    if eff_nodes > 0.6 * max_fanout_nodes:
        sweep_epochs = min(sweep_epochs, 4)
        trial.set_user_attr("epochs_clamped", sweep_epochs)
    seed = int(trial.user_attrs.get("seed", 42))

    # Check timeout before training
    elapsed = time.time() - t_start
    if elapsed > trial_timeout_sec:
        logger.warning(
            f"Trial {trial.number} exceeded timeout {trial_timeout_sec}s before training; pruning"
        )
        raise optuna.TrialPruned()

    # Progress callback for ASHA: report -loss at each epoch (maximize)
    def _progress_cb(epoch: int, epoch_loss: float) -> None:
        try:
            trial.report(float(-epoch_loss), step=int(epoch))
            if trial.should_prune():
                raise optuna.TrialPruned()
        except Exception:  # nosec B110 -- best-effort Optuna progress reporting, pass is intentional
            pass

    try:
        rge.train_graphsage.progress_callback = _progress_cb  # type: ignore[attr-defined]
        # Configure train_graphsage runtime toggles
        rge.train_graphsage.deg_frac = float(deg_frac)  # type: ignore[attr-defined]
        rge.train_graphsage.twohop_frac = float(twohop_frac)  # type: ignore[attr-defined]
        rge.train_graphsage.sem_frac = float(sem_frac)  # type: ignore[attr-defined]
        rge.train_graphsage.deg_exp = 0.75  # type: ignore[attr-defined]
        rge.train_graphsage.no_mp = bool(neighbors == (0, 0))  # type: ignore[attr-defined]
        # Activate semantic indices for mix/mix_v3
        rge.train_graphsage.sem_keys = ["primary_sic_code", "gr_country"]  # type: ignore[attr-defined]
        rge.train_graphsage.feat_df = feat_df  # type: ignore[attr-defined]
        rge.train_graphsage.sem_indices = build_semantic_indices(
            feat_df, rge.train_graphsage.sem_keys
        )  # type: ignore[attr-defined]
        model, _ = rge.train_graphsage(
            g=g,
            x=x,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
            lr=lr,
            epochs=sweep_epochs,
            batch_size=batch,
            neigh1=n1,
            neigh2=n2,
            neg_ratio=2.0,
            neg_strategy=neg_strategy,
            device=device,
            seed=seed,
        )
    except RuntimeError as e:
        emsg = str(e).lower()
        if "out of memory" in emsg or "cuda" in emsg:
            # Retry with safer settings
            safe_batch = 1024
            safe_neighbors = (10, 10)
            trial.set_user_attr("oom_retry", True)
            logger.warning(f"Trial {trial.number} OOM, retrying with safe settings")
            rge.train_graphsage.progress_callback = _progress_cb  # type: ignore[attr-defined]
            rge.train_graphsage.deg_frac = float(deg_frac)  # type: ignore[attr-defined]
            rge.train_graphsage.twohop_frac = float(twohop_frac)  # type: ignore[attr-defined]
            rge.train_graphsage.sem_frac = float(sem_frac)  # type: ignore[attr-defined]
            rge.train_graphsage.deg_exp = 0.75  # type: ignore[attr-defined]
            rge.train_graphsage.no_mp = bool(neighbors == (0, 0))  # type: ignore[attr-defined]
            rge.train_graphsage.sem_keys = ["primary_sic_code", "gr_country"]  # type: ignore[attr-defined]
            rge.train_graphsage.feat_df = feat_df  # type: ignore[attr-defined]
            rge.train_graphsage.sem_indices = build_semantic_indices(
                feat_df, rge.train_graphsage.sem_keys
            )  # type: ignore[attr-defined]
            model, _ = rge.train_graphsage(
                g=g,
                x=x,
                hidden=hidden,
                layers=layers,
                dropout=dropout,
                lr=lr,
                epochs=max(4, sweep_epochs // 2),
                batch_size=safe_batch,
                neigh1=safe_neighbors[0],
                neigh2=safe_neighbors[1],
                neg_ratio=neg_ratio,
                neg_strategy=neg_strategy,
                device=device,
                seed=seed,
            )
        else:
            raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed during training: {e}")
        raise optuna.TrialPruned() from e
    finally:
        with contextlib.suppress(Exception):
            rge.train_graphsage.progress_callback = None  # type: ignore[attr-defined]

    # Check timeout after training
    elapsed = time.time() - t_start
    if elapsed > trial_timeout_sec:
        logger.warning(
            f"Trial {trial.number} exceeded timeout {trial_timeout_sec}s after training; pruning"
        )
        raise optuna.TrialPruned()

    with torch.no_grad():
        x_full = x.to(device)
        if use_id_emb and hasattr(model, "id_emb"):
            x_full = torch.cat([x_full, model.id_emb.weight], dim=1)  # type: ignore[attr-defined]
        z = model.encode(x_full, csr_to_edge_index(g.csr).to(device)).detach()
        if normalize_emb:
            import torch.nn.functional as F  # type: ignore

            z = F.normalize(z, p=2, dim=1)

    # Evaluate val subset; disable slices for speed
    gdf, _, _, _ = evaluate_split(
        name="val",
        Z=z,
        device=device,
        g=g,
        Ks=Ks,
        cand_path=cand_val_path,
        out_dir=Path("/tmp"),  # nosec B108 -- ephemeral HPO trial scratch space
        batch_size=2_000_000,
        splits_root=splits_root,
        collect_slices=False,
        max_sources=max_sources,
    )

    # Final timeout check
    elapsed = time.time() - t_start
    if elapsed > trial_timeout_sec:
        logger.warning(
            f"Trial {trial.number} exceeded timeout {trial_timeout_sec}s (elapsed {int(elapsed)}s); pruning"
        )
        raise optuna.TrialPruned()

    score = (
        float(gdf.iloc[0][select_metric])
        if select_metric in gdf.columns
        else float(gdf.iloc[0]["mrr"])
    )
    trial.report(score, step=1)
    logger.info(f"Trial {trial.number} completed in {int(elapsed)}s with score {score:.6f}")
    return score


def main() -> None:
    args = parse_args()
    # Logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "graphsage_hpo.log"), logging.StreamHandler()],
    )
    logging.getLogger("graphsage.hpo")
    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )
    Ks = [1, 10, 50]
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    # Load data once (optionally append struct feats if provided)
    g = load_graph(Path(args.adj), undirected=False)
    struct_path = Path(args.struct_feats) if getattr(args, "struct_feats", "") else None
    if struct_path is not None and struct_path.exists():
        x, feat_df = load_features(Path(args.features), g.num_nodes, struct_feats_path=struct_path)
    else:
        x, feat_df = load_features(Path(args.features), g.num_nodes)
    # Optional concatenate external embeddings (e.g., Node2Vec)
    if getattr(args, "concat_emb", ""):
        emb_path = Path(str(args.concat_emb))
        if emb_path.exists():
            try:
                ext = emb_path.suffix.lower()
                import numpy as _np

                if ext == ".pt":
                    import torch as _torch

                    obj = _torch.load(emb_path, map_location="cpu")  # nosec B614 -- local pipeline artifacts
                    if isinstance(obj, dict) and "embeddings" in obj:
                        arr = obj["embeddings"]
                        emb = (
                            arr.detach().cpu().numpy()
                            if hasattr(arr, "detach")
                            else _np.asarray(arr)
                        )
                    else:
                        raise RuntimeError("PT file missing embeddings key")
                elif ext == ".npy":
                    emb = _np.load(emb_path)
                elif ext == ".npz":
                    npz = _np.load(emb_path)
                    key = next(iter(npz.keys()))
                    emb = npz[key]
                elif ext == ".parquet":
                    import pandas as _pd

                    df_emb = _pd.read_parquet(emb_path)
                    if "node_id" in df_emb.columns:
                        df_emb = (
                            df_emb.sort_values("node_id")
                            .reset_index(drop=True)
                            .drop(columns=["node_id"])
                        )
                    emb = df_emb.to_numpy(dtype=_np.float32)
                else:
                    raise RuntimeError(f"Unsupported embeddings format: {ext}")
                if emb.shape[0] != g.num_nodes:
                    raise RuntimeError(
                        f"Concat embeddings rows {emb.shape[0]} != num_nodes {g.num_nodes}"
                    )
                if bool(getattr(args, "concat_emb_normalize", False)):
                    denom = _np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
                    emb = (emb / denom).astype(_np.float32)
                x = torch.from_numpy(_np.concatenate([x.numpy(), emb.astype(_np.float32)], axis=1))
                print(f"[HPO] Concatenated external embeddings: {emb_path}, dim={emb.shape[1]}")
            except Exception as e:
                print(f"[HPO][WARN] Failed to concatenate embeddings from {emb_path}: {e}")
    cand_val_path = Path(args.candidates_val)
    splits_root = Path(args.splits_root)

    # Study
    sampler = TPESampler(seed=int(args.seed))
    pruner = SuccessiveHalvingPruner(min_resource=3, reduction_factor=3, min_early_stopping_rate=0)
    # Persistent study storage
    study_name = "graphsage_hpo_core_v1"
    storage_url = f"sqlite:///{art_dir}/graphsage_optuna_study.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    print(f"[HPO] Using persistent study: {storage_url}")
    # Store auxiliary params
    study.set_user_attr("sweep_epochs", int(args.sweep_epochs))
    study.set_user_attr("seed", int(args.seed))

    def obj(trial: optuna.Trial) -> float:
        # propagate attrs
        trial.set_user_attr("sweep_epochs", int(args.sweep_epochs))
        trial.set_user_attr("seed", int(args.seed))
        if args.fixed_neg_strategy:
            trial.set_user_attr("fixed_neg_strategy", str(args.fixed_neg_strategy))
        trial.set_user_attr("use_id_emb", bool(args.use_id_emb))
        trial.set_user_attr("id_emb_dim", int(args.id_emb_dim))
        trial.set_user_attr("normalize_emb", bool(args.normalize_emb))
        return objective(
            trial,
            g,
            x,
            feat_df,
            device,
            Ks,
            cand_val_path,
            splits_root,
            int(args.sweep_max_sources),
            args.select_metric,
            int(args.max_fanout_nodes),
            int(args.trial_timeout_sec),
        )

    print(f"[HPO] Starting Optuna with {args.trials} trials on device={device}")
    print(
        f"[HPO] Enhanced guardrails: max_fanout={args.max_fanout_nodes}, timeout={args.trial_timeout_sec}s"
    )
    study.optimize(obj, n_trials=int(args.trials), n_jobs=int(args.n_jobs), show_progress_bar=False)
    print("[HPO] Best value:", study.best_value)
    print("[HPO] Best params:", study.best_params)

    # Persist results
    # Normalize best_params (ensure JSON-friendly types and include deg_frac default)
    bp = dict(study.best_params)
    if isinstance(bp.get("neighbors"), tuple):
        bp["neighbors"] = list(bp["neighbors"])  # JSON-friendly
    if bp.get("neg_strategy") == "mix" and "deg_frac" not in bp:
        bp["deg_frac"] = 0.5

    best = {
        "best_value": study.best_value,
        "best_params": bp,
        "metric": args.select_metric,
        "trials": int(args.trials),
        "sweep_epochs": int(args.sweep_epochs),
        "seed": int(args.seed),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (art_dir / "best_params.json").write_text(json.dumps(best, indent=2))

    # Trials dataframe
    try:
        df = study.trials_dataframe()
        df.to_csv(art_dir / "optuna_trials.csv", index=False)
    except Exception:  # nosec B110 -- best-effort artifact save, pass is intentional
        pass

    # Summary json
    summary = {
        "best": best,
        "study_direction": study.direction.name,
        "n_trials": len(study.trials),
    }
    (art_dir / "optuna_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[HPO] Wrote results to {art_dir}")


if __name__ == "__main__":
    main()
