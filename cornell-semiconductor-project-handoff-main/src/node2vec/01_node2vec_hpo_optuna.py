#!/usr/bin/env python3
"""Optuna-based hyperparameter optimization for Node2Vec on core_v1."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import optuna
import torch
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

# Reuse training/eval helpers from the main Node2Vec runner
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from node2vec.run_node2vec_eval import (  # type: ignore
    _now_iso,
    evaluate_split,
    get_embeddings,
    load_graph,
    set_seed,
    train_node2vec,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna HPO for Node2Vec on core_v1")
    p.add_argument(
        "--adj", required=True, type=str, help="Path to train adjacency CSR (train_adj_T0.npz)"
    )
    p.add_argument(
        "--candidates-val", required=True, type=str, help="Validation candidate pool parquet"
    )
    p.add_argument(
        "--splits-root",
        required=True,
        type=str,
        help="Root containing splits/{train,val,test}_edges.parquet",
    )
    p.add_argument("--artifacts-dir", type=str, default="artifacts/node2vec/node2vec_hpo_v1")
    p.add_argument("--log-dir", type=str, default="logs/node2vec")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument(
        "--sweep-epochs",
        type=int,
        default=10,
        help="Epochs per trial during HPO (lighter than finalize runs)",
    )
    p.add_argument(
        "--max-sources",
        type=int,
        default=4000,
        help="Cap validation sources per trial to keep evaluation tractable",
    )
    p.add_argument(
        "--select-metric",
        type=str,
        default="ndcg@100",
        choices=["ndcg@100", "mrr", "map", "hit@10"],
        help="Validation metric to maximise",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Optuna parallel jobs (keep small unless multiple GPUs are available)",
    )
    p.add_argument(
        "--timeout", type=int, default=0, help="Optional global timeout in seconds (0 disables)"
    )
    return p.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cuda" or (arg == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def objective_factory(
    g,
    device: torch.device,
    args: argparse.Namespace,
    cand_val: Path,
    splits_root: Path,
    Ks: list[int],
    logger: logging.Logger,
):
    def objective(trial: optuna.Trial) -> float:
        trial_seed = args.seed + trial.number
        set_seed(trial_seed)

        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 96, 128, 192, 256])
        walk_length = trial.suggest_int("walk_length", 10, 40, step=5)
        walks_per_node = trial.suggest_int("walks_per_node", 5, 25, step=5)
        p = trial.suggest_float("p", 0.25, 4.0, log=True)
        q = trial.suggest_float("q", 0.25, 4.0, log=True)
        lr = trial.suggest_float("lr", 5e-4, 5e-2, log=True)

        cfg: dict[str, float] = {
            "embedding_dim": embedding_dim,
            "walk_length": walk_length,
            "walks_per_node": walks_per_node,
            "p": p,
            "q": q,
            "lr": lr,
            "epochs": int(args.sweep_epochs),
        }

        start_train = time.time()
        model, tr_stats = train_node2vec(
            g=g,
            embedding_dim=embedding_dim,
            p=p,
            q=q,
            walks_per_node=walks_per_node,
            walk_length=walk_length,
            epochs=int(args.sweep_epochs),
            lr=lr,
            device=device,
        )
        train_time = time.time() - start_train

        Z = get_embeddings(model, device=device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        val_global, _, _, counts = evaluate_split(
            name="val",
            Z=Z,
            device=device,
            g=g,
            Ks=Ks,
            cand_path=cand_val,
            out_dir=Path("/tmp"),  # nosec B108 -- ephemeral HPO trial scratch space; evaluate_split does not emit files by itself
            batch_size=2_000_000,
            splits_root=splits_root,
            collect_slices=False,
            max_sources=int(args.max_sources) if args.max_sources > 0 else None,
            emit_strict_warm=False,
        )
        metric_value = float(val_global.iloc[0][args.select_metric])

        trial.set_user_attr("config", cfg)
        trial.set_user_attr("train_time_sec", float(tr_stats.get("training_time_sec", train_time)))
        trial.set_user_attr("sources_eval", int(counts.get("sources", 0)))

        trial.report(metric_value, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        logger.info(
            "Trial %d metric %.6f (sources=%s) cfg=%s",
            trial.number,
            metric_value,
            counts.get("sources", "-"),
            cfg,
        )
        return metric_value

    return objective


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "node2vec_hpo.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("node2vec.hpo")

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    logger.info("Loading graph from %s", args.adj)
    g = load_graph(Path(args.adj), undirected=False)
    Ks = [1, 10, 50]

    cand_val = Path(args.candidates_val)
    splits_root = Path(args.splits_root)

    storage = f"sqlite:///{artifacts_dir / 'node2vec_optuna.db'}"
    sampler = TPESampler(seed=args.seed)
    pruner = SuccessiveHalvingPruner()
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name="node2vec_hpo",
        load_if_exists=True,
    )

    objective = objective_factory(g, device, args, cand_val, splits_root, Ks, logger)

    timeout = args.timeout if args.timeout > 0 else None
    logger.info("Starting Optuna study: trials=%d timeout=%s", args.trials, timeout)
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=timeout,
        n_jobs=args.n_jobs,
        gc_after_trial=True,
    )

    best = study.best_trial
    logger.info("Best trial %d metric %.6f params=%s", best.number, best.value, best.params)

    best_record = {
        "timestamp": _now_iso(),
        "metric": args.select_metric,
        "score": float(best.value),  # type: ignore[arg-type]
        "params": best.params,
        "user_attrs": best.user_attrs,
        "n_trials": len(study.trials),
    }
    (artifacts_dir / "best_params.json").write_text(json.dumps(best_record, indent=2))

    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials_df.to_csv(artifacts_dir / "optuna_trials.csv", index=False)

    summary = {
        "timestamp": _now_iso(),
        "adjacency": str(Path(args.adj)),
        "candidates_val": str(cand_val),
        "splits_root": str(splits_root),
        "device": str(device),
        "metric": args.select_metric,
        "trials": int(args.trials),
        "sweep_epochs": int(args.sweep_epochs),
        "max_sources": int(args.max_sources),
        "timeout": int(args.timeout),
        "best": best_record,
    }
    (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
