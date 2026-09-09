#!/usr/bin/env python3
"""
00_graphsage_hpo_two_phase.py
-------------------
GraphSAGE two-phase HPO orchestrator with fixed negative strategy (mix_deg50):

Phase A (screen): run Optuna HPO with short sweep epochs to find promising configs.
Phase B (refine): re-evaluate top-M configs with higher epochs (single-seed) on the
full scorecard, then select top-K and invoke the robustness finalize script to run
multi-seed and emit robust_best_params.json.

Artifacts layout under --artifacts-root and --version-tag (e.g., graphsage_mixdeg50_v1):
- artifacts/graphsage/<tag>/phaseA/* (Optuna study + CSV)
- artifacts/graphsage/<tag>/phaseB_t<trial>/* (per-config refine runs)
- artifacts/graphsage/<tag>/finalize/* (robust summaries + robust_best_params.json)

Usage example (core_v1):
python src/graphsage/00_graphsage_hpo_two_phase.py \
  --adj data/processed/core/releases/core_v1/adjacency/train_adj_T0.npz \
  --features data/processed/core/releases/core_v1/features/node_features_T0.parquet \
  --candidates-val data/processed/core/releases/core_v1/candidates/val_candidates.parquet \
  --candidates-test data/processed/core/releases/core_v1/candidates/test_candidates.parquet \
  --splits-root data/processed/core/releases/core_v1/splits \
  --version-tag graphsage_mixdeg50_v1 --trials-a 60 --epochs-a 12 --top-m 12 --epochs-b 40 --top-k 3
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
        description="Two-phase HPO orchestrator for GraphSAGE (rank-aligned mix_v3)"
    )
    # Data
    p.add_argument("--adj", required=True, type=str)
    p.add_argument("--features", required=True, type=str)
    p.add_argument(
        "--struct-feats",
        type=str,
        default="",
        help="Path to node_structural_v1.parquet",
    )
    p.add_argument("--candidates-val", required=True, type=str)
    p.add_argument("--candidates-test", required=True, type=str)
    p.add_argument("--splits-root", required=True, type=str)
    # Layout / versioning
    p.add_argument("--version-tag", type=str, default="graphsage_mix_rank_v3")
    p.add_argument("--artifacts-root", type=str, default="artifacts/graphsage")
    p.add_argument("--results-root", type=str, default="results/graphsage")
    p.add_argument("--logs-dir", type=str, default="logs/graphsage")
    # Device
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    # Structural embeddings settings
    p.add_argument(
        "--use-id-emb",
        action="store_true",
        help="Enable learnable node-ID embeddings in all phases",
    )
    p.add_argument(
        "--id-emb-dim",
        type=int,
        default=64,
        help="Dimensionality of node-ID embeddings",
    )
    p.add_argument(
        "--normalize-emb",
        action="store_true",
        help="L2-normalize embeddings before ranking",
    )
    # Phase A
    p.add_argument("--trials-a", type=int, default=60)
    p.add_argument("--epochs-a", type=int, default=12)
    p.add_argument("--max-sources-a", type=int, default=5000)
    p.add_argument(
        "--n-jobs-a",
        type=int,
        default=1,
        help="Optuna n_jobs for Phase A (parallel trials within process)",
    )
    # Phase B
    p.add_argument("--top-m", type=int, default=10)
    p.add_argument("--epochs-b", type=int, default=60)
    # Finalize (multi-seed)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--seeds", type=str, default="42,123,456,789,999")
    p.add_argument("--max-fanout-nodes", type=int, default=800_000)
    p.add_argument("--auto-clamp", type=str, default="true")
    return p.parse_args()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], log_file: Path | None = None) -> None:
    # Override with os.environ["SUPPLYCHAIN_ROOT"] if auto-detection fails
    project_root = os.environ.get("SUPPLYCHAIN_ROOT", str(Path(__file__).resolve().parents[2]))
    if log_file:
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


def load_topm(csv_path: Path, top_m: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[(df.get("state") == "COMPLETE") & (~df.get("value").isna())].copy()  # type: ignore[union-attr]
    df = df.sort_values("value", ascending=False).head(int(top_m)).reset_index(drop=True)  # type: ignore[call-overload]
    return df


def main() -> None:
    args = parse_args()
    tag = args.version_tag
    art_root = Path(args.artifacts_root)
    res_root = Path(args.results_root)
    logs_dir = Path(args.logs_dir)
    phaseA_art = art_root / tag / "phaseA"
    phaseB_art_root = art_root / tag
    phaseB_res_root = res_root / tag
    _ensure_dir(phaseA_art)
    _ensure_dir(phaseB_art_root)
    _ensure_dir(phaseB_res_root)
    _ensure_dir(logs_dir)

    # Phase A: HPO with fixed mix (v2) and short epochs (kept for compatibility)
    adj_path = str(Path(args.adj).resolve())
    print(f"[DEBUG] Resolved adj path: {adj_path}")
    a_cmd = [
        sys.executable,
        str(Path(__file__).parent / "01_graphsage_hpo_optuna.py"),
        "--adj",
        adj_path,
        "--features",
        str(Path(args.features).resolve()),
        "--struct-feats",
        str(Path(args.struct_feats).resolve()) if args.struct_feats else "",
        "--candidates-val",
        str(Path(args.candidates_val).resolve()),
        "--splits-root",
        str(Path(args.splits_root).resolve()),
        "--artifacts-dir",
        str(phaseA_art),
        "--trials",
        str(int(args.trials_a)),
        "--sweep-epochs",
        str(int(args.epochs_a)),
        "--sweep-max-sources",
        str(int(args.max_sources_a)),
        "--select-metric",
        "ndcg@100",
        "--device",
        args.device,
        "--fixed-neg-strategy",
        "mix_v3",
        "--n-jobs",
        str(int(args.n_jobs_a)),
    ]
    if args.use_id_emb:
        a_cmd += ["--use-id-emb"]
    a_cmd += ["--id-emb-dim", str(int(args.id_emb_dim))]
    if args.normalize_emb:
        a_cmd += ["--normalize-emb"]
    run_cmd(a_cmd, logs_dir / f"{tag}_phaseA_hpo.log")

    # Load top-M configs from Phase A
    trials_csv = phaseA_art / "optuna_trials.csv"
    if not trials_csv.exists():
        raise FileNotFoundError(f"Phase A trials CSV not found: {trials_csv}")
    topm = load_topm(trials_csv, int(args.top_m))

    # Phase B: refine each config with higher epochs (single-seed full scorecard)
    phaseB_records: list[dict[str, object]] = []
    for _, r in topm.iterrows():
        trial = int(r["number"]) if "number" in r else -1
        params = {
            "hidden": int(r.get("params_hidden", 128)),  # type: ignore[arg-type]
            "layers": int(r.get("params_layers", 2)),  # type: ignore[arg-type]
            "dropout": float(r.get("params_dropout", 0.1)),  # type: ignore[arg-type]
            "lr": float(r.get("params_lr", 1e-3)),  # type: ignore[arg-type]
            "batch": int(r.get("params_batch", 2048)),  # type: ignore[arg-type]
            "neighbors": [int(x) for x in str(r.get("params_neighbors", "15,10")).split(",")[:2]],
            "neg_ratio": float(r.get("params_neg_ratio", 1.0)),  # type: ignore[arg-type]
            "neg_strategy": "mix_v3",
            "use_id_emb": bool(args.use_id_emb),
            "id_emb_dim": int(args.id_emb_dim),
            "normalize_emb": bool(args.normalize_emb),
        }
        out_dir = phaseB_res_root / f"phaseB_t{trial}"
        art_dir = phaseB_art_root / f"phaseB_t{trial}"
        _ensure_dir(out_dir)
        _ensure_dir(art_dir)
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "03_graphsage_train_eval.py"),
            "--adj",
            str(Path(args.adj).resolve()),
            "--features",
            str(Path(args.features).resolve()),
            "--struct-feats",
            str(Path(args.struct_feats).resolve()) if args.struct_feats else "",
            "--candidates-val",
            str(Path(args.candidates_val).resolve()),
            "--candidates-test",
            str(Path(args.candidates_test).resolve()),
            "--splits-root",
            str(Path(args.splits_root).resolve()),
            "--out-dir",
            str(out_dir),
            "--artifacts-dir",
            str(art_dir),
            "--device",
            args.device,
            "--hidden",
            str(params["hidden"]),
            "--layers",
            str(params["layers"]),
            "--dropout",
            str(params["dropout"]),
            "--lr",
            str(params["lr"]),
            "--epochs",
            str(int(args.epochs_b)),
            "--batch",
            str(params["batch"]),
            "--neighbors",
            str(params["neighbors"][0]),
            str(params["neighbors"][1]),
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
            "--sem-keys",
            "primary_sic_code,gr_country",
            "--deg-exp",
            "0.75",
            "--emit-strict-warm",
            "true",
            "--calibrate",
            "true",
            "--use-struct-feats",
            "false",
            "--val-probe-interval",
            "2",
            "--val-probe-max-sources",
            "1000",
            "--early-stop-patience",
            "2",
            "--early-stop-min-delta",
            "0.0005",
            "--loss",
            "bpr",
            "--pairwise-negs",
            "2",
            "--batch-size",
            "2000000",
            "--max-fanout-nodes",
            str(int(args.max_fanout_nodes)),
            "--auto-clamp",
            ("true" if str(args.auto_clamp).lower() in {"1", "true", "t", "yes", "y"} else "false"),
        ]
        # Pass booleans as explicit values (run_graphsage_eval expects type=_bool)
        # Force no ID embeddings (defensible baseline)
        cmd += ["--use-id-emb", "false"]
        cmd += ["--id-emb-dim", "64"]
        cmd += ["--normalize-emb", "true"]
        run_cmd(cmd, logs_dir / f"{tag}_phaseB_t{trial}.log")
        # Read value from global_val
        gv = out_dir / "global_val.csv"
        if not gv.exists():
            # alternate naming in run_graphsage_eval writes global_val.csv in out_dir
            gv = out_dir / "global_val.csv"
        gdf = pd.read_csv(gv)
        val = float(gdf.iloc[0]["ndcg@100"]) if "ndcg@100" in gdf.columns else float("nan")
        phaseB_records.append(
            {"trial": trial, "value": val, "params": params, "out_dir": str(out_dir)}
        )

    # Rank Phase B and build explicit configs for finalize top-K
    phaseB_records.sort(key=lambda x: x.get("value", 0.0), reverse=True)  # type: ignore[arg-type]
    topk = phaseB_records[: int(args.top_k)]
    explicit = []
    for rec in topk:
        explicit.append(
            {
                "trial": int(rec["trial"]),  # type: ignore[arg-type]
                "value": float(rec["value"]),  # type: ignore[arg-type]
                "params": rec["params"],
            }
        )
    explicit_path = phaseB_art_root / "phaseB_topk_explicit.json"
    explicit_path.write_text(json.dumps(explicit, indent=2))

    # Finalize robustness on top-K using explicit configs
    fin_cmd = [
        sys.executable,
        str(Path(__file__).parent / "02_graphsage_finalize_hpo.py"),
        "--adj",
        str(Path(args.adj).resolve()),
        "--features",
        str(Path(args.features).resolve()),
        "--candidates-val",
        str(Path(args.candidates_val).resolve()),
        "--candidates-test",
        str(Path(args.candidates_test).resolve()),
        "--splits-root",
        str(Path(args.splits_root).resolve()),
        "--artifacts-dir",
        str(phaseB_art_root),
        "--results-dir",
        str(res_root / tag),
        "--logs-dir",
        str(logs_dir),
        "--explicit-configs",
        str(explicit_path),
        "--epochs",
        str(int(args.epochs_b)),
        "--device",
        args.device,
        "--batch-size",
        "2000000",
        "--max-fanout-nodes",
        str(int(args.max_fanout_nodes)),
        "--auto-clamp",
        str(args.auto_clamp),
        "--seeds",
        args.seeds,
    ]
    if args.struct_feats:
        fin_cmd += ["--struct-feats", str(Path(args.struct_feats).resolve())]
    run_cmd(fin_cmd, logs_dir / f"{tag}_finalize.log")

    print(f"[ORCH] Done. Robust best at {phaseB_art_root / 'robust_best_params.json'}")


if __name__ == "__main__":
    main()
