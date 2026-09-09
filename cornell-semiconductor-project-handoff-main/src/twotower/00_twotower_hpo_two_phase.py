#!/usr/bin/env python3
"""
00_twotower_hpo_two_phase.py
----------------------------
Two‑Tower two‑phase orchestrator aligned to the runbook:

- Phase A (screen): small random/grid search with short epochs; record ndcg@100.
- Phase B (refine): re‑run top‑M configs with higher epochs; select top‑K.
- Finalize: multi‑seed runs on top‑K with fixed seeds and save aggregates.

Outputs under --artifacts-root/--results-root and --version-tag.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], log_file: Path | None = None) -> None:
    # Ensure unbuffered Python output for live logs
    cmd2 = cmd[:]
    if len(cmd2) > 0 and Path(cmd2[0]).name in {Path(sys.executable).name}:
        cmd2.insert(1, "-u")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    if log_file:
        with log_file.open("w") as f:
            f.write("[CMD] " + " ".join(cmd2) + "\n")
            f.flush()
            proc = subprocess.Popen(
                cmd2,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Command failed: {' '.join(cmd2)} (exit {proc.returncode})")
    else:
        subprocess.check_call(cmd2, env=env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-phase HPO orchestrator for Two-Tower")
    # Data
    p.add_argument("--adj", required=True, type=str)
    p.add_argument("--features", required=True, type=str)
    p.add_argument("--struct-feats", required=True, type=str)
    p.add_argument("--candidates-val", required=True, type=str)
    p.add_argument("--candidates-test", required=True, type=str)
    p.add_argument("--splits-root", required=True, type=str)
    # Layout / version
    p.add_argument("--version-tag", type=str, default="twotower_inductive_v1")
    p.add_argument("--artifacts-root", type=str, default="artifacts/twotower")
    p.add_argument("--results-root", type=str, default="results/twotower")
    p.add_argument("--logs-dir", type=str, default="logs/twotower")
    # Device
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    # Phase A
    p.add_argument("--trials-a", type=int, default=60)
    p.add_argument("--epochs-a", type=int, default=8)
    p.add_argument("--top-m", type=int, default=10)
    p.add_argument(
        "--parallel-a", type=int, default=1, help="Run up to N Phase A trials concurrently"
    )
    # Phase B
    p.add_argument("--epochs-b", type=int, default=40)
    p.add_argument("--top-k", type=int, default=3)
    # Finalize seeds
    p.add_argument("--seeds", type=str, default="42,123,456,789,999")
    return p.parse_args()


def sample_configs(n: int) -> list[dict[str, object]]:
    dims = [64, 128]
    struct_hidden = ["64,64", "128,64"]
    attr_emb = [16, 32]
    attr_hidden = ["64,64", "128,64"]
    dropout = [0.0, 0.1, 0.2]
    lr = [1e-3, 5e-4]
    configs: list[dict[str, object]] = []
    for _ in range(n):
        configs.append(
            {
                "embed_dim": random.choice(dims),  # nosec B311 -- seeded for reproducible sampling, not cryptographic
                "struct_hidden": random.choice(struct_hidden),  # nosec B311 -- seeded for reproducible sampling, not cryptographic
                "attr_emb_dim": random.choice(attr_emb),  # nosec B311 -- seeded for reproducible sampling, not cryptographic
                "attr_hidden": random.choice(attr_hidden),  # nosec B311 -- seeded for reproducible sampling, not cryptographic
                "dropout": random.choice(dropout),  # nosec B311 -- seeded for reproducible sampling, not cryptographic
                "lr": random.choice(lr),  # nosec B311 -- seeded for reproducible sampling, not cryptographic
                "batch_size": 2048,
                # Phase-A will override to 2 explicitly; keep 4 here for Phase-B defaults
                "pairwise_negs": 4,
            }
        )
    return configs


def main() -> None:
    args = parse_args()
    tag = args.version_tag
    art_root = Path(args.artifacts_root) / tag
    res_root = Path(args.results_root) / tag
    logs_dir = Path(args.logs_dir)
    _ensure_dir(art_root)
    _ensure_dir(res_root)
    _ensure_dir(logs_dir)

    # Phase A
    phaseA_dir = art_root / "phaseA"
    _ensure_dir(phaseA_dir)
    cfgs = sample_configs(int(args.trials_a))
    print(f"[ORCH] Phase A: {len(cfgs)} trials, epochs={int(args.epochs_a)}")
    trials_rows = []
    # Optional parallel execution for Phase A
    parallel = int(getattr(args, "parallel_a", 1)) if hasattr(args, "parallel_a") else 1
    cmds = []
    for i, cfg in enumerate(cfgs, 1):
        out_dir = res_root / f"phaseA_t{i}"
        art_dir = art_root / f"phaseA_t{i}"
        _ensure_dir(out_dir)
        _ensure_dir(art_dir)
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_twotower_eval.py"),
            "--adj",
            args.adj,
            "--features",
            args.features,
            "--struct-feats",
            args.struct_feats,
            "--candidates-val",
            args.candidates_val,
            "--candidates-test",
            args.candidates_test,
            "--splits-root",
            args.splits_root,
            "--out-dir",
            str(out_dir),
            "--artifacts-dir",
            str(art_dir),
            "--device",
            args.device,
            "--epochs",
            str(int(args.epochs_a)),
            "--embed-dim",
            str(int(cfg["embed_dim"])),  # type: ignore[arg-type]
            "--struct-hidden",
            str(cfg["struct_hidden"]),
            "--attr-emb-dim",
            str(int(cfg["attr_emb_dim"])),  # type: ignore[arg-type]
            "--attr-hidden",
            str(cfg["attr_hidden"]),
            "--dropout",
            str(float(cfg["dropout"])),  # type: ignore[arg-type]
            "--lr",
            str(float(cfg["lr"])),  # type: ignore[arg-type]
            "--batch-size",
            str(int(cfg["batch_size"])),  # type: ignore[arg-type]
            # Phase-A: cheaper negatives for faster screening
            "--pairwise-negs",
            "2",
            "--emit-strict-warm",
            "true",
            "--calibrate",
            "false",
            "--group-by-src",
            "true",
            "--async-sampler",
            "true",
            "--use-amp",
            "true",
        ]
        cmds.append((i, cfg, cmd))

    if parallel <= 1:
        for i, cfg, cmd in cmds:
            print(f"[ORCH][A] Trial {i}/{len(cmds)}: cfg={cfg}")
            run_cmd(cmd, logs_dir / f"{tag}_phaseA_t{i}.log")
    else:
        print(f"[ORCH] Phase A parallel: launching up to {parallel} trials concurrently")
        with ProcessPoolExecutor(max_workers=parallel) as ex:
            futs = []
            for i, _cfg, cmd in cmds:
                futs.append(ex.submit(run_cmd, cmd, logs_dir / f"{tag}_phaseA_t{i}.log"))
            # Wait for completion
            for _ in futs:
                pass

    # Collect results after all trials finish
    for i, cfg, _ in cmds:
        out_dir = res_root / f"phaseA_t{i}"
        gv = out_dir / "global_val.csv"
        if not gv.exists():
            print(f"[ORCH][A][WARN] Missing global_val.csv for trial {i}")
            ndcg = float("nan")
        else:
            gdf = pd.read_csv(gv)
            ndcg = float(gdf.iloc[0]["ndcg@100"]) if "ndcg@100" in gdf.columns else float("nan")
        print(f"[ORCH][A] Trial {i} done: ndcg@100={ndcg:.6f}")
        trials_rows.append({"trial": i, "ndcg@100": ndcg, **cfg})
    trials_df = (
        pd.DataFrame(trials_rows).sort_values("ndcg@100", ascending=False).reset_index(drop=True)
    )
    trials_df.to_csv(phaseA_dir / "trials_summary.csv", index=False)
    print(
        f"[ORCH] Phase A complete. Top-M={int(args.top_m)} moving to Phase B (epochs={int(args.epochs_b)})"
    )

    # Phase B
    top_m = int(args.top_m)
    top_df = trials_df.head(top_m).copy()
    phaseB_records: list[dict[str, object]] = []
    for _, r in top_df.iterrows():
        trial = int(r["trial"])
        out_dir = res_root / f"phaseB_t{trial}"
        art_dir = art_root / f"phaseB_t{trial}"
        _ensure_dir(out_dir)
        _ensure_dir(art_dir)
        print(f"[ORCH][B] Refining trial {trial} with epochs={int(args.epochs_b)}")
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_twotower_eval.py"),
            "--adj",
            args.adj,
            "--features",
            args.features,
            "--struct-feats",
            args.struct_feats,
            "--candidates-val",
            args.candidates_val,
            "--candidates-test",
            args.candidates_test,
            "--splits-root",
            args.splits_root,
            "--out-dir",
            str(out_dir),
            "--artifacts-dir",
            str(art_dir),
            "--device",
            args.device,
            "--epochs",
            str(int(args.epochs_b)),
            "--embed-dim",
            str(int(r["embed_dim"])),
            "--struct-hidden",
            str(r["struct_hidden"]),
            "--attr-emb-dim",
            str(int(r["attr_emb_dim"])),
            "--attr-hidden",
            str(r["attr_hidden"]),
            "--dropout",
            str(float(r["dropout"])),
            "--lr",
            str(float(r["lr"])),
            "--batch-size",
            str(int(r["batch_size"])),
            # Phase-B: restore K=4 for quality
            "--pairwise-negs",
            "4",
            "--emit-strict-warm",
            "true",
            "--calibrate",
            "true",
            "--group-by-src",
            "true",
            "--async-sampler",
            "true",
            "--use-amp",
            "true",
        ]
        run_cmd(cmd, logs_dir / f"{tag}_phaseB_t{trial}.log")
        gv = out_dir / "global_val.csv"
        gdf = pd.read_csv(gv)
        ndcg = float(gdf.iloc[0]["ndcg@100"]) if "ndcg@100" in gdf.columns else float("nan")
        print(f"[ORCH][B] Trial {trial} refine done: ndcg@100={ndcg:.6f}")
        phaseB_records.append(
            {
                "trial": trial,
                "ndcg@100": ndcg,
                "params": {
                    "embed_dim": int(r["embed_dim"]),
                    "struct_hidden": str(r["struct_hidden"]),
                    "attr_emb_dim": int(r["attr_emb_dim"]),
                    "attr_hidden": str(r["attr_hidden"]),
                    "dropout": float(r["dropout"]),
                    "lr": float(r["lr"]),
                    "batch_size": int(r["batch_size"]),
                    "pairwise_negs": int(r["pairwise_negs"]),
                },
            }
        )
    phaseB_df = (
        pd.DataFrame(phaseB_records).sort_values("ndcg@100", ascending=False).reset_index(drop=True)
    )
    phaseB_df.to_csv(art_root / "phaseB_summary.csv", index=False)

    # Finalize (multi-seed) on top-K
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    top_k = phaseB_df.head(int(args.top_k))
    finalize_dir = art_root / "finalize"
    _ensure_dir(finalize_dir)
    robust: list[dict[str, object]] = []
    print(f"[ORCH] Finalize: top-K={int(args.top_k)} seeds={args.seeds}")
    for _, r in top_k.iterrows():
        trial = int(r["trial"])
        params = r["params"]
        fin_out = res_root / f"finalize_t{trial}"
        fin_art = art_root / f"finalize_t{trial}"
        _ensure_dir(fin_out)
        _ensure_dir(fin_art)
        for s in seeds:
            print(f"[ORCH][F] Trial {trial} seed {s}")
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "run_twotower_eval.py"),
                "--adj",
                args.adj,
                "--features",
                args.features,
                "--struct-feats",
                args.struct_feats,
                "--candidates-val",
                args.candidates_val,
                "--candidates-test",
                args.candidates_test,
                "--splits-root",
                args.splits_root,
                "--out-dir",
                str(fin_out / f"seed_{s}"),
                "--artifacts-dir",
                str(fin_art / f"seed_{s}"),
                "--device",
                args.device,
                "--epochs",
                str(int(args.epochs_b)),
                "--embed-dim",
                str(int(params["embed_dim"])),
                "--struct-hidden",
                str(params["struct_hidden"]),
                "--attr-emb-dim",
                str(int(params["attr_emb_dim"])),
                "--attr-hidden",
                str(params["attr_hidden"]),
                "--dropout",
                str(float(params["dropout"])),
                "--lr",
                str(float(params["lr"])),
                "--batch-size",
                str(int(params["batch_size"])),
                # Finalize: use K=4
                "--pairwise-negs",
                "4",
                "--emit-strict-warm",
                "true",
                "--calibrate",
                "true",
                "--group-by-src",
                "true",
                "--async-sampler",
                "true",
                "--use-amp",
                "true",
                "--seed",
                str(int(s)),
            ]
            run_cmd(cmd, logs_dir / f"{tag}_finalize_t{trial}_seed{s}.log")
        # Aggregate across seeds
        vals = []
        tests = []
        for s in seeds:
            gv = fin_out / f"seed_{s}" / "global_val.csv"
            tv = fin_out / f"seed_{s}" / "global_test.csv"
            gdf = pd.read_csv(gv)
            tdf = pd.read_csv(tv)
            vals.append(float(gdf.iloc[0]["ndcg@100"]))
            tests.append(float(tdf.iloc[0]["ndcg@100"]))
        robust.append(
            {
                "trial": trial,
                "val_mean_ndcg@100": float(pd.Series(vals).mean()),
                "test_mean_ndcg@100": float(pd.Series(tests).mean()),
                "params": params,
            }
        )
    robust_df = (
        pd.DataFrame(robust)
        .sort_values("val_mean_ndcg@100", ascending=False)
        .reset_index(drop=True)
    )
    robust_df.to_csv(finalize_dir / "robust_selection.csv", index=False)
    # Best params json
    best = robust_df.iloc[0].to_dict()
    (finalize_dir / "robust_best_params.json").write_text(json.dumps(best, indent=2))
    print(f"[ORCH] Done. Best params written to {finalize_dir / 'robust_best_params.json'}")


if __name__ == "__main__":
    main()
