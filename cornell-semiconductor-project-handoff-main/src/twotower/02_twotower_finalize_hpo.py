#!/usr/bin/env python3
"""
02_twotower_finalize_hpo.py
---------------------------
Finalize top-K Two-Tower configs across multiple seeds and write robust summary.

This script mirrors the GraphSAGE finalize pattern: it takes an explicit configs
JSON (produced by an orchestrator) and spawns 03_twotower_train_eval.py runs
for each config x seed, collecting ndcg@100 and writing aggregates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finalize Two-Tower HPO on top-K configs with multi-seed evaluation"
    )
    # Data
    p.add_argument("--adj", required=True, type=str)
    p.add_argument("--features", required=True, type=str)
    p.add_argument("--struct-feats", required=True, type=str)
    p.add_argument("--candidates-val", required=True, type=str)
    p.add_argument("--candidates-test", required=True, type=str)
    p.add_argument("--splits-root", required=True, type=str)
    # Paths
    p.add_argument("--artifacts-dir", required=True, type=str)
    p.add_argument("--results-dir", required=True, type=str)
    p.add_argument("--logs-dir", type=str, default="logs/twotower")
    p.add_argument(
        "--explicit-configs", required=True, type=str, help="JSON with list of {trial,value,params}"
    )
    # Runtime
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seeds", type=str, default="42,123,456,789,999")
    p.add_argument(
        "--batch-size", type=int, default=2_000_000, help="Candidate streaming batch size"
    )
    return p.parse_args()


def run_cmd(cmd: list[str], log_file: Path | None = None) -> None:
    if log_file:
        with log_file.open("w") as f:
            f.write("[CMD] " + " ".join(cmd) + "\n")
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
                raise RuntimeError(f"Command failed: {' '.join(cmd)} (exit {proc.returncode})")
    else:
        subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    logs = Path(args.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    art_dir = Path(args.artifacts_dir)
    res_dir = Path(args.results_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]

    topk = json.loads(Path(args.explicit_configs).read_text())
    robust: list[dict[str, object]] = []
    for rec in topk:
        trial = int(rec.get("trial", -1))
        params = rec["params"]
        fin_out = res_dir / f"finalize_t{trial}"
        fin_art = art_dir / f"finalize_t{trial}"
        fin_out.mkdir(parents=True, exist_ok=True)
        fin_art.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            print(f"[FINALIZE] Trial {trial} seed {s}")
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "03_twotower_train_eval.py"),
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
                str(int(args.epochs)),
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
                str(int(params.get("batch_size", 2048))),
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
            run_cmd(cmd, logs / f"twotower_finalize_t{trial}_seed{s}.log")
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
    robust_df.to_csv(art_dir / "robust_selection.csv", index=False)
    best = robust_df.iloc[0].to_dict()
    (art_dir / "robust_best_params.json").write_text(json.dumps(best, indent=2))
    print(f"[FINALIZE] Done. Best params written to {art_dir / 'robust_best_params.json'}")


if __name__ == "__main__":
    main()
