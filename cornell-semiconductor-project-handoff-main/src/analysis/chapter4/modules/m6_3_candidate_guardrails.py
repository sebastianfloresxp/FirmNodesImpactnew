#!/usr/bin/env python3
"""Module 6.3: candidate-selection guardrails and stability checks."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 6.3 candidate guardrails")
    parser.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2_fix01.yaml",
        help="Config YAML path",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config must parse to mapping")
    return cfg


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(  # nosec B607 -- git is a well-known system executable
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            or None
        )
    except Exception:
        return None


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 0.0
    return float(len(a & b) / len(u))


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    m6_1_dir = out_root / snapshot / "m6_1"
    out_dir = out_root / snapshot / "m6_3"
    out_dir.mkdir(parents=True, exist_ok=True)

    random_cfg = cfg.get("random", {})
    seed = int(random_cfg.get("global_seed", 7))
    rng = random.Random(seed + 6300)  # nosec B311 -- seeded for reproducible sampling, not cryptographic

    views = [
        str(v)
        for v in cfg.get("m6_3", {}).get(
            "views", cfg.get("m6_1", {}).get("views", ["disclosed", "observed", "full"])
        )
    ]
    top_n = int(cfg.get("m6_3", {}).get("top_n", 25))
    pool_sizes = [int(x) for x in cfg.get("m6_3", {}).get("pool_sizes", [3000, 5000, 7000, 10000])]
    random_draws = int(cfg.get("m6_3", {}).get("random_draws", 200))

    input_hashes: dict[str, str] = {}
    outputs: dict[str, dict[str, str]] = {}
    runtime_by_view: dict[str, float] = {}

    for view in views:
        t0 = dt.datetime.now(dt.timezone.utc)
        pool_path = m6_1_dir / f"candidate_pool_{view}.csv"
        h1_path = m6_1_dir / f"node_single_removal_screen_h1_{view}.csv"
        run_meta_path = m6_1_dir / f"run_metadata_{view}.json"
        for p in [pool_path, h1_path, run_meta_path]:
            if not p.exists():
                raise FileNotFoundError(p)
            input_hashes[str(p)] = file_sha256(p)

        pool = pd.read_csv(pool_path)
        h1 = pd.read_csv(h1_path)
        run_meta = json.loads(run_meta_path.read_text())
        deep_eval_n = int(run_meta.get("settings", {}).get("deep_eval_n", 1500))

        merged = pool[["analysis_uid", "candidate_rank"]].merge(
            h1[["analysis_uid", "h1_reach_loss"]], how="left", on="analysis_uid"
        )
        merged["h1_reach_loss"] = pd.to_numeric(merged["h1_reach_loss"], errors="coerce").fillna(
            0.0
        )

        # Guardrail 1: random deep-eval baseline against signal-selected top deep_eval_n.
        signal_top = (
            merged.sort_values(["h1_reach_loss", "analysis_uid"], ascending=[False, True])
            .head(deep_eval_n)
            .copy()
        )
        signal_best = float(signal_top["h1_reach_loss"].max()) if len(signal_top) else np.nan
        signal_p95 = (
            float(signal_top["h1_reach_loss"].quantile(0.95)) if len(signal_top) else np.nan
        )

        random_best: list[float] = []
        random_p95: list[float] = []
        choices = merged["analysis_uid"].tolist()
        for _ in range(random_draws):
            sample_uids = rng.sample(choices, min(deep_eval_n, len(choices)))
            sample = merged[merged["analysis_uid"].isin(set(sample_uids))]
            random_best.append(float(sample["h1_reach_loss"].max()) if len(sample) else 0.0)
            random_p95.append(float(sample["h1_reach_loss"].quantile(0.95)) if len(sample) else 0.0)

        guardrail_random_df = pd.DataFrame(
            {
                "view": view,
                "random_draw_id": np.arange(1, random_draws + 1, dtype=np.int32),
                "random_best_h1": random_best,
                "random_p95_h1": random_p95,
                "signal_best_h1": signal_best,
                "signal_p95_h1": signal_p95,
            }
        )
        guardrail_random_path = out_dir / f"guardrail_random_deepeval_{view}.csv"
        guardrail_random_df.to_csv(guardrail_random_path, index=False)

        summary_row = {
            "view": view,
            "deep_eval_n": deep_eval_n,
            "random_draws": random_draws,
            "signal_best_h1": signal_best,
            "signal_p95_h1": signal_p95,
            "random_best_h1_mean": float(np.mean(random_best)) if random_best else np.nan,
            "random_best_h1_p95": float(np.quantile(random_best, 0.95)) if random_best else np.nan,
            "random_p95_h1_mean": float(np.mean(random_p95)) if random_p95 else np.nan,
            "p_random_best_ge_signal_best": float(np.mean(np.array(random_best) >= signal_best))
            if random_best
            else np.nan,
        }
        guardrail_summary_df = pd.DataFrame([summary_row])
        guardrail_summary_path = out_dir / f"guardrail_random_deepeval_summary_{view}.csv"
        guardrail_summary_df.to_csv(guardrail_summary_path, index=False)

        # Guardrail 2: pool-size stability of top-N impacted nodes.
        max_pool = max(pool_sizes)
        top_sets: dict[int, set[str]] = {}
        for n in sorted(set(pool_sizes)):
            sub = merged[merged["candidate_rank"] <= n].copy()
            top = set(
                sub.sort_values(["h1_reach_loss", "analysis_uid"], ascending=[False, True])
                .head(top_n)["analysis_uid"]
                .astype(str)
                .tolist()
            )
            top_sets[n] = top
        ref = top_sets.get(max_pool, set())

        stability_rows: list[dict[str, Any]] = []
        for n in sorted(top_sets):
            s = top_sets[n]
            stability_rows.append(
                {
                    "view": view,
                    "top_n": top_n,
                    "candidate_pool_n": int(n),
                    "overlap_with_max_pool_topn": len(s & ref),
                    "jaccard_with_max_pool_topn": jaccard(s, ref),
                }
            )
        pool_stability_path = out_dir / f"pool_stability_top{top_n}_{view}.csv"
        pd.DataFrame(stability_rows).to_csv(pool_stability_path, index=False)

        pairwise_rows: list[dict[str, Any]] = []
        sizes = sorted(top_sets)
        for i, a in enumerate(sizes):
            for b in sizes[i + 1 :]:
                pairwise_rows.append(
                    {
                        "view": view,
                        "top_n": top_n,
                        "pool_a": int(a),
                        "pool_b": int(b),
                        "overlap": len(top_sets[a] & top_sets[b]),
                        "jaccard": jaccard(top_sets[a], top_sets[b]),
                    }
                )
        pairwise_path = out_dir / f"pool_stability_pairwise_top{top_n}_{view}.csv"
        pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)

        runtime = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
        runtime_by_view[view] = float(runtime)
        outputs[view] = {
            "guardrail_random_deepeval": str(guardrail_random_path),
            "guardrail_random_deepeval_summary": str(guardrail_summary_path),
            "pool_stability_topn": str(pool_stability_path),
            "pool_stability_pairwise": str(pairwise_path),
        }
        print(f"[m6_3] completed view={view} runtime_s={runtime:.2f}")

    run_metadata = {
        "module": "m6_3",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "platform": platform.platform(),
        "settings": {
            "views": views,
            "top_n": top_n,
            "pool_sizes": pool_sizes,
            "random_draws": random_draws,
            "seed": seed + 6300,
        },
        "runtime_seconds_by_view": runtime_by_view,
    }
    run_metadata_path = out_dir / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m6_3",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs_sha256": input_hashes,
        "outputs": outputs,
        "run_metadata": str(run_metadata_path),
    }
    manifest_path = out_dir / "manifest_m6_3.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {manifest_path}")
    print(f"[done] wrote {run_metadata_path}")


if __name__ == "__main__":
    main()
