#!/usr/bin/env python3
"""Module 6.2: stratify disruption impacts by tier/corridor classes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 6.2 stratified impacts")
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


def stratum_label(row: pd.Series) -> str:
    if bool(row.get("is_intermediary_corridor", False)):
        return "corridor_intermediary"
    if bool(row.get("is_deep_upstream", False)):
        return "deep_upstream"
    tier = row.get("tier_prime_dist", np.nan)
    if pd.notna(tier) and float(tier) >= 0 and float(tier) <= 2:
        return "near_prime"
    if pd.notna(tier):
        return "other_upstream"
    return "unclassified"


def tier_bin(v: float | int | None) -> str:
    if v is None or pd.isna(v):
        return "tier_na"
    x = int(v)
    if x <= 1:
        return "tier_0_1"
    if x <= 3:
        return "tier_2_3"
    if x <= 5:
        return "tier_4_5"
    return "tier_6_plus"


def topn_summary(df: pd.DataFrame, top_n: int, h1_col: str, view: str) -> pd.DataFrame:
    sub = df.sort_values([h1_col, "analysis_uid"], ascending=[False, True]).head(top_n).copy()
    rows: list[dict[str, Any]] = []
    total = len(sub)
    by_stratum = sub.groupby("impact_stratum", dropna=False).size()
    for stratum, count in by_stratum.items():
        rows.append(
            {
                "view": view,
                "scope": f"top_{top_n}",
                "group_type": "impact_stratum",
                "group_value": str(stratum),
                "count": int(count),
                "share": float(count / max(total, 1)),
            }
        )
    by_tier = sub.groupby("tier_bin", dropna=False).size()
    for tbin, count in by_tier.items():
        rows.append(
            {
                "view": view,
                "scope": f"top_{top_n}",
                "group_type": "tier_bin",
                "group_value": str(tbin),
                "count": int(count),
                "share": float(count / max(total, 1)),
            }
        )
    return pd.DataFrame(rows)


def additive_proxy_curve(
    df: pd.DataFrame, h1_col: str, view: str, fracs: list[float], n_removable: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratum in sorted(df["impact_stratum"].dropna().astype(str).unique().tolist()):
        sub = (
            df[df["impact_stratum"] == stratum]
            .sort_values([h1_col, "analysis_uid"], ascending=[False, True])
            .copy()
        )
        if sub.empty:
            continue
        values = sub[h1_col].fillna(0.0).to_numpy(np.float64)
        cumsum = np.cumsum(values)
        for frac in fracs:
            k = max(1, round(frac * n_removable))
            k = min(k, len(values))
            rows.append(
                {
                    "view": view,
                    "impact_stratum": stratum,
                    "frac_removed": float(frac),
                    "k_removed": int(k),
                    "additive_h1_proxy": float(cumsum[k - 1]),
                    "proxy_capped_1": float(min(cumsum[k - 1], 1.0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    m1_dir = out_root / snapshot / "m1"
    m4r_dir = out_root / snapshot / "m4_refine"
    m6_1_dir = out_root / snapshot / "m6_1"
    out_dir = out_root / snapshot / "m6_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    views = [
        str(v)
        for v in cfg.get("m6_2", {}).get(
            "views", cfg.get("m6_1", {}).get("views", ["disclosed", "observed", "full"])
        )
    ]
    top_n_values = [int(x) for x in cfg.get("m6_2", {}).get("top_n_values", [25, 100, 500])]
    fracs = [
        float(x)
        for x in cfg.get("m6_2", {}).get(
            "proxy_fracs",
            cfg.get("m6_1", {}).get("removal_fracs", [0.001, 0.005, 0.01, 0.02, 0.05]),
        )
    ]

    runtime_by_view: dict[str, float] = {}
    output_paths: dict[str, dict[str, str]] = {}
    input_hashes: dict[str, str] = {}

    for view in views:
        t0 = dt.datetime.now(dt.timezone.utc)
        impact_path = m6_1_dir / f"node_single_removal_impacts_{view}.csv"
        baseline_path = m6_1_dir / f"baseline_metrics_{view}.json"
        m1_path = m1_dir / f"node_metrics_baseline_{view}.parquet"
        m4_path = m4r_dir / f"corridor_nodes_refined_{view}.parquet"
        for p in [impact_path, baseline_path, m1_path, m4_path]:
            if not p.exists():
                raise FileNotFoundError(p)
            input_hashes[str(p)] = file_sha256(p)

        impacts = pd.read_csv(impact_path)
        baseline = json.loads(baseline_path.read_text())
        h1_col = str(cfg.get("m6_2", {}).get("h1_column_override", "h1_reach_loss"))
        if h1_col not in impacts.columns:
            h1_col = "h1_reach_loss"

        m1 = pd.read_parquet(m1_path)[
            [
                "analysis_uid",
                "tier_prime_dist",
                "is_deep_upstream",
                "is_corridor_candidate",
                "is_tier1_prime",
            ]
        ]
        m4 = pd.read_parquet(m4_path)[
            ["analysis_uid", "is_intermediary_corridor", "dist_from_semi", "dist_to_prime"]
        ]
        merged = impacts.merge(m1, how="left", on="analysis_uid").merge(
            m4, how="left", on="analysis_uid"
        )
        merged["impact_stratum"] = merged.apply(stratum_label, axis=1)
        merged["tier_bin"] = merged["tier_prime_dist"].apply(tier_bin)

        stratified_out_csv = out_dir / f"node_impacts_stratified_{view}.csv"
        stratified_out_parquet = out_dir / f"node_impacts_stratified_{view}.parquet"
        merged.to_csv(stratified_out_csv, index=False)
        merged.to_parquet(stratified_out_parquet, index=False)

        summary_parts = [topn_summary(merged, n, h1_col=h1_col, view=view) for n in top_n_values]
        summary_df = (
            pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
        )
        summary_out = out_dir / f"impact_strata_summary_{view}.csv"
        summary_df.to_csv(summary_out, index=False)

        n_removable = int(baseline.get("n_removable_firms", len(merged)))
        proxy_df = additive_proxy_curve(
            merged, h1_col=h1_col, view=view, fracs=fracs, n_removable=n_removable
        )
        proxy_out = out_dir / f"impact_strata_proxy_curves_{view}.csv"
        proxy_df.to_csv(proxy_out, index=False)

        runtime = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
        runtime_by_view[view] = float(runtime)
        output_paths[view] = {
            "node_impacts_stratified_csv": str(stratified_out_csv),
            "node_impacts_stratified_parquet": str(stratified_out_parquet),
            "impact_strata_summary": str(summary_out),
            "impact_strata_proxy_curves": str(proxy_out),
        }
        print(f"[m6_2] completed view={view} runtime_s={runtime:.2f}")

    run_metadata = {
        "module": "m6_2",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "platform": platform.platform(),
        "settings": {
            "views": views,
            "top_n_values": top_n_values,
            "proxy_fracs": fracs,
        },
        "note": "proxy curves are additive single-node approximations, not exact set-removal simulation",
        "runtime_seconds_by_view": runtime_by_view,
    }
    run_metadata_path = out_dir / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m6_2",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs_sha256": input_hashes,
        "outputs": output_paths,
        "run_metadata": str(run_metadata_path),
    }
    manifest_path = out_dir / "manifest_m6_2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {manifest_path}")
    print(f"[done] wrote {run_metadata_path}")


if __name__ == "__main__":
    main()
