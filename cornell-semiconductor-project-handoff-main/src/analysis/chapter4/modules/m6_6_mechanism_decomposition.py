#!/usr/bin/env python3
"""Module 6.6: mechanism decomposition via cohort-restricted interdiction."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from m6_1_weighted_disruption import evaluate_metrics
from m6_5_prime_exposure_profiles import PrimeViewContext, build_view_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 6.6 mechanism decomposition")
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


def cohort_mask(strata: pd.DataFrame, cohort: str) -> pd.Series:
    if cohort == "corridor_intermediary":
        if "is_intermediary_corridor" in strata.columns:
            return strata["is_intermediary_corridor"].fillna(False).astype(bool)
        return strata["impact_stratum"].astype(str).eq("corridor_intermediary")
    if cohort == "upstream_of_semi":
        return strata["impact_stratum"].astype(str).isin(["deep_upstream", "other_upstream"])
    if cohort == "prime_adjacent":
        return strata["impact_stratum"].astype(str).eq("near_prime")
    if cohort == "all_removable":
        return pd.Series(np.ones(len(strata), dtype=bool), index=strata.index)
    raise ValueError(f"unsupported cohort: {cohort}")


def evaluate_prefix_sets(
    ctx: PrimeViewContext,
    ordered_candidate_indices: list[int],
    max_k: int,
) -> list[dict[str, Any]]:
    selected = ordered_candidate_indices[:max_k]
    rows: list[dict[str, Any]] = []
    for step in range(1, len(selected) + 1):
        removed = np.array(selected[:step], dtype=np.int32)
        full_metrics = evaluate_metrics(
            context=ctx.graph_context,
            removed_nodes=removed,
            need_h2=True,
            need_h3=True,
        )
        sel_idx = selected[step - 1]
        rows.append(
            {
                "step": int(step),
                "selected_uid": str(ctx.node_uids[sel_idx]),
                "selected_name": str(ctx.node_names[sel_idx]),
                **full_metrics,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    m0_dir = out_root / snapshot / "m0"
    m6_1_dir = out_root / snapshot / "m6_1"
    m6_2_dir = out_root / snapshot / "m6_2"
    out_dir = out_root / snapshot / "m6_6"
    out_dir.mkdir(parents=True, exist_ok=True)

    node_path = m0_dir / "node_table_contract.parquet"
    edge_path = m0_dir / "edge_table_contract.parquet"
    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)

    m6_cfg = cfg.get("m6_1", cfg.get("m6", {}))
    k_values = [
        int(x)
        for x in cfg.get("m6_6", {}).get(
            "interdiction_k", m6_cfg.get("interdiction_k", [5, 10, 25])
        )
    ]
    shortlist_n = int(cfg.get("m6_6", {}).get("shortlist_n", 30))
    cohorts = [
        str(x)
        for x in cfg.get("m6_6", {}).get(
            "cohorts", ["corridor_intermediary", "upstream_of_semi", "prime_adjacent"]
        )
    ]
    objective_list = [
        str(x) for x in cfg.get("m6_6", {}).get("objectives", ["deny_h1", "delay_h2"])
    ]
    h1_key = str(
        cfg.get("m6_6", {}).get(
            "deny_h1_key", m6_cfg.get("h1_primary_key", "log_obligation_any_support")
        )
    )
    views = [
        str(v)
        for v in cfg.get("m6_6", {}).get(
            "views", cfg.get("m6_1", {}).get("views", ["disclosed", "observed", "full"])
        )
    ]

    output_map: dict[str, dict[str, str]] = {}
    runtime_by_view: dict[str, float] = {}

    for view_name in views:
        t0 = dt.datetime.now(dt.timezone.utc)
        ctx: PrimeViewContext = build_view_context(
            cfg=cfg, nodes=nodes, edges=edges, view_name=view_name
        )

        impacts_path = m6_1_dir / f"node_single_removal_impacts_{view_name}.csv"
        strata_path = m6_2_dir / f"node_impacts_stratified_{view_name}.parquet"
        perf_path = m6_1_dir / f"interdiction_performance_{view_name}.csv"
        if not impacts_path.exists():
            raise FileNotFoundError(impacts_path)
        if not strata_path.exists():
            raise FileNotFoundError(strata_path)
        if not perf_path.exists():
            raise FileNotFoundError(perf_path)

        impacts = pd.read_csv(impacts_path)
        strata = pd.read_parquet(strata_path)
        base_perf = pd.read_csv(perf_path)

        imp = impacts[["analysis_uid", "h1_reach_loss"]].copy()
        if f"h1_{h1_key}" in impacts.columns:
            imp[f"h1_{h1_key}"] = impacts[f"h1_{h1_key}"]
        merged = strata.merge(imp, how="left", on="analysis_uid")
        if "h1_reach_loss" not in merged.columns:
            for alt in ["h1_reach_loss_x", "h1_reach_loss_y"]:
                if alt in merged.columns:
                    merged["h1_reach_loss"] = merged[alt]
                    break
        if f"h1_{h1_key}" not in merged.columns:
            for alt in [f"h1_{h1_key}_x", f"h1_{h1_key}_y"]:
                if alt in merged.columns:
                    merged[f"h1_{h1_key}"] = merged[alt]
                    break

        rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for cohort in cohorts:
            mask = cohort_mask(merged, cohort)
            sub = merged.loc[mask].copy()
            sub = sub[sub["analysis_uid"].isin(set(ctx.node_uids[ctx.removable_indices].tolist()))]
            if sub.empty:
                continue

            for objective in objective_list:
                score_col = (
                    "h2_path_growth"
                    if objective == "delay_h2"
                    else (f"h1_{h1_key}" if f"h1_{h1_key}" in sub.columns else "h1_reach_loss")
                )
                sub_ranked = (
                    sub.sort_values([score_col, "analysis_uid"], ascending=[False, True])
                    .head(shortlist_n)
                    .copy()
                )
                candidate_indices = [
                    ctx.uid_to_idx[uid]
                    for uid in sub_ranked["analysis_uid"].astype(str).tolist()
                    if uid in ctx.uid_to_idx
                ]
                if not candidate_indices:
                    continue
                max_k = max(k_values)
                selected_full = evaluate_prefix_sets(
                    ctx=ctx,
                    ordered_candidate_indices=candidate_indices,
                    max_k=max_k,
                )
                if not selected_full:
                    continue
                for k_target in k_values:
                    used_k = min(int(k_target), len(selected_full))
                    selected = selected_full[:used_k]
                    for rec in selected:
                        rows.append(
                            {
                                "view": view_name,
                                "cohort": cohort,
                                "objective": objective
                                if objective == "delay_h2"
                                else f"deny_h1_{h1_key}",
                                "k_target": int(k_target),
                                "selection_score_col": score_col,
                                **rec,
                            }
                        )
                    final = selected[-1]
                    objective_key = objective if objective == "delay_h2" else f"deny_h1_{h1_key}"
                    base_match = base_perf[
                        (base_perf["objective"] == objective_key)
                        & (base_perf["k_target"] == k_target)
                    ]
                    base_h1 = (
                        float(base_match["h1_reach_loss"].iloc[0])
                        if len(base_match) > 0
                        else np.nan
                    )
                    base_h2 = (
                        float(base_match["h2_path_growth"].iloc[0])
                        if len(base_match) > 0
                        else np.nan
                    )
                    summary_rows.append(
                        {
                            "view": view_name,
                            "cohort": cohort,
                            "objective": objective_key,
                            "k_target": int(k_target),
                            "k_used": int(used_k),
                            "h1_reach_loss": float(final.get("h1_reach_loss", np.nan)),
                            "h2_path_growth": float(final.get("h2_path_growth", np.nan)),
                            "h2_disconnect_share": float(final.get("h2_disconnect_share", np.nan)),
                            "h3_share_no_support": float(final.get("h3_share_no_support", np.nan)),
                            "candidate_pool_size": len(candidate_indices),
                            "share_of_unrestricted_h1": (
                                float(final.get("h1_reach_loss", np.nan) / base_h1)
                                if np.isfinite(base_h1) and base_h1 > 0
                                else np.nan
                            ),
                            "share_of_unrestricted_h2": (
                                float(final.get("h2_path_growth", np.nan) / base_h2)
                                if np.isfinite(base_h2) and base_h2 > 0
                                else np.nan
                            ),
                            "unrestricted_h1_reach_loss": base_h1,
                            "unrestricted_h2_path_growth": base_h2,
                        }
                    )

        sets_df = pd.DataFrame(rows)
        summary_df = pd.DataFrame(summary_rows)
        sets_out = out_dir / f"mechanism_interdiction_sets_{view_name}.csv"
        summary_out = out_dir / f"mechanism_interdiction_summary_{view_name}.csv"
        sets_df.to_csv(sets_out, index=False)
        summary_df.to_csv(summary_out, index=False)

        runtime = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
        runtime_by_view[view_name] = float(runtime)
        output_map[view_name] = {
            "mechanism_interdiction_sets": str(sets_out),
            "mechanism_interdiction_summary": str(summary_out),
        }
        print(f"[m6_6] completed view={view_name} runtime_s={runtime:.2f}")

    run_metadata = {
        "module": "m6_6",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "yaml": yaml.__version__,
        },
        "settings": {
            "views": views,
            "cohorts": cohorts,
            "objectives": objective_list,
            "interdiction_k": k_values,
            "shortlist_n": shortlist_n,
            "deny_h1_key": h1_key,
        },
        "runtime_seconds_by_view": runtime_by_view,
    }
    run_metadata_path = out_dir / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m6_6",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs": {
            "node_table_contract": str(node_path),
            "edge_table_contract": str(edge_path),
            "node_sha256": file_sha256(node_path),
            "edge_sha256": file_sha256(edge_path),
        },
        "outputs": output_map,
        "run_metadata": str(run_metadata_path),
    }
    manifest_path = out_dir / "manifest_m6_6.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {manifest_path}")
    print(f"[done] wrote {run_metadata_path}")


if __name__ == "__main__":
    main()
