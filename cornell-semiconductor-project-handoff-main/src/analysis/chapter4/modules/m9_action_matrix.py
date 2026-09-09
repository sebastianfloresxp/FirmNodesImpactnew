#!/usr/bin/env python3
"""Module 9: action matrix translating impact-confidence into decision buckets."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 9 action matrix")
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


def confidence_bin(presence_count: int, min_views_high: int) -> str:
    return "high_confidence" if presence_count >= min_views_high else "low_confidence"


def action_bucket(impact_bin: str, conf_bin: str) -> str:
    if impact_bin == "high_impact" and conf_bin == "high_confidence":
        return "act_now"
    if impact_bin == "high_impact" and conf_bin == "low_confidence":
        return "validate_then_act"
    if impact_bin == "lower_impact" and conf_bin == "high_confidence":
        return "monitor_routine"
    return "deprioritize"


def action_text(bucket: str) -> str:
    if bucket == "act_now":
        return "Prioritize mitigation and continuity planning immediately."
    if bucket == "validate_then_act":
        return "Prioritize data validation and traceability, then mitigate."
    if bucket == "monitor_routine":
        return "Track routinely and include in standard resilience reviews."
    return "Keep on watchlist with lower immediate priority."


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    m6_1_dir = out_root / snapshot / "m6_1"
    m7_dir = out_root / snapshot / "m7"
    out_dir = out_root / snapshot / "m9"
    out_dir.mkdir(parents=True, exist_ok=True)

    views = [
        str(v)
        for v in cfg.get("m9", {}).get(
            "views", cfg.get("m7", {}).get("views", ["disclosed", "observed", "full"])
        )
    ]
    ranking_metric = str(
        cfg.get("m9", {}).get(
            "ranking_metric", cfg.get("m7", {}).get("ranking_metric", "h1_reach_loss")
        )
    )
    top_n_universe = int(cfg.get("m9", {}).get("top_n_universe", 500))
    top_n_high_impact = int(cfg.get("m9", {}).get("top_n_high_impact", 100))
    min_views_high_conf = int(cfg.get("m9", {}).get("min_views_high_confidence", 2))

    view_tables: dict[str, pd.DataFrame] = {}
    for view in views:
        path = m6_1_dir / f"node_single_removal_impacts_{view}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        table = pd.read_csv(path)
        if ranking_metric not in table.columns:
            raise ValueError(f"ranking metric {ranking_metric} missing in {path}")
        table["analysis_uid"] = table["analysis_uid"].astype(str)
        table = table.sort_values(
            [ranking_metric, "analysis_uid"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        table["rank_position"] = np.arange(1, len(table) + 1, dtype=np.int32)
        view_tables[view] = table

    top_sets = {
        view: set(df.head(top_n_universe)["analysis_uid"].tolist())
        for view, df in view_tables.items()
    }
    universe = sorted(set.union(*(top_sets[v] for v in views)))

    rank_maps = {
        view: dict(zip(df["analysis_uid"], df["rank_position"], strict=False))
        for view, df in view_tables.items()
    }
    metric_maps = {
        view: dict(zip(df["analysis_uid"], df[ranking_metric].astype(float), strict=False))
        for view, df in view_tables.items()
    }
    name_maps = {
        view: dict(
            zip(
                df["analysis_uid"],
                df.get("name", pd.Series([""] * len(df))).fillna("").astype(str),
                strict=False,
            )
        )
        for view, df in view_tables.items()
    }
    role_maps = {
        view: dict(
            zip(
                df["analysis_uid"],
                df.get("entity_role", pd.Series([""] * len(df))).fillna("").astype(str),
                strict=False,
            )
        )
        for view, df in view_tables.items()
    }

    quadrants_path = m7_dir / "high_impact_confidence_quadrants.csv"
    low_conf_reason_map: dict[str, str] = {}
    if quadrants_path.exists():
        q = pd.read_csv(quadrants_path)
        if "analysis_uid" in q.columns and "low_conf_reason" in q.columns:
            low_conf_reason_map = dict(
                zip(
                    q["analysis_uid"].astype(str),
                    q["low_conf_reason"].fillna("").astype(str),
                    strict=False,
                )
            )

    rows: list[dict[str, Any]] = []
    for uid in universe:
        present_in = [view for view in views if uid in top_sets[view]]
        presence_count = len(present_in)
        min_rank = min(int(rank_maps[view].get(uid, 10**9)) for view in views)
        max_h1 = max(float(metric_maps[view].get(uid, np.nan)) for view in views)
        impact_bin = "high_impact" if min_rank <= top_n_high_impact else "lower_impact"
        conf_bin = confidence_bin(presence_count=presence_count, min_views_high=min_views_high_conf)
        bucket = action_bucket(impact_bin=impact_bin, conf_bin=conf_bin)

        best_name = ""
        best_role = ""
        for view in views:
            if not best_name:
                best_name = name_maps[view].get(uid, "")
            if not best_role:
                best_role = role_maps[view].get(uid, "")

        rec = {
            "analysis_uid": uid,
            "name": best_name if best_name else pd.NA,
            "entity_role": best_role if best_role else pd.NA,
            "presence_count": int(presence_count),
            "present_views": "|".join(present_in),
            "min_rank_across_views": int(min_rank),
            "max_h1_across_views": float(max_h1),
            "impact_bin": impact_bin,
            "confidence_bin": conf_bin,
            "action_bucket": bucket,
            "recommended_action": action_text(bucket),
            "low_conf_reason": low_conf_reason_map.get(uid, ""),
        }
        for view in views:
            rec[f"rank_{view}"] = rank_maps[view].get(uid, pd.NA)
            rec[f"{ranking_metric}_{view}"] = metric_maps[view].get(uid, np.nan)
        rows.append(rec)

    nodes_df = pd.DataFrame(rows).sort_values(
        ["impact_bin", "confidence_bin", "min_rank_across_views", "analysis_uid"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    summary_df = (
        nodes_df.groupby(["impact_bin", "confidence_bin", "action_bucket"], as_index=False)
        .size()
        .rename(columns={"size": "n_nodes"})
        .sort_values(
            ["impact_bin", "confidence_bin", "action_bucket"], ascending=[True, True, True]
        )
    )
    summary_df["share_of_universe"] = summary_df["n_nodes"] / max(len(nodes_df), 1)

    nodes_out = out_dir / "action_matrix_nodes.csv"
    summary_out = out_dir / "action_matrix_summary.csv"
    nodes_df.to_csv(nodes_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    run_metadata = {
        "module": "m9",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "settings": {
            "views": views,
            "ranking_metric": ranking_metric,
            "top_n_universe": top_n_universe,
            "top_n_high_impact": top_n_high_impact,
            "min_views_high_confidence": min_views_high_conf,
        },
        "counts": {
            "n_universe_nodes": len(nodes_df),
            "n_summary_rows": len(summary_df),
        },
    }
    run_metadata_out = out_dir / "run_metadata.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m9",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs": {
            **{
                f"m6_1_node_impacts_{view}": str(
                    m6_1_dir / f"node_single_removal_impacts_{view}.csv"
                )
                for view in views
            },
            **{
                f"m6_1_node_impacts_{view}_sha256": file_sha256(
                    m6_1_dir / f"node_single_removal_impacts_{view}.csv"
                )
                for view in views
            },
            "m7_confidence_quadrants": str(quadrants_path) if quadrants_path.exists() else None,
        },
        "outputs": {
            "action_matrix_nodes": str(nodes_out),
            "action_matrix_summary": str(summary_out),
            "run_metadata": str(run_metadata_out),
        },
    }
    manifest_out = out_dir / "manifest_m9.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {nodes_out}")
    print(f"[done] wrote {summary_out}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")


if __name__ == "__main__":
    main()
