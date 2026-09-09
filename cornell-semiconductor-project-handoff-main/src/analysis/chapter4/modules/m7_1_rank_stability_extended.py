#!/usr/bin/env python3
"""Module 7.1: extended cross-view stability on matched candidate universes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 7.1 rank-stability extensions")
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


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    corr = spearmanr(a, b).correlation
    return float(corr) if corr is not None else np.nan


def load_impacts(path: Path, ranking_metric: str, tie_breaker_metric: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if ranking_metric not in df.columns:
        raise ValueError(f"ranking metric {ranking_metric} missing from {path}")
    if tie_breaker_metric not in df.columns:
        df[tie_breaker_metric] = 0.0
    if "analysis_uid" not in df.columns:
        raise ValueError(f"analysis_uid missing from {path}")
    out = df.copy()
    out["analysis_uid"] = out["analysis_uid"].astype(str)
    out = out.sort_values(
        [ranking_metric, tie_breaker_metric, "analysis_uid"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["rank_position"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


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
    out_dir = out_root / snapshot / "m7_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    m7_cfg = cfg.get("m7", {})
    cfg_71 = cfg.get("m7_1", {})
    views = [
        str(v) for v in cfg_71.get("views", m7_cfg.get("views", ["disclosed", "observed", "full"]))
    ]
    ranking_metric = str(
        cfg_71.get("ranking_metric", m7_cfg.get("ranking_metric", "h1_reach_loss"))
    )
    tie_breaker_metric = str(
        cfg_71.get("tie_breaker_metric", m7_cfg.get("tie_breaker_metric", "h2_path_growth"))
    )
    top_n_grid = [int(x) for x in cfg_71.get("top_n_grid", [25, 50, 100, 250, 500])]

    view_tables: dict[str, pd.DataFrame] = {}
    rank_maps: dict[str, dict[str, int]] = {}
    name_maps: dict[str, dict[str, str]] = {}
    role_maps: dict[str, dict[str, str]] = {}
    for view in views:
        table = load_impacts(
            path=m6_1_dir / f"node_single_removal_impacts_{view}.csv",
            ranking_metric=ranking_metric,
            tie_breaker_metric=tie_breaker_metric,
        )
        view_tables[view] = table
        rank_maps[view] = dict(zip(table["analysis_uid"], table["rank_position"], strict=False))
        name_maps[view] = dict(
            zip(
                table["analysis_uid"],
                table.get("name", pd.Series([""] * len(table))).fillna("").astype(str),
                strict=False,
            )
        )
        role_maps[view] = dict(
            zip(
                table["analysis_uid"],
                table.get("entity_role", pd.Series([""] * len(table))).fillna("").astype(str),
                strict=False,
            )
        )

    max_n = min(len(x) for x in view_tables.values())
    top_n_grid = sorted({n for n in top_n_grid if n > 0 and n <= max_n})
    if not top_n_grid:
        raise ValueError("no valid top_n after filtering to available rows")

    overlap_rows: list[dict[str, Any]] = []
    pairwise_fixed_rows: list[dict[str, Any]] = []
    for top_n in top_n_grid:
        top_sets = {
            view: set(view_tables[view].head(top_n)["analysis_uid"].tolist()) for view in views
        }
        for va, vb in itertools.combinations(views, 2):
            set_a = top_sets[va]
            set_b = top_sets[vb]
            union = sorted(set_a | set_b)
            inter = sorted(set_a & set_b)

            overlap_rows.append(
                {
                    "top_n": int(top_n),
                    "view_a": va,
                    "view_b": vb,
                    "overlap_count": len(inter),
                    "union_count": len(union),
                    "jaccard_overlap": float(len(inter) / len(union)) if len(union) > 0 else np.nan,
                    "overlap_share_of_top_n": float(len(inter) / top_n),
                }
            )

            rank_a_union = np.array(
                [rank_maps[va].get(uid, len(view_tables[va]) + 1) for uid in union],
                dtype=np.float64,
            )
            rank_b_union = np.array(
                [rank_maps[vb].get(uid, len(view_tables[vb]) + 1) for uid in union],
                dtype=np.float64,
            )
            rank_a_inter = (
                np.array([rank_maps[va][uid] for uid in inter], dtype=np.float64)
                if inter
                else np.array([], dtype=np.float64)
            )
            rank_b_inter = (
                np.array([rank_maps[vb][uid] for uid in inter], dtype=np.float64)
                if inter
                else np.array([], dtype=np.float64)
            )

            pairwise_fixed_rows.append(
                {
                    "top_n": int(top_n),
                    "view_a": va,
                    "view_b": vb,
                    "n_union": len(union),
                    "n_intersection": len(inter),
                    "spearman_union_rank": safe_spearman(rank_a_union, rank_b_union),
                    "spearman_intersection_rank": safe_spearman(rank_a_inter, rank_b_inter),
                }
            )

    full_union = sorted(set.union(*(set(t["analysis_uid"].tolist()) for t in view_tables.values())))
    matched_rows: list[dict[str, Any]] = []
    for uid in full_union:
        row: dict[str, Any] = {"analysis_uid": uid}
        best_name = ""
        best_role = ""
        for view in views:
            rank_val = rank_maps[view].get(uid, np.nan)
            row[f"rank_{view}"] = rank_val
            if not best_name:
                best_name = name_maps[view].get(uid, "")
            if not best_role:
                best_role = role_maps[view].get(uid, "")
        row["name"] = best_name if best_name else pd.NA
        row["entity_role"] = best_role if best_role else pd.NA
        matched_rows.append(row)
    matched_df = pd.DataFrame(matched_rows)

    matrix_rows: list[dict[str, Any]] = []
    for va, vb in itertools.combinations(views, 2):
        arr_a = matched_df[f"rank_{va}"].to_numpy(np.float64)
        arr_b = matched_df[f"rank_{vb}"].to_numpy(np.float64)
        valid = np.isfinite(arr_a) & np.isfinite(arr_b)
        matrix_rows.append(
            {
                "view_a": va,
                "view_b": vb,
                "n_matched": int(valid.sum()),
                "spearman_matched_universe": safe_spearman(arr_a[valid], arr_b[valid]),
            }
        )
    matrix_df = pd.DataFrame(matrix_rows)

    overlap_df = pd.DataFrame(overlap_rows).sort_values(
        ["top_n", "view_a", "view_b"], ascending=[True, True, True]
    )
    pairwise_df = pd.DataFrame(pairwise_fixed_rows).sort_values(
        ["top_n", "view_a", "view_b"], ascending=[True, True, True]
    )

    overlap_out = out_dir / "overlap_curve_topn.csv"
    pairwise_out = out_dir / "pairwise_rank_corr_topn.csv"
    matched_out = out_dir / "matched_universe_ranks.csv"
    matrix_out = out_dir / "matched_universe_spearman_matrix.csv"
    overlap_df.to_csv(overlap_out, index=False)
    pairwise_df.to_csv(pairwise_out, index=False)
    matched_df.to_csv(matched_out, index=False)
    matrix_df.to_csv(matrix_out, index=False)

    run_metadata = {
        "module": "m7_1",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "yaml": yaml.__version__,
        },
        "settings": {
            "views": views,
            "ranking_metric": ranking_metric,
            "tie_breaker_metric": tie_breaker_metric,
            "top_n_grid": top_n_grid,
        },
        "counts": {
            "n_overlap_rows": len(overlap_df),
            "n_pairwise_rows": len(pairwise_df),
            "n_matched_nodes": len(matched_df),
        },
    }
    run_metadata_out = out_dir / "run_metadata.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m7_1",
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
        },
        "outputs": {
            "overlap_curve_topn": str(overlap_out),
            "pairwise_rank_corr_topn": str(pairwise_out),
            "matched_universe_ranks": str(matched_out),
            "matched_universe_spearman_matrix": str(matrix_out),
            "run_metadata": str(run_metadata_out),
        },
    }
    manifest_out = out_dir / "manifest_m7_1.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {overlap_out}")
    print(f"[done] wrote {pairwise_out}")
    print(f"[done] wrote {matched_out}")
    print(f"[done] wrote {matrix_out}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")


if __name__ == "__main__":
    main()
