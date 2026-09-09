#!/usr/bin/env python3
"""Module 7: cross-view stability and confidence classification."""

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
from scipy.stats import kendalltau, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chapter 4 Module 7 stability across evidence views"
    )
    parser.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2.template.yaml",
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
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit() -> str | None:
    try:
        out = subprocess.check_output(  # nosec B607 -- git is a well-known system executable
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out or None
    except Exception:
        return None


def load_view_impacts(
    m6_dir: Path,
    view_name: str,
    ranking_metric: str,
    tie_breaker_metric: str,
) -> pd.DataFrame:
    impacts_path = m6_dir / f"node_single_removal_impacts_{view_name}.csv"
    top_path = m6_dir / f"top100_high_impact_{view_name}.csv"
    if impacts_path.exists():
        df = pd.read_csv(impacts_path)
        src_path = impacts_path
    elif top_path.exists():
        df = pd.read_csv(top_path)
        src_path = top_path
    else:
        raise FileNotFoundError(
            f"M7 requires M6 outputs for view={view_name}: missing {impacts_path} and {top_path}"
        )

    if ranking_metric not in df.columns:
        raise ValueError(f"ranking metric {ranking_metric} not in {src_path}")
    if "analysis_uid" not in df.columns:
        raise ValueError(f"analysis_uid missing in {src_path}")
    if tie_breaker_metric not in df.columns:
        df[tie_breaker_metric] = 0.0

    if "name" not in df.columns:
        df["name"] = pd.NA
    if "entity_role" not in df.columns:
        df["entity_role"] = pd.NA

    out = df.copy()
    out["analysis_uid"] = out["analysis_uid"].astype(str)
    out = out.sort_values(
        [ranking_metric, tie_breaker_metric, "analysis_uid"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["rank_position"] = np.arange(1, len(out) + 1, dtype=np.int32)
    out["view"] = view_name
    return out


def safe_rank_correlation(rank_a: np.ndarray, rank_b: np.ndarray) -> tuple[float, float]:
    if len(rank_a) < 2 or len(rank_b) < 2:
        return np.nan, np.nan
    sp = spearmanr(rank_a, rank_b).correlation
    kd = kendalltau(rank_a, rank_b).correlation
    return float(sp) if sp is not None else np.nan, float(kd) if kd is not None else np.nan


def quadrant_label(
    presence_count: int,
    high_conf_min_views: int,
    medium_conf_min_views: int,
) -> str:
    if presence_count >= high_conf_min_views:
        return "high_impact_high_confidence"
    if presence_count >= medium_conf_min_views:
        return "high_impact_medium_confidence"
    return "high_impact_low_confidence"


def low_conf_reason(uid: str, membership: dict[str, set[str]]) -> str:
    in_full = uid in membership.get("full", set())
    in_disclosed = uid in membership.get("disclosed", set())
    in_observed = uid in membership.get("observed", set())

    if in_full and (not in_disclosed) and (not in_observed):
        return "full_only_predicted_sensitive"
    if in_observed and (not in_disclosed) and (not in_full):
        return "observed_only_shipping_sensitive"
    if in_disclosed and (not in_observed) and (not in_full):
        return "disclosed_only"
    if in_disclosed and in_full and (not in_observed):
        return "missing_in_observed"
    if in_observed and in_full and (not in_disclosed):
        return "missing_in_disclosed"
    return "single_view_only_or_mixed"


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "unspecified_snapshot"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2")))

    m7_cfg = cfg.get("m7", {})
    preferred_source = str(m7_cfg.get("source_module", "m6_1"))
    m6_dir_candidates = [preferred_source, "m6_1", "m6"]
    m6_dir = None
    for name in m6_dir_candidates:
        cand = out_root / snapshot / name
        if cand.exists():
            m6_dir = cand
            break
    if m6_dir is None:
        raise FileNotFoundError(
            f"Could not locate M6 impacts directory under {out_root / snapshot} (tried {m6_dir_candidates})"
        )
    m7_dir = out_root / snapshot / "m7"
    m7_dir.mkdir(parents=True, exist_ok=True)

    views = [str(v) for v in m7_cfg.get("views", ["disclosed", "observed", "full"])]
    if len(views) < 2:
        raise ValueError("m7.views must include at least two views")

    top_n_primary = int(m7_cfg.get("top_n_primary", 50))
    top_n_grid = sorted({int(n) for n in m7_cfg.get("top_n_grid", [25, 50, 100]) if int(n) > 0})
    ranking_metric = str(m7_cfg.get("ranking_metric", "h1_reach_loss"))
    tie_breaker_metric = str(m7_cfg.get("tie_breaker_metric", "h2_path_growth"))
    high_conf_min_views = int(m7_cfg.get("high_confidence_min_views", len(views)))
    medium_conf_min_views = int(m7_cfg.get("medium_confidence_min_views", max(2, len(views) - 1)))

    if top_n_primary not in top_n_grid:
        top_n_grid = sorted({*top_n_grid, top_n_primary})

    view_tables: dict[str, pd.DataFrame] = {}
    view_rank_maps: dict[str, dict[str, int]] = {}
    view_metric_maps: dict[str, dict[str, float]] = {}
    view_name_maps: dict[str, dict[str, str]] = {}
    view_role_maps: dict[str, dict[str, str]] = {}

    for view in views:
        table = load_view_impacts(
            m6_dir=m6_dir,
            view_name=view,
            ranking_metric=ranking_metric,
            tie_breaker_metric=tie_breaker_metric,
        )
        view_tables[view] = table
        view_rank_maps[view] = dict(
            zip(table["analysis_uid"], table["rank_position"], strict=False)
        )
        view_metric_maps[view] = dict(
            zip(table["analysis_uid"], table[ranking_metric].astype(float), strict=False)
        )
        view_name_maps[view] = dict(
            zip(table["analysis_uid"], table["name"].fillna("").astype(str), strict=False)
        )
        view_role_maps[view] = dict(
            zip(table["analysis_uid"], table["entity_role"].fillna("").astype(str), strict=False)
        )

    max_n_available = min(len(df) for df in view_tables.values())
    top_n_grid = [n for n in top_n_grid if n <= max_n_available]
    if not top_n_grid:
        raise ValueError("no valid top_n values remain after checking available rows")

    stability_rows: list[dict[str, Any]] = []
    for top_n in top_n_grid:
        top_sets = {
            view: set(view_tables[view].head(top_n)["analysis_uid"].tolist()) for view in views
        }

        for view_a, view_b in itertools.combinations(views, 2):
            a = top_sets[view_a]
            b = top_sets[view_b]
            inter = a & b
            union = a | b
            overlap_count = len(inter)
            union_count = len(union)
            jaccard = float(overlap_count / union_count) if union_count > 0 else np.nan
            overlap_share = float(overlap_count / top_n)

            inter_sorted = sorted(inter)
            rank_a = np.array(
                [view_rank_maps[view_a][uid] for uid in inter_sorted], dtype=np.float64
            )
            rank_b = np.array(
                [view_rank_maps[view_b][uid] for uid in inter_sorted], dtype=np.float64
            )
            spearman_corr, kendall_corr = safe_rank_correlation(rank_a=rank_a, rank_b=rank_b)

            stability_rows.append(
                {
                    "scope": "pairwise",
                    "top_n": int(top_n),
                    "view_a": view_a,
                    "view_b": view_b,
                    "view_combo": f"{view_a}|{view_b}",
                    "overlap_count": overlap_count,
                    "union_count": union_count,
                    "jaccard_overlap": jaccard,
                    "overlap_share_of_top_n": overlap_share,
                    "n_intersection_for_rank_corr": len(inter_sorted),
                    "spearman_rank_corr": spearman_corr,
                    "kendall_rank_corr": kendall_corr,
                }
            )

        all_inter = set.intersection(*(top_sets[v] for v in views))
        all_union = set.union(*(top_sets[v] for v in views))
        all_inter_count = len(all_inter)
        all_union_count = len(all_union)
        all_jaccard = float(all_inter_count / all_union_count) if all_union_count > 0 else np.nan

        pair_spearman = [
            row["spearman_rank_corr"]
            for row in stability_rows
            if row["scope"] == "pairwise"
            and row["top_n"] == top_n
            and np.isfinite(row["spearman_rank_corr"])
        ]
        pair_kendall = [
            row["kendall_rank_corr"]
            for row in stability_rows
            if row["scope"] == "pairwise"
            and row["top_n"] == top_n
            and np.isfinite(row["kendall_rank_corr"])
        ]

        stability_rows.append(
            {
                "scope": "all_views",
                "top_n": int(top_n),
                "view_a": pd.NA,
                "view_b": pd.NA,
                "view_combo": "|".join(views),
                "overlap_count": all_inter_count,
                "union_count": all_union_count,
                "jaccard_overlap": all_jaccard,
                "overlap_share_of_top_n": float(all_inter_count / top_n),
                "n_intersection_for_rank_corr": int(all_inter_count),
                "spearman_rank_corr": float(np.mean(pair_spearman)) if pair_spearman else np.nan,
                "kendall_rank_corr": float(np.mean(pair_kendall)) if pair_kendall else np.nan,
            }
        )

    stability_df = pd.DataFrame(stability_rows)
    stability_df = stability_df.sort_values(
        ["top_n", "scope", "view_combo"], ascending=[True, True, True], kind="mergesort"
    )
    stability_out = m7_dir / "stability_across_views.csv"
    stability_df.to_csv(stability_out, index=False)

    membership = {
        view: set(view_tables[view].head(top_n_primary)["analysis_uid"].tolist()) for view in views
    }
    candidates = set.union(*(membership[v] for v in views))

    quadrant_rows: list[dict[str, Any]] = []
    for uid in sorted(candidates):
        row: dict[str, Any] = {"analysis_uid": uid}
        present_scores: list[float] = []
        present_ranks: list[int] = []
        present_count = 0
        best_name = ""
        best_role = ""

        for view in views:
            in_view = uid in membership[view]
            row[f"in_{view}"] = bool(in_view)
            rank_val = view_rank_maps[view].get(uid, pd.NA)
            metric_val = view_metric_maps[view].get(uid, np.nan)
            row[f"rank_{view}"] = rank_val
            row[f"{ranking_metric}_{view}"] = metric_val

            if in_view:
                present_count += 1
            if np.isfinite(metric_val):
                present_scores.append(float(metric_val))
            if rank_val is not pd.NA:
                present_ranks.append(int(rank_val))

            if not best_name:
                candidate_name = view_name_maps[view].get(uid, "")
                if candidate_name:
                    best_name = candidate_name
            if not best_role:
                candidate_role = view_role_maps[view].get(uid, "")
                if candidate_role:
                    best_role = candidate_role

        qlabel = quadrant_label(
            presence_count=present_count,
            high_conf_min_views=high_conf_min_views,
            medium_conf_min_views=medium_conf_min_views,
        )
        row["presence_count"] = int(present_count)
        row["confidence_class"] = qlabel
        row["low_conf_reason"] = (
            low_conf_reason(uid=uid, membership=membership)
            if qlabel == "high_impact_low_confidence"
            else ""
        )
        row[ranking_metric + "_max_present"] = (
            float(max(present_scores)) if present_scores else np.nan
        )
        row[ranking_metric + "_mean_present"] = (
            float(np.mean(present_scores)) if present_scores else np.nan
        )
        row["best_rank_present"] = int(min(present_ranks)) if present_ranks else pd.NA
        row["name"] = best_name if best_name else pd.NA
        row["entity_role"] = best_role if best_role else pd.NA
        quadrant_rows.append(row)

    quadrants_df = pd.DataFrame(quadrant_rows)
    class_order = {
        "high_impact_high_confidence": 0,
        "high_impact_medium_confidence": 1,
        "high_impact_low_confidence": 2,
    }
    quadrants_df["class_order"] = (
        quadrants_df["confidence_class"].map(class_order).fillna(99).astype(int)
    )
    rank_full_col = "rank_full" if "rank_full" in quadrants_df.columns else "best_rank_present"
    quadrants_df = quadrants_df.sort_values(
        ["class_order", rank_full_col, "best_rank_present", "analysis_uid"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).drop(columns=["class_order"])
    quadrants_out = m7_dir / "high_impact_confidence_quadrants.csv"
    quadrants_df.to_csv(quadrants_out, index=False)

    class_summary = (
        quadrants_df.groupby("confidence_class", as_index=False)
        .size()
        .rename(columns={"size": "n_nodes"})
        .sort_values("n_nodes", ascending=False)
    )
    summary_out = m7_dir / "high_impact_confidence_summary.csv"
    class_summary.to_csv(summary_out, index=False)

    input_files = {
        f"m6_node_impacts_{view}": str(m6_dir / f"node_single_removal_impacts_{view}.csv")
        for view in views
    }
    input_hashes = {key + "_sha256": file_sha256(Path(path)) for key, path in input_files.items()}

    run_metadata = {
        "module": "m7",
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
            "top_n_primary": top_n_primary,
            "top_n_grid": top_n_grid,
            "ranking_metric": ranking_metric,
            "tie_breaker_metric": tie_breaker_metric,
            "high_confidence_min_views": high_conf_min_views,
            "medium_confidence_min_views": medium_conf_min_views,
        },
        "counts": {
            "n_rows_stability": len(stability_df),
            "n_rows_quadrants": len(quadrants_df),
        },
    }
    run_metadata_out = m7_dir / "run_metadata.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m7",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs": {**input_files, **input_hashes},
        "params": run_metadata["settings"],
        "outputs": {
            "stability_across_views": str(stability_out),
            "high_impact_confidence_quadrants": str(quadrants_out),
            "high_impact_confidence_summary": str(summary_out),
            "run_metadata": str(run_metadata_out),
        },
    }
    manifest_out = m7_dir / "manifest_m7.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))

    print(f"[done] wrote {stability_out}")
    print(f"[done] wrote {quadrants_out}")
    print(f"[done] wrote {summary_out}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")


if __name__ == "__main__":
    main()
