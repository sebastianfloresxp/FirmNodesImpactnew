#!/usr/bin/env python3
"""Module 4.1: corridor bottleneck refinement to break large tie blocks."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 4.1 corridor refinement")
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


def get_view_mask(edges: pd.DataFrame, include_any: list[str]) -> np.ndarray:
    if not include_any:
        raise ValueError("include_any must not be empty")
    missing = [col for col in include_any if col not in edges.columns]
    if missing:
        raise ValueError(f"missing view flags: {missing}")
    mask = np.zeros(len(edges), dtype=bool)
    for col in include_any:
        mask |= edges[col].astype(bool).to_numpy()
    return mask


def rank_desc(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values.astype(float))
    return series.rank(method="min", ascending=False).astype(np.int32).to_numpy()


def rank_asc(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values.astype(float))
    return series.rank(method="min", ascending=True).astype(np.int32).to_numpy()


def build_view_edges(
    edges: pd.DataFrame,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    needed_cols = ["src_idx", "dst_idx", "is_predicted", "is_disclosed", "is_observed_ship"]
    ve = edges.loc[mask, needed_cols].copy()
    ve = ve[ve["src_idx"] != ve["dst_idx"]]
    ve = (
        ve.groupby(["src_idx", "dst_idx"], as_index=False)[
            ["is_predicted", "is_disclosed", "is_observed_ship"]
        ]
        .max()
        .sort_values(["src_idx", "dst_idx"], ascending=[True, True], kind="mergesort")
    )

    src = ve["src_idx"].to_numpy(np.int32, copy=False)
    dst = ve["dst_idx"].to_numpy(np.int32, copy=False)
    is_pred = ve["is_predicted"].astype(bool).to_numpy()
    is_disc = ve["is_disclosed"].astype(bool).to_numpy()
    is_obs = ve["is_observed_ship"].astype(bool).to_numpy()
    return src, dst, is_pred, is_disc, is_obs


def compute_scc_labels(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = csr_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n_nodes, n_nodes))
    _, labels = connected_components(matrix, directed=True, connection="strong", return_labels=True)
    labels = labels.astype(np.int32, copy=False)
    sizes = np.bincount(labels).astype(np.int32, copy=False)
    return labels, sizes


def compute_boundary_counts(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    scc_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    in_external = np.zeros(n_nodes, dtype=np.int32)
    out_external = np.zeros(n_nodes, dtype=np.int32)
    cross = scc_labels[src] != scc_labels[dst]
    if int(cross.sum()) == 0:
        return in_external, out_external
    src_cross = src[cross]
    dst_cross = dst[cross]
    np.add.at(out_external, src_cross, 1)
    np.add.at(in_external, dst_cross, 1)
    return in_external, out_external


def multi_source_weighted_distance(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    edge_cost: np.ndarray,
    source_indices: np.ndarray,
) -> np.ndarray:
    super_source = n_nodes
    rows = np.concatenate([src, np.full(len(source_indices), super_source, dtype=np.int32)])
    cols = np.concatenate([dst, source_indices.astype(np.int32, copy=False)])
    data = np.concatenate(
        [edge_cost.astype(np.float64, copy=False), np.zeros(len(source_indices), dtype=np.float64)]
    )
    matrix = csr_matrix((data, (rows, cols)), shape=(n_nodes + 1, n_nodes + 1))
    dist = dijkstra(matrix, directed=True, indices=super_source, unweighted=False).astype(
        np.float64, copy=False
    )
    return dist[:n_nodes]


def compute_geodesic_lane_counts(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    edge_cost: np.ndarray,
    dist_from_semi_w: np.ndarray,
    dist_to_prime_w: np.ndarray,
    lane_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    in_lane_count = np.zeros(n_nodes, dtype=np.int32)
    out_lane_count = np.zeros(n_nodes, dtype=np.int32)

    finite_forward = np.isfinite(dist_from_semi_w[src]) & np.isfinite(dist_from_semi_w[dst])
    on_forward_geodesic = finite_forward & np.isclose(
        dist_from_semi_w[src] + edge_cost,
        dist_from_semi_w[dst],
        atol=lane_tol,
        rtol=0.0,
    )
    if int(on_forward_geodesic.sum()) > 0:
        np.add.at(in_lane_count, dst[on_forward_geodesic], 1)

    finite_reverse = np.isfinite(dist_to_prime_w[src]) & np.isfinite(dist_to_prime_w[dst])
    on_reverse_geodesic = finite_reverse & np.isclose(
        edge_cost + dist_to_prime_w[dst],
        dist_to_prime_w[src],
        atol=lane_tol,
        rtol=0.0,
    )
    if int(on_reverse_geodesic.sum()) > 0:
        np.add.at(out_lane_count, src[on_reverse_geodesic], 1)

    return in_lane_count, out_lane_count


def load_m4_base(
    m4_dir: Path,
    view_name: str,
    node_uids: np.ndarray,
) -> pd.DataFrame:
    path = m4_dir / f"corridor_nodes_{view_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"M4.1 requires M4 output missing: {path}")
    cols = [
        "analysis_uid",
        "dist_from_semi",
        "dist_to_prime",
        "is_corridor",
        "is_intermediary_corridor",
        "semi_reach_count",
        "prime_reach_count",
        "corridor_score",
    ]
    m4_df = pd.read_parquet(path, columns=cols)
    merged = pd.DataFrame({"analysis_uid": node_uids}).merge(m4_df, on="analysis_uid", how="left")
    if merged.isna().any().any():
        missing = int(merged["corridor_score"].isna().sum())
        raise ValueError(f"M4 base metrics missing for {missing} nodes in view {view_name}")
    return merged


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "unspecified_snapshot"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2")))
    m0_dir = out_root / snapshot / "m0"
    m4_dir = out_root / snapshot / "m4"
    m4r_dir = out_root / snapshot / "m4_refine"
    m4r_dir.mkdir(parents=True, exist_ok=True)

    node_path = m0_dir / "node_table_contract.parquet"
    edge_path = m0_dir / "edge_table_contract.parquet"
    if not node_path.exists():
        raise FileNotFoundError(node_path)
    if not edge_path.exists():
        raise FileNotFoundError(edge_path)

    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)

    node_uids = nodes["analysis_uid"].astype(str).to_numpy()
    uid_to_idx = {uid: i for i, uid in enumerate(node_uids.tolist())}
    n_nodes = len(node_uids)

    src = edges["src_uid"].astype(str).map(uid_to_idx)
    dst = edges["dst_uid"].astype(str).map(uid_to_idx)
    valid = src.notna() & dst.notna()
    edges = edges.loc[valid].copy()
    edges["src_idx"] = src.loc[valid].astype(np.int32).to_numpy()
    edges["dst_idx"] = dst.loc[valid].astype(np.int32).to_numpy()

    role = (
        nodes["entity_role"].astype(str)
        if "entity_role" in nodes.columns
        else pd.Series("unknown", index=nodes.index)
    )
    is_prime = role.eq("prime_vendor").to_numpy(bool)
    is_dod = role.eq("dod_component").to_numpy(bool)
    is_semi = (
        nodes["is_semi_strict"].fillna(False).to_numpy(bool)
        if "is_semi_strict" in nodes.columns
        else np.zeros(n_nodes, dtype=bool)
    )
    name = nodes["name"] if "name" in nodes.columns else pd.Series([None] * n_nodes)

    semi_indices = np.flatnonzero(is_semi).astype(np.int32, copy=False)
    prime_indices = np.flatnonzero(is_prime).astype(np.int32, copy=False)
    if len(semi_indices) == 0:
        raise ValueError("No strict semiconductor nodes found for M4.1")
    if len(prime_indices) == 0:
        raise ValueError("No prime nodes found for M4.1")

    views = cfg.get("views", {})
    if not isinstance(views, dict) or not views:
        raise ValueError("config.views must be a non-empty mapping")

    m4r_cfg = cfg.get("m4_refine", {})
    pred_only_cost = float(m4r_cfg.get("predicted_only_edge_cost", 3.0))
    if pred_only_cost < 1.0:
        raise ValueError("m4_refine.predicted_only_edge_cost must be >= 1")
    top_n = int(m4r_cfg.get("top_n_export", 100))
    top_n_intermediary = int(m4r_cfg.get("top_n_intermediary_export", 100))
    lane_tol = float(m4r_cfg.get("lane_tolerance", 1e-9))
    primary_view = str(m4r_cfg.get("primary_view", "full"))

    refined_outputs: dict[str, str] = {}
    top_outputs: dict[str, str] = {}
    top_intermediary_outputs: dict[str, str] = {}
    tie_outputs: dict[str, str] = {}
    view_stats: dict[str, Any] = {}

    for view_name, view_spec in views.items():
        t_view = time.perf_counter()
        include_any = view_spec.get("include_any", [])
        mask = get_view_mask(edges, include_any)

        t0 = time.perf_counter()
        src_idx, dst_idx, is_pred, is_disc, is_obs = build_view_edges(edges, mask)
        t_edges = time.perf_counter() - t0

        t0 = time.perf_counter()
        base = load_m4_base(m4_dir=m4_dir, view_name=view_name, node_uids=node_uids)
        t_m4_load = time.perf_counter() - t0

        t0 = time.perf_counter()
        scc_labels, scc_sizes = compute_scc_labels(n_nodes=n_nodes, src=src_idx, dst=dst_idx)
        node_scc_size = scc_sizes[scc_labels]
        t_scc = time.perf_counter() - t0

        t0 = time.perf_counter()
        boundary_in_external, boundary_out_external = compute_boundary_counts(
            n_nodes=n_nodes,
            src=src_idx,
            dst=dst_idx,
            scc_labels=scc_labels,
        )
        boundary_gate_score = boundary_in_external.astype(np.int64) * boundary_out_external.astype(
            np.int64
        )
        t_boundary = time.perf_counter() - t0

        pred_only = is_pred & (~is_disc) & (~is_obs)
        edge_cost = np.where(pred_only, pred_only_cost, 1.0).astype(np.float64, copy=False)

        t0 = time.perf_counter()
        dist_from_semi_w = multi_source_weighted_distance(
            n_nodes=n_nodes,
            src=src_idx,
            dst=dst_idx,
            edge_cost=edge_cost,
            source_indices=semi_indices,
        )
        dist_to_prime_w = multi_source_weighted_distance(
            n_nodes=n_nodes,
            src=dst_idx,
            dst=src_idx,
            edge_cost=edge_cost,
            source_indices=prime_indices,
        )
        t_weighted_dist = time.perf_counter() - t0

        t0 = time.perf_counter()
        in_lane_count, out_lane_count = compute_geodesic_lane_counts(
            n_nodes=n_nodes,
            src=src_idx,
            dst=dst_idx,
            edge_cost=edge_cost,
            dist_from_semi_w=dist_from_semi_w,
            dist_to_prime_w=dist_to_prime_w,
            lane_tol=lane_tol,
        )
        t_lanes = time.perf_counter() - t0

        is_corridor = base["is_corridor"].to_numpy(bool)
        is_intermediary = base["is_intermediary_corridor"].to_numpy(bool)
        corridor_score = base["corridor_score"].astype(np.float64).to_numpy()

        finite_pair = np.isfinite(dist_from_semi_w) & np.isfinite(dist_to_prime_w)
        weighted_dist_sum = np.full(n_nodes, np.inf, dtype=np.float64)
        weighted_dist_sum[finite_pair] = (
            dist_from_semi_w[finite_pair] + dist_to_prime_w[finite_pair]
        )
        geodesic_proximity = np.zeros(n_nodes, dtype=np.float64)
        geodesic_proximity[finite_pair] = 1.0 / (1.0 + weighted_dist_sum[finite_pair])

        coverage_scc_norm = np.zeros(n_nodes, dtype=np.float64)
        valid_norm = node_scc_size > 0
        coverage_scc_norm[valid_norm] = corridor_score[valid_norm] / node_scc_size[
            valid_norm
        ].astype(np.float64)

        geodesic_lane_product = np.maximum(
            in_lane_count.astype(np.int64) * out_lane_count.astype(np.int64),
            1,
        )
        local_narrowness = 1.0 / geodesic_lane_product.astype(np.float64)

        bottleneck_score = (
            coverage_scc_norm
            * (1.0 + np.log1p(boundary_gate_score.astype(np.float64)))
            * local_narrowness
            * geodesic_proximity
        )
        bottleneck_score[~is_corridor] = 0.0

        weighted_dist_for_rank = weighted_dist_sum.copy()
        finite_sum = np.isfinite(weighted_dist_for_rank)
        fallback = (
            weighted_dist_for_rank[finite_sum].max() + 1.0 if int(finite_sum.sum()) > 0 else 1e9
        )
        weighted_dist_for_rank[~finite_sum] = fallback

        refined_df = pd.DataFrame(
            {
                "view": view_name,
                "analysis_uid": node_uids,
                "name": name.to_numpy(),
                "entity_role": role.to_numpy(),
                "is_dod_component": is_dod,
                "is_tier1_prime": is_prime,
                "is_semi_strict": is_semi,
                "dist_from_semi": base["dist_from_semi"].astype(np.int32).to_numpy(),
                "dist_to_prime": base["dist_to_prime"].astype(np.int32).to_numpy(),
                "dist_from_semi_weighted": dist_from_semi_w,
                "dist_to_prime_weighted": dist_to_prime_w,
                "weighted_dist_sum": weighted_dist_sum,
                "is_corridor": is_corridor,
                "is_intermediary_corridor": is_intermediary,
                "semi_reach_count": base["semi_reach_count"].astype(np.int32).to_numpy(),
                "prime_reach_count": base["prime_reach_count"].astype(np.int32).to_numpy(),
                "corridor_score": corridor_score.astype(np.int64),
                "scc_id": scc_labels.astype(np.int32),
                "scc_size": node_scc_size.astype(np.int32),
                "coverage_scc_norm": coverage_scc_norm,
                "boundary_in_external": boundary_in_external.astype(np.int32),
                "boundary_out_external": boundary_out_external.astype(np.int32),
                "boundary_gate_score": boundary_gate_score.astype(np.int64),
                "in_lane_count": in_lane_count.astype(np.int32),
                "out_lane_count": out_lane_count.astype(np.int32),
                "geodesic_lane_product": geodesic_lane_product.astype(np.int64),
                "local_narrowness": local_narrowness,
                "geodesic_proximity": geodesic_proximity,
                "bottleneck_score_refined": bottleneck_score,
                "rank_corridor_score": rank_desc(corridor_score.astype(np.float64)),
                "rank_coverage_scc_norm": rank_desc(coverage_scc_norm),
                "rank_boundary_gate_score": rank_desc(boundary_gate_score.astype(np.float64)),
                "rank_local_narrowness": rank_desc(local_narrowness),
                "rank_weighted_dist_sum": rank_asc(weighted_dist_for_rank),
                "rank_bottleneck_score_refined": rank_desc(bottleneck_score),
            }
        )

        top_all = (
            refined_df.loc[refined_df["is_corridor"]]
            .sort_values(
                [
                    "bottleneck_score_refined",
                    "boundary_gate_score",
                    "local_narrowness",
                    "analysis_uid",
                ],
                ascending=[False, False, False, True],
            )
            .head(top_n)
        )
        top_intermediary = (
            refined_df.loc[refined_df["is_intermediary_corridor"]]
            .sort_values(
                [
                    "bottleneck_score_refined",
                    "boundary_gate_score",
                    "local_narrowness",
                    "analysis_uid",
                ],
                ascending=[False, False, False, True],
            )
            .head(top_n_intermediary)
        )

        corridor_sub = refined_df.loc[refined_df["is_corridor"]]
        raw_top_score = (
            float(corridor_sub["corridor_score"].max()) if not corridor_sub.empty else 0.0
        )
        refined_top_score = (
            float(corridor_sub["bottleneck_score_refined"].max()) if not corridor_sub.empty else 0.0
        )
        tie_df = pd.DataFrame(
            [
                {
                    "view": view_name,
                    "corridor_nodes": int(corridor_sub.shape[0]),
                    "corridor_raw_score_unique": int(corridor_sub["corridor_score"].nunique())
                    if not corridor_sub.empty
                    else 0,
                    "corridor_refined_score_unique": int(
                        corridor_sub["bottleneck_score_refined"].nunique()
                    )
                    if not corridor_sub.empty
                    else 0,
                    "raw_top_score_nodes": int(
                        (corridor_sub["corridor_score"] == raw_top_score).sum()
                    )
                    if not corridor_sub.empty
                    else 0,
                    "refined_top_score_nodes": int(
                        (corridor_sub["bottleneck_score_refined"] == refined_top_score).sum()
                    )
                    if not corridor_sub.empty
                    else 0,
                }
            ]
        )

        refined_out = m4r_dir / f"corridor_nodes_refined_{view_name}.parquet"
        top_out = m4r_dir / f"top_corridor_nodes_refined_{view_name}.csv"
        top_intermediary_out = m4r_dir / f"top_corridor_nodes_refined_intermediary_{view_name}.csv"
        tie_out = m4r_dir / f"tie_diagnostics_{view_name}.csv"
        refined_df.to_parquet(refined_out, index=False)
        top_all.to_csv(top_out, index=False)
        top_intermediary.to_csv(top_intermediary_out, index=False)
        tie_df.to_csv(tie_out, index=False)

        refined_outputs[view_name] = str(refined_out)
        top_outputs[view_name] = str(top_out)
        top_intermediary_outputs[view_name] = str(top_intermediary_out)
        tie_outputs[view_name] = str(tie_out)

        view_stats[view_name] = {
            "edges_view": len(src_idx),
            "predicted_only_edges_view": int(pred_only.sum()),
            "corridor_nodes_count": int(is_corridor.sum()),
            "intermediary_corridor_count": int(is_intermediary.sum()),
            "scc_count": len(scc_sizes),
            "largest_scc_size": int(scc_sizes.max()) if len(scc_sizes) else 0,
            "corridor_raw_score_unique": int(corridor_sub["corridor_score"].nunique())
            if not corridor_sub.empty
            else 0,
            "corridor_refined_score_unique": int(corridor_sub["bottleneck_score_refined"].nunique())
            if not corridor_sub.empty
            else 0,
            "runtime_seconds": {
                "build_view_edges": t_edges,
                "load_m4": t_m4_load,
                "scc": t_scc,
                "boundary_counts": t_boundary,
                "weighted_distances": t_weighted_dist,
                "geodesic_lanes": t_lanes,
                "view_total": time.perf_counter() - t_view,
            },
        }

    run_metadata = {
        "module": "m4_refine",
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
            "predicted_only_edge_cost": pred_only_cost,
            "top_n_export": top_n,
            "top_n_intermediary_export": top_n_intermediary,
            "lane_tolerance": lane_tol,
            "primary_view": primary_view,
        },
        "view_stats": view_stats,
    }
    run_metadata_out = m4r_dir / "run_metadata.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m4_refine",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs": {
            "node_table_contract": str(node_path),
            "edge_table_contract": str(edge_path),
            "node_sha256": file_sha256(node_path),
            "edge_sha256": file_sha256(edge_path),
            "m4_corridor_disclosed": str(m4_dir / "corridor_nodes_disclosed.parquet"),
            "m4_corridor_observed": str(m4_dir / "corridor_nodes_observed.parquet"),
            "m4_corridor_full": str(m4_dir / "corridor_nodes_full.parquet"),
        },
        "params": {
            "views": views,
            "predicted_only_edge_cost": pred_only_cost,
            "top_n_export": top_n,
            "top_n_intermediary_export": top_n_intermediary,
            "lane_tolerance": lane_tol,
            "primary_view": primary_view,
        },
        "outputs": {
            "corridor_nodes_refined": refined_outputs,
            "top_corridor_nodes_refined": top_outputs,
            "top_corridor_nodes_refined_intermediary": top_intermediary_outputs,
            "tie_diagnostics": tie_outputs,
            "run_metadata": str(run_metadata_out),
        },
    }
    manifest_out = m4r_dir / "manifest_m4_refine.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))

    for _view_name, out in refined_outputs.items():
        print(f"[done] wrote {out}")
    for _view_name, out in top_outputs.items():
        print(f"[done] wrote {out}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")


if __name__ == "__main__":
    main()
