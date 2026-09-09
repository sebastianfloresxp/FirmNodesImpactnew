#!/usr/bin/env python3
"""Module 2.1: exact seam severity across all articulation nodes and bridges."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 2.1 seam refinement")
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


def build_undirected_adjacency(
    n_nodes: int,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    keep = src_idx != dst_idx
    s = src_idx[keep]
    d = dst_idx[keep]
    left = np.minimum(s, d)
    right = np.maximum(s, d)
    dedup = pd.DataFrame({"u": left, "v": right}).drop_duplicates(ignore_index=True)
    u = dedup["u"].to_numpy(np.int32, copy=False)
    v = dedup["v"].to_numpy(np.int32, copy=False)

    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    for a, b in zip(u.tolist(), v.tolist(), strict=False):
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency, u, v


def connected_components(adjacency: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = len(adjacency)
    comp_id = np.full(n_nodes, -1, dtype=np.int32)
    comp_sizes: list[int] = []
    next_comp = 0
    for root in range(n_nodes):
        if comp_id[root] != -1:
            continue
        stack = [root]
        comp_id[root] = next_comp
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nbr in adjacency[node]:
                if comp_id[nbr] == -1:
                    comp_id[nbr] = next_comp
                    stack.append(nbr)
        comp_sizes.append(size)
        next_comp += 1
    return comp_id, np.array(comp_sizes, dtype=np.int32)


def exact_seam_stats(
    adjacency: list[list[int]],
    comp_id: np.ndarray,
    comp_sizes: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    n_nodes = len(adjacency)
    disc = np.full(n_nodes, -1, dtype=np.int64)
    low = np.full(n_nodes, -1, dtype=np.int64)
    parent = np.full(n_nodes, -1, dtype=np.int64)
    subtree_size = np.ones(n_nodes, dtype=np.int64)
    child_count = np.zeros(n_nodes, dtype=np.int64)
    art_parts: list[list[int]] = [[] for _ in range(n_nodes)]
    bridges: list[tuple[int, int, int, int]] = []

    time_counter = 0
    for root in range(n_nodes):
        if disc[root] != -1:
            continue
        disc[root] = time_counter
        low[root] = time_counter
        time_counter += 1
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            node, cursor = stack[-1]
            if cursor < len(adjacency[node]):
                nbr = adjacency[node][cursor]
                stack[-1] = (node, cursor + 1)
                if disc[nbr] == -1:
                    parent[nbr] = node
                    child_count[node] += 1
                    disc[nbr] = time_counter
                    low[nbr] = time_counter
                    time_counter += 1
                    stack.append((nbr, 0))
                elif nbr != parent[node]:
                    if disc[nbr] < low[node]:
                        low[node] = disc[nbr]
            else:
                stack.pop()
                par = parent[node]
                if par != -1:
                    subtree_size[par] += subtree_size[node]
                    if low[node] < low[par]:
                        low[par] = low[node]
                    if low[node] > disc[par]:
                        comp_size = int(comp_sizes[comp_id[node]])
                        side_node = int(subtree_size[node])
                        side_other = int(comp_size - subtree_size[node])
                        bridges.append((int(par), int(node), side_node, side_other))
                    if low[node] >= disc[par]:
                        art_parts[par].append(int(subtree_size[node]))

    articulation_rows: list[dict[str, Any]] = []
    for node in range(n_nodes):
        comp_size = int(comp_sizes[comp_id[node]])
        separated_parts = art_parts[node]
        if parent[node] == -1:
            if child_count[node] < 2:
                continue
            part_sizes = list(separated_parts)
        else:
            if not separated_parts:
                continue
            part_sizes = list(separated_parts)
            rest_size = comp_size - 1 - int(sum(part_sizes))
            if rest_size > 0:
                part_sizes.append(int(rest_size))

        part_sizes.sort(reverse=True)
        largest = int(part_sizes[0]) if part_sizes else 0
        second = int(part_sizes[1]) if len(part_sizes) > 1 else 0
        articulation_rows.append(
            {
                "node_idx": int(node),
                "component_size": comp_size,
                "components_after_removal": len(part_sizes),
                "largest_component": largest,
                "second_component": second,
                "split_imbalance": int(largest - second),
                "largest_component_share": float(largest / comp_size) if comp_size > 0 else np.nan,
                "second_component_share": float(second / comp_size) if comp_size > 0 else np.nan,
            }
        )

    bridge_rows: list[dict[str, Any]] = []
    for u, v, side_u, side_v in bridges:
        comp_size = int(side_u + side_v)
        largest = int(max(side_u, side_v))
        second = int(min(side_u, side_v))
        bridge_rows.append(
            {
                "src_idx": int(u),
                "dst_idx": int(v),
                "component_size": comp_size,
                "components_after_removal": 2,
                "largest_component": largest,
                "second_component": second,
                "split_imbalance": int(abs(side_u - side_v)),
                "component_u": int(side_u),
                "component_v": int(side_v),
                "largest_component_share": float(largest / comp_size) if comp_size > 0 else np.nan,
                "second_component_share": float(second / comp_size) if comp_size > 0 else np.nan,
            }
        )

    deg = np.fromiter((len(nbrs) for nbrs in adjacency), dtype=np.int32, count=n_nodes)
    art_df = pd.DataFrame(articulation_rows)
    br_df = pd.DataFrame(bridge_rows)
    return art_df, br_df, deg


def seam_summary_table(seam_df: pd.DataFrame, view_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seam_type in ["articulation", "bridge"]:
        sub = seam_df[seam_df["seam_type"] == seam_type]
        if sub.empty:
            rows.append(
                {
                    "view": view_name,
                    "seam_type": seam_type,
                    "scope": "all",
                    "n_seams": 0,
                    "n_prime_touch": 0,
                    "pct_prime_touch": 0.0,
                    "second_median": np.nan,
                    "second_p95": np.nan,
                    "second_p99": np.nan,
                    "second_max": np.nan,
                    "n_second_ge_2": 0,
                    "n_second_ge_5": 0,
                    "n_second_ge_10": 0,
                }
            )
            continue
        rows.append(
            {
                "view": view_name,
                "seam_type": seam_type,
                "scope": "all",
                "n_seams": len(sub),
                "n_prime_touch": int(sub["is_touch_prime"].sum()),
                "pct_prime_touch": float(100.0 * sub["is_touch_prime"].mean()),
                "second_median": float(sub["second_component"].median()),
                "second_p95": float(sub["second_component"].quantile(0.95)),
                "second_p99": float(sub["second_component"].quantile(0.99)),
                "second_max": float(sub["second_component"].max()),
                "n_second_ge_2": int((sub["second_component"] >= 2).sum()),
                "n_second_ge_5": int((sub["second_component"] >= 5).sum()),
                "n_second_ge_10": int((sub["second_component"] >= 10).sum()),
            }
        )
        touch = sub[sub["is_touch_prime"]]
        rows.append(
            {
                "view": view_name,
                "seam_type": seam_type,
                "scope": "prime_touch",
                "n_seams": len(touch),
                "n_prime_touch": len(touch),
                "pct_prime_touch": 100.0 if len(sub) > 0 else 0.0,
                "second_median": float(touch["second_component"].median())
                if len(touch)
                else np.nan,
                "second_p95": float(touch["second_component"].quantile(0.95))
                if len(touch)
                else np.nan,
                "second_p99": float(touch["second_component"].quantile(0.99))
                if len(touch)
                else np.nan,
                "second_max": float(touch["second_component"].max()) if len(touch) else np.nan,
                "n_second_ge_2": int((touch["second_component"] >= 2).sum()),
                "n_second_ge_5": int((touch["second_component"] >= 5).sum()),
                "n_second_ge_10": int((touch["second_component"] >= 10).sum()),
            }
        )
    return pd.DataFrame(rows)


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
    m2_refine_dir = out_root / snapshot / "m2_refine"
    m2_refine_dir.mkdir(parents=True, exist_ok=True)

    node_path = m0_dir / "node_table_contract.parquet"
    edge_path = m0_dir / "edge_table_contract.parquet"
    if not node_path.exists():
        raise FileNotFoundError(node_path)
    if not edge_path.exists():
        raise FileNotFoundError(edge_path)

    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)

    node_uids = nodes["analysis_uid"].astype(str).tolist()
    uid_to_idx = {uid: i for i, uid in enumerate(node_uids)}
    idx_to_uid = np.array(node_uids, dtype=object)
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
    name = nodes["name"] if "name" in nodes.columns else pd.Series([None] * n_nodes)

    views = cfg.get("views", {})
    if not isinstance(views, dict) or not views:
        raise ValueError("config.views must be a non-empty mapping")

    m2r_cfg = cfg.get("m2_refine", {})
    top_n = int(m2r_cfg.get("top_n_export", 200))

    seam_csv_outputs: dict[str, str] = {}
    seam_parquet_outputs: dict[str, str] = {}
    summary_outputs: dict[str, str] = {}
    top_outputs: dict[str, str] = {}
    top_prime_outputs: dict[str, str] = {}
    view_stats: dict[str, Any] = {}

    for view_name, view_spec in views.items():
        t_view = time.perf_counter()
        include_any = view_spec.get("include_any", [])
        mask = get_view_mask(edges, include_any)
        ve = edges.loc[mask, ["src_idx", "dst_idx"]]
        src_idx = ve["src_idx"].to_numpy(np.int32, copy=False)
        dst_idx = ve["dst_idx"].to_numpy(np.int32, copy=False)

        t0 = time.perf_counter()
        adjacency, u_edges, _v_edges = build_undirected_adjacency(n_nodes, src_idx, dst_idx)
        t_adj = time.perf_counter() - t0

        t0 = time.perf_counter()
        comp_id, comp_sizes = connected_components(adjacency)
        t_comp = time.perf_counter() - t0

        t0 = time.perf_counter()
        art_df, br_df, deg = exact_seam_stats(adjacency, comp_id, comp_sizes)
        t_exact = time.perf_counter() - t0

        if art_df.empty:
            art_out = pd.DataFrame(
                columns=[
                    "view",
                    "seam_type",
                    "analysis_uid",
                    "analysis_name",
                    "analysis_role",
                    "src_uid",
                    "src_name",
                    "src_role",
                    "dst_uid",
                    "dst_name",
                    "dst_role",
                    "proxy_score",
                    "is_touch_prime",
                    "split_computed",
                    "component_size",
                    "components_after_removal",
                    "largest_component",
                    "second_component",
                    "split_imbalance",
                    "largest_component_share",
                    "second_component_share",
                    "component_u",
                    "component_v",
                ]
            )
        else:
            art_out = pd.DataFrame(
                {
                    "view": view_name,
                    "seam_type": "articulation",
                    "analysis_uid": [idx_to_uid[i] for i in art_df["node_idx"].to_numpy(np.int32)],
                    "analysis_name": art_df["node_idx"].map(lambda i: name.iloc[int(i)]).to_numpy(),
                    "analysis_role": art_df["node_idx"].map(lambda i: role.iloc[int(i)]).to_numpy(),
                    "src_uid": None,
                    "src_name": None,
                    "src_role": None,
                    "dst_uid": None,
                    "dst_name": None,
                    "dst_role": None,
                    "proxy_score": art_df["node_idx"]
                    .map(lambda i, _deg=deg: float(_deg[int(i)]))
                    .to_numpy(),
                    "is_touch_prime": art_df["node_idx"]
                    .map(lambda i: bool(is_prime[int(i)]))
                    .to_numpy(),
                    "split_computed": True,
                    "component_size": art_df["component_size"].astype(np.int32).to_numpy(),
                    "components_after_removal": art_df["components_after_removal"]
                    .astype(np.int16)
                    .to_numpy(),
                    "largest_component": art_df["largest_component"].astype(np.int32).to_numpy(),
                    "second_component": art_df["second_component"].astype(np.int32).to_numpy(),
                    "split_imbalance": art_df["split_imbalance"].astype(np.int32).to_numpy(),
                    "largest_component_share": art_df["largest_component_share"]
                    .astype(np.float64)
                    .to_numpy(),
                    "second_component_share": art_df["second_component_share"]
                    .astype(np.float64)
                    .to_numpy(),
                    "component_u": np.nan,
                    "component_v": np.nan,
                }
            )

        if br_df.empty:
            br_out = pd.DataFrame(columns=art_out.columns.tolist())
        else:
            br_out = pd.DataFrame(
                {
                    "view": view_name,
                    "seam_type": "bridge",
                    "analysis_uid": None,
                    "analysis_name": None,
                    "analysis_role": None,
                    "src_uid": [idx_to_uid[i] for i in br_df["src_idx"].to_numpy(np.int32)],
                    "src_name": [name.iloc[int(i)] for i in br_df["src_idx"].to_numpy(np.int32)],
                    "src_role": [role.iloc[int(i)] for i in br_df["src_idx"].to_numpy(np.int32)],
                    "dst_uid": [idx_to_uid[i] for i in br_df["dst_idx"].to_numpy(np.int32)],
                    "dst_name": [name.iloc[int(i)] for i in br_df["dst_idx"].to_numpy(np.int32)],
                    "dst_role": [role.iloc[int(i)] for i in br_df["dst_idx"].to_numpy(np.int32)],
                    "proxy_score": np.minimum(
                        br_df["src_idx"].map(lambda i, _deg=deg: _deg[int(i)]).to_numpy(np.int32),
                        br_df["dst_idx"].map(lambda i, _deg=deg: _deg[int(i)]).to_numpy(np.int32),
                    ).astype(np.float64),
                    "is_touch_prime": (
                        br_df["src_idx"].map(lambda i: is_prime[int(i)]).to_numpy(bool)
                        | br_df["dst_idx"].map(lambda i: is_prime[int(i)]).to_numpy(bool)
                    ),
                    "split_computed": True,
                    "component_size": br_df["component_size"].astype(np.int32).to_numpy(),
                    "components_after_removal": br_df["components_after_removal"]
                    .astype(np.int16)
                    .to_numpy(),
                    "largest_component": br_df["largest_component"].astype(np.int32).to_numpy(),
                    "second_component": br_df["second_component"].astype(np.int32).to_numpy(),
                    "split_imbalance": br_df["split_imbalance"].astype(np.int32).to_numpy(),
                    "largest_component_share": br_df["largest_component_share"]
                    .astype(np.float64)
                    .to_numpy(),
                    "second_component_share": br_df["second_component_share"]
                    .astype(np.float64)
                    .to_numpy(),
                    "component_u": br_df["component_u"].astype(np.int32).to_numpy(),
                    "component_v": br_df["component_v"].astype(np.int32).to_numpy(),
                }
            )

        seam_df = pd.concat([art_out, br_out], ignore_index=True)
        seam_df = seam_df.sort_values(
            ["second_component", "component_size", "proxy_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        seam_summary = seam_summary_table(seam_df, view_name)
        top_all = seam_df.head(top_n).copy()
        top_prime = seam_df[seam_df["is_touch_prime"]].head(top_n).copy()

        seam_csv = m2_refine_dir / f"seam_nodes_refined_{view_name}.csv"
        seam_parquet = m2_refine_dir / f"seam_nodes_refined_{view_name}.parquet"
        summary_csv = m2_refine_dir / f"seam_summary_{view_name}.csv"
        top_csv = m2_refine_dir / f"top_seams_by_second_component_{view_name}.csv"
        top_prime_csv = m2_refine_dir / f"top_prime_touch_seams_{view_name}.csv"

        seam_df.to_csv(seam_csv, index=False)
        seam_df.to_parquet(seam_parquet, index=False)
        seam_summary.to_csv(summary_csv, index=False)
        top_all.to_csv(top_csv, index=False)
        top_prime.to_csv(top_prime_csv, index=False)

        seam_csv_outputs[view_name] = str(seam_csv)
        seam_parquet_outputs[view_name] = str(seam_parquet)
        summary_outputs[view_name] = str(summary_csv)
        top_outputs[view_name] = str(top_csv)
        top_prime_outputs[view_name] = str(top_prime_csv)
        view_stats[view_name] = {
            "edges_view_directed": len(ve),
            "edges_view_undirected_dedup": len(u_edges),
            "components_total": len(comp_sizes),
            "largest_component_size": int(comp_sizes.max()) if len(comp_sizes) else 0,
            "articulation_count": len(art_out),
            "bridge_count": len(br_out),
            "runtime_seconds": {
                "build_adjacency": t_adj,
                "components": t_comp,
                "exact_seams": t_exact,
                "view_total": time.perf_counter() - t_view,
            },
        }

    run_metadata = {
        "module": "m2_refine",
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
        "view_stats": view_stats,
    }
    run_metadata_out = m2_refine_dir / "run_metadata.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m2_refine",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs": {
            "node_table_contract": str(node_path),
            "edge_table_contract": str(edge_path),
            "node_sha256": file_sha256(node_path),
            "edge_sha256": file_sha256(edge_path),
        },
        "params": {
            "views": views,
            "top_n_export": top_n,
            "exact_seam_scope": "all articulation + all bridges",
        },
        "outputs": {
            "seam_nodes_csv": seam_csv_outputs,
            "seam_nodes_parquet": seam_parquet_outputs,
            "seam_summary_csv": summary_outputs,
            "top_seams_csv": top_outputs,
            "top_prime_touch_seams_csv": top_prime_outputs,
            "run_metadata": str(run_metadata_out),
        },
    }
    manifest_out = m2_refine_dir / "manifest_m2_refine.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))

    for _view_name, out in seam_csv_outputs.items():
        print(f"[done] wrote {out}")
    for _view_name, out in summary_outputs.items():
        print(f"[done] wrote {out}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")


if __name__ == "__main__":
    main()
