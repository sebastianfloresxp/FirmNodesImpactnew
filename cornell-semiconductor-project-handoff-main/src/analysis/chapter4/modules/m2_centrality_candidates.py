#!/usr/bin/env python3
"""Module 2: centrality candidate lists and seam diagnostics per view."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import yaml

THIS_FILE = Path(__file__).resolve()
CH4_DIR = THIS_FILE.parents[1]
if str(CH4_DIR) not in sys.path:
    sys.path.append(str(CH4_DIR))

from ch4_common import eigenvector_centrality_power


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chapter 4 Module 2 centrality candidate lists")
    p.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2.template.yaml",
        help="Config YAML path",
    )
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must parse to mapping")
    return cfg


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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
    missing = [c for c in include_any if c not in edges.columns]
    if missing:
        raise ValueError(f"missing view flags: {missing}")
    mask = np.zeros(len(edges), dtype=bool)
    for c in include_any:
        mask |= edges[c].astype(bool).to_numpy()
    return mask


def rank_desc(values: np.ndarray) -> np.ndarray:
    ser = pd.Series(values.astype(float))
    return ser.rank(method="min", ascending=False).astype(np.int32).to_numpy()


def articulation_split_stats(graph: nx.Graph, node: int) -> dict[str, float]:
    neighbors = list(graph.neighbors(node))
    if len(neighbors) <= 1:
        return {
            "components_after_removal": 0.0,
            "largest_component": 0.0,
            "second_component": 0.0,
            "split_imbalance": 0.0,
        }

    seen: set[int] = {node}
    comp_sizes: list[int] = []
    for start in neighbors:
        if start in seen:
            continue
        q: deque[int] = deque([start])
        seen.add(start)
        size = 0
        while q:
            u = q.popleft()
            size += 1
            for v in graph.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comp_sizes.append(size)

    if not comp_sizes:
        return {
            "components_after_removal": 0.0,
            "largest_component": 0.0,
            "second_component": 0.0,
            "split_imbalance": 0.0,
        }
    comp_sizes.sort(reverse=True)
    largest = float(comp_sizes[0])
    second = float(comp_sizes[1]) if len(comp_sizes) > 1 else 0.0
    return {
        "components_after_removal": float(len(comp_sizes)),
        "largest_component": largest,
        "second_component": second,
        "split_imbalance": largest - second,
    }


def bridge_split_stats(graph: nx.Graph, edge: tuple[int, int]) -> dict[str, float]:
    u, v = edge
    if not graph.has_edge(u, v):
        return {"component_u": 0.0, "component_v": 0.0, "split_imbalance": 0.0}

    graph.remove_edge(u, v)
    seen: set[int] = {u}
    q: deque[int] = deque([u])
    size_u = 0
    while q:
        x = q.popleft()
        size_u += 1
        for y in graph.neighbors(x):
            if y not in seen:
                seen.add(y)
                q.append(y)
    graph.add_edge(u, v)
    size_v = graph.number_of_nodes() - size_u
    return {
        "component_u": float(size_u),
        "component_v": float(size_v),
        "split_imbalance": float(abs(size_u - size_v)),
    }


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
    m2_dir = out_root / snapshot / "m2"
    m2_dir.mkdir(parents=True, exist_ok=True)

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
    if "is_semi_strict" in nodes.columns:
        is_semi = nodes["is_semi_strict"].fillna(False).to_numpy(bool)
    else:
        is_semi = np.zeros(n_nodes, dtype=bool)

    views = cfg.get("views", {})
    if not isinstance(views, dict) or not views:
        raise ValueError("config.views must be a non-empty mapping")

    approx_cfg = cfg.get("approximation", {})
    bet_k = int(approx_cfg.get("betweenness_sample_k", 1024))
    random_cfg = cfg.get("random", {})
    seed = int(random_cfg.get("global_seed", 7))

    m2_cfg = cfg.get("m2", {})
    top_art = int(m2_cfg.get("seam_split_top_n_articulation", 2000))
    top_br = int(m2_cfg.get("seam_split_top_n_bridges", 2000))

    node_outputs: dict[str, str] = {}
    seam_outputs: dict[str, str] = {}
    seam_parquet_outputs: dict[str, str] = {}
    view_stats: dict[str, Any] = {}

    for view_name, view_spec in views.items():
        t_view = time.perf_counter()
        include_any = view_spec.get("include_any", [])
        mask = get_view_mask(edges, include_any)
        ve = edges.loc[mask, ["src_idx", "dst_idx"]].copy()
        ve = ve[ve["src_idx"] != ve["dst_idx"]]

        s = ve["src_idx"].to_numpy(np.int32, copy=False)
        d = ve["dst_idx"].to_numpy(np.int32, copy=False)

        in_deg = np.bincount(d, minlength=n_nodes).astype(np.int32)
        out_deg = np.bincount(s, minlength=n_nodes).astype(np.int32)

        dg = nx.DiGraph()
        dg.add_nodes_from(range(n_nodes))
        dg.add_edges_from(zip(s.tolist(), d.tolist(), strict=False))

        ug = nx.Graph()
        ug.add_nodes_from(range(n_nodes))
        ug.add_edges_from(zip(s.tolist(), d.tolist(), strict=False))
        ug_deg = np.zeros(n_nodes, dtype=np.int32)
        for node_id, degree in ug.degree():
            ug_deg[int(node_id)] = int(degree)

        # Centrality candidates
        t0 = time.perf_counter()
        pr = nx.pagerank(dg, alpha=0.85, max_iter=200, tol=1e-8)
        t_pagerank = time.perf_counter() - t0
        pr_arr = np.fromiter((pr.get(i, 0.0) for i in range(n_nodes)), dtype=np.float64)

        t0 = time.perf_counter()
        eig_arr = eigenvector_centrality_power(
            n=n_nodes,
            src=s,
            dst=d,
            weight=None,
            undirected=True,
            tol=1e-10,
            max_iter=200,
        )
        t_eig = time.perf_counter() - t0

        k_eff = max(1, min(bet_k, n_nodes))
        t0 = time.perf_counter()
        btw = nx.betweenness_centrality(dg, k=k_eff, seed=seed, normalized=True)
        t_btw = time.perf_counter() - t0
        btw_arr = np.fromiter((btw.get(i, 0.0) for i in range(n_nodes)), dtype=np.float64)

        node_df = pd.DataFrame(
            {
                "analysis_uid": idx_to_uid,
                "entity_role": role,
                "name": nodes["name"] if "name" in nodes.columns else None,
                "is_semi_strict": is_semi,
                "is_tier1_prime": is_prime,
                "deg_in": in_deg,
                "deg_out": out_deg,
                "pagerank": pr_arr,
                "eigenvector_undirected": eig_arr,
                "betweenness_approx": btw_arr,
                "rank_deg_out": rank_desc(out_deg.astype(np.float64)),
                "rank_pagerank": rank_desc(pr_arr),
                "rank_eigenvector_undirected": rank_desc(eig_arr),
                "rank_betweenness_approx": rank_desc(btw_arr),
                "view": view_name,
            }
        )

        node_out = m2_dir / f"node_centralities_{view_name}.parquet"
        node_df.to_parquet(node_out, index=False)
        node_outputs[view_name] = str(node_out)

        # Seam diagnostics
        t0 = time.perf_counter()
        art_nodes = list(nx.articulation_points(ug))
        t_art_enum = time.perf_counter() - t0
        t0 = time.perf_counter()
        bridge_edges = list(nx.bridges(ug))
        t_br_enum = time.perf_counter() - t0

        art_df = pd.DataFrame(
            {
                "view": view_name,
                "seam_type": "articulation",
                "analysis_uid": [idx_to_uid[i] for i in art_nodes],
                "src_uid": None,
                "dst_uid": None,
                "proxy_score": [float(ug_deg[i]) for i in art_nodes],
                "is_touch_prime": [bool(is_prime[i]) for i in art_nodes],
                "split_computed": False,
                "components_after_removal": np.nan,
                "largest_component": np.nan,
                "second_component": np.nan,
                "split_imbalance": np.nan,
                "component_u": np.nan,
                "component_v": np.nan,
            }
        )
        if not art_df.empty:
            art_df = art_df.sort_values(
                ["proxy_score", "analysis_uid"], ascending=[False, True]
            ).reset_index(drop=True)
            k_art = min(top_art, len(art_df))
            t0 = time.perf_counter()
            for i in range(k_art):
                uid = str(art_df.at[i, "analysis_uid"])
                idx = uid_to_idx[uid]
                stats = articulation_split_stats(ug, idx)
                art_df.at[i, "split_computed"] = True
                art_df.at[i, "components_after_removal"] = stats["components_after_removal"]
                art_df.at[i, "largest_component"] = stats["largest_component"]
                art_df.at[i, "second_component"] = stats["second_component"]
                art_df.at[i, "split_imbalance"] = stats["split_imbalance"]
            t_art_split = time.perf_counter() - t0
        else:
            t_art_split = 0.0

        br_df = pd.DataFrame(
            {
                "view": view_name,
                "seam_type": "bridge",
                "analysis_uid": None,
                "src_uid": [idx_to_uid[u] for u, _ in bridge_edges],
                "dst_uid": [idx_to_uid[v] for _, v in bridge_edges],
                "proxy_score": [float(min(ug_deg[u], ug_deg[v])) for u, v in bridge_edges],
                "is_touch_prime": [bool(is_prime[u] or is_prime[v]) for u, v in bridge_edges],
                "split_computed": False,
                "components_after_removal": np.nan,
                "largest_component": np.nan,
                "second_component": np.nan,
                "split_imbalance": np.nan,
                "component_u": np.nan,
                "component_v": np.nan,
            }
        )
        if not br_df.empty:
            br_df = br_df.sort_values(
                ["proxy_score", "src_uid", "dst_uid"], ascending=[False, True, True]
            ).reset_index(drop=True)
            k_br = min(top_br, len(br_df))
            t0 = time.perf_counter()
            for i in range(k_br):
                suid = str(br_df.at[i, "src_uid"])
                duid = str(br_df.at[i, "dst_uid"])
                u = uid_to_idx[suid]
                v = uid_to_idx[duid]
                stats = bridge_split_stats(ug, (u, v))
                br_df.at[i, "split_computed"] = True
                br_df.at[i, "split_imbalance"] = stats["split_imbalance"]
                br_df.at[i, "component_u"] = stats["component_u"]
                br_df.at[i, "component_v"] = stats["component_v"]
            t_br_split = time.perf_counter() - t0
        else:
            t_br_split = 0.0

        seam_df = pd.concat([art_df, br_df], ignore_index=True)
        seam_out = m2_dir / f"seam_nodes_{view_name}.csv"
        seam_parquet_out = m2_dir / f"seam_nodes_{view_name}.parquet"
        seam_df.to_csv(seam_out, index=False)
        seam_df.to_parquet(seam_parquet_out, index=False)
        seam_outputs[view_name] = str(seam_out)
        seam_parquet_outputs[view_name] = str(seam_parquet_out)

        view_stats[view_name] = {
            "edges_view": len(ve),
            "betweenness_k": int(k_eff),
            "articulation_count": len(art_df),
            "bridge_count": len(br_df),
            "articulation_split_computed": int(min(top_art, len(art_df))),
            "bridge_split_computed": int(min(top_br, len(br_df))),
            "runtime_seconds": {
                "pagerank": t_pagerank,
                "eigenvector_undirected": t_eig,
                "betweenness_approx": t_btw,
                "articulation_enumeration": t_art_enum,
                "bridge_enumeration": t_br_enum,
                "articulation_split_eval": t_art_split,
                "bridge_split_eval": t_br_split,
                "view_total": time.perf_counter() - t_view,
            },
        }

    run_metadata = {
        "module": "m2",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "scipy": scipy.__version__,
            "yaml": yaml.__version__,
        },
        "view_stats": view_stats,
    }
    run_metadata_out = m2_dir / "run_metadata.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m2",
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
            "betweenness_sample_k": bet_k,
            "seam_split_top_n_articulation": top_art,
            "seam_split_top_n_bridges": top_br,
            "seed": seed,
        },
        "outputs": {
            "node_centralities": node_outputs,
            "seam_nodes_csv": seam_outputs,
            "seam_nodes_parquet": seam_parquet_outputs,
            "run_metadata": str(run_metadata_out),
        },
    }
    manifest_out = m2_dir / "manifest_m2.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))

    for _view_name, out in node_outputs.items():
        print(f"[done] wrote {out}")
    for _view_name, out in seam_outputs.items():
        print(f"[done] wrote {out}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")


if __name__ == "__main__":
    main()
