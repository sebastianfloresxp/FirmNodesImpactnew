#!/usr/bin/env python3
"""Module 1: baseline topology and lightweight node typology per graph view."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chapter 4 Module 1 baseline topology")
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
        raise ValueError("config must be a mapping")
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
        raise ValueError(f"missing edge flag columns: {missing}")
    mask = np.zeros(len(edges), dtype=bool)
    for c in include_any:
        mask |= edges[c].astype(bool).to_numpy()
    return mask


def build_adj_lists(
    n: int, src: np.ndarray, dst: np.ndarray
) -> tuple[list[list[int]], list[list[int]]]:
    fwd = [[] for _ in range(n)]
    rev = [[] for _ in range(n)]
    for s, d in zip(src.tolist(), dst.tolist(), strict=False):
        if s == d:
            continue
        fwd[s].append(d)
        rev[d].append(s)
    return fwd, rev


def multisource_bfs(sources: np.ndarray, adj: list[list[int]]) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, -1, dtype=np.int32)
    q: deque[int] = deque()
    for s in sources.tolist():
        if dist[s] == -1:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        nd = dist[u] + 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = nd
                q.append(v)
    return dist


def deg_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    arrf = arr.astype(np.float64)
    return {
        f"{prefix}_mean": float(arrf.mean()),
        f"{prefix}_median": float(np.median(arrf)),
        f"{prefix}_p90": float(np.quantile(arrf, 0.90)),
        f"{prefix}_p99": float(np.quantile(arrf, 0.99)),
        f"{prefix}_max": float(arrf.max()),
        f"{prefix}_nonzero_share": float((arr > 0).mean()),
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
    m1_dir = out_root / snapshot / "m1"
    m1_dir.mkdir(parents=True, exist_ok=True)

    node_path = m0_dir / "node_table_contract.parquet"
    edge_path = m0_dir / "edge_table_contract.parquet"
    if not node_path.exists():
        raise FileNotFoundError(node_path)
    if not edge_path.exists():
        raise FileNotFoundError(edge_path)

    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)

    if "analysis_uid" not in nodes.columns:
        raise ValueError("node_table_contract must include analysis_uid")
    if "src_uid" not in edges.columns or "dst_uid" not in edges.columns:
        raise ValueError("edge_table_contract must include src_uid,dst_uid")

    node_uids = nodes["analysis_uid"].astype(str).tolist()
    uid_to_idx = {uid: i for i, uid in enumerate(node_uids)}
    n_nodes = len(node_uids)

    idx_src = edges["src_uid"].astype(str).map(uid_to_idx)
    idx_dst = edges["dst_uid"].astype(str).map(uid_to_idx)
    valid = idx_src.notna() & idx_dst.notna()
    edges = edges.loc[valid].copy()
    edges["src_idx"] = idx_src.loc[valid].astype(np.int32).to_numpy()
    edges["dst_idx"] = idx_dst.loc[valid].astype(np.int32).to_numpy()

    role = (
        nodes["entity_role"].astype(str)
        if "entity_role" in nodes.columns
        else pd.Series("unknown", index=nodes.index)
    )
    is_prime = role.eq("prime_vendor").to_numpy()
    is_dod = role.eq("dod_component").to_numpy()
    if "is_semi_strict" in nodes.columns:
        is_semi = nodes["is_semi_strict"].fillna(False).to_numpy(bool)
    else:
        is_semi = np.zeros(n_nodes, dtype=bool)

    prime_idx = np.where(is_prime)[0].astype(np.int32)
    semi_idx = np.where(is_semi)[0].astype(np.int32)

    deep_threshold = int(cfg.get("m1", {}).get("deep_upstream_min_tier", 3))

    views = cfg.get("views", {})
    if not isinstance(views, dict) or not views:
        raise ValueError("config.views must be a non-empty mapping")

    summary_rows: list[dict[str, Any]] = []
    node_outputs: dict[str, str] = {}
    summary_outputs: dict[str, str] = {}

    for view_name, view_spec in views.items():
        include_any = view_spec.get("include_any", [])
        mask = get_view_mask(edges, include_any)
        view_edges = edges.loc[mask, ["src_idx", "dst_idx"]].copy()
        view_edges = view_edges[view_edges["src_idx"] != view_edges["dst_idx"]]

        src = view_edges["src_idx"].to_numpy(np.int32, copy=False)
        dst = view_edges["dst_idx"].to_numpy(np.int32, copy=False)
        m_edges = len(view_edges)

        in_deg = np.bincount(dst, minlength=n_nodes).astype(np.int32)
        out_deg = np.bincount(src, minlength=n_nodes).astype(np.int32)

        data = np.ones(m_edges, dtype=np.int8)
        adj = sp.csr_matrix((data, (src, dst)), shape=(n_nodes, n_nodes), dtype=np.int8)

        n_wcc, labels_w = csgraph.connected_components(
            adj, directed=True, connection="weak", return_labels=True
        )
        n_scc, labels_s = csgraph.connected_components(
            adj, directed=True, connection="strong", return_labels=True
        )

        w_sizes = np.bincount(labels_w, minlength=n_wcc)
        s_sizes = np.bincount(labels_s, minlength=n_scc)
        giant_w_label = int(np.argmax(w_sizes)) if n_wcc > 0 else -1
        giant_s_label = int(np.argmax(s_sizes)) if n_scc > 0 else -1
        giant_w_size = int(w_sizes[giant_w_label]) if n_wcc > 0 else 0
        giant_s_size = int(s_sizes[giant_s_label]) if n_scc > 0 else 0

        fwd, rev = build_adj_lists(n_nodes, src, dst)

        # dist_to_prime: node -> prime distance (downstream), computed as reverse BFS from primes.
        dist_to_prime = (
            multisource_bfs(prime_idx, rev)
            if len(prime_idx)
            else np.full(n_nodes, -1, dtype=np.int32)
        )
        # dist_from_semi: semi -> node distance (downstream).
        dist_from_semi = (
            multisource_bfs(semi_idx, fwd)
            if len(semi_idx)
            else np.full(n_nodes, -1, dtype=np.int32)
        )

        tier_prime_dist = np.where(dist_to_prime >= 0, dist_to_prime + 1, -1).astype(np.int32)
        tier_prime_dist[is_dod] = 0
        tier_prime_dist[is_prime] = 1

        # Undirected core structure.
        g_und = nx.Graph()
        g_und.add_nodes_from(range(n_nodes))
        g_und.add_edges_from(zip(src.tolist(), dst.tolist(), strict=False))
        core = nx.core_number(g_und)
        k_core = np.zeros(n_nodes, dtype=np.int32)
        for k, v in core.items():
            k_core[int(k)] = int(v)

        node_metrics = pd.DataFrame(
            {
                "analysis_uid": nodes["analysis_uid"].astype(str),
                "entity_role": role,
                "name": nodes["name"] if "name" in nodes.columns else None,
                "is_semi_strict": is_semi,
                "is_tier0_dod": is_dod,
                "is_tier1_prime": is_prime,
                "deg_in": in_deg,
                "deg_out": out_deg,
                "weak_component_id": labels_w.astype(np.int32),
                "strong_component_id": labels_s.astype(np.int32),
                "is_in_giant_weak": (labels_w == giant_w_label),
                "is_in_giant_strong": (labels_s == giant_s_label),
                "k_core": k_core,
                "dist_to_prime": dist_to_prime.astype(np.int32),
                "dist_from_semi": dist_from_semi.astype(np.int32),
                "tier_prime_dist": tier_prime_dist.astype(np.int32),
                "is_corridor_candidate": (dist_to_prime >= 0) & (dist_from_semi >= 0),
                "is_deep_upstream": tier_prime_dist >= deep_threshold,
                "view": view_name,
            }
        )

        node_out = m1_dir / f"node_metrics_baseline_{view_name}.parquet"
        node_metrics.to_parquet(node_out, index=False)
        node_outputs[view_name] = str(node_out)

        density = float(m_edges) / float(max(1, n_nodes * (n_nodes - 1)))
        summary = {
            "snapshot": snapshot,
            "run_id": run_id,
            "view": view_name,
            "nodes_total": int(n_nodes),
            "edges_total": int(m_edges),
            "density_directed": density,
            "wcc_count": int(n_wcc),
            "scc_count": int(n_scc),
            "giant_weak_size": giant_w_size,
            "giant_weak_share": float(giant_w_size / n_nodes),
            "giant_strong_size": giant_s_size,
            "giant_strong_share": float(giant_s_size / n_nodes),
            "k_core_max": int(k_core.max()),
            "k_core_median": float(np.median(k_core)),
            "k_core_p90": float(np.quantile(k_core, 0.90)),
            "prime_upstream_coverage_share": float((dist_to_prime >= 0).mean()),
            "corridor_candidate_share": float(
                ((dist_to_prime >= 0) & (dist_from_semi >= 0)).mean()
            ),
            "deep_upstream_share": float((tier_prime_dist >= deep_threshold).mean()),
            "semis_count": int(is_semi.sum()),
            "primes_count": int(is_prime.sum()),
            "dod_count": int(is_dod.sum()),
            **deg_stats(in_deg, "in_deg"),
            **deg_stats(out_deg, "out_deg"),
        }
        summary_rows.append(summary)

        summary_df = pd.DataFrame([summary])
        summary_out = m1_dir / f"graph_summary_{view_name}.csv"
        summary_df.to_csv(summary_out, index=False)
        summary_outputs[view_name] = str(summary_out)

    all_summary = pd.DataFrame(summary_rows).sort_values("view")
    all_summary_out = m1_dir / "graph_summary_all_views.csv"
    all_summary.to_csv(all_summary_out, index=False)

    run_metadata = {
        "module": "m1",
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
    }
    run_metadata_out = m1_dir / "run_metadata.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m1",
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
            "deep_upstream_min_tier": deep_threshold,
        },
        "outputs": {
            "graph_summary_all_views": str(all_summary_out),
            "graph_summary_by_view": summary_outputs,
            "node_metrics_by_view": node_outputs,
            "run_metadata": str(run_metadata_out),
        },
    }
    manifest_out = m1_dir / "manifest_m1.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))

    print(f"[done] wrote {all_summary_out}")
    for _view_name, path in summary_outputs.items():
        print(f"[done] wrote {path}")
    for _view_name, path in node_outputs.items():
        print(f"[done] wrote {path}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")


if __name__ == "__main__":
    main()
