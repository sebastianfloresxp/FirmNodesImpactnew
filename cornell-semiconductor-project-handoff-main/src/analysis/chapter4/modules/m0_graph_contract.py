#!/usr/bin/env python3
"""Module 0: build and validate the Chapter 4 analysis graph contract.

Outputs:
  - graph_contract.json
  - node_table_contract.parquet
  - edge_table_contract.parquet
  - direction_sanity_report.csv
  - manifest_m0.json
  - run_metadata.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import random
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Reuse Chapter 4 collapse logic from the existing exploratory code.
THIS_FILE = Path(__file__).resolve()
CH4_DIR = THIS_FILE.parents[1]
if str(CH4_DIR) not in sys.path:
    sys.path.append(str(CH4_DIR))

from ch4_common import collapse_identity_graph

CANON_EDGE_FLAGS = [
    "is_contract_seed",
    "is_contract",
    "is_disclosed",
    "is_predicted",
    "is_observed_ship",
    "is_shipping",
    "is_identity_bridge",
    "is_maps_to_scr",
    "is_maps_to_factset",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chapter 4 Module 0 graph-object contract")
    p.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2.template.yaml",
        help="YAML config path (copy template to a run-specific file for production runs)",
    )
    return p.parse_args()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_edge_schema(edges: pd.DataFrame) -> pd.DataFrame:
    edges = edges.copy()
    if "src_uid" not in edges.columns or "dst_uid" not in edges.columns:
        raise ValueError("edges must include src_uid and dst_uid")

    # Normalize naming variants from Chapter 3 vs Chapter 4 prompt.
    if "is_contract_seed" not in edges.columns and "is_contract" in edges.columns:
        edges["is_contract_seed"] = edges["is_contract"]
    if "is_contract" not in edges.columns and "is_contract_seed" in edges.columns:
        edges["is_contract"] = edges["is_contract_seed"]

    if "is_observed_ship" not in edges.columns and "is_shipping" in edges.columns:
        edges["is_observed_ship"] = edges["is_shipping"]
    if "is_shipping" not in edges.columns and "is_observed_ship" in edges.columns:
        edges["is_shipping"] = edges["is_observed_ship"]

    if "is_identity_bridge" not in edges.columns:
        if "is_maps_to_scr" in edges.columns or "is_maps_to_factset" in edges.columns:
            edges["is_identity_bridge"] = edges.get("is_maps_to_scr", 0).fillna(0) + edges.get(
                "is_maps_to_factset", 0
            ).fillna(0)
            edges["is_identity_bridge"] = (edges["is_identity_bridge"] > 0).astype(np.int8)
        else:
            edges["is_identity_bridge"] = 0

    if "is_maps_to_scr" not in edges.columns:
        edges["is_maps_to_scr"] = edges.get("is_identity_bridge", 0)
    if "is_maps_to_factset" not in edges.columns:
        edges["is_maps_to_factset"] = 0

    for col in CANON_EDGE_FLAGS:
        if col not in edges.columns:
            edges[col] = 0
        edges[col] = edges[col].fillna(0).astype(np.int8)
    return edges


def dedup_edges(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    edges = edges.copy()
    before = len(edges)
    non_flags = [c for c in edges.columns if c not in {"src_uid", "dst_uid", *CANON_EDGE_FLAGS}]
    flags_agg = {c: "max" for c in CANON_EDGE_FLAGS if c in edges.columns}
    other_agg = dict.fromkeys(non_flags, "first")
    agg = {**flags_agg, **other_agg}
    dedup = edges.groupby(["src_uid", "dst_uid"], as_index=False).agg(agg)
    after = len(dedup)
    return dedup, {
        "edges_before_dedup": before,
        "edges_after_dedup": after,
        "edges_removed_by_dedup": before - after,
    }


def get_view_mask(edges: pd.DataFrame, include_any: list[str]) -> np.ndarray:
    if not include_any:
        raise ValueError("view include_any must contain at least one flag")
    missing = [c for c in include_any if c not in edges.columns]
    if missing:
        raise ValueError(f"view references missing columns: {missing}")
    mask = np.zeros(len(edges), dtype=bool)
    for col in include_any:
        mask |= edges[col].astype(bool).to_numpy()
    return mask


def build_adjacency(
    nodes: pd.DataFrame, edges: pd.DataFrame, src_col: str = "src_uid", dst_col: str = "dst_uid"
) -> tuple[list[list[int]], list[list[int]], dict[str, int]]:
    uid_col = "analysis_uid"
    uid_to_idx = {uid: i for i, uid in enumerate(nodes[uid_col].astype(str).tolist())}
    n = len(uid_to_idx)
    fwd = [[] for _ in range(n)]
    rev = [[] for _ in range(n)]
    for src_uid, dst_uid in edges[[src_col, dst_col]].itertuples(index=False, name=None):
        si = uid_to_idx.get(str(src_uid))
        di = uid_to_idx.get(str(dst_uid))
        if si is None or di is None or si == di:
            continue
        fwd[si].append(di)
        rev[di].append(si)
    return fwd, rev, uid_to_idx


def bfs_limited(start: int, adj: list[list[int]], max_hops: int) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, -1, dtype=np.int32)
    q: deque[int] = deque([start])
    dist[start] = 0
    while q:
        u = q.popleft()
        if dist[u] >= max_hops:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def prime_mask(nodes: pd.DataFrame) -> pd.Series:
    if "entity_role" in nodes.columns:
        return nodes["entity_role"].astype(str).eq("prime_vendor")
    if "node_type" in nodes.columns:
        return nodes["node_type"].astype(str).str.contains("prime", case=False, na=False)
    return pd.Series(False, index=nodes.index)


def semi_mask(nodes: pd.DataFrame) -> pd.Series:
    for col in ["is_semi_strict", "is_semi", "semi"]:
        if col in nodes.columns:
            return nodes[col].fillna(False).astype(bool)
    return pd.Series(False, index=nodes.index)


def build_direction_sanity(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seed: int,
    max_hops: int,
    sample_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    supply_cols = ["is_disclosed", "is_predicted", "is_observed_ship"]
    for col in supply_cols:
        if col not in edges.columns:
            edges[col] = 0
    supply = edges[(edges[supply_cols].sum(axis=1) > 0)].copy()

    fwd, rev, uid_to_idx = build_adjacency(nodes, supply, "src_uid", "dst_uid")
    {idx: uid for uid, idx in uid_to_idx.items()}

    primes = nodes.loc[prime_mask(nodes), "analysis_uid"].astype(str).tolist()
    semis_bool = semi_mask(nodes).to_numpy()

    rng = random.Random(seed)  # nosec B311 -- seeded for reproducible sampling, not cryptographic
    if len(primes) > sample_size:
        primes = rng.sample(primes, sample_size)

    rows: list[dict[str, Any]] = []
    for puid in primes:
        pidx = uid_to_idx[puid]
        rev_dist = bfs_limited(pidx, rev, max_hops=max_hops)
        fwd_dist = bfs_limited(pidx, fwd, max_hops=max_hops)

        rev_reached = rev_dist >= 1
        fwd_reached = fwd_dist >= 1

        rev_nodes = int(rev_reached.sum())
        fwd_nodes = int(fwd_reached.sum())
        rev_semis = int((rev_reached & semis_bool).sum())
        fwd_semis = int((fwd_reached & semis_bool).sum())

        rows.append(
            {
                "prime_uid": puid,
                "prime_name": nodes.loc[nodes["analysis_uid"] == puid, "name"].iloc[0]
                if "name" in nodes.columns
                else None,
                "rev_upstream_nodes": rev_nodes,
                "rev_upstream_semis": rev_semis,
                "fwd_nodes": fwd_nodes,
                "fwd_semis": fwd_semis,
                "direction_pass_prime": bool((rev_semis >= fwd_semis) and (rev_nodes >= fwd_nodes)),
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        summary = {
            "sampled_primes": 0,
            "share_primes_with_rev_semis": 0.0,
            "share_primes_with_fwd_semis": 0.0,
            "share_direction_pass_prime": 0.0,  # nosec B105 -- "pass" in key name is not a password
            "recommended_traversal": "reversed_for_upstream",
        }
        return report, summary

    share_rev_semis = float((report["rev_upstream_semis"] > 0).mean())
    share_fwd_semis = float((report["fwd_semis"] > 0).mean())
    share_pass = float(report["direction_pass_prime"].mean())
    recommended = "reversed_for_upstream"
    if share_fwd_semis > (share_rev_semis + 0.15):
        recommended = "flip_required_check_data_direction"

    summary = {
        "sampled_primes": len(report),
        "share_primes_with_rev_semis": share_rev_semis,
        "share_primes_with_fwd_semis": share_fwd_semis,
        "share_direction_pass_prime": share_pass,
        "recommended_traversal": recommended,
    }
    return report, summary


def get_git_commit() -> str | None:
    try:
        out = subprocess.check_output(  # nosec B607 -- git is a well-known system executable
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out or None
    except Exception:
        return None


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must parse into a mapping")
    return cfg


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "unspecified_snapshot"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    random_cfg = cfg.get("random", {})
    global_seed = int(random_cfg.get("global_seed", 7))
    np.random.seed(global_seed)
    random.seed(global_seed)

    paths = cfg.get("paths", {})
    nodes_path = Path(str(paths.get("nodes")))
    edges_path = Path(str(paths.get("edges")))
    if not nodes_path.exists():
        raise FileNotFoundError(nodes_path)
    if not edges_path.exists():
        raise FileNotFoundError(edges_path)

    out_root = Path(str(paths.get("out_root", "artifacts/ch4/v2")))
    m0_dir = out_root / snapshot / "m0"
    m0_dir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_parquet(nodes_path)
    edges_raw = pd.read_parquet(edges_path)
    edges_norm = normalize_edge_schema(edges_raw)
    edges_dedup, dedup_stats = dedup_edges(edges_norm)

    identity_mode = str(cfg.get("graph", {}).get("identity_mode", "collapse")).strip().lower()
    if identity_mode not in {"collapse", "zero_length_traversal"}:
        raise ValueError(f"Unsupported graph.identity_mode: {identity_mode}")

    node_map_out = None
    if identity_mode == "collapse":
        entity_graph = collapse_identity_graph(nodes=nodes, edges=edges_dedup)
        node_table = entity_graph.nodes.copy()
        node_table["analysis_uid"] = node_table["entity_uid"].astype(str)

        edge_table = entity_graph.edges.copy()
        edge_table["is_contract_seed"] = edge_table.get("is_contract", 0).fillna(0).astype(np.int8)
        edge_table["is_observed_ship"] = edge_table.get("is_shipping", 0).fillna(0).astype(np.int8)
        edge_table["is_identity_bridge"] = 0
        edge_table["src_uid"] = edge_table["src_uid"].astype(str)
        edge_table["dst_uid"] = edge_table["dst_uid"].astype(str)

        node_map_out = m0_dir / "raw_to_analysis_node_map.parquet"
        map_df = entity_graph.node_map.copy()
        map_df["analysis_uid"] = map_df["entity_uid"].astype(str)
        map_df.to_parquet(node_map_out, index=False)
    else:
        node_table = nodes.copy()
        node_table["analysis_uid"] = node_table["node_uid"].astype(str)
        edge_table = edges_dedup.copy()
        edge_table["src_uid"] = edge_table["src_uid"].astype(str)
        edge_table["dst_uid"] = edge_table["dst_uid"].astype(str)

    for col in CANON_EDGE_FLAGS:
        if col not in edge_table.columns:
            edge_table[col] = 0
        edge_table[col] = edge_table[col].fillna(0).astype(np.int8)

    node_out = m0_dir / "node_table_contract.parquet"
    edge_out = m0_dir / "edge_table_contract.parquet"
    node_table.to_parquet(node_out, index=False)
    edge_table.to_parquet(edge_out, index=False)

    views_cfg = cfg.get("views", {})
    view_counts: dict[str, int] = {}
    for view_name, spec in views_cfg.items():
        include_any = spec.get("include_any", [])
        mask = get_view_mask(edge_table, include_any)
        view_counts[view_name] = int(mask.sum())

    monotone_ok = True
    if {"disclosed", "observed", "full"}.issubset(set(view_counts)):
        monotone_ok = view_counts["disclosed"] <= view_counts["observed"] <= view_counts["full"]

    direction_cfg = cfg.get("direction", {})
    sanity_sample = int(direction_cfg.get("sanity_sample_primes", 20))
    sanity_hops = int(direction_cfg.get("max_sanity_hops", 6))
    sanity_df, sanity_summary = build_direction_sanity(
        nodes=node_table,
        edges=edge_table,
        seed=global_seed,
        max_hops=sanity_hops,
        sample_size=sanity_sample,
    )
    sanity_out = m0_dir / "direction_sanity_report.csv"
    sanity_df.to_csv(sanity_out, index=False)

    graph_contract = {
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "identity_mode": identity_mode,
        "inputs": {
            "nodes_path": str(nodes_path),
            "edges_path": str(edges_path),
            "nodes_sha256": file_sha256(nodes_path),
            "edges_sha256": file_sha256(edges_path),
        },
        "counts": {
            "nodes_raw": len(nodes),
            "edges_raw": len(edges_raw),
            **dedup_stats,
            "nodes_contract": len(node_table),
            "edges_contract": len(edge_table),
        },
        "views_edge_counts": view_counts,
        "checks": {
            "view_monotonicity_pass": bool(monotone_ok),
            "direction_recommended_traversal": sanity_summary.get("recommended_traversal"),
            "direction_share_primes_with_rev_semis": sanity_summary.get(
                "share_primes_with_rev_semis"
            ),
            "direction_share_primes_with_fwd_semis": sanity_summary.get(
                "share_primes_with_fwd_semis"
            ),
            "direction_share_pass_prime": sanity_summary.get("share_direction_pass_prime"),
        },
        "outputs": {
            "node_table_contract": str(node_out),
            "edge_table_contract": str(edge_out),
            "direction_sanity_report": str(sanity_out),
            "raw_to_analysis_node_map": str(node_map_out) if node_map_out is not None else None,
        },
    }
    graph_contract_out = m0_dir / "graph_contract.json"
    graph_contract_out.write_text(json.dumps(graph_contract, indent=2))

    run_metadata = {
        "snapshot": snapshot,
        "run_id": run_id,
        "module": "m0",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "yaml": yaml.__version__,
        },
        "seeds": random_cfg,
    }
    run_metadata_out = m0_dir / "run_metadata.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m0",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs": graph_contract["inputs"],
        "params": {
            "identity_mode": identity_mode,
            "views": views_cfg,
            "direction": direction_cfg,
            "graph": cfg.get("graph", {}),
        },
        "outputs": graph_contract["outputs"]
        | {"graph_contract": str(graph_contract_out), "run_metadata": str(run_metadata_out)},
    }
    manifest_out = m0_dir / "manifest_m0.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))

    print(f"[done] wrote {graph_contract_out}")
    print(f"[done] wrote {node_out}")
    print(f"[done] wrote {edge_out}")
    print(f"[done] wrote {sanity_out}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")
    if node_map_out is not None:
        print(f"[done] wrote {node_map_out}")


if __name__ == "__main__":
    main()
