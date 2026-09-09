#!/usr/bin/env python3
"""
Chapter 3: prune DoD network to a semiconductor-focused subgraph.

Keep rules:
- Nodes on any directed path from semis -> primes (downstream path).
- All one-hop upstream suppliers of semis.
- Upstream nodes beyond one hop only if they connect to >=2 distinct semis,
  plus the paths from those shared nodes down to semis.
- Always keep DoD components and primes; keep contract + mapping edges if endpoints kept.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def bfs_reach(sources: Iterable[int], adj: list[list[int]]) -> np.ndarray:
    n = len(adj)
    visited = np.zeros(n, dtype=bool)
    q: deque[int] = deque()
    for s in sources:
        if not visited[s]:
            visited[s] = True
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    return visited


def bfs_reach_restricted(
    sources: Iterable[int], adj: list[list[int]], allowed: np.ndarray
) -> np.ndarray:
    n = len(adj)
    visited = np.zeros(n, dtype=bool)
    q: deque[int] = deque()
    for s in sources:
        if allowed[s] and not visited[s]:
            visited[s] = True
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if allowed[v] and not visited[v]:
                visited[v] = True
                q.append(v)
    return visited


def multi_source_dist(sources: Iterable[int], rev_adj: list[list[int]]) -> np.ndarray:
    n = len(rev_adj)
    dist = np.full(n, -1, dtype=np.int32)
    q: deque[int] = deque()
    for s in sources:
        if dist[s] == -1:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        for v in rev_adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def semis_reach_counts(sources: list[int], rev_adj: list[list[int]]) -> np.ndarray:
    n = len(rev_adj)
    a = np.full(n, -1, dtype=np.int32)
    b = np.full(n, -1, dtype=np.int32)
    count = np.zeros(n, dtype=np.int8)
    q: deque[tuple[int, int]] = deque()
    for sid, node in enumerate(sources):
        if a[node] == -1:
            a[node] = sid
            count[node] = 1
            q.append((node, sid))
    while q:
        u, sid = q.popleft()
        for v in rev_adj[u]:
            if count[v] == 2:
                continue
            if a[v] == sid or b[v] == sid:
                continue
            if count[v] == 0:
                a[v] = sid
                count[v] = 1
            else:
                b[v] = sid
                count[v] = 2
            q.append((v, sid))
    return count


def main() -> None:
    p = argparse.ArgumentParser(description="Prune DoD network to semiconductor subgraph")
    p.add_argument(
        "--edges",
        type=Path,
        default=Path("artifacts/ch3/network_upstream/dod_network_edges_top5_d99_shipping.parquet"),
    )
    p.add_argument(
        "--nodes",
        type=Path,
        default=Path("artifacts/ch3/network_upstream/dod_network_nodes_top5_d99_shipping.parquet"),
    )
    p.add_argument(
        "--semis-flags",
        type=Path,
        default=Path("artifacts/ch3/reference/semis_flags_strict.parquet"),
    )
    p.add_argument(
        "--include-adjacent", action="store_true", help="Include adjacent semis in seed set"
    )
    p.add_argument(
        "--min-shared", type=int, default=2, help="Min semis shared for >1 hop upstream nodes"
    )
    p.add_argument("--out-root", type=Path, default=Path("artifacts/ch3/network_upstream"))
    p.add_argument("--tag", type=str, default="top5_d99_shipping_strict")
    args = p.parse_args()

    for path in [args.edges, args.nodes, args.semis_flags]:
        if not path.exists():
            raise FileNotFoundError(path)

    edges = pd.read_parquet(args.edges)
    nodes = pd.read_parquet(args.nodes)

    node_uids = nodes["node_uid"].tolist()
    node_to_idx = {uid: i for i, uid in enumerate(node_uids)}
    n = len(node_uids)

    semis_df = pd.read_parquet(args.semis_flags)
    if "node_uid" not in semis_df.columns and "node_id" in semis_df.columns:
        semis_df["node_uid"] = "scr:" + semis_df["node_id"].astype(str)
    semis_df = semis_df.drop_duplicates(subset=["node_uid"])

    if "is_semi_strict" in semis_df.columns:
        semis_mask = semis_df["is_semi_strict"].astype(bool)
    else:
        semis_df["is_semi_core"] = semis_df["is_semi_core"].astype(bool)
        semis_df["is_semi_adjacent"] = semis_df["is_semi_adjacent"].astype(bool)
        if args.include_adjacent:
            semis_mask = semis_df["is_semi_core"] | semis_df["is_semi_adjacent"]
        else:
            semis_mask = semis_df["is_semi_core"]
    semis_uids = semis_df.loc[semis_mask, "node_uid"].tolist()
    semis_idx = [node_to_idx[uid] for uid in semis_uids if uid in node_to_idx]

    primes_idx = nodes.index[nodes["node_type"] == "prime_vendor"].tolist()
    dod_idx = nodes.index[nodes["node_type"] == "dod_component"].tolist()

    fwd: list[list[int]] = [[] for _ in range(n)]
    rev: list[list[int]] = [[] for _ in range(n)]

    for row in edges.itertuples(index=False):
        src = node_to_idx.get(row.src_uid)
        dst = node_to_idx.get(row.dst_uid)
        if src is None or dst is None:
            continue
        is_map = bool(row.is_maps_to_scr) or bool(row.is_maps_to_factset)
        is_supply = bool(row.is_disclosed) or bool(row.is_predicted) or bool(row.is_shipping)
        if is_map:
            fwd[src].append(dst)
            fwd[dst].append(src)
            rev[src].append(dst)
            rev[dst].append(src)
        if is_supply:
            fwd[src].append(dst)
            rev[dst].append(src)

    reachable_down = bfs_reach(semis_idx, fwd)
    reachable_up = bfs_reach(primes_idx, rev)
    path_nodes = reachable_down & reachable_up
    semis_idx_conn = [idx for idx in semis_idx if reachable_up[idx]]

    dist_up = multi_source_dist(semis_idx_conn, rev)
    semis_counts = semis_reach_counts(semis_idx_conn, rev)

    direct_up = dist_up == 1
    shared_up = (dist_up >= 2) & (semis_counts >= args.min_shared)
    upstream_allowed = dist_up >= 0
    shared_paths = bfs_reach_restricted(np.where(shared_up)[0].tolist(), fwd, upstream_allowed)
    upstream_keep = direct_up | shared_paths

    keep = path_nodes | upstream_keep
    for idx in dod_idx:
        keep[idx] = True

    keep_uids = set(nodes.loc[keep, "node_uid"])
    edges_keep = edges[edges["src_uid"].isin(keep_uids) & edges["dst_uid"].isin(keep_uids)].copy()
    endpoint_uids = set(edges_keep["src_uid"]).union(set(edges_keep["dst_uid"]))

    nodes_keep = nodes.loc[nodes["node_uid"].isin(endpoint_uids)].copy()
    semis_cols = [
        "node_uid",
        "is_semi_core",
        "is_semi_adjacent",
        "is_semi_strict",
        "is_semi_rbics_l4",
        "is_semi_sic3674",
        "semi_source",
        "rbics_l4_ids",
        "rbics_l4_names",
    ]
    semis_cols = [c for c in semis_cols if c in semis_df.columns]
    if semis_cols:
        nodes_keep = nodes_keep.merge(semis_df[semis_cols], on="node_uid", how="left")
    for col in [
        "is_semi_core",
        "is_semi_adjacent",
        "is_semi_strict",
        "is_semi_rbics_l4",
        "is_semi_sic3674",
    ]:
        if col in nodes_keep.columns:
            nodes_keep[col] = nodes_keep[col].fillna(False).astype(bool)
    if "is_semi_strict" in nodes_keep.columns:
        nodes_keep["is_semi"] = nodes_keep["is_semi_strict"]
    else:
        nodes_keep["is_semi"] = nodes_keep.get("is_semi_core", False) | nodes_keep.get(
            "is_semi_adjacent", False
        )

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    edges_out = out_root / f"dod_semiconductor_edges_{args.tag}.parquet"
    nodes_out = out_root / f"dod_semiconductor_nodes_{args.tag}.parquet"
    summary_out = out_root / f"summary_dod_semiconductor_{args.tag}.json"

    edges_keep.sort_values(["src_uid", "dst_uid"]).reset_index(drop=True).to_parquet(edges_out, index=False)
    nodes_keep.sort_values("node_uid").reset_index(drop=True).to_parquet(nodes_out, index=False)

    summary = {
        "tag": args.tag,
        "include_adjacent": bool(args.include_adjacent),
        "min_shared": int(args.min_shared),
        "nodes_total": len(nodes_keep),
        "edges_total": len(edges_keep),
        "nodes_semis_core": int(nodes_keep["is_semi_core"].sum())
        if "is_semi_core" in nodes_keep.columns
        else None,
        "nodes_semis_adjacent": int(nodes_keep["is_semi_adjacent"].sum())
        if "is_semi_adjacent" in nodes_keep.columns
        else None,
        "nodes_semis_strict": int(nodes_keep["is_semi_strict"].sum())
        if "is_semi_strict" in nodes_keep.columns
        else None,
        "nodes_semis_rbics_l4": int(nodes_keep["is_semi_rbics_l4"].sum())
        if "is_semi_rbics_l4" in nodes_keep.columns
        else None,
        "nodes_semis_sic3674": int(nodes_keep["is_semi_sic3674"].sum())
        if "is_semi_sic3674" in nodes_keep.columns
        else None,
        "nodes_path_semis_to_primes": int(path_nodes.sum()),
        "nodes_upstream_onehop": int(direct_up.sum()),
        "nodes_upstream_shared": int(shared_up.sum()),
        "edges_contract": int(edges_keep["is_contract"].sum()),
        "edges_maps_to_scr": int(edges_keep["is_maps_to_scr"].sum()),
        "edges_maps_to_factset": int(edges_keep["is_maps_to_factset"].sum()),
        "edges_disclosed": int(edges_keep["is_disclosed"].sum()),
        "edges_predicted": int(edges_keep["is_predicted"].sum()),
        "edges_shipping": int(edges_keep["is_shipping"].sum()),
        "outputs": {"edges": str(edges_out), "nodes": str(nodes_out)},
    }
    summary_out.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
