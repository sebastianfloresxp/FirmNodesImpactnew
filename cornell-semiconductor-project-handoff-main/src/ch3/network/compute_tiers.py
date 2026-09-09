#!/usr/bin/env python3
"""
Chapter 3: compute tier depth from DoD primes.

Produces two tier assignments (mapping edges treated as zero-length identity):
  - disclosed-only (edges where is_disclosed=1)
  - evidence-augmented (disclosed + predicted + shipping)

Outputs (default under artifacts/ch3/network):
  - dod_network_nodes_top{K}_d99_shipping_tiers.parquet
  - summary_dod_network_top{K}_d99_shipping_tiers.json
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def build_adjacency(edges: pd.DataFrame) -> dict[str, list[str]]:
    """Adjacency for upstream traversal: dst -> list[src]."""
    if edges.empty:
        return {}
    grouped = edges.groupby("dst_uid")["src_uid"].apply(list)
    return grouped.to_dict()


def bfs_tiers(seeds: Iterable[str], adj: dict[str, list[str]]) -> dict[str, int]:
    """BFS upstream from seeds; returns shortest tier (seed=1)."""
    tiers: dict[str, int] = {}
    q: deque[str] = deque()
    for s in seeds:
        if s not in tiers:
            tiers[s] = 1
            q.append(s)
    while q:
        node = q.popleft()
        tier = tiers[node]
        for src in adj.get(node, []):
            if src not in tiers:
                tiers[src] = tier + 1
                q.append(src)
    return tiers


def tier_counts(tiers: pd.Series) -> dict[str, int]:
    counts = tiers.dropna().astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in counts.items()}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute disclosed-only and augmented tier depth from DoD primes"
    )
    p.add_argument(
        "--edges",
        default="artifacts/ch3/network_upstream/dod_network_edges_top5_d99_shipping.parquet",
    )
    p.add_argument(
        "--nodes",
        default="artifacts/ch3/network_upstream/dod_network_nodes_top5_d99_shipping.parquet",
    )
    p.add_argument("--out-root", default="artifacts/ch3/network_upstream")
    p.add_argument("--k", type=int, default=5, choices=[5, 10, 50])
    args = p.parse_args()

    edges_path = Path(args.edges)
    nodes_path = Path(args.nodes)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not edges_path.exists():
        raise FileNotFoundError(edges_path)
    if not nodes_path.exists():
        raise FileNotFoundError(nodes_path)

    edges = pd.read_parquet(
        edges_path,
        columns=[
            "src_uid",
            "dst_uid",
            "is_disclosed",
            "is_predicted",
            "is_shipping",
            "is_maps_to_scr",
            "is_maps_to_factset",
        ],
    )
    nodes = pd.read_parquet(nodes_path)

    # Prime vendors as Tier-1 seeds.
    primes = nodes.loc[nodes["node_type"] == "prime_vendor", "node_uid"].dropna().tolist()
    prime_set: set[str] = set(primes)

    # Add mapped SCR/FactSet nodes as Tier-1 seeds (identity bridges).
    # This collapses mapping edges for tier depth (no extra tier from mapping).
    maps = edges[(edges["is_maps_to_scr"] == 1) | (edges["is_maps_to_factset"] == 1)]
    mapped = maps[maps["src_uid"].isin(prime_set)]["dst_uid"].dropna().tolist()
    seed_nodes = list(prime_set.union(mapped))

    # Disclosed-only adjacency.
    disclosed_edges = edges[edges["is_disclosed"] == 1][["src_uid", "dst_uid"]]
    adj_disclosed = build_adjacency(disclosed_edges)
    tiers_disclosed = bfs_tiers(seed_nodes, adj_disclosed)

    # Augmented adjacency: disclosed + predicted + shipping.
    any_mask = (
        (edges["is_disclosed"] == 1) | (edges["is_predicted"] == 1) | (edges["is_shipping"] == 1)
    )
    any_edges = edges[any_mask][["src_uid", "dst_uid"]]
    adj_any = build_adjacency(any_edges)
    tiers_any = bfs_tiers(seed_nodes, adj_any)

    # Attach tiers to node table.
    nodes = nodes.copy()
    nodes["tier_disclosed"] = nodes["node_uid"].map(tiers_disclosed)
    nodes["tier_any_evidence"] = nodes["node_uid"].map(tiers_any)

    # Ensure DoD components show Tier-0.
    nodes.loc[nodes["node_type"] == "dod_component", "tier_disclosed"] = 0
    nodes.loc[nodes["node_type"] == "dod_component", "tier_any_evidence"] = 0

    # Ensure primes are Tier-1 (even if no upstream edges).
    nodes.loc[nodes["node_type"] == "prime_vendor", "tier_disclosed"] = 1
    nodes.loc[nodes["node_type"] == "prime_vendor", "tier_any_evidence"] = 1

    tiers_out = out_root / f"dod_network_nodes_top{args.k}_d99_shipping_tiers.parquet"
    nodes.to_parquet(tiers_out, index=False)

    summary = {
        "k": int(args.k),
        "nodes_total": len(nodes),
        "tier_disclosed_counts": tier_counts(nodes["tier_disclosed"]),
        "tier_any_evidence_counts": tier_counts(nodes["tier_any_evidence"]),
        "tier_disclosed_max": int(nodes["tier_disclosed"].max()),
        "tier_any_evidence_max": int(nodes["tier_any_evidence"].max()),
        "outputs": {
            "nodes_with_tiers": str(tiers_out),
        },
    }
    summary_path = out_root / f"summary_dod_network_top{args.k}_d99_shipping_tiers.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
