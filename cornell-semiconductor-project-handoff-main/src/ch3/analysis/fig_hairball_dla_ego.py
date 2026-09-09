#!/usr/bin/env python3
"""Render a DLA-centric ego network with layered tiers (strict semis graph)."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

EDGE_STYLES = {
    "disclosed": {"color": "#2F4858", "alpha": 0.25, "lw": 0.6, "ls": "solid"},
    "predicted": {"color": "#3D9970", "alpha": 0.45, "lw": 0.7, "ls": "solid"},
    "shipping": {"color": "#F18F01", "alpha": 0.45, "lw": 0.7, "ls": "solid"},
    "contract": {"color": "#8D1B1B", "alpha": 0.8, "lw": 1.1, "ls": "solid"},
    "mapping": {"color": "#9E9E9E", "alpha": 0.35, "lw": 0.5, "ls": "dashed"},
}


def bfs_reachable(seeds: Iterable[str], adj: dict[str, list[str]]) -> set[str]:
    visited = set()
    q = deque()
    for s in seeds:
        if s not in visited:
            visited.add(s)
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
    return visited


def bfs_tiers(seeds: Iterable[str], adj: dict[str, list[str]]) -> dict[str, int]:
    tiers: dict[str, int] = {}
    q = deque()
    for s in seeds:
        tiers[s] = 1
        q.append(s)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in tiers:
                tiers[v] = tiers[u] + 1
                q.append(v)
    return tiers


def main() -> None:
    ap = argparse.ArgumentParser(description="DLA ego-network hairball (layered)")
    ap.add_argument(
        "--edges",
        default="artifacts/ch3/network_upstream/dod_semiconductor_edges_top5_d99_shipping_strict.parquet",
    )
    ap.add_argument(
        "--nodes",
        default="artifacts/ch3/network_upstream/dod_semiconductor_nodes_top5_d99_shipping_strict.parquet",
    )
    ap.add_argument("--semis-flags", default="artifacts/ch3/reference/semis_flags_strict.parquet")
    ap.add_argument("--dla-uid", default="dod:defense_logistics_agency")
    ap.add_argument("--max-per-tier", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="figs/chapter3/fig_hairball_dla_ego.pdf")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    edges = pd.read_parquet(args.edges)
    nodes = pd.read_parquet(args.nodes)
    semis = pd.read_parquet(args.semis_flags)

    node_type = nodes.set_index("node_uid")["node_type"].to_dict()
    node_name = nodes.set_index("node_uid")["name"].to_dict()

    semis_uids = set(semis.loc[semis["is_semi_strict"], "node_uid"].tolist())

    # Contract edges to identify DLA primes.
    contract = edges[edges["is_contract"] == 1][["src_uid", "dst_uid"]]
    dla = args.dla_uid
    primes = set()
    for row in contract.itertuples(index=False):
        if row.src_uid == dla:
            primes.add(row.dst_uid)
        elif row.dst_uid == dla:
            primes.add(row.src_uid)

    if not primes:
        raise ValueError("No primes found for DLA in contract edges.")

    # Build supply adjacency (supplier -> customer) and reverse (customer -> supplier).
    supply = edges[
        (edges["is_disclosed"] == 1) | (edges["is_predicted"] == 1) | (edges["is_shipping"] == 1)
    ]
    adj_fwd = defaultdict(list)
    adj_rev = defaultdict(list)
    for row in supply.itertuples(index=False):
        adj_fwd[row.src_uid].append(row.dst_uid)
        adj_rev[row.dst_uid].append(row.src_uid)

    # Identify nodes on semis->prime paths (within DLA primes).
    upstream_from_primes = bfs_reachable(primes, adj_rev)
    semis_in_dla = [u for u in semis_uids if u in upstream_from_primes]
    downstream_from_semis = bfs_reachable(semis_in_dla, adj_fwd)
    path_nodes = upstream_from_primes.intersection(downstream_from_semis)

    # Add one-hop upstream suppliers of semis.
    onehop_up = set()
    for s in semis_in_dla:
        for sup in adj_rev.get(s, []):
            onehop_up.add(sup)

    keep_nodes = {dla} | primes | path_nodes | onehop_up

    # Include mapping edges for connectivity but keep them separate for styling.
    mapping = edges[(edges["is_maps_to_scr"] == 1) | (edges["is_maps_to_factset"] == 1)][
        ["src_uid", "dst_uid"]
    ]

    # Build tiers (reverse supply edges only; primes are tier 1).
    tiers = bfs_tiers(primes, adj_rev)
    tiers[dla] = 0
    # Map any SCR nodes directly linked via mapping to primes into tier 1 for layout.
    for row in mapping.itertuples(index=False):
        if row.src_uid in primes:
            tiers[row.dst_uid] = 1
        if row.dst_uid in primes:
            tiers[row.src_uid] = 1

    # Subselect nodes per tier for readability (keep all primes/semis/DLA).
    degree = defaultdict(int)
    for row in supply.itertuples(index=False):
        degree[row.src_uid] += 1
        degree[row.dst_uid] += 1

    tier_buckets = defaultdict(list)
    for uid in keep_nodes:
        t = tiers.get(uid, None)
        if t is None:
            continue
        tier_buckets[t].append(uid)

    selected = {dla} | primes | set(semis_in_dla)
    for t, bucket in tier_buckets.items():
        if t in (0, 1):
            selected.update(bucket)
            continue
        if len(bucket) <= args.max_per_tier:
            selected.update(bucket)
            continue
        # keep highest-degree nodes + all semis
        bucket_sorted = sorted(bucket, key=lambda u: degree.get(u, 0), reverse=True)
        chosen = set(bucket_sorted[: args.max_per_tier])
        chosen.update([u for u in bucket if u in semis_in_dla])
        selected.update(chosen)

    # Filter edges.
    supply_keep = supply[supply["src_uid"].isin(selected) & supply["dst_uid"].isin(selected)].copy()
    contract_keep = contract[
        (contract["src_uid"].isin(selected)) & (contract["dst_uid"].isin(selected))
    ].copy()
    mapping_keep = mapping[
        (mapping["src_uid"].isin(selected)) & (mapping["dst_uid"].isin(selected))
    ].copy()

    # Build positions (layered by tier).
    tier_levels = sorted({tiers.get(u, 0) for u in selected})
    pos: dict[str, tuple[float, float]] = {}
    for t in tier_levels:
        layer = [u for u in selected if tiers.get(u, 0) == t]
        if not layer:
            continue
        y = np.linspace(0, 1, num=len(layer), endpoint=True)
        rng.shuffle(y)
        x = np.full(len(layer), t, dtype=float)
        # jitter
        x += rng.normal(0, 0.02, size=len(layer))
        y += rng.normal(0, 0.02, size=len(layer))
        for uid, xi, yi in zip(layer, x, y, strict=False):
            pos[uid] = (xi, yi)

    # Build edge segments by type.
    def segments(df: pd.DataFrame) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        segs = []
        for row in df.itertuples(index=False):
            if row.src_uid not in pos or row.dst_uid not in pos:
                continue
            segs.append((pos[row.src_uid], pos[row.dst_uid]))
        return segs

    seg_disclosed = segments(supply_keep[supply_keep["is_disclosed"] == 1])
    seg_predicted = segments(supply_keep[supply_keep["is_predicted"] == 1])
    seg_shipping = segments(supply_keep[supply_keep["is_shipping"] == 1])
    seg_contract = segments(contract_keep)
    seg_mapping = segments(mapping_keep)

    # Node styling
    colors = {
        "dod_component": "#C73E3A",
        "prime_vendor": "#2A4B8D",
        "semi": "#E3B341",
        "other": "#B0B0B0",
    }
    sizes = {
        "dod_component": 120,
        "prime_vendor": 50,
        "semi": 36,
        "other": 16,
    }

    node_colors = []
    node_sizes = []
    node_list = list(pos.keys())
    for uid in node_list:
        ntype = node_type.get(uid, "other")
        if uid in semis_in_dla:
            node_colors.append(colors["semi"])
            node_sizes.append(sizes["semi"])
        elif ntype in colors:
            node_colors.append(colors[ntype])
            node_sizes.append(sizes[ntype])
        else:
            node_colors.append(colors["other"])
            node_sizes.append(sizes["other"])

    # Plot
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9})
    fig = plt.figure(figsize=(10, 6.5))
    ax = plt.gca()
    ax.set_axis_off()

    for segs, style in [
        (seg_disclosed, EDGE_STYLES["disclosed"]),
        (seg_predicted, EDGE_STYLES["predicted"]),
        (seg_shipping, EDGE_STYLES["shipping"]),
        (seg_mapping, EDGE_STYLES["mapping"]),
        (seg_contract, EDGE_STYLES["contract"]),
    ]:
        if not segs:
            continue
        lc = LineCollection(
            segs, colors=style["color"], linewidths=style["lw"], alpha=style["alpha"]
        )
        if style["ls"] == "dashed":
            lc.set_linestyle("dashed")
        ax.add_collection(lc)

    xs = [pos[u][0] for u in node_list]
    ys = [pos[u][1] for u in node_list]
    ax.scatter(xs, ys, s=node_sizes, c=node_colors, edgecolors="none")

    # Labels: DLA + top primes by degree (limit 8).
    label_uids = [dla]
    prime_sorted = sorted(primes, key=lambda u: -degree.get(u, 0))
    label_uids += [u for u in prime_sorted if node_name.get(u)][:8]
    for uid in label_uids:
        if uid not in pos:
            continue
        label = node_name.get(uid) or uid
        x, y = pos[uid]
        ax.text(x + 0.03, y, label, fontsize=7, color="#111111")

    # Legend
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="DoD component",
            markerfacecolor=colors["dod_component"],
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Prime vendor",
            markerfacecolor=colors["prime_vendor"],
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Semiconductor firm",
            markerfacecolor=colors["semi"],
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Other supplier",
            markerfacecolor=colors["other"],
            markersize=5,
        ),
        Line2D([0], [0], color=EDGE_STYLES["disclosed"]["color"], lw=1.0, label="Disclosed SCR"),
        Line2D(
            [0], [0], color=EDGE_STYLES["predicted"]["color"], lw=1.0, label="Predicted (Top-5)"
        ),
        Line2D([0], [0], color=EDGE_STYLES["shipping"]["color"], lw=1.0, label="Observed shipping"),
        Line2D([0], [0], color=EDGE_STYLES["contract"]["color"], lw=1.0, label="DoD contract"),
        Line2D(
            [0],
            [0],
            color=EDGE_STYLES["mapping"]["color"],
            lw=1.0,
            linestyle="dashed",
            label="Identity map",
        ),
    ]
    ax.legend(handles=legend_items, loc="lower left", frameon=False, ncol=2, fontsize=7)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(
        f"Wrote {out_path} (nodes={len(node_list)}, edges={len(seg_disclosed) + len(seg_predicted) + len(seg_shipping) + len(seg_contract) + len(seg_mapping)})"
    )


if __name__ == "__main__":
    main()
