#!/usr/bin/env python3
"""Render a publication-quality hairball view of the strict semis DoD network (subset)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

EDGE_PRIORITY = ["contract", "shipping", "predicted", "disclosed"]


def pick_edge_type(row: pd.Series) -> str | None:
    types = []
    if row.get("is_contract", 0) == 1:
        types.append("contract")
    if row.get("is_shipping", 0) == 1:
        types.append("shipping")
    if row.get("is_predicted", 0) == 1:
        types.append("predicted")
    if row.get("is_disclosed", 0) == 1:
        types.append("disclosed")
    for t in EDGE_PRIORITY:
        if t in types:
            return t
    return None


def build_degree(edges: pd.DataFrame) -> dict[str, int]:
    deg = {}
    for uid in pd.concat([edges["src_uid"], edges["dst_uid"]]):
        deg[uid] = deg.get(uid, 0) + 1
    return deg


def main() -> None:
    ap = argparse.ArgumentParser(description="Hairball figure for strict semis network")
    ap.add_argument(
        "--edges",
        default="artifacts/ch3/network_upstream/dod_semiconductor_edges_top5_d99_shipping_strict.parquet",
    )
    ap.add_argument(
        "--nodes",
        default="artifacts/ch3/network_upstream/dod_semiconductor_nodes_top5_d99_shipping_strict.parquet",
    )
    ap.add_argument("--max-nodes", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="figs/chapter3/fig_hairball_semis_strict.pdf")
    args = ap.parse_args()

    edges = pd.read_parquet(args.edges)
    nodes = pd.read_parquet(args.nodes)

    # Drop mapping edges to avoid tier noise in visualization.
    edges = edges[(edges["is_maps_to_scr"] == 0) & (edges["is_maps_to_factset"] == 0)].copy()

    # Select nodes: all DoD, primes, semis + top degree others.
    nodes["is_semi_strict"] = nodes.get("is_semi_strict", False)
    base_mask = (nodes["node_type"] == "dod_component") | (nodes["node_type"] == "prime_vendor")
    semis_mask = nodes["is_semi_strict"]
    keep = set(nodes.loc[base_mask | semis_mask, "node_uid"].tolist())

    deg = build_degree(edges)
    remaining = [uid for uid in nodes["node_uid"].tolist() if uid not in keep]
    remaining = sorted(remaining, key=lambda u: deg.get(u, 0), reverse=True)
    budget = max(0, args.max_nodes - len(keep))
    keep.update(remaining[:budget])

    edges_keep = edges[edges["src_uid"].isin(keep) & edges["dst_uid"].isin(keep)].copy()
    nodes_keep = nodes[nodes["node_uid"].isin(keep)].copy()

    # Build graph with edge type.
    G = nx.Graph()
    for row in edges_keep.itertuples(index=False):
        etype = pick_edge_type(row._asdict())
        if not etype:
            continue
        if G.has_edge(row.src_uid, row.dst_uid):
            # Keep higher-priority evidence if already present.
            existing = G[row.src_uid][row.dst_uid]["etype"]
            if EDGE_PRIORITY.index(etype) < EDGE_PRIORITY.index(existing):
                G[row.src_uid][row.dst_uid]["etype"] = etype
        else:
            G.add_edge(row.src_uid, row.dst_uid, etype=etype)

    # Node styling
    node_type = nodes_keep.set_index("node_uid")["node_type"].to_dict()
    is_semi = nodes_keep.set_index("node_uid")["is_semi_strict"].to_dict()

    color_map = {
        "dod_component": "#C73E3A",
        "prime_vendor": "#2A4B8D",
        "semi": "#E3B341",
        "other": "#B0B0B0",
    }
    size_map = {
        "dod_component": 110,
        "prime_vendor": 40,
        "semi": 30,
        "other": 12,
    }

    node_colors = []
    node_sizes = []
    for n in G.nodes():
        ntype = node_type.get(n, "other")
        if is_semi.get(n, False):
            node_colors.append(color_map["semi"])
            node_sizes.append(size_map["semi"])
        elif ntype in color_map:
            node_colors.append(color_map[ntype])
            node_sizes.append(size_map[ntype])
        else:
            node_colors.append(color_map["other"])
            node_sizes.append(size_map["other"])

    # Edge styling
    edge_colors = {
        "disclosed": "#2F4858",
        "predicted": "#3D9970",
        "shipping": "#F18F01",
        "contract": "#8D1B1B",
    }
    edge_alpha = {
        "disclosed": 0.15,
        "predicted": 0.35,
        "shipping": 0.35,
        "contract": 0.6,
    }
    edge_width = {
        "disclosed": 0.3,
        "predicted": 0.4,
        "shipping": 0.4,
        "contract": 0.6,
    }

    # Layout
    pos = nx.spring_layout(G, seed=args.seed, k=1 / np.sqrt(max(1, G.number_of_nodes())))

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = plt.gca()
    ax.set_axis_off()

    # Draw edges by type to control alpha/width.
    for etype in EDGE_PRIORITY[::-1]:
        edgelist = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == etype]
        if not edgelist:
            continue
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            edge_color=edge_colors[etype],
            alpha=edge_alpha[etype],
            width=edge_width[etype],
            ax=ax,
        )

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, linewidths=0, ax=ax
    )

    # Label only DoD components.
    dod_nodes = [n for n in G.nodes() if node_type.get(n) == "dod_component"]
    labels = {n: nodes_keep.loc[nodes_keep["node_uid"] == n, "name"].iloc[0] for n in dod_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color="#111111")

    # Legend
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="DoD component",
            markerfacecolor=color_map["dod_component"],
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Prime vendor",
            markerfacecolor=color_map["prime_vendor"],
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Semiconductor firm",
            markerfacecolor=color_map["semi"],
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Other supplier",
            markerfacecolor=color_map["other"],
            markersize=5,
        ),
        Line2D([0], [0], color=edge_colors["disclosed"], lw=1.2, label="Disclosed SCR"),
        Line2D([0], [0], color=edge_colors["predicted"], lw=1.2, label="Predicted (Top-5)"),
        Line2D([0], [0], color=edge_colors["shipping"], lw=1.2, label="Observed shipping"),
        Line2D([0], [0], color=edge_colors["contract"], lw=1.2, label="DoD contract"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower left",
        frameon=False,
        ncol=2,
        fontsize=7,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path} (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")


if __name__ == "__main__":
    main()
