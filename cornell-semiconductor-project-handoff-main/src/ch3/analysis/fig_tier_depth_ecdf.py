#!/usr/bin/env python3
"""Render ECDF of tier depth from DoD primes (strict semis network).

Tier convention used for this figure:
- Tier-1 is the set of prime *firms* in the commercial graph (SCR/factset nodes) that
  are mapped from USAspending primes via identity bridges (maps_to_scr/maps_to_factset).
- Tier-(k+1) are upstream suppliers at graph distance k from those prime firms when
  traversing supply-like edges upstream (supplier -> customer edges reversed).

We exclude Tier-0 (DoD components) and contracting-identity nodes (prime_vendor) from the
ECDF denominator so the curve describes the depth profile of *commercial firms* in the
locked DoD semiconductor network.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, LogLocator


def build_adjacency(edges: pd.DataFrame) -> dict[str, list[str]]:
    grouped = edges.groupby("dst_uid")["src_uid"].apply(list)
    return grouped.to_dict()


def bfs_tiers(seeds: Iterable[str], adj: dict[str, list[str]]) -> dict[str, int]:
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


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    vals = values.dropna().astype(int)
    vals = vals[vals >= 1]
    if vals.empty:
        return np.array([]), np.array([])
    counts = vals.value_counts().sort_index()
    cum = counts.cumsum() / counts.sum()
    return counts.index.to_numpy(), cum.to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Tier depth ECDF for strict semis network")
    ap.add_argument(
        "--edges",
        default="artifacts/ch3/network_upstream/dod_semiconductor_edges_top5_d99_shipping_strict.parquet",
    )
    ap.add_argument(
        "--nodes",
        default="artifacts/ch3/network_upstream/dod_semiconductor_nodes_top5_d99_shipping_strict.parquet",
    )
    ap.add_argument("--out", default="figs/chapter3/fig_3_1_tier_depth_profile.pdf")
    args = ap.parse_args()

    edges = pd.read_parquet(
        args.edges,
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
    nodes = pd.read_parquet(args.nodes, columns=["node_uid", "node_type", "is_semi_strict"]).copy()
    nodes["is_semi_strict"] = nodes["is_semi_strict"].fillna(False).astype(bool)

    supply_mask = (
        (edges["is_disclosed"] == 1) | (edges["is_predicted"] == 1) | (edges["is_shipping"] == 1)
    )
    supply_edges = edges.loc[supply_mask, ["src_uid", "dst_uid"]]
    rev_adj = build_adjacency(supply_edges)

    # Tier-1 seed firms are the mapped SCR/FactSet nodes (identity bridges from prime vendors).
    maps = edges[(edges["is_maps_to_scr"] == 1) | (edges["is_maps_to_factset"] == 1)]
    seed_nodes = maps["dst_uid"].dropna().unique().tolist()

    tiers = bfs_tiers(seed_nodes, rev_adj)
    nodes["tier_from_primes"] = nodes["node_uid"].map(tiers)

    # ECDF over commercial firm nodes only (exclude Tier-0 + contracting identity nodes).
    is_commercial = nodes["node_type"].isin(["scr_firm", "factset_firm"])
    all_nodes = nodes[is_commercial].copy()
    semis = all_nodes[all_nodes["is_semi_strict"]].copy()

    _x_all, _y_all = ecdf(all_nodes["tier_from_primes"])
    _x_sem, _y_sem = ecdf(semis["tier_from_primes"])

    counts_all = all_nodes["tier_from_primes"].dropna().astype(int).value_counts().sort_index()
    counts_sem = semis["tier_from_primes"].dropna().astype(int).value_counts().sort_index()
    tiers = sorted(set(counts_all.index.to_list()) | set(counts_sem.index.to_list()))
    y_counts_all = np.array([int(counts_all.get(t, 0)) for t in tiers], dtype=int)
    np.array([int(counts_sem.get(t, 0)) for t in tiers], dtype=int)
    tiers_sem = np.array([t for t in tiers if counts_sem.get(t, 0) > 0], dtype=int)
    y_counts_sem_pos = np.array([int(counts_sem.get(t, 0)) for t in tiers_sem], dtype=int)

    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9})
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.bar(
        tiers,
        y_counts_all,
        color="#2F4858",
        alpha=0.85,
        width=0.75,
        label="All firms (disclosed + predicted + observed)",
    )
    if len(tiers_sem) > 0:
        ax.plot(
            tiers_sem,
            y_counts_sem_pos,
            color="#3D9970",
            marker="o",
            markersize=3.8,
            linewidth=1.7,
            label="Strict semiconductor firms",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Tier depth (from primes, upstream)")
    ax.set_ylabel("Firms at tier (log scale)")
    ax.set_xticks(tiers)

    # Log axis but show numeric tick labels (1, 10, 100, 1,000) instead of 10^x.
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}" if y >= 1 else ""))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout(pad=0.2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[done] wrote {out_path} and {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
