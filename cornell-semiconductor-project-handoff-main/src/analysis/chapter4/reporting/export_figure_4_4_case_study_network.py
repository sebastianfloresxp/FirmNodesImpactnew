#!/usr/bin/env python3
"""Export a case-study network visual (baseline vs focal-node removal).

This is an illustrative mechanism figure for Chapter 4. It does not alter
core estimands and is built from observed-view artifacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG = "src/analysis/chapter4/config/ch4_v2_fix01.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Figure 4.4 case-study network visual")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument(
        "--focal-uid",
        default=None,
        help="Optional focal analysis_uid (default: rank 1 from strict observed Top-25)",
    )
    parser.add_argument(
        "--max-semis",
        type=int,
        default=18,
        help="Number of semiconductor nodes to include (default: 18)",
    )
    parser.add_argument(
        "--max-primes",
        type=int,
        default=16,
        help="Number of prime endpoints to include (default: 16)",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=130,
        help="Minimum node count after neighborhood augmentation (default: 130)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=170,
        help="Maximum node count after pruning (default: 170)",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=10,
        help="Shortest path/BFS cutoff for selection (default: 10)",
    )
    parser.add_argument(
        "--k-shortest",
        type=int,
        default=2,
        help="Max shortest-simple paths per endpoint pair segment (default: 2)",
    )
    parser.add_argument(
        "--augment-neighbors-per-node",
        type=int,
        default=5,
        help="Neighbor augmentation fan-out per seed node (default: 5)",
    )
    parser.add_argument(
        "--max-context-edges",
        type=int,
        default=220,
        help="Maximum non-highlight edges to draw per panel (default: 220)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional override output directory (default: <figs_root>/main_text)",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config must parse to mapping")
    return cfg


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def short_name(text: Any, max_chars: int = 30) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return ""
    s = " ".join(s.split())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def safe_name(uid: str, name_map: dict[str, str]) -> str:
    s = short_name(name_map.get(uid, ""), max_chars=34)
    return s if s else uid


def select_focal_uid(top25_path: Path, override: str | None) -> str:
    if override:
        return str(override)
    top = pd.read_csv(top25_path)
    if top.empty:
        raise ValueError(f"top25 file is empty: {top25_path}")
    if "rank_semiconductor_lens" in top.columns:
        top = top.sort_values(["rank_semiconductor_lens", "analysis_uid"], ascending=[True, True])
    return str(top.iloc[0]["analysis_uid"])


def build_role(
    uid: str,
    focal_uid: str,
    dod_uid: str | None,
    meta: pd.DataFrame,
    semis: set[str],
    primes: set[str],
) -> str:
    if dod_uid and uid == dod_uid:
        return "DoD Agency"
    if uid == focal_uid:
        return "Focal Chokepoint"
    if uid in primes:
        return "Prime Endpoint"
    if uid in semis:
        return "Semiconductor Node"
    row = meta.loc[uid] if uid in meta.index else None
    if row is not None:
        if bool(row.get("is_intermediary_corridor", False)):
            return "Corridor Intermediary"
        if pd.notna(row.get("dist_to_prime")) and float(row.get("dist_to_prime")) == 1.0:
            return "Prime Adjacent"
    return "Upstream Context"


def build_layout(
    graph: nx.DiGraph,
    node_list: list[str],
    role_map: dict[str, str],
    dod_uid: str | None,
    seed: int = 7,
) -> dict[str, tuple[float, float]]:
    """Spring-first layout with soft role anchors (avoids line-like layering)."""
    rng = np.random.default_rng(seed)
    x_anchor = {
        "DoD Agency": -3.2,
        "Prime Endpoint": -2.1,
        "Prime Adjacent": -1.2,
        "Focal Chokepoint": -0.1,
        "Corridor Intermediary": 0.9,
        "Semiconductor Node": 2.0,
        "Upstream Context": 2.7,
    }
    init_pos: dict[str, tuple[float, float]] = {}
    for uid in node_list:
        role = role_map.get(uid, "Upstream Context")
        x0 = float(x_anchor.get(role, 0.0)) + float(rng.normal(0.0, 0.28))
        y0 = float(rng.normal(0.0, 1.2))
        init_pos[uid] = (x0, y0)
    fixed_nodes: list[str] | None = None
    if dod_uid and dod_uid in node_list:
        init_pos[dod_uid] = (-3.5, 0.0)
        fixed_nodes = [dod_uid]

    # Spring layout on undirected projection for visual complexity and spacing.
    g_u = graph.subgraph(node_list).to_undirected()
    spring_pos = nx.spring_layout(
        g_u,
        pos=init_pos,
        fixed=fixed_nodes,
        seed=seed,
        k=max(1.35, 8.5 / math.sqrt(max(1, len(node_list)))),
        iterations=460,
        weight=None,
        scale=4.0,
    )

    # Affine cleanup and role anchor blend.
    pos: dict[str, tuple[float, float]] = {}
    if dod_uid and dod_uid in spring_pos:
        # Ensure DoD anchor is on left side.
        x_dod = float(spring_pos[dod_uid][0])
        x_med = float(np.median([float(spring_pos[u][0]) for u in node_list]))
        flip = -1.0 if x_dod > x_med else 1.0
    else:
        flip = 1.0
    for uid in node_list:
        sx, sy = spring_pos[uid]
        role = role_map.get(uid, "Upstream Context")
        ax = x_anchor.get(role, 0.0)
        x = 0.80 * (flip * float(sx)) + 0.20 * float(ax)
        y = float(sy) * 1.15
        pos[uid] = (x, y)

    # Mild radial expansion to reduce local clumping while preserving structure.
    for uid in node_list:
        x, y = pos[uid]
        pos[uid] = (x * 1.12, y * 1.16)
    return pos


def draw_panel(
    ax: plt.Axes,
    graph: nx.DiGraph,
    node_list: list[str],
    pos: dict[str, tuple[float, float]],
    role_map: dict[str, str],
    highlight_edges: set[tuple[str, str]],
    label_nodes: set[str],
    name_map: dict[str, str],
    focal_uid: str,
    removed_mode: bool,
    disconnected_primes: set[str],
    degraded_primes: set[str],
    max_context_edges: int,
) -> None:
    color_map = {
        "DoD Agency": "#8B1E3F",
        "Prime Endpoint": "#4E79A7",
        "Semiconductor Node": "#59A14F",
        "Corridor Intermediary": "#9C755F",
        "Prime Adjacent": "#BAB0AC",
        "Upstream Context": "#D6D6D6",
        "Focal Chokepoint": "#F2BE5C",
    }

    ax.set_axis_off()
    ax.set_aspect("equal")

    # Edges first (deterministically thinned for readability).
    draw_edges_all = [e for e in graph.edges() if e[0] in node_list and e[1] in node_list]
    hi_set = set(highlight_edges)
    hi_edges = [e for e in draw_edges_all if e in hi_set]
    context_edges = [e for e in draw_edges_all if e not in hi_set]
    key_roles = {"DoD Agency", "Prime Endpoint", "Focal Chokepoint"}
    protected_context = [
        e
        for e in context_edges
        if role_map.get(e[0], "Upstream Context") in key_roles
        or role_map.get(e[1], "Upstream Context") in key_roles
    ]
    protected_set = set(protected_context)
    remaining_context = [e for e in context_edges if e not in protected_set]
    remaining_context = sorted(
        remaining_context,
        key=lambda e: (
            -int(role_map.get(e[0], "Upstream Context") == "Corridor Intermediary")
            - int(role_map.get(e[1], "Upstream Context") == "Corridor Intermediary"),
            e[0],
            e[1],
        ),
    )
    keep_n = max(0, int(max_context_edges) - len(protected_context))
    draw_edges = protected_context + remaining_context[:keep_n]
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        edgelist=draw_edges,
        ax=ax,
        width=0.34,
        edge_color="#8a8a8a",
        alpha=0.23 if not removed_mode else 0.15,
        arrows=False,
    )
    # Highlight a subset of structurally relevant local path edges.
    if hi_edges:
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            edgelist=hi_edges,
            ax=ax,
            width=0.95,
            edge_color="#4f4f4f",
            alpha=0.80 if not removed_mode else 0.56,
            arrows=False,
        )

    node_size = {
        "DoD Agency": 70,
        "Upstream Context": 18,
        "Semiconductor Node": 26,
        "Corridor Intermediary": 22,
        "Prime Adjacent": 24,
        "Prime Endpoint": 30,
        "Focal Chokepoint": 52,
    }
    node_lw = {
        "Upstream Context": 0.35,
        "Semiconductor Node": 0.45,
        "Corridor Intermediary": 0.45,
        "Prime Adjacent": 0.52,
        "Prime Endpoint": 0.62,
    }

    # Node drawing by role.
    for role in [
        "Upstream Context",
        "Semiconductor Node",
        "Corridor Intermediary",
        "Prime Adjacent",
        "Prime Endpoint",
    ]:
        nodes = [u for u in node_list if role_map.get(u) == role and u != focal_uid]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            nodelist=nodes,
            node_size=node_size.get(role, 50),
            node_color=color_map[role],
            edgecolors="#4c4c4c",
            linewidths=node_lw.get(role, 0.55),
            ax=ax,
            alpha=0.97,
        )

    # DoD node style (square marker so the anchor is obvious).
    dod_nodes = [u for u in node_list if role_map.get(u) == "DoD Agency" and u != focal_uid]
    if dod_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            nodelist=dod_nodes,
            node_size=node_size["DoD Agency"],
            node_shape="D",
            node_color=color_map["DoD Agency"],
            edgecolors="#2a0f1a",
            linewidths=1.1,
            ax=ax,
            alpha=1.0,
        )

    # Focal node style.
    if focal_uid in node_list:
        if removed_mode:
            nx.draw_networkx_nodes(
                graph,
                pos=pos,
                nodelist=[focal_uid],
                node_size=58,
                node_color="#ffffff",
                edgecolors="#C03A2B",
                linewidths=1.2,
                ax=ax,
                alpha=1.0,
            )
            x, y = pos[focal_uid]
            ax.plot([x - 0.07, x + 0.07], [y - 0.07, y + 0.07], color="#C03A2B", lw=1.2, zorder=6)
            ax.plot([x - 0.07, x + 0.07], [y + 0.07, y - 0.07], color="#C03A2B", lw=1.2, zorder=6)
        else:
            nx.draw_networkx_nodes(
                graph,
                pos=pos,
                nodelist=[focal_uid],
                node_size=node_size["Focal Chokepoint"],
                node_color=color_map["Focal Chokepoint"],
                edgecolors="#333333",
                linewidths=1.15,
                ax=ax,
                alpha=1.0,
            )

    # Mark disconnected/degraded primes in removal panel.
    if removed_mode and disconnected_primes:
        nodes = [u for u in disconnected_primes if u in node_list]
        if nodes:
            nx.draw_networkx_nodes(
                graph,
                pos=pos,
                nodelist=nodes,
                node_size=38,
                node_color="#f7f7f7",
                edgecolors="#C03A2B",
                linewidths=1.0,
                ax=ax,
                alpha=1.0,
            )
    if removed_mode and degraded_primes:
        nodes = [u for u in degraded_primes if u in node_list and u not in disconnected_primes]
        if nodes:
            nx.draw_networkx_nodes(
                graph,
                pos=pos,
                nodelist=nodes,
                node_size=38,
                node_color="#ffffff",
                edgecolors="#D98E04",
                linewidths=1.0,
                ax=ax,
                alpha=1.0,
            )

    # Labels for an optional selected subset.
    for uid in sorted(label_nodes):
        if uid not in pos:
            continue
        x, y = pos[uid]
        key = sum(ord(c) for c in uid) % 8
        offsets = [
            (0.085, 0.055, "left"),
            (0.085, -0.055, "left"),
            (-0.085, 0.055, "right"),
            (-0.085, -0.055, "right"),
            (0.11, 0.0, "left"),
            (-0.11, 0.0, "right"),
            (0.04, 0.09, "left"),
            (-0.04, -0.09, "right"),
        ]
        dx, dy, ha = offsets[key]
        label = safe_name(uid, name_map)
        ax.text(
            x + dx,
            y + dy,
            label,
            fontsize=6.4,
            ha=ha,
            va="center",
            color="#1f1f1f",
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
        )

    # Solid panel outline for print clarity.
    panel_outline = plt.Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax.transAxes,
        fill=False,
        edgecolor="#000000",
        linewidth=1.0,
        zorder=50,
        clip_on=False,
    )
    ax.add_patch(panel_outline)
    ax.margins(x=0.045, y=0.05)

    return


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    figs_root = Path(str(cfg.get("paths", {}).get("figs_root", "figs/chapter4/v2_fix01")))
    out_dir = Path(args.out_dir) if args.out_dir else (figs_root / "main_text")
    out_dir.mkdir(parents=True, exist_ok=True)

    m0_dir = out_root / snapshot / "m0"
    m4_dir = out_root / snapshot / "m4"
    m6_2_dir = out_root / snapshot / "m6_2"
    m0_7_dir = out_root / snapshot / "m0_7"

    edges_path = m0_dir / "edge_table_contract.parquet"
    corridor_path = m4_dir / "corridor_nodes_observed.parquet"
    strata_path = m6_2_dir / "node_impacts_stratified_observed.parquet"
    names_path = m0_7_dir / "node_semiconductor_lens_decisions.csv"
    top25_path = m0_7_dir / "top25_high_impact_semiconductor_value_chain_strict_observed.csv"
    prime_weights_path = Path(
        str(cfg.get("paths", {}).get("prime_weights", "artifacts/ch4/inputs/prime_weights.parquet"))
    )
    node_table_path = m0_dir / "node_table_contract.parquet"

    for p in [
        edges_path,
        corridor_path,
        strata_path,
        names_path,
        top25_path,
        prime_weights_path,
        node_table_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    edges = pd.read_parquet(edges_path)
    corridor = pd.read_parquet(corridor_path)
    strata = pd.read_parquet(strata_path)
    names_df = pd.read_csv(names_path)
    top25 = pd.read_csv(top25_path)
    prime_weights = pd.read_parquet(prime_weights_path)
    node_table = pd.read_parquet(node_table_path, columns=["analysis_uid", "entity_role", "name"])

    focal_uid = select_focal_uid(top25_path=top25_path, override=args.focal_uid)

    # Observed-view graph: disclosed OR observed shipping.
    is_disclosed = edges["is_disclosed"].fillna(0).astype(bool)
    is_observed_ship = edges["is_observed_ship"].fillna(0).astype(bool)
    e = edges[is_disclosed | is_observed_ship].copy()
    e = e.drop_duplicates(subset=["src_uid", "dst_uid"])
    G = nx.DiGraph()
    G.add_edges_from(zip(e["src_uid"].astype(str), e["dst_uid"].astype(str), strict=False))
    if focal_uid not in G:
        raise ValueError(f"focal uid not found in observed graph: {focal_uid}")

    # Node metadata join.
    meta = corridor.merge(
        strata[["analysis_uid", "h1_log_obligation_any_support", "impact_stratum"]],
        on="analysis_uid",
        how="left",
    ).drop_duplicates(subset=["analysis_uid"])
    meta = meta.set_index("analysis_uid", drop=False)

    name_map = (
        names_df[["analysis_uid", "name"]]
        .drop_duplicates(subset=["analysis_uid"])
        .set_index("analysis_uid")["name"]
        .astype(str)
        .to_dict()
    )
    node_name_fallback = (
        node_table[["analysis_uid", "name"]]
        .drop_duplicates(subset=["analysis_uid"])
        .set_index("analysis_uid")["name"]
        .astype(str)
        .to_dict()
    )
    for uid, nm in node_name_fallback.items():
        if (
            uid not in name_map
            or not str(name_map.get(uid, "")).strip()
            or str(name_map.get(uid, "")).strip().lower() == "nan"
        ):
            name_map[uid] = nm

    score_map = (
        strata[["analysis_uid", "h1_log_obligation_any_support"]]
        .drop_duplicates(subset=["analysis_uid"])
        .set_index("analysis_uid")["h1_log_obligation_any_support"]
        .fillna(0.0)
        .astype(float)
        .to_dict()
    )

    # Role anchors.
    primes = set(corridor.loc[corridor["is_tier1_prime"].fillna(False), "analysis_uid"].astype(str))
    semis = set(corridor.loc[corridor["is_semi_strict"].fillna(False), "analysis_uid"].astype(str))
    dod_nodes = set(
        node_table.loc[node_table["entity_role"] == "dod_component", "analysis_uid"].astype(str)
    )
    top25_set = set(top25["analysis_uid"].astype(str))
    prime_weight_map = (
        prime_weights[["analysis_uid", "weight"]]
        .drop_duplicates(subset=["analysis_uid"])
        .set_index("analysis_uid")["weight"]
        .fillna(0.0)
        .astype(float)
        .to_dict()
    )

    # Reachability from focal.
    G_rev = G.reverse(copy=False)
    max_hops = int(args.max_hops)
    k_shortest = max(1, int(args.k_shortest))
    up_hops = nx.single_source_shortest_path_length(G_rev, focal_uid, cutoff=max_hops)
    down_hops = nx.single_source_shortest_path_length(G, focal_uid, cutoff=max_hops)

    semi_candidates = [u for u in up_hops if u in semis and u != focal_uid]
    semi_candidates = sorted(
        semi_candidates,
        key=lambda u: (
            up_hops.get(u, 999),
            -int(u in top25_set),
            -float(score_map.get(u, 0.0)),
            u,
        ),
    )
    selected_semis = semi_candidates[: int(args.max_semis)]

    prime_candidates = [u for u in down_hops if u in primes and u != focal_uid]
    # Pre-filter by mission salience then score dependency on focal.
    prime_prefilter = sorted(
        prime_candidates,
        key=lambda u: (-float(prime_weight_map.get(u, 0.0)), down_hops.get(u, 999), u),
    )[: max(250, int(args.max_primes) * 20)]

    # Contract-seed links (DoD component -> prime vendor). Reverse these for visual flow prime -> DoD.
    contract_seed = edges.loc[
        edges["is_contract_seed"].fillna(0).astype(bool), ["src_uid", "dst_uid"]
    ].copy()
    contract_seed["src_uid"] = contract_seed["src_uid"].astype(str)
    contract_seed["dst_uid"] = contract_seed["dst_uid"].astype(str)
    contract_seed = contract_seed[
        contract_seed["src_uid"].isin(dod_nodes) & contract_seed["dst_uid"].isin(primes)
    ]
    dod_uid: str | None = None
    if not contract_seed.empty:
        overlap = contract_seed[contract_seed["dst_uid"].isin(prime_prefilter)]
        if not overlap.empty:
            dod_rank = overlap.groupby("src_uid")["dst_uid"].nunique().sort_values(ascending=False)
            dod_uid = str(dod_rank.index[0])
        else:
            dod_rank = (
                contract_seed.groupby("src_uid")["dst_uid"].nunique().sort_values(ascending=False)
            )
            dod_uid = str(dod_rank.index[0])

    G_no_focal_full = G.copy()
    if focal_uid in G_no_focal_full:
        G_no_focal_full.remove_node(focal_uid)

    dependent_prime_flag: dict[str, bool] = {}
    for p_uid in prime_prefilter:
        has_support = False
        for s_uid in selected_semis:
            if (
                s_uid in G_no_focal_full
                and p_uid in G_no_focal_full
                and nx.has_path(G_no_focal_full, s_uid, p_uid)
            ):
                has_support = True
                break
        dependent_prime_flag[p_uid] = not has_support

    dod_prime_set: set[str] = set()
    if dod_uid:
        dod_prime_set = set(
            contract_seed.loc[contract_seed["src_uid"] == dod_uid, "dst_uid"].astype(str)
        )

    prime_ranked = sorted(
        prime_prefilter,
        key=lambda u: (
            -int(dependent_prime_flag.get(u, False)),
            -int(u in dod_prime_set),
            -float(prime_weight_map.get(u, 0.0)),
            down_hops.get(u, 999),
            u,
        ),
    )
    selected_primes = prime_ranked[: int(args.max_primes)]

    if not selected_semis or not selected_primes:
        raise ValueError(
            f"insufficient case-study endpoints for focal={focal_uid}: "
            f"semis={len(selected_semis)}, primes={len(selected_primes)}"
        )

    def add_path(
        path_nodes: list[str], node_set: set[str], path_edges: set[tuple[str, str]]
    ) -> None:
        if not path_nodes or len(path_nodes) < 2:
            return
        node_set.update(path_nodes)
        path_edges.update((path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1))

    def add_k_shortest_paths(
        src: str, dst: str, node_set: set[str], path_edges: set[tuple[str, str]]
    ) -> None:
        try:
            gen = nx.shortest_simple_paths(G, source=src, target=dst)
            for kept, path in enumerate(gen, 1):
                if len(path) - 1 > max_hops:
                    break
                add_path(path, node_set, path_edges)
                if kept >= k_shortest:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            return

    # Build multi-path core around focal.
    node_set: set[str] = {focal_uid}
    path_edge_set: set[tuple[str, str]] = set()

    for s_uid in selected_semis:
        add_k_shortest_paths(s_uid, focal_uid, node_set, path_edge_set)
    for p_uid in selected_primes:
        add_k_shortest_paths(focal_uid, p_uid, node_set, path_edge_set)
    if dod_uid:
        node_set.add(dod_uid)
        for p_uid in selected_primes:
            if p_uid in dod_prime_set:
                path_edge_set.add((p_uid, dod_uid))

    # Add explicit semi->prime paths through focal for additional local structure.
    pair_semis = selected_semis[: min(6, len(selected_semis))]
    pair_primes = selected_primes[: min(6, len(selected_primes))]
    for s_uid in pair_semis:
        for p_uid in pair_primes:
            try:
                path = nx.shortest_path(G, source=s_uid, target=p_uid)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if focal_uid in path and len(path) - 1 <= (max_hops + 2):
                add_path(path, node_set, path_edge_set)

    target_min = int(args.min_nodes)
    target_max = int(args.max_nodes)
    augment_per_node = max(1, int(args.augment_neighbors_per_node))
    deg_map = dict(G.degree())

    def node_rank_key(u: str) -> tuple[Any, ...]:
        is_inter = bool(meta.at[u, "is_intermediary_corridor"]) if u in meta.index else False
        is_prime = bool(u in primes)
        is_semi = bool(u in semis)
        return (
            -int(u in top25_set),
            -int(is_inter),
            -int(is_prime),
            -int(is_semi),
            -float(score_map.get(u, 0.0)),
            -int(deg_map.get(u, 0)),
            u,
        )

    def augment_with_neighbors(seed_nodes: list[str], per_seed: int) -> int:
        candidates: set[str] = set()
        for u in seed_nodes:
            if u not in G:
                continue
            preds = sorted(G.predecessors(u), key=node_rank_key)[:per_seed]
            succs = sorted(G.successors(u), key=node_rank_key)[:per_seed]
            candidates.update(preds)
            candidates.update(succs)
        added = 0
        for u in sorted((c for c in candidates if c not in node_set), key=node_rank_key):
            node_set.add(u)
            added += 1
            if len(node_set) >= target_max:
                break
            if len(node_set) >= target_min:
                break
        return added

    protected = set(selected_semis) | set(selected_primes) | {focal_uid}

    def prune_to_max() -> None:
        if len(node_set) <= target_max:
            return
        removable = [u for u in node_set if u not in protected]
        removable = sorted(
            removable,
            key=lambda u: (
                int(u in top25_set),
                int(bool(meta.at[u, "is_intermediary_corridor"])) if u in meta.index else 0,
                float(score_map.get(u, 0.0)),
                int(deg_map.get(u, 0)),
                u,
            ),
        )
        for u in removable:
            if len(node_set) <= target_max:
                break
            node_set.remove(u)

    # First pass: expand from core corridor seeds.
    core_nodes = set(node_set)
    seed_core = sorted(core_nodes, key=node_rank_key)
    rounds = 0
    while len(node_set) < target_min and rounds < 4:
        added = augment_with_neighbors(seed_core, per_seed=augment_per_node)
        rounds += 1
        if added == 0:
            break
        seed_core = sorted(node_set, key=node_rank_key)

    # Second pass: widen context if still sparse.
    rounds = 0
    while len(node_set) < target_min and rounds < 4:
        added = augment_with_neighbors(
            sorted(node_set, key=node_rank_key), per_seed=max(1, augment_per_node // 2)
        )
        rounds += 1
        if added == 0:
            break

    # Hard prune if still too large.
    prune_to_max()

    node_list = sorted(node_set)

    # Edge set: all observed edges inside selected nodes; highlight core path edges.
    local_edges = [(u, v) for u, v in G.edges(node_list) if v in node_set]
    edge_set = set(local_edges)
    if dod_uid and dod_uid in node_set:
        for p_uid in selected_primes:
            if p_uid in node_set and p_uid in dod_prime_set:
                edge_set.add((p_uid, dod_uid))
    highlight_edge_set = {e for e in path_edge_set if e[0] in node_set and e[1] in node_set}

    # Optional densification: if local graph remains too sparse, add a few more neighbors.
    density_rounds = 0
    while len(edge_set) < 100 and len(node_set) < target_max and density_rounds < 3:
        before_nodes = len(node_set)
        augment_with_neighbors(
            sorted(node_set, key=node_rank_key), per_seed=max(1, augment_per_node // 2)
        )
        if len(node_set) == before_nodes:
            break
        node_list = sorted(node_set)
        local_edges = [(u, v) for u, v in G.edges(node_list) if v in node_set]
        edge_set = set(local_edges)
        if dod_uid and dod_uid in node_set:
            for p_uid in selected_primes:
                if p_uid in node_set and p_uid in dod_prime_set:
                    edge_set.add((p_uid, dod_uid))
        highlight_edge_set = {e for e in path_edge_set if e[0] in node_set and e[1] in node_set}
        density_rounds += 1

    # Final hard cap after densification.
    prune_to_max()
    node_list = sorted(node_set)
    local_edges = [(u, v) for u, v in G.edges(node_list) if v in node_set]
    edge_set = set(local_edges)
    if dod_uid and dod_uid in node_set:
        for p_uid in selected_primes:
            if p_uid in node_set and p_uid in dod_prime_set:
                edge_set.add((p_uid, dod_uid))
    highlight_edge_set = {e for e in path_edge_set if e[0] in node_set and e[1] in node_set}

    G_local = nx.DiGraph()
    G_local.add_nodes_from(node_list)
    G_local.add_edges_from(sorted(edge_set))

    # Role and layout.
    role_map = {
        u: build_role(
            u,
            focal_uid=focal_uid,
            dod_uid=dod_uid,
            meta=meta,
            semis=semis,
            primes=primes,
        )
        for u in node_list
    }
    pos = build_layout(
        graph=G_local,
        node_list=node_list,
        role_map=role_map,
        dod_uid=dod_uid,
        seed=int(cfg.get("random", {}).get("python_seed", 7)),
    )

    # Labels intentionally suppressed for readability in dense network rendering.
    label_nodes: set[str] = set()

    # Removal panel graph (focal node removed from connectivity, kept as visual marker).
    G_removed = nx.DiGraph()
    G_removed.add_nodes_from(node_list)
    G_removed.add_edges_from([(u, v) for (u, v) in edge_set if u != focal_uid and v != focal_uid])
    highlight_removed = {e for e in highlight_edge_set if e[0] != focal_uid and e[1] != focal_uid}

    # Disconnected/degraded primes from selected semis after focal removal in the local case-study graph.
    def best_path_len(graph: nx.DiGraph, src_nodes: list[str], dst: str) -> float:
        best = math.inf
        if dst not in graph:
            return best
        for s_uid in src_nodes:
            if s_uid not in graph:
                continue
            try:
                d = nx.shortest_path_length(graph, source=s_uid, target=dst)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if d < best:
                best = float(d)
        return best

    baseline_best: dict[str, float] = {
        p_uid: best_path_len(G_local, selected_semis, p_uid) for p_uid in selected_primes
    }
    baseline_support_count: dict[str, int] = {}
    for p_uid in selected_primes:
        c = 0
        for s_uid in selected_semis:
            if s_uid in G_local and p_uid in G_local and nx.has_path(G_local, s_uid, p_uid):
                c += 1
        baseline_support_count[p_uid] = c

    G_no_focal = G_removed.copy()
    if focal_uid in G_no_focal:
        G_no_focal.remove_node(focal_uid)
    disconnected_primes: set[str] = set()
    degraded_primes: set[str] = set()
    for p_uid in selected_primes:
        b = baseline_best.get(p_uid, math.inf)
        a = best_path_len(G_no_focal, selected_semis, p_uid)
        a_support = 0
        for s_uid in selected_semis:
            if (
                s_uid in G_no_focal
                and p_uid in G_no_focal
                and nx.has_path(G_no_focal, s_uid, p_uid)
            ):
                a_support += 1
        if math.isinf(a):
            disconnected_primes.add(p_uid)
            continue
        if a_support < baseline_support_count.get(p_uid, 0):
            degraded_primes.add(p_uid)
            continue
        if math.isfinite(b) and a > b:
            degraded_primes.add(p_uid)

    # Draw figure.
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.0), constrained_layout=False)
    draw_panel(
        ax=axes[0],
        graph=G_local,
        node_list=node_list,
        pos=pos,
        role_map=role_map,
        highlight_edges=highlight_edge_set,
        label_nodes=label_nodes,
        name_map=name_map,
        focal_uid=focal_uid,
        removed_mode=False,
        disconnected_primes=set(),
        degraded_primes=set(),
        max_context_edges=int(args.max_context_edges),
    )
    draw_panel(
        ax=axes[1],
        graph=G_removed,
        node_list=node_list,
        pos=pos,
        role_map=role_map,
        highlight_edges=highlight_removed,
        label_nodes=label_nodes,
        name_map=name_map,
        focal_uid=focal_uid,
        removed_mode=True,
        disconnected_primes=disconnected_primes,
        degraded_primes=degraded_primes,
        max_context_edges=int(args.max_context_edges),
    )

    axes[0].set_title("DoD-Linked Local Network (Baseline)", fontsize=11, pad=8)
    axes[1].set_title("After Focal Removal", fontsize=11, pad=8)

    # Compact legend (single shared).
    from matplotlib.lines import Line2D

    dod_label = f"DoD agency ({safe_name(dod_uid, name_map)})" if dod_uid else "DoD agency"
    focal_label = f"Focal chokepoint ({safe_name(focal_uid, name_map)})"
    legend_items = [
        Line2D([0, 1], [0, 0], color="#4f4f4f", lw=1.15, label="Highlighted corridor paths"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#8B1E3F",
            markeredgecolor="#2a0f1a",
            markersize=8,
            label=dod_label,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#F2BE5C",
            markeredgecolor="#333333",
            markersize=8,
            label=focal_label,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#4E79A7",
            markeredgecolor="#4c4c4c",
            markersize=8,
            label="Prime endpoint",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#59A14F",
            markeredgecolor="#4c4c4c",
            markersize=8,
            label="Semiconductor node",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9C755F",
            markeredgecolor="#4c4c4c",
            markersize=8,
            label="Corridor intermediary",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#BAB0AC",
            markeredgecolor="#4c4c4c",
            markersize=8,
            label="Prime adjacent",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#ffffff",
            markeredgecolor="#D98E04",
            markersize=8,
            label="Prime degraded after removal",
        ),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8.0,
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.3,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.91, bottom=0.13, wspace=0.02)

    out_pdf = out_dir / "fig_4_4_case_study_network_observed.pdf"
    out_svg = out_dir / "fig_4_4_case_study_network_observed.svg"
    out_png = out_dir / "fig_4_4_case_study_network_observed.png"
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_png, format="png", dpi=600)
    plt.close(fig)

    # Export node/edge selections for auditability.
    node_rows = []
    for uid in node_list:
        node_rows.append(
            {
                "analysis_uid": uid,
                "name": name_map.get(uid, ""),
                "role": role_map.get(uid, ""),
                "x": pos[uid][0],
                "y": pos[uid][1],
                "is_focal": uid == focal_uid,
                "is_selected_semi": uid in set(selected_semis),
                "is_selected_prime": uid in set(selected_primes),
                "is_top25_strict": uid in top25_set,
                "is_prime_disconnected_after_removal": uid in disconnected_primes,
                "is_prime_degraded_after_removal": uid in degraded_primes,
            }
        )
    node_csv = out_dir / "fig_4_4_case_study_network_nodes.csv"
    pd.DataFrame(node_rows).to_csv(node_csv, index=False)

    edge_rows = [{"src_uid": u, "dst_uid": v} for (u, v) in sorted(edge_set)]
    edge_csv = out_dir / "fig_4_4_case_study_network_edges.csv"
    pd.DataFrame(edge_rows).to_csv(edge_csv, index=False)

    spec = {
        "figure_name": "fig_4_4_case_study_network_observed",
        "view": "observed",
        "focal_uid": focal_uid,
        "focal_name": safe_name(focal_uid, name_map),
        "selection_params": {
            "max_semis": int(args.max_semis),
            "max_primes": int(args.max_primes),
            "min_nodes": int(args.min_nodes),
            "max_nodes": int(args.max_nodes),
            "max_hops": int(args.max_hops),
            "k_shortest": int(args.k_shortest),
            "augment_neighbors_per_node": int(args.augment_neighbors_per_node),
            "max_context_edges": int(args.max_context_edges),
        },
        "counts": {
            "n_nodes": len(node_list),
            "n_edges": len(edge_set),
            "n_highlight_edges": len(highlight_edge_set),
            "n_selected_semis": len(selected_semis),
            "n_selected_primes": len(selected_primes),
            "n_selected_dod_agency": (1 if dod_uid and dod_uid in node_set else 0),
            "n_disconnected_primes_after_removal": len(disconnected_primes),
            "n_degraded_primes_after_removal": len(degraded_primes),
        },
        "selected_dod_agency": dod_uid,
        "selected_dod_agency_name": safe_name(dod_uid, name_map) if dod_uid else None,
        "selected_semis": selected_semis,
        "selected_primes": selected_primes,
        "output_files": {
            "pdf": str(out_pdf),
            "svg": str(out_svg),
            "png": str(out_png),
            "nodes_csv": str(node_csv),
            "edges_csv": str(edge_csv),
        },
    }
    spec_json = out_dir / "fig_4_4_case_study_network_observed_spec.json"
    spec_json.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    run_meta = {
        "module": "export_fig_4_4_case_study_network",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "inputs": {
            "edges_parquet": str(edges_path),
            "edges_sha256": file_sha256(edges_path),
            "corridor_nodes_parquet": str(corridor_path),
            "corridor_nodes_sha256": file_sha256(corridor_path),
            "strata_parquet": str(strata_path),
            "strata_sha256": file_sha256(strata_path),
            "names_csv": str(names_path),
            "names_sha256": file_sha256(names_path),
            "top25_csv": str(top25_path),
            "top25_sha256": file_sha256(top25_path),
            "prime_weights_parquet": str(prime_weights_path),
            "prime_weights_sha256": file_sha256(prime_weights_path),
        },
        "outputs": {
            "pdf": str(out_pdf),
            "svg": str(out_svg),
            "png": str(out_png),
            "nodes_csv": str(node_csv),
            "edges_csv": str(edge_csv),
            "spec_json": str(spec_json),
        },
    }
    run_meta_json = out_dir / "fig_4_4_case_study_network_observed_run_metadata.json"
    run_meta_json.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"[done] wrote {out_pdf}")
    print(f"[done] wrote {out_svg}")
    print(f"[done] wrote {out_png}")
    print(f"[done] wrote {node_csv}")
    print(f"[done] wrote {edge_csv}")
    print(f"[done] wrote {spec_json}")
    print(f"[done] wrote {run_meta_json}")


if __name__ == "__main__":
    main()
