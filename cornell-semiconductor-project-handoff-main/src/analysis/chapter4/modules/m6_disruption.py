#!/usr/bin/env python3
"""Module 6: disruption severity analysis."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 6 disruption analysis")
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


def build_view_edges(
    edges: pd.DataFrame, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = ["src_idx", "dst_idx", "is_disclosed", "is_observed_ship", "is_predicted"]
    view_edges = edges.loc[mask, cols].copy()
    view_edges = view_edges[view_edges["src_idx"] != view_edges["dst_idx"]]
    view_edges = (
        view_edges.groupby(["src_idx", "dst_idx"], as_index=False)[
            ["is_disclosed", "is_observed_ship", "is_predicted"]
        ]
        .max()
        .sort_values(["src_idx", "dst_idx"], ascending=[True, True], kind="mergesort")
    )
    src = view_edges["src_idx"].to_numpy(np.int32, copy=False)
    dst = view_edges["dst_idx"].to_numpy(np.int32, copy=False)
    is_disclosed = view_edges["is_disclosed"].astype(bool).to_numpy()
    is_observed = view_edges["is_observed_ship"].astype(bool).to_numpy()
    is_predicted = view_edges["is_predicted"].astype(bool).to_numpy()
    return src, dst, is_disclosed, is_observed, is_predicted


def rank_desc(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values.astype(float))
        .rank(method="min", ascending=False)
        .astype(np.int32)
        .to_numpy()
    )


def multi_source_weighted_distance(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    edge_cost: np.ndarray,
    source_indices: np.ndarray,
) -> np.ndarray:
    super_source = n_nodes
    rows = np.concatenate([src, np.full(len(source_indices), super_source, dtype=np.int32)])
    cols = np.concatenate([dst, source_indices.astype(np.int32, copy=False)])
    data = np.concatenate(
        [edge_cost.astype(np.float64, copy=False), np.zeros(len(source_indices), dtype=np.float64)]
    )
    matrix = csr_matrix((data, (rows, cols)), shape=(n_nodes + 1, n_nodes + 1))
    dist = dijkstra(matrix, directed=True, indices=super_source, unweighted=False).astype(
        np.float64, copy=False
    )
    return dist[:n_nodes]


def compute_source_reach_and_scc(
    n_nodes: int,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(src_idx) == 0:
        labels = np.arange(n_nodes, dtype=np.int32)
        sizes = np.ones(n_nodes, dtype=np.int32)
        reach = np.zeros(n_nodes, dtype=np.int32)
        if len(source_indices) > 0:
            reach[source_indices] = 1
        return reach, labels, sizes

    matrix = csr_matrix(
        (np.ones(len(src_idx), dtype=np.int8), (src_idx, dst_idx)), shape=(n_nodes, n_nodes)
    )
    n_comp, labels = connected_components(
        matrix, directed=True, connection="strong", return_labels=True
    )
    labels = labels.astype(np.int32, copy=False)
    comp_sizes = np.bincount(labels, minlength=n_comp).astype(np.int32, copy=False)

    comp_src = labels[src_idx]
    comp_dst = labels[dst_idx]
    cross = comp_src != comp_dst
    if int(cross.sum()) > 0:
        condensed = np.unique(np.stack([comp_src[cross], comp_dst[cross]], axis=1), axis=0)
        cu = condensed[:, 0].astype(np.int32, copy=False)
        cv = condensed[:, 1].astype(np.int32, copy=False)
    else:
        cu = np.array([], dtype=np.int32)
        cv = np.array([], dtype=np.int32)

    succ: list[list[int]] = [[] for _ in range(n_comp)]
    indeg = np.zeros(n_comp, dtype=np.int32)
    for u, v in zip(cu.tolist(), cv.tolist(), strict=False):
        succ[u].append(v)
        indeg[v] += 1

    topo: list[int] = []
    queue = list(np.flatnonzero(indeg == 0))
    head = 0
    while head < len(queue):
        node = int(queue[head])
        head += 1
        topo.append(node)
        for nxt in succ[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(int(nxt))
    if len(topo) != n_comp:
        raise RuntimeError("condensation graph topological sort failed")

    source_pos = np.full(n_nodes, -1, dtype=np.int32)
    source_pos[source_indices] = np.arange(len(source_indices), dtype=np.int32)
    comp_source_bits: list[int] = [0] * n_comp
    for idx in source_indices.tolist():
        bit = 1 << int(source_pos[idx])
        comp_source_bits[int(labels[idx])] |= bit

    comp_reach_bits = comp_source_bits.copy()
    for comp in topo:
        bits = comp_reach_bits[comp]
        if bits == 0:
            continue
        for nxt in succ[comp]:
            comp_reach_bits[nxt] |= bits

    comp_reach_count = np.fromiter(
        (int(bits.bit_count()) for bits in comp_reach_bits), dtype=np.int32, count=n_comp
    )
    node_reach_count = comp_reach_count[labels]
    return node_reach_count, labels, comp_sizes


@dataclass
class GraphContext:
    n_nodes: int
    src: np.ndarray
    dst: np.ndarray
    edge_cost: np.ndarray
    prime_indices: np.ndarray
    semi_indices: np.ndarray
    is_firm: np.ndarray
    node_uids: np.ndarray
    node_names: np.ndarray
    role: np.ndarray
    prime_incoming: list[list[tuple[int, float]]]
    baseline_total_support: float
    baseline_prime_dist: np.ndarray
    baseline_avg_len: float
    baseline_finite_mask: np.ndarray
    baseline_share_single_point_supported: float
    baseline_share_no_support: float
    geodesic_tol: float


def build_prime_incoming(
    n_nodes: int, src: np.ndarray, dst: np.ndarray, edge_cost: np.ndarray, prime_indices: np.ndarray
) -> list[list[tuple[int, float]]]:
    incoming: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    prime_set = set(prime_indices.tolist())
    for u, v, cost in zip(src.tolist(), dst.tolist(), edge_cost.tolist(), strict=False):
        if v in prime_set:
            incoming[v].append((u, float(cost)))
    return incoming


def evaluate_metrics(
    context: GraphContext,
    removed_nodes: np.ndarray,
    need_h2: bool,
    need_h3: bool,
) -> dict[str, Any]:
    alive = np.ones(context.n_nodes, dtype=bool)
    if len(removed_nodes) > 0:
        alive[removed_nodes] = False

    edge_mask = alive[context.src] & alive[context.dst]
    src_active = context.src[edge_mask]
    dst_active = context.dst[edge_mask]
    cost_active = context.edge_cost[edge_mask]

    semi_active = context.semi_indices[alive[context.semi_indices]]
    semi_reach_count, scc_labels, _ = compute_source_reach_and_scc(
        n_nodes=context.n_nodes,
        src_idx=src_active,
        dst_idx=dst_active,
        source_indices=semi_active.astype(np.int32, copy=False),
    )

    prime_support = semi_reach_count[context.prime_indices].astype(np.float64, copy=False)
    total_support = float(prime_support.sum())
    h1_reach_loss = (
        float((context.baseline_total_support - total_support) / context.baseline_total_support)
        if context.baseline_total_support > 0
        else np.nan
    )

    result: dict[str, Any] = {
        "h1_reach_loss": h1_reach_loss,
        "support_total": total_support,
        "removed_nodes_count": len(removed_nodes),
    }

    dist: np.ndarray | None = None
    prime_dist: np.ndarray | None = None
    if need_h2 or need_h3:
        if len(semi_active) == 0:
            dist = np.full(context.n_nodes, np.inf, dtype=np.float64)
        else:
            dist = multi_source_weighted_distance(
                n_nodes=context.n_nodes,
                src=src_active,
                dst=dst_active,
                edge_cost=cost_active,
                source_indices=semi_active,
            )
        prime_dist = dist[context.prime_indices]
        finite_current = np.isfinite(prime_dist)
        finite_overlap = context.baseline_finite_mask & finite_current
        if int(finite_overlap.sum()) > 0:
            avg_len = float(np.mean(prime_dist[finite_overlap]))
        else:
            avg_len = np.nan
        disconnect_share = (
            float(np.mean(context.baseline_finite_mask & (~finite_current)))
            if int(context.baseline_finite_mask.sum()) > 0
            else np.nan
        )
        h2_growth = (
            float(avg_len - context.baseline_avg_len)
            if np.isfinite(avg_len) and np.isfinite(context.baseline_avg_len)
            else np.nan
        )
        result.update(
            {
                "h2_avg_path_len": avg_len,
                "h2_path_growth": h2_growth,
                "h2_disconnect_share": disconnect_share,
            }
        )

    if need_h3:
        if prime_dist is None or dist is None:
            raise RuntimeError("distance outputs must be available when need_h3 is True")
        no_support = 0
        single_point = 0
        supported = 0
        for prime_node in context.prime_indices.tolist():
            support_count = int(semi_reach_count[prime_node])
            if support_count <= 0:
                no_support += 1
                continue
            supported += 1

            candidates = []
            for pred, cost in context.prime_incoming[prime_node]:
                if not alive[pred]:
                    continue
                if not context.is_firm[pred]:
                    continue
                if semi_reach_count[pred] <= 0:
                    continue
                candidates.append((pred, cost))

            entry_branch_count = len(candidates)
            entry_scc_count = len({int(scc_labels[pred]) for pred, _ in candidates})

            geodesic_entry_count = 0
            prime_distance = float(dist[prime_node])
            if np.isfinite(prime_distance):
                for pred, cost in candidates:
                    pred_distance = float(dist[pred])
                    if np.isfinite(pred_distance) and np.isclose(
                        pred_distance + cost, prime_distance, atol=context.geodesic_tol, rtol=0.0
                    ):
                        geodesic_entry_count += 1

            redundancy_proxy = int(
                min(support_count, entry_branch_count, entry_scc_count, geodesic_entry_count)
            )
            if redundancy_proxy <= 1:
                single_point += 1

        share_no_support = float(no_support / len(context.prime_indices))
        share_single_point_supported = float(single_point / supported) if supported > 0 else np.nan
        result.update(
            {
                "h3_share_no_support": share_no_support,
                "h3_share_single_point_supported": share_single_point_supported,
                "h3_redundancy_collapse": float(
                    share_single_point_supported - context.baseline_share_single_point_supported
                )
                if np.isfinite(share_single_point_supported)
                else np.nan,
                "h3_no_support_increase": float(
                    share_no_support - context.baseline_share_no_support
                ),
            }
        )

    return result


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
    rng = random.Random(global_seed)  # nosec B311 -- seeded for reproducible sampling, not cryptographic

    m6_cfg = cfg.get("m6", cfg.get("m6_pilot", {}))
    primary_view = str(m6_cfg.get("primary_view", "full"))
    candidate_pool_n = int(m6_cfg.get("candidate_pool_n", 500))
    deep_eval_n = int(m6_cfg.get("deep_eval_n", 100))
    random_repeats = int(m6_cfg.get("random_repeats", 10))
    removal_fracs = m6_cfg.get("removal_fracs", [0.001, 0.005, 0.01])
    removal_fracs = [float(x) for x in removal_fracs]
    interdiction_k = [int(x) for x in m6_cfg.get("interdiction_k", [5, 10])]
    interdiction_shortlist_n = int(m6_cfg.get("interdiction_shortlist_n", 50))
    geodesic_tol = float(
        m6_cfg.get("geodesic_tolerance", cfg.get("m5", {}).get("geodesic_tolerance", 1e-9))
    )
    pred_only_cost = float(
        m6_cfg.get(
            "predicted_only_edge_cost", cfg.get("m5", {}).get("predicted_only_edge_cost", 3.0)
        )
    )

    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2")))
    m0_dir = out_root / snapshot / "m0"
    m2_dir = out_root / snapshot / "m2"
    m3_dir = out_root / snapshot / "m3"
    m4r_dir = out_root / snapshot / "m4_refine"
    m5_dir = out_root / snapshot / "m5"
    m6_dir = out_root / snapshot / "m6"
    m6_dir.mkdir(parents=True, exist_ok=True)

    node_path = m0_dir / "node_table_contract.parquet"
    edge_path = m0_dir / "edge_table_contract.parquet"
    for path in [node_path, edge_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)

    node_uids = nodes["analysis_uid"].astype(str).to_numpy()
    uid_to_idx = {uid: i for i, uid in enumerate(node_uids.tolist())}
    n_nodes = len(node_uids)

    src_map = edges["src_uid"].astype(str).map(uid_to_idx)
    dst_map = edges["dst_uid"].astype(str).map(uid_to_idx)
    valid = src_map.notna() & dst_map.notna()
    edges = edges.loc[valid].copy()
    edges["src_idx"] = src_map.loc[valid].astype(np.int32).to_numpy()
    edges["dst_idx"] = dst_map.loc[valid].astype(np.int32).to_numpy()

    views = cfg.get("views", {})
    if primary_view not in views:
        raise ValueError(f"primary view {primary_view} not found in config.views")
    include_any = views[primary_view].get("include_any", [])
    view_mask = get_view_mask(edges, include_any)
    src, dst, is_disclosed, is_observed, is_predicted = build_view_edges(edges, view_mask)
    pred_only = is_predicted & (~is_disclosed) & (~is_observed)
    edge_cost = np.where(pred_only, pred_only_cost, 1.0).astype(np.float64, copy=False)

    role = nodes["entity_role"].astype(str).to_numpy()
    is_prime = role == "prime_vendor"
    is_dod = role == "dod_component"
    is_firm = role == "firm"
    is_semi = (
        nodes["is_semi_strict"].fillna(False).to_numpy(bool)
        if "is_semi_strict" in nodes.columns
        else np.zeros(n_nodes, dtype=bool)
    )
    node_names = (
        nodes["name"].fillna("").astype(str).to_numpy()
        if "name" in nodes.columns
        else np.array([""] * n_nodes, dtype=object)
    )

    prime_indices = np.flatnonzero(is_prime).astype(np.int32, copy=False)
    semi_indices = np.flatnonzero(is_semi).astype(np.int32, copy=False)
    removable_mask = is_firm & (~is_prime) & (~is_dod)
    removable_indices = np.flatnonzero(removable_mask).astype(np.int32, copy=False)
    removable_uids = node_uids[removable_indices]

    if len(prime_indices) == 0:
        raise ValueError("No primes found for M6")
    if len(semi_indices) == 0:
        raise ValueError("No strict semis found for M6")

    # Candidate pool signals
    m2 = pd.read_parquet(m2_dir / f"node_centralities_{primary_view}.parquet")
    m3 = pd.read_parquet(m3_dir / f"node_prime_reach_{primary_view}.parquet")
    m4r = pd.read_parquet(m4r_dir / f"corridor_nodes_refined_{primary_view}.parquet")
    m5 = pd.read_csv(m5_dir / f"prime_redundancy_{primary_view}.csv")

    candidate = pd.DataFrame({"analysis_uid": removable_uids})
    candidate["node_idx"] = candidate["analysis_uid"].map(uid_to_idx).astype(np.int32)
    candidate["name"] = candidate["analysis_uid"].map(
        dict(zip(node_uids.tolist(), node_names.tolist(), strict=False))
    )
    candidate["entity_role"] = "firm"

    m3_map = m3.set_index("analysis_uid")["prime_reach_count"].to_dict()
    m4_map = m4r.set_index("analysis_uid")["bottleneck_score_refined"].to_dict()
    m2_pr_map = m2.set_index("analysis_uid")["pagerank"].to_dict()
    m2_bw_map = m2.set_index("analysis_uid")["betweenness_approx"].to_dict()

    indeg = np.zeros(n_nodes, dtype=np.int32)
    outdeg = np.zeros(n_nodes, dtype=np.int32)
    np.add.at(indeg, dst, 1)
    np.add.at(outdeg, src, 1)
    degree_total = indeg + outdeg

    candidate["signal_m3_prime_reach"] = (
        candidate["analysis_uid"].map(m3_map).fillna(0.0).astype(float)
    )
    candidate["signal_m4_bottleneck"] = (
        candidate["analysis_uid"].map(m4_map).fillna(0.0).astype(float)
    )
    candidate["signal_pagerank"] = (
        candidate["analysis_uid"].map(m2_pr_map).fillna(0.0).astype(float)
    )
    candidate["signal_betweenness"] = (
        candidate["analysis_uid"].map(m2_bw_map).fillna(0.0).astype(float)
    )
    candidate["signal_degree_total"] = degree_total[candidate["node_idx"].to_numpy()]

    single_point_primes = set(
        m5.loc[(m5["is_single_point_exposed"]) & (m5["semi_support_count"] > 0), "analysis_uid"]
        .astype(str)
        .tolist()
    )
    edge_df = pd.DataFrame({"src_uid": node_uids[src], "dst_uid": node_uids[dst]})
    feeders = (
        edge_df[
            edge_df["dst_uid"].isin(single_point_primes)
            & edge_df["src_uid"].isin(set(removable_uids.tolist()))
        ]
        .drop_duplicates()
        .groupby("src_uid")
        .size()
        .to_dict()
    )
    candidate["signal_m5_single_point_feeders"] = (
        candidate["analysis_uid"].map(feeders).fillna(0.0).astype(float)
    )

    signal_cols = [
        "signal_m3_prime_reach",
        "signal_m4_bottleneck",
        "signal_pagerank",
        "signal_betweenness",
        "signal_m5_single_point_feeders",
        "signal_degree_total",
    ]
    for col in signal_cols:
        candidate[f"{col}_rank"] = rank_desc(candidate[col].to_numpy(np.float64))
        denom = max(len(candidate) - 1, 1)
        candidate[f"{col}_norm"] = 1.0 - ((candidate[f"{col}_rank"] - 1) / denom)

    top_each = int(m6_cfg.get("candidate_signal_top_n", 200))
    selected_flags: list[str] = []
    for col in signal_cols:
        flag = f"selected_by_{col.replace('signal_', '')}"
        selected_flags.append(flag)
        candidate[flag] = False
        top_uids = (
            candidate.sort_values([col, "analysis_uid"], ascending=[False, True])
            .head(top_each)["analysis_uid"]
            .tolist()
        )
        candidate.loc[candidate["analysis_uid"].isin(set(top_uids)), flag] = True

    candidate["selected_by_any_signal"] = candidate[selected_flags].any(axis=1)
    candidate["aggregate_score"] = candidate[[f"{col}_norm" for col in signal_cols]].sum(axis=1)

    union = candidate[candidate["selected_by_any_signal"]].copy()
    if len(union) > candidate_pool_n:
        pool = (
            union.sort_values(["aggregate_score", "analysis_uid"], ascending=[False, True])
            .head(candidate_pool_n)
            .copy()
        )
    else:
        need = candidate_pool_n - len(union)
        fill = (
            candidate[~candidate["analysis_uid"].isin(set(union["analysis_uid"].tolist()))]
            .sort_values(["aggregate_score", "analysis_uid"], ascending=[False, True])
            .head(max(need, 0))
        )
        pool = pd.concat([union, fill], ignore_index=True)
    pool = pool.sort_values(
        ["aggregate_score", "analysis_uid"], ascending=[False, True]
    ).reset_index(drop=True)
    pool["candidate_rank"] = np.arange(1, len(pool) + 1, dtype=np.int32)
    pool_out = m6_dir / f"candidate_pool_{primary_view}.csv"
    pool.to_csv(pool_out, index=False)

    # Baseline context
    t0 = time.perf_counter()
    baseline_support, baseline_scc_labels, _ = compute_source_reach_and_scc(
        n_nodes=n_nodes,
        src_idx=src,
        dst_idx=dst,
        source_indices=semi_indices,
    )
    baseline_total_support = float(baseline_support[prime_indices].astype(np.float64).sum())
    baseline_dist = multi_source_weighted_distance(
        n_nodes=n_nodes,
        src=src,
        dst=dst,
        edge_cost=edge_cost,
        source_indices=semi_indices,
    )
    baseline_prime_dist = baseline_dist[prime_indices]
    baseline_finite_mask = np.isfinite(baseline_prime_dist)
    baseline_avg_len = (
        float(np.mean(baseline_prime_dist[baseline_finite_mask]))
        if int(baseline_finite_mask.sum()) > 0
        else np.nan
    )

    prime_incoming = build_prime_incoming(
        n_nodes=n_nodes,
        src=src,
        dst=dst,
        edge_cost=edge_cost,
        prime_indices=prime_indices,
    )

    baseline_no_support = 0
    baseline_single_point = 0
    baseline_supported = 0
    for prime_node in prime_indices.tolist():
        support_count = int(baseline_support[prime_node])
        if support_count <= 0:
            baseline_no_support += 1
            continue
        baseline_supported += 1
        candidates = [
            (pred, cost)
            for pred, cost in prime_incoming[prime_node]
            if is_firm[pred] and baseline_support[pred] > 0
        ]
        entry_branch_count = len(candidates)
        entry_scc_count = len({int(baseline_scc_labels[pred]) for pred, _ in candidates})
        prime_distance = float(baseline_dist[prime_node])
        geodesic_entry_count = 0
        if np.isfinite(prime_distance):
            for pred, cost in candidates:
                pred_distance = float(baseline_dist[pred])
                if np.isfinite(pred_distance) and np.isclose(
                    pred_distance + cost, prime_distance, atol=geodesic_tol, rtol=0.0
                ):
                    geodesic_entry_count += 1
        redundancy_proxy = int(
            min(support_count, entry_branch_count, entry_scc_count, geodesic_entry_count)
        )
        if redundancy_proxy <= 1:
            baseline_single_point += 1

    baseline_share_no_support = float(baseline_no_support / len(prime_indices))
    baseline_share_single_point_supported = (
        float(baseline_single_point / baseline_supported) if baseline_supported > 0 else np.nan
    )

    context = GraphContext(
        n_nodes=n_nodes,
        src=src,
        dst=dst,
        edge_cost=edge_cost,
        prime_indices=prime_indices,
        semi_indices=semi_indices,
        is_firm=is_firm,
        node_uids=node_uids,
        node_names=node_names,
        role=role,
        prime_incoming=prime_incoming,
        baseline_total_support=baseline_total_support,
        baseline_prime_dist=baseline_prime_dist,
        baseline_avg_len=baseline_avg_len,
        baseline_finite_mask=baseline_finite_mask,
        baseline_share_single_point_supported=baseline_share_single_point_supported,
        baseline_share_no_support=baseline_share_no_support,
        geodesic_tol=geodesic_tol,
    )
    baseline_runtime = time.perf_counter() - t0

    baseline_metrics = evaluate_metrics(
        context=context, removed_nodes=np.array([], dtype=np.int32), need_h2=True, need_h3=True
    )
    baseline_metrics.update(
        {
            "n_nodes_total": int(n_nodes),
            "n_edges_view": len(src),
            "n_primes": len(prime_indices),
            "n_semis": len(semi_indices),
            "n_removable_firms": len(removable_indices),
            "baseline_runtime_seconds": baseline_runtime,
        }
    )
    baseline_out = m6_dir / f"baseline_metrics_{primary_view}.json"
    baseline_out.write_text(json.dumps(baseline_metrics, indent=2))

    # M6B: single-node H1 screening
    candidate_indices = pool["node_idx"].astype(np.int32).to_numpy()
    h1_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for idx, node_idx in enumerate(candidate_indices.tolist(), start=1):
        metrics = evaluate_metrics(
            context=context,
            removed_nodes=np.array([node_idx], dtype=np.int32),
            need_h2=False,
            need_h3=False,
        )
        h1_rows.append(
            {
                "view": primary_view,
                "analysis_uid": node_uids[node_idx],
                "name": node_names[node_idx],
                "entity_role": role[node_idx],
                "h1_reach_loss": metrics["h1_reach_loss"],
                "support_total": metrics["support_total"],
            }
        )
        if idx % 50 == 0:
            print(f"[m6] H1 screened {idx}/{len(candidate_indices)}")
    h1_df = (
        pd.DataFrame(h1_rows)
        .sort_values(["h1_reach_loss", "analysis_uid"], ascending=[False, True])
        .reset_index(drop=True)
    )
    h1_df["rank_h1_reach_loss"] = np.arange(1, len(h1_df) + 1, dtype=np.int32)
    h1_out = m6_dir / f"node_single_removal_screen_h1_{primary_view}.csv"
    h1_df.to_csv(h1_out, index=False)
    h1_runtime = time.perf_counter() - t0

    # M6C: deep re-eval top N
    deep_uids = h1_df.head(deep_eval_n)["analysis_uid"].tolist()
    deep_indices = np.array([uid_to_idx[uid] for uid in deep_uids], dtype=np.int32)
    deep_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for idx, node_idx in enumerate(deep_indices.tolist(), start=1):
        metrics = evaluate_metrics(
            context=context,
            removed_nodes=np.array([node_idx], dtype=np.int32),
            need_h2=True,
            need_h3=True,
        )
        deep_rows.append(
            {
                "view": primary_view,
                "analysis_uid": node_uids[node_idx],
                "name": node_names[node_idx],
                "entity_role": role[node_idx],
                **metrics,
            }
        )
        if idx % 25 == 0:
            print(f"[m6] deep eval {idx}/{len(deep_indices)}")
    deep_df = (
        pd.DataFrame(deep_rows)
        .sort_values(["h1_reach_loss", "analysis_uid"], ascending=[False, True])
        .reset_index(drop=True)
    )
    deep_df["rank_h1_reach_loss"] = np.arange(1, len(deep_df) + 1, dtype=np.int32)
    deep_out_csv = m6_dir / f"node_single_removal_impacts_{primary_view}.csv"
    deep_out_parquet = m6_dir / f"node_single_removal_impacts_{primary_view}.parquet"
    top100_out = m6_dir / f"top100_high_impact_{primary_view}.csv"
    deep_df.to_csv(deep_out_csv, index=False)
    deep_df.to_parquet(deep_out_parquet, index=False)
    deep_df.head(100).to_csv(top100_out, index=False)
    deep_runtime = time.perf_counter() - t0

    # M6D: random vs targeted robustness
    removable_list = removable_indices.tolist()
    ranking_maps = {
        "target_m3_prime_reach": m3_map,
        "target_m4_bottleneck": m4_map,
        "target_pagerank": m2_pr_map,
    }
    strategy_ranked_indices: dict[str, list[int]] = {}
    for strategy, score_map in ranking_maps.items():
        ranked = sorted(
            removable_list,
            key=lambda idx: (float(score_map.get(node_uids[idx], 0.0)), node_uids[idx]),
            reverse=True,
        )
        strategy_ranked_indices[strategy] = ranked

    robustness_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for frac in removal_fracs:
        remove_k = max(1, round(frac * len(removable_list)))
        for strategy, ranked in strategy_ranked_indices.items():
            removed = np.array(ranked[:remove_k], dtype=np.int32)
            metrics = evaluate_metrics(
                context=context, removed_nodes=removed, need_h2=True, need_h3=True
            )
            robustness_rows.append(
                {
                    "view": primary_view,
                    "strategy": strategy,
                    "frac_removed": frac,
                    "k_removed": remove_k,
                    "repeat_id": 0,
                    **metrics,
                }
            )
        for rep in range(1, random_repeats + 1):
            removed = np.array(rng.sample(removable_list, remove_k), dtype=np.int32)
            metrics = evaluate_metrics(
                context=context, removed_nodes=removed, need_h2=True, need_h3=True
            )
            robustness_rows.append(
                {
                    "view": primary_view,
                    "strategy": "random",
                    "frac_removed": frac,
                    "k_removed": remove_k,
                    "repeat_id": rep,
                    **metrics,
                }
            )
            if rep % 5 == 0:
                print(f"[m6] random robustness frac={frac:.4f} repeat={rep}/{random_repeats}")
    robustness_df = pd.DataFrame(robustness_rows)
    robustness_out = m6_dir / f"robustness_curves_{primary_view}.csv"
    robustness_df.to_csv(robustness_out, index=False)
    robustness_runtime = time.perf_counter() - t0

    agg_rows: list[dict[str, Any]] = []
    for (strategy, frac, k_removed), sub in robustness_df.groupby(
        ["strategy", "frac_removed", "k_removed"], dropna=False
    ):
        agg_rows.append(
            {
                "view": primary_view,
                "strategy": strategy,
                "frac_removed": float(frac),
                "k_removed": int(k_removed),
                "n_runs": len(sub),
                "h1_reach_loss_mean": float(sub["h1_reach_loss"].mean()),
                "h1_reach_loss_p05": float(sub["h1_reach_loss"].quantile(0.05)),
                "h1_reach_loss_p95": float(sub["h1_reach_loss"].quantile(0.95)),
                "h2_path_growth_mean": float(sub["h2_path_growth"].mean()),
                "h2_disconnect_share_mean": float(sub["h2_disconnect_share"].mean()),
                "h3_redundancy_collapse_mean": float(sub["h3_redundancy_collapse"].mean()),
            }
        )
    robustness_summary_df = pd.DataFrame(agg_rows).sort_values(
        ["frac_removed", "strategy"], ascending=[True, True]
    )
    robustness_summary_out = m6_dir / f"robustness_summary_{primary_view}.csv"
    robustness_summary_df.to_csv(robustness_summary_out, index=False)

    random_lookup = (
        robustness_summary_df[robustness_summary_df["strategy"] == "random"]
        .set_index("frac_removed")["h1_reach_loss_mean"]
        .to_dict()
    )
    contrast_rows: list[dict[str, Any]] = []
    for row in robustness_summary_df.itertuples(index=False):
        if row.strategy == "random":
            continue
        baseline_random = float(random_lookup.get(row.frac_removed, np.nan))
        ratio = (
            float(row.h1_reach_loss_mean / baseline_random)
            if np.isfinite(baseline_random) and baseline_random > 0
            else np.nan
        )
        contrast_rows.append(
            {
                "view": primary_view,
                "strategy": row.strategy,
                "frac_removed": float(row.frac_removed),
                "h1_reach_loss_target_mean": float(row.h1_reach_loss_mean),
                "h1_reach_loss_random_mean": baseline_random,
                "target_vs_random_ratio_h1": ratio,
            }
        )
    contrast_df = pd.DataFrame(contrast_rows)
    contrast_out = m6_dir / f"robustness_target_vs_random_{primary_view}.csv"
    contrast_df.to_csv(contrast_out, index=False)

    # M6E: greedy interdiction
    shortlisted = h1_df.head(interdiction_shortlist_n)["analysis_uid"].tolist()
    shortlisted_idx = [uid_to_idx[uid] for uid in shortlisted]

    def greedy_interdiction(objective: str, k_value: int) -> list[dict[str, Any]]:
        current: list[int] = []
        remaining = shortlisted_idx.copy()
        rows: list[dict[str, Any]] = []
        for step in range(1, k_value + 1):
            best_node = None
            best_value = -np.inf
            for cand in remaining:
                removed = np.array([*current, cand], dtype=np.int32)
                score = evaluate_metrics(
                    context=context,
                    removed_nodes=removed,
                    need_h2=(objective == "delay_h2"),
                    need_h3=False,
                )
                value = float(
                    score["h2_path_growth"] if objective == "delay_h2" else score["h1_reach_loss"]
                )
                if np.isnan(value):
                    value = -np.inf
                if value > best_value:
                    best_value = value
                    best_node = cand
            if best_node is None:
                break
            current.append(int(best_node))
            remaining.remove(int(best_node))
            full_metrics = evaluate_metrics(
                context=context,
                removed_nodes=np.array(current, dtype=np.int32),
                need_h2=True,
                need_h3=True,
            )
            rows.append(
                {
                    "view": primary_view,
                    "objective": objective,
                    "k_target": int(k_value),
                    "step": int(step),
                    "selected_uid": node_uids[best_node],
                    "selected_name": node_names[best_node],
                    "objective_value": float(best_value),
                    **full_metrics,
                }
            )
            print(f"[m6] interdiction objective={objective} k={k_value} step={step}/{k_value}")
        return rows

    t0 = time.perf_counter()
    interdiction_rows: list[dict[str, Any]] = []
    for k_value in interdiction_k:
        interdiction_rows.extend(greedy_interdiction("deny_h1", k_value))
        interdiction_rows.extend(greedy_interdiction("delay_h2", k_value))
    interdiction_df = pd.DataFrame(interdiction_rows)
    interdiction_sets_out = m6_dir / f"interdiction_sets_{primary_view}.csv"
    interdiction_df.to_csv(interdiction_sets_out, index=False)
    interdiction_runtime = time.perf_counter() - t0

    final_perf = (
        interdiction_df.sort_values(["objective", "k_target", "step"], ascending=[True, True, True])
        .groupby(["objective", "k_target"], as_index=False)
        .tail(1)
        .copy()
    )
    final_perf_out = m6_dir / f"interdiction_performance_{primary_view}.csv"
    final_perf.to_csv(final_perf_out, index=False)

    run_metadata = {
        "module": "m6",
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
        "settings": {
            "primary_view": primary_view,
            "candidate_pool_n": candidate_pool_n,
            "deep_eval_n": deep_eval_n,
            "candidate_signal_top_n": top_each,
            "random_repeats": random_repeats,
            "removal_fracs": removal_fracs,
            "interdiction_k": interdiction_k,
            "interdiction_shortlist_n": interdiction_shortlist_n,
            "predicted_only_edge_cost": pred_only_cost,
            "geodesic_tolerance": geodesic_tol,
            "global_seed": global_seed,
        },
        "graph": {
            "n_nodes": int(n_nodes),
            "n_edges_view": len(src),
            "n_primes": len(prime_indices),
            "n_semis": len(semi_indices),
            "n_removable_firms": len(removable_indices),
        },
        "runtime_seconds": {
            "baseline": baseline_runtime,
            "single_node_screen_h1": h1_runtime,
            "single_node_deep_h2_h3": deep_runtime,
            "robustness": robustness_runtime,
            "interdiction": interdiction_runtime,
        },
    }
    run_metadata_out = m6_dir / f"run_metadata_{primary_view}.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m6",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs": {
            "node_table_contract": str(node_path),
            "edge_table_contract": str(edge_path),
            "node_sha256": file_sha256(node_path),
            "edge_sha256": file_sha256(edge_path),
            "m2_centralities": str(m2_dir / f"node_centralities_{primary_view}.parquet"),
            "m3_prime_reach": str(m3_dir / f"node_prime_reach_{primary_view}.parquet"),
            "m4_refine_corridor": str(m4r_dir / f"corridor_nodes_refined_{primary_view}.parquet"),
            "m5_prime_redundancy": str(m5_dir / f"prime_redundancy_{primary_view}.csv"),
        },
        "outputs": {
            "candidate_pool": str(pool_out),
            "baseline_metrics": str(baseline_out),
            "single_node_h1_screen": str(h1_out),
            "single_node_impacts_csv": str(deep_out_csv),
            "single_node_impacts_parquet": str(deep_out_parquet),
            "top100_high_impact": str(top100_out),
            "robustness_curves": str(robustness_out),
            "robustness_summary": str(robustness_summary_out),
            "robustness_target_vs_random": str(contrast_out),
            "interdiction_sets": str(interdiction_sets_out),
            "interdiction_performance": str(final_perf_out),
            "run_metadata": str(run_metadata_out),
        },
    }
    manifest_out = m6_dir / f"manifest_m6_{primary_view}.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))

    print(f"[done] wrote {pool_out}")
    print(f"[done] wrote {baseline_out}")
    print(f"[done] wrote {h1_out}")
    print(f"[done] wrote {deep_out_csv}")
    print(f"[done] wrote {deep_out_parquet}")
    print(f"[done] wrote {top100_out}")
    print(f"[done] wrote {robustness_out}")
    print(f"[done] wrote {robustness_summary_out}")
    print(f"[done] wrote {contrast_out}")
    print(f"[done] wrote {interdiction_sets_out}")
    print(f"[done] wrote {final_perf_out}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")


if __name__ == "__main__":
    main()
