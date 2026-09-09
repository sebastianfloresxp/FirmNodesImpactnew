#!/usr/bin/env python3
"""Module 6.5: prime-level vulnerability profiles under targeted scenarios."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from m6_1_weighted_disruption import (
    GraphContext,
    build_prime_incoming,
    build_view_edges,
    compute_source_reach_and_scc,
    evaluate_metrics,
    get_view_mask,
    load_prime_weight_table,
    parse_h1_profiles,
)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 6.5 prime exposure profiles")
    parser.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2_fix01.yaml",
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
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(  # nosec B607 -- git is a well-known system executable
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            or None
        )
    except Exception:
        return None


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


@dataclass
class PrimeViewContext:
    view: str
    n_nodes: int
    src: np.ndarray
    dst: np.ndarray
    edge_cost: np.ndarray
    node_uids: np.ndarray
    node_names: np.ndarray
    role: np.ndarray
    is_firm: np.ndarray
    prime_indices: np.ndarray
    semi_indices: np.ndarray
    removable_indices: np.ndarray
    uid_to_idx: dict[str, int]
    h1_profiles: list[dict[str, str]]
    h1_weights: dict[str, np.ndarray]
    h1_baseline_total: dict[str, float]
    h1_primary_key: str
    graph_context: GraphContext
    baseline_prime_support_count: np.ndarray
    baseline_prime_support_any: np.ndarray
    baseline_prime_dist: np.ndarray
    baseline_finite_mask: np.ndarray
    baseline_avg_len: float
    prime_uids: np.ndarray
    prime_names: np.ndarray
    weight_unit: np.ndarray
    weight_log: np.ndarray
    weight_raw: np.ndarray


def build_view_context(
    cfg: dict[str, Any],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    view_name: str,
) -> PrimeViewContext:
    m6_cfg = cfg.get("m6_1", cfg.get("m6", {}))
    h1_profiles = parse_h1_profiles(m6_cfg)
    h1_primary_key = str(m6_cfg.get("primary_h1_key", "log_obligation_any_support"))
    if not any(f"{p['weight_mode']}_{p['support_form']}" == h1_primary_key for p in h1_profiles):
        h1_primary_key = f"{h1_profiles[0]['weight_mode']}_{h1_profiles[0]['support_form']}"

    pred_only_cost = float(
        m6_cfg.get(
            "predicted_only_edge_cost", cfg.get("m5", {}).get("predicted_only_edge_cost", 3.0)
        )
    )
    geodesic_tol = float(
        m6_cfg.get("geodesic_tolerance", cfg.get("m5", {}).get("geodesic_tolerance", 1.0e-9))
    )

    node_uids = nodes["analysis_uid"].astype(str).to_numpy()
    uid_to_idx = {uid: i for i, uid in enumerate(node_uids.tolist())}
    n_nodes = len(node_uids)

    src_map = edges["src_uid"].astype(str).map(uid_to_idx)
    dst_map = edges["dst_uid"].astype(str).map(uid_to_idx)
    valid = src_map.notna() & dst_map.notna()
    edges_m = edges.loc[valid].copy()
    edges_m["src_idx"] = src_map.loc[valid].astype(np.int32).to_numpy()
    edges_m["dst_idx"] = dst_map.loc[valid].astype(np.int32).to_numpy()

    include_any = cfg.get("views", {}).get(view_name, {}).get("include_any", [])
    view_mask = get_view_mask(edges_m, include_any)
    src, dst, is_disclosed, is_observed, is_predicted = build_view_edges(edges_m, view_mask)
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
    if len(prime_indices) == 0:
        raise ValueError(f"No primes found for view={view_name}")
    if len(semi_indices) == 0:
        raise ValueError(f"No semis found for view={view_name}")

    prime_uids = node_uids[prime_indices]
    prime_names = node_names[prime_indices]
    weights_path = Path(
        str(cfg.get("paths", {}).get("prime_weights", "artifacts/ch4/inputs/prime_weights.parquet"))
    )
    weight_table, _ = load_prime_weight_table(weights_path=weights_path, prime_uids=prime_uids)
    weight_map = weight_table.set_index("analysis_uid")

    h1_weights: dict[str, np.ndarray] = {}
    for profile in h1_profiles:
        mode = profile["weight_mode"]
        form = profile["support_form"]
        key = f"{mode}_{form}"
        if mode == "unit":
            arr = np.ones(len(prime_indices), dtype=np.float64)
        elif mode == "log_obligation":
            arr = (
                weight_map["weight_log_obligation"]
                .reindex(prime_uids)
                .fillna(0.0)
                .to_numpy(np.float64)
            )
        elif mode == "raw_obligation":
            arr = (
                weight_map["weight_raw_obligation"]
                .reindex(prime_uids)
                .fillna(0.0)
                .to_numpy(np.float64)
            )
        else:
            arr = np.ones(len(prime_indices), dtype=np.float64)
        h1_weights[key] = arr

    baseline_support, baseline_scc, _ = compute_source_reach_and_scc(
        n_nodes=n_nodes,
        src_idx=src,
        dst_idx=dst,
        source_indices=semi_indices,
    )
    baseline_prime_support_count = baseline_support[prime_indices].astype(np.float64, copy=False)
    baseline_prime_support_any = (baseline_prime_support_count > 0).astype(np.float64, copy=False)

    h1_baseline_total: dict[str, float] = {}
    for profile in h1_profiles:
        key = f"{profile['weight_mode']}_{profile['support_form']}"
        support_vec = (
            baseline_prime_support_count
            if profile["support_form"] == "support_count"
            else baseline_prime_support_any
        )
        h1_baseline_total[key] = float(np.dot(h1_weights[key], support_vec))

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
        entry_scc_count = len({int(baseline_scc[pred]) for pred, _ in candidates})
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
    baseline_share_single = (
        float(baseline_single_point / baseline_supported) if baseline_supported > 0 else np.nan
    )

    graph_context = GraphContext(
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
        h1_profiles=h1_profiles,
        h1_weights=h1_weights,
        h1_baseline_total=h1_baseline_total,
        h1_primary_key=h1_primary_key,
        prime_incoming=prime_incoming,
        baseline_prime_dist=baseline_prime_dist,
        baseline_avg_len=baseline_avg_len,
        baseline_finite_mask=baseline_finite_mask,
        baseline_share_single_point_supported=baseline_share_single,
        baseline_share_no_support=baseline_share_no_support,
        geodesic_tol=geodesic_tol,
    )

    weight_unit = weight_map["weight_unit"].reindex(prime_uids).fillna(1.0).to_numpy(np.float64)
    weight_log = (
        weight_map["weight_log_obligation"].reindex(prime_uids).fillna(0.0).to_numpy(np.float64)
    )
    weight_raw = (
        weight_map["weight_raw_obligation"].reindex(prime_uids).fillna(0.0).to_numpy(np.float64)
    )

    return PrimeViewContext(
        view=view_name,
        n_nodes=n_nodes,
        src=src,
        dst=dst,
        edge_cost=edge_cost,
        node_uids=node_uids,
        node_names=node_names,
        role=role,
        is_firm=is_firm,
        prime_indices=prime_indices,
        semi_indices=semi_indices,
        removable_indices=removable_indices,
        uid_to_idx=uid_to_idx,
        h1_profiles=h1_profiles,
        h1_weights=h1_weights,
        h1_baseline_total=h1_baseline_total,
        h1_primary_key=h1_primary_key,
        graph_context=graph_context,
        baseline_prime_support_count=baseline_prime_support_count,
        baseline_prime_support_any=baseline_prime_support_any,
        baseline_prime_dist=baseline_prime_dist,
        baseline_finite_mask=baseline_finite_mask,
        baseline_avg_len=baseline_avg_len,
        prime_uids=prime_uids,
        prime_names=prime_names,
        weight_unit=weight_unit,
        weight_log=weight_log,
        weight_raw=weight_raw,
    )


def evaluate_prime_arrays(
    ctx: PrimeViewContext, removed_nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alive = np.ones(ctx.n_nodes, dtype=bool)
    if len(removed_nodes) > 0:
        alive[removed_nodes] = False

    edge_mask = alive[ctx.src] & alive[ctx.dst]
    src_active = ctx.src[edge_mask]
    dst_active = ctx.dst[edge_mask]
    cost_active = ctx.edge_cost[edge_mask]
    semi_active = ctx.semi_indices[alive[ctx.semi_indices]]

    support_count, _, _ = compute_source_reach_and_scc(
        n_nodes=ctx.n_nodes,
        src_idx=src_active,
        dst_idx=dst_active,
        source_indices=semi_active.astype(np.int32, copy=False),
    )
    prime_support_count = support_count[ctx.prime_indices].astype(np.float64, copy=False)
    prime_support_any = (prime_support_count > 0).astype(np.float64, copy=False)

    if len(semi_active) == 0:
        dist = np.full(ctx.n_nodes, np.inf, dtype=np.float64)
    else:
        dist = multi_source_weighted_distance(
            n_nodes=ctx.n_nodes,
            src=src_active,
            dst=dst_active,
            edge_cost=cost_active,
            source_indices=semi_active,
        )
    prime_dist = dist[ctx.prime_indices]
    return prime_support_count, prime_support_any, prime_dist


def build_targeted_scenarios(
    cfg: dict[str, Any],
    ctx: PrimeViewContext,
    view_name: str,
    out_root: Path,
) -> list[dict[str, Any]]:
    m6_cfg = cfg.get("m6_1", cfg.get("m6", {}))
    fracs = [float(x) for x in m6_cfg.get("removal_fracs", [0.001, 0.005, 0.01, 0.02, 0.05])]

    m2 = pd.read_parquet(
        out_root / cfg["snapshot"] / "m2" / f"node_centralities_{view_name}.parquet"
    )
    m3 = pd.read_parquet(
        out_root / cfg["snapshot"] / "m3" / f"node_prime_reach_{view_name}.parquet"
    )
    m4 = pd.read_parquet(
        out_root / cfg["snapshot"] / "m4_refine" / f"corridor_nodes_refined_{view_name}.parquet"
    )

    m2_pr_map = (
        m2.set_index("analysis_uid")["pagerank"].to_dict() if "pagerank" in m2.columns else {}
    )
    m3_map = (
        m3.set_index("analysis_uid")["prime_reach_count"].to_dict()
        if "prime_reach_count" in m3.columns
        else {}
    )
    m4_map = (
        m4.set_index("analysis_uid")["bottleneck_score_refined"].to_dict()
        if "bottleneck_score_refined" in m4.columns
        else {}
    )

    removable_list = ctx.removable_indices.tolist()
    ranking_maps = {
        "target_pagerank": m2_pr_map,
        "target_m3_prime_reach": m3_map,
        "target_m4_bottleneck": m4_map,
    }

    scenarios: list[dict[str, Any]] = []
    for strategy, score_map in ranking_maps.items():
        ranked = sorted(
            removable_list,
            key=lambda idx: (float(score_map.get(ctx.node_uids[idx], 0.0)), ctx.node_uids[idx]),
            reverse=True,
        )
        for frac in fracs:
            k = max(1, round(frac * len(removable_list)))
            removed = np.array(ranked[:k], dtype=np.int32)
            scenarios.append(
                {
                    "scenario_type": "targeted_frac",
                    "scenario_id": f"{strategy}_frac_{frac:.4f}",
                    "strategy": strategy,
                    "frac_removed": float(frac),
                    "k_removed": int(k),
                    "objective": pd.NA,
                    "removed_indices": removed,
                }
            )
    return scenarios


def build_interdiction_scenarios(
    m6_1_dir: Path, ctx: PrimeViewContext, view_name: str
) -> list[dict[str, Any]]:
    path = m6_1_dir / f"interdiction_sets_{view_name}.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    scenarios: list[dict[str, Any]] = []
    for (objective, k_target), sub in df.groupby(["objective", "k_target"], dropna=False):
        ordered = sub.sort_values("step", ascending=True)
        uids = ordered["selected_uid"].astype(str).tolist()
        idxs = np.array([ctx.uid_to_idx[u] for u in uids if u in ctx.uid_to_idx], dtype=np.int32)
        scenarios.append(
            {
                "scenario_type": "interdiction",
                "scenario_id": f"{objective!s}_k_{int(k_target)}",
                "strategy": "greedy_interdiction",
                "frac_removed": pd.NA,
                "k_removed": int(k_target),
                "objective": str(objective),
                "removed_indices": idxs,
            }
        )
    return scenarios


def scenario_summary(
    ctx: PrimeViewContext,
    prime_support_count: np.ndarray,
    prime_support_any: np.ndarray,
    prime_dist: np.ndarray,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = dict(aggregate)
    baseline_any = ctx.baseline_prime_support_any > 0
    lost_any = baseline_any & (prime_support_any <= 0)

    denom_unit = float(np.dot(ctx.weight_unit, ctx.baseline_prime_support_any))
    denom_log = float(np.dot(ctx.weight_log, ctx.baseline_prime_support_any))
    denom_raw = float(np.dot(ctx.weight_raw, ctx.baseline_prime_support_any))
    row["share_primes_lost_any_unit"] = float(np.mean(lost_any))
    row["share_primes_lost_any_log_weighted"] = (
        float(np.dot(ctx.weight_log, lost_any.astype(np.float64)) / denom_log)
        if denom_log > 0
        else np.nan
    )
    row["share_primes_lost_any_raw_weighted"] = (
        float(np.dot(ctx.weight_raw, lost_any.astype(np.float64)) / denom_raw)
        if denom_raw > 0
        else np.nan
    )
    row["baseline_supported_unit_total"] = denom_unit
    row["baseline_supported_log_total"] = denom_log
    row["baseline_supported_raw_total"] = denom_raw

    row["mean_support_count_drop"] = float(
        np.mean(ctx.baseline_prime_support_count - prime_support_count)
    )
    finite_current = np.isfinite(prime_dist)
    finite_overlap = ctx.baseline_finite_mask & finite_current
    row["mean_path_growth_finite_overlap"] = (
        float(np.mean(prime_dist[finite_overlap] - ctx.baseline_prime_dist[finite_overlap]))
        if int(finite_overlap.sum()) > 0
        else np.nan
    )
    row["disconnect_share_from_baseline_finite"] = (
        float(np.mean(ctx.baseline_finite_mask & (~finite_current)))
        if int(ctx.baseline_finite_mask.sum()) > 0
        else np.nan
    )
    return row


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    m0_dir = out_root / snapshot / "m0"
    m6_1_dir = out_root / snapshot / "m6_1"
    out_dir = out_root / snapshot / "m6_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    node_path = m0_dir / "node_table_contract.parquet"
    edge_path = m0_dir / "edge_table_contract.parquet"
    if not node_path.exists():
        raise FileNotFoundError(node_path)
    if not edge_path.exists():
        raise FileNotFoundError(edge_path)

    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)

    views = [
        str(v)
        for v in cfg.get("m6_5", {}).get(
            "views", cfg.get("m6_1", {}).get("views", ["disclosed", "observed", "full"])
        )
    ]
    output_map: dict[str, dict[str, str]] = {}
    runtime_by_view: dict[str, float] = {}

    for view_name in views:
        t0 = dt.datetime.now(dt.timezone.utc)
        ctx = build_view_context(cfg=cfg, nodes=nodes, edges=edges, view_name=view_name)
        targeted = build_targeted_scenarios(
            cfg=cfg, ctx=ctx, view_name=view_name, out_root=out_root
        )
        interdiction = build_interdiction_scenarios(m6_1_dir=m6_1_dir, ctx=ctx, view_name=view_name)
        scenarios = targeted + interdiction

        prime_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for scenario in scenarios:
            removed = scenario["removed_indices"]
            metrics = evaluate_metrics(
                context=ctx.graph_context,
                removed_nodes=removed,
                need_h2=True,
                need_h3=True,
            )
            support_count, support_any, prime_dist = evaluate_prime_arrays(
                ctx=ctx, removed_nodes=removed
            )
            aggregate = {
                "view": view_name,
                "scenario_type": scenario["scenario_type"],
                "scenario_id": scenario["scenario_id"],
                "strategy": scenario["strategy"],
                "objective": scenario["objective"],
                "frac_removed": scenario["frac_removed"],
                "k_removed": scenario["k_removed"],
                "removed_nodes_count": len(removed),
                "h1_reach_loss": float(metrics.get("h1_reach_loss", np.nan)),
                "h2_path_growth": float(metrics.get("h2_path_growth", np.nan)),
                "h2_disconnect_share": float(metrics.get("h2_disconnect_share", np.nan)),
                "h3_share_no_support": float(metrics.get("h3_share_no_support", np.nan)),
                "h3_share_single_point_supported": float(
                    metrics.get("h3_share_single_point_supported", np.nan)
                ),
            }
            for profile in ctx.h1_profiles:
                key = f"{profile['weight_mode']}_{profile['support_form']}"
                aggregate[f"h1_{key}"] = float(metrics.get(f"h1_{key}", np.nan))
                aggregate[f"support_total_{key}"] = float(
                    metrics.get(f"support_total_{key}", np.nan)
                )

            summary_rows.append(
                scenario_summary(
                    ctx=ctx,
                    prime_support_count=support_count,
                    prime_support_any=support_any,
                    prime_dist=prime_dist,
                    aggregate=aggregate,
                )
            )

            finite_current = np.isfinite(prime_dist)
            for pos, pidx in enumerate(ctx.prime_indices.tolist()):
                prime_rows.append(
                    {
                        "view": view_name,
                        "scenario_type": scenario["scenario_type"],
                        "scenario_id": scenario["scenario_id"],
                        "strategy": scenario["strategy"],
                        "objective": scenario["objective"],
                        "frac_removed": scenario["frac_removed"],
                        "k_removed": scenario["k_removed"],
                        "prime_uid": ctx.node_uids[pidx],
                        "prime_name": ctx.node_names[pidx],
                        "prime_weight_unit": float(ctx.weight_unit[pos]),
                        "prime_weight_log_obligation": float(ctx.weight_log[pos]),
                        "prime_weight_raw_obligation": float(ctx.weight_raw[pos]),
                        "baseline_support_count": float(ctx.baseline_prime_support_count[pos]),
                        "support_count": float(support_count[pos]),
                        "support_count_loss": float(
                            ctx.baseline_prime_support_count[pos] - support_count[pos]
                        ),
                        "baseline_any_support": int(ctx.baseline_prime_support_any[pos] > 0),
                        "any_support": int(support_any[pos] > 0),
                        "lost_any_support": int(
                            (ctx.baseline_prime_support_any[pos] > 0) and (support_any[pos] <= 0)
                        ),
                        "baseline_path_len": float(ctx.baseline_prime_dist[pos])
                        if np.isfinite(ctx.baseline_prime_dist[pos])
                        else np.nan,
                        "path_len": float(prime_dist[pos]) if finite_current[pos] else np.nan,
                        "path_growth": (
                            float(prime_dist[pos] - ctx.baseline_prime_dist[pos])
                            if (
                                np.isfinite(prime_dist[pos])
                                and np.isfinite(ctx.baseline_prime_dist[pos])
                            )
                            else np.nan
                        ),
                        "disconnected_from_baseline_path": int(
                            np.isfinite(ctx.baseline_prime_dist[pos]) and (not finite_current[pos])
                        ),
                    }
                )

        prime_df = pd.DataFrame(prime_rows)
        summary_df = pd.DataFrame(summary_rows)
        if not prime_df.empty:
            worst = (
                prime_df.groupby(["prime_uid", "prime_name"], as_index=False)
                .agg(
                    prime_weight_log_obligation=("prime_weight_log_obligation", "max"),
                    prime_weight_raw_obligation=("prime_weight_raw_obligation", "max"),
                    baseline_support_count=("baseline_support_count", "max"),
                    max_support_count_loss=("support_count_loss", "max"),
                    max_path_growth=("path_growth", "max"),
                    scenarios_with_any_loss=("lost_any_support", "sum"),
                    scenarios_with_disconnect=("disconnected_from_baseline_path", "sum"),
                )
                .sort_values(
                    ["max_support_count_loss", "max_path_growth", "prime_uid"],
                    ascending=[False, False, True],
                )
            )
        else:
            worst = pd.DataFrame()

        prime_csv = out_dir / f"prime_exposure_profiles_{view_name}.csv"
        prime_parquet = out_dir / f"prime_exposure_profiles_{view_name}.parquet"
        summary_csv = out_dir / f"prime_exposure_summary_{view_name}.csv"
        worst_csv = out_dir / f"prime_exposure_worstcase_{view_name}.csv"
        prime_df.to_csv(prime_csv, index=False)
        prime_df.to_parquet(prime_parquet, index=False)
        summary_df.to_csv(summary_csv, index=False)
        worst.to_csv(worst_csv, index=False)

        runtime = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
        runtime_by_view[view_name] = float(runtime)
        output_map[view_name] = {
            "prime_exposure_profiles_csv": str(prime_csv),
            "prime_exposure_profiles_parquet": str(prime_parquet),
            "prime_exposure_summary_csv": str(summary_csv),
            "prime_exposure_worstcase_csv": str(worst_csv),
        }
        print(f"[m6_5] completed view={view_name} runtime_s={runtime:.2f}")

    run_metadata = {
        "module": "m6_5",
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
            "views": views,
            "scenario_types": ["targeted_frac", "interdiction"],
        },
        "runtime_seconds_by_view": runtime_by_view,
    }
    run_metadata_path = out_dir / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m6_5",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs": {
            "node_table_contract": str(node_path),
            "edge_table_contract": str(edge_path),
            "node_sha256": file_sha256(node_path),
            "edge_sha256": file_sha256(edge_path),
        },
        "outputs": output_map,
        "run_metadata": str(run_metadata_path),
    }
    manifest_path = out_dir / "manifest_m6_5.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {manifest_path}")
    print(f"[done] wrote {run_metadata_path}")


if __name__ == "__main__":
    main()
