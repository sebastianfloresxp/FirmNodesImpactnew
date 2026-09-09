#!/usr/bin/env python3
"""Module 6.4: effective-reach disruption using cost/hop constrained support."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def bool_series_with_fallback(
    df: pd.DataFrame, columns: list[str], *, default: bool = False
) -> pd.Series:
    for col in columns:
        if col in df.columns:
            return df[col].fillna(default).astype(bool)
    return pd.Series(np.full(len(df), default, dtype=bool), index=df.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 6.4 effective disruption")
    parser.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2_fix03_effective_extensions.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--view",
        default=None,
        help="Optional single view override (disclosed|observed|full)",
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
        return subprocess.check_output(  # nosec B607 -- git is a well-known system executable
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
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


def build_view_edges(edges: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    cols = ["src_idx", "dst_idx", "is_disclosed", "is_observed_ship", "is_predicted"]
    df = edges.loc[mask, cols].copy()
    df = df[df["src_idx"] != df["dst_idx"]]
    df = (
        df.groupby(["src_idx", "dst_idx"], as_index=False)[
            ["is_disclosed", "is_observed_ship", "is_predicted"]
        ]
        .max()
        .sort_values(["src_idx", "dst_idx"], ascending=[True, True], kind="mergesort")
    )
    return df


def load_prime_weight_table(weights_path: Path, prime_uids: np.ndarray) -> pd.DataFrame:
    if not weights_path.exists():
        return pd.DataFrame(
            {
                "analysis_uid": prime_uids.astype(str),
                "weight_unit": np.ones(len(prime_uids), dtype=np.float64),
                "weight_log_obligation": np.ones(len(prime_uids), dtype=np.float64),
                "weight_raw_obligation": np.ones(len(prime_uids), dtype=np.float64),
            }
        )

    if weights_path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(weights_path)
    else:
        raw = pd.read_csv(weights_path)

    uid_col = next(
        (c for c in ["analysis_uid", "prime_uid", "uid", "node_uid"] if c in raw.columns), None
    )
    if uid_col is None:
        return pd.DataFrame(
            {
                "analysis_uid": prime_uids.astype(str),
                "weight_unit": np.ones(len(prime_uids), dtype=np.float64),
                "weight_log_obligation": np.ones(len(prime_uids), dtype=np.float64),
                "weight_raw_obligation": np.ones(len(prime_uids), dtype=np.float64),
            }
        )

    df = raw.copy()
    df["analysis_uid"] = df[uid_col].astype(str)
    for col in [
        "obligation_log1p",
        "obligation_weight",
        "weight",
        "obligation_nonneg",
        "total_obligations",
        "obligation_raw",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    out = pd.DataFrame({"analysis_uid": prime_uids.astype(str)})
    merged = out.merge(df, how="left", on="analysis_uid")
    merged["weight_unit"] = 1.0

    if "obligation_log1p" in merged.columns:
        merged["weight_log_obligation"] = merged["obligation_log1p"].fillna(0.0)
    elif "obligation_weight" in merged.columns:
        merged["weight_log_obligation"] = merged["obligation_weight"].fillna(0.0)
    elif "weight" in merged.columns:
        merged["weight_log_obligation"] = merged["weight"].fillna(0.0)
    else:
        merged["weight_log_obligation"] = 0.0

    if "obligation_nonneg" in merged.columns:
        merged["weight_raw_obligation"] = merged["obligation_nonneg"].fillna(0.0)
    elif "total_obligations" in merged.columns:
        merged["weight_raw_obligation"] = merged["total_obligations"].fillna(0.0)
    elif "obligation_raw" in merged.columns:
        merged["weight_raw_obligation"] = merged["obligation_raw"].clip(lower=0).fillna(0.0)
    else:
        merged["weight_raw_obligation"] = 0.0

    return merged[["analysis_uid", "weight_unit", "weight_log_obligation", "weight_raw_obligation"]]


def build_super_source_distances(
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    edge_weight: np.ndarray,
    semi_idx_active: np.ndarray,
    *,
    unweighted: bool,
    limit: float,
) -> np.ndarray:
    super_source = n_nodes
    rows = np.concatenate([src, np.full(len(semi_idx_active), super_source, dtype=np.int32)])
    cols = np.concatenate([dst, semi_idx_active.astype(np.int32, copy=False)])
    if unweighted:
        data = np.concatenate(
            [np.ones(len(src), dtype=np.float64), np.ones(len(semi_idx_active), dtype=np.float64)]
        )
    else:
        data = np.concatenate(
            [
                edge_weight.astype(np.float64, copy=False),
                np.zeros(len(semi_idx_active), dtype=np.float64),
            ]
        )
    matrix = csr_matrix((data, (rows, cols)), shape=(n_nodes + 1, n_nodes + 1))
    dist = dijkstra(
        matrix,
        directed=True,
        indices=super_source,
        unweighted=unweighted,
        limit=limit,
    ).astype(np.float64, copy=False)
    return dist[:n_nodes]


def evaluate_metrics(
    *,
    n_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    edge_cost: np.ndarray,
    semi_idx: np.ndarray,
    prime_idx: np.ndarray,
    prime_w_log: np.ndarray,
    prime_w_unit: np.ndarray,
    removed_nodes: np.ndarray,
    cost_cap: float,
    hop_cap: float,
    baseline: dict[str, float],
) -> dict[str, float]:
    alive = np.ones(n_nodes, dtype=bool)
    if len(removed_nodes) > 0:
        alive[removed_nodes] = False

    edge_mask = alive[src] & alive[dst]
    src_a = src[edge_mask]
    dst_a = dst[edge_mask]
    cost_a = edge_cost[edge_mask]
    semi_active = semi_idx[alive[semi_idx]]

    if len(semi_active) == 0:
        return {
            "h1_cost_any_support_loss": 1.0,
            "h1_hop_any_support_loss": 1.0,
            "h2_cost_path_growth": 0.0,
            "h2_hop_path_growth": 0.0,
            "h2_cost_disconnect_share": 1.0,
            "h2_hop_disconnect_share": 1.0,
            "support_total_cost_log": 0.0,
            "support_total_hop_log": 0.0,
            "support_total_cost_unit": 0.0,
            "support_total_hop_unit": 0.0,
        }

    dist_cost = build_super_source_distances(
        n_nodes=n_nodes,
        src=src_a,
        dst=dst_a,
        edge_weight=cost_a,
        semi_idx_active=semi_active,
        unweighted=False,
        limit=cost_cap,
    )
    prime_cost = dist_cost[prime_idx]
    support_cost = np.isfinite(prime_cost) & (prime_cost <= cost_cap)

    dist_hop = build_super_source_distances(
        n_nodes=n_nodes,
        src=src_a,
        dst=dst_a,
        edge_weight=np.ones(len(src_a), dtype=np.float64),
        semi_idx_active=semi_active,
        unweighted=True,
        limit=hop_cap,
    )
    prime_hop = dist_hop[prime_idx]
    support_hop = np.isfinite(prime_hop) & (prime_hop <= hop_cap)

    support_total_cost_log = float((support_cost.astype(np.float64) * prime_w_log).sum())
    support_total_hop_log = float((support_hop.astype(np.float64) * prime_w_log).sum())
    support_total_cost_unit = float((support_cost.astype(np.float64) * prime_w_unit).sum())
    support_total_hop_unit = float((support_hop.astype(np.float64) * prime_w_unit).sum())

    h1_cost = float(
        (baseline["support_total_cost_log"] - support_total_cost_log)
        / max(baseline["support_total_cost_log"], 1e-12)
    )
    h1_hop = float(
        (baseline["support_total_hop_log"] - support_total_hop_log)
        / max(baseline["support_total_hop_log"], 1e-12)
    )

    if int(support_cost.sum()) > 0:
        avg_cost = float(np.nanmean(prime_cost[support_cost]))
    else:
        avg_cost = np.nan
    if int(support_hop.sum()) > 0:
        avg_hop = float(np.nanmean(prime_hop[support_hop]))
    else:
        avg_hop = np.nan

    h2_cost_path_growth = (
        float(avg_cost - baseline["avg_cost_supported"]) if np.isfinite(avg_cost) else 0.0
    )
    h2_hop_path_growth = (
        float(avg_hop - baseline["avg_hop_supported"]) if np.isfinite(avg_hop) else 0.0
    )
    h2_cost_disconnect_share = float(1.0 - support_cost.mean()) if len(support_cost) else 0.0
    h2_hop_disconnect_share = float(1.0 - support_hop.mean()) if len(support_hop) else 0.0

    return {
        "h1_cost_any_support_loss": h1_cost,
        "h1_hop_any_support_loss": h1_hop,
        "h2_cost_path_growth": h2_cost_path_growth,
        "h2_hop_path_growth": h2_hop_path_growth,
        "h2_cost_disconnect_share": h2_cost_disconnect_share,
        "h2_hop_disconnect_share": h2_hop_disconnect_share,
        "support_total_cost_log": support_total_cost_log,
        "support_total_hop_log": support_total_hop_log,
        "support_total_cost_unit": support_total_cost_unit,
        "support_total_hop_unit": support_total_hop_unit,
    }


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "ch4_unknown"))
    paths_cfg = cfg.get("paths", {})
    out_root = Path(str(paths_cfg.get("out_root", "artifacts/ch4/v2_fix01")))
    m0_dir = out_root / snapshot / "m0"
    m6_1_dir = out_root / snapshot / "m6_1"
    m6_4_dir = out_root / snapshot / "m6_4"
    m6_4_dir.mkdir(parents=True, exist_ok=True)

    node_path = m0_dir / "node_table_contract.parquet"
    edge_path = m0_dir / "edge_table_contract.parquet"
    if not node_path.exists() or not edge_path.exists():
        raise FileNotFoundError("M0 contract outputs are required before M6.4")

    nodes = pd.read_parquet(node_path).reset_index(drop=True)
    edges = pd.read_parquet(edge_path).reset_index(drop=True)
    nodes["node_idx"] = np.arange(len(nodes), dtype=np.int32)
    uid_to_idx = nodes.set_index("analysis_uid")["node_idx"]
    edges["src_idx"] = edges["src_uid"].astype(str).map(uid_to_idx).astype(np.int32)
    edges["dst_idx"] = edges["dst_uid"].astype(str).map(uid_to_idx).astype(np.int32)

    is_prime = bool_series_with_fallback(
        nodes,
        ["is_tier1_prime", "has_prime_vendor"],
        default=False,
    ).to_numpy(bool)
    is_semi = bool_series_with_fallback(nodes, ["is_semi_strict"], default=False).to_numpy(bool)
    is_firm_nonsemi = (nodes["entity_role"].astype(str) == "firm").to_numpy() & ~is_semi
    prime_idx = np.flatnonzero(is_prime).astype(np.int32)
    semi_idx = np.flatnonzero(is_semi).astype(np.int32)
    removable_idx = np.flatnonzero(is_firm_nonsemi).astype(np.int32)
    n_nodes = len(nodes)

    prime_uids = nodes.loc[prime_idx, "analysis_uid"].astype(str).to_numpy()
    weights_path = Path(
        str(paths_cfg.get("prime_weights", "artifacts/ch4/inputs/prime_weights.parquet"))
    )
    prime_weights = load_prime_weight_table(weights_path, prime_uids)
    prime_weights = prime_weights.set_index("analysis_uid").reindex(prime_uids)
    prime_w_log = prime_weights["weight_log_obligation"].fillna(1.0).to_numpy(np.float64)
    prime_w_unit = prime_weights["weight_unit"].fillna(1.0).to_numpy(np.float64)

    m_cfg = cfg.get("m6_4", {})
    views = [str(v) for v in m_cfg.get("views", list(cfg.get("views", {}).keys()))]
    if args.view:
        views = [str(args.view)]
    candidate_pool_n = int(m_cfg.get("candidate_pool_n", 6000))
    deep_eval_n = int(m_cfg.get("deep_eval_n", 1000))
    random_repeats = int(m_cfg.get("random_repeats", 30))
    removal_fracs = [float(x) for x in m_cfg.get("removal_fracs", [0.001, 0.005, 0.01, 0.02, 0.05])]
    interdiction_k = [int(x) for x in m_cfg.get("interdiction_k", [5, 10, 25])]
    interdiction_shortlist_n = int(m_cfg.get("interdiction_shortlist_n", 100))
    predicted_only_edge_cost = float(m_cfg.get("predicted_only_edge_cost", 3.0))
    cost_cap_primary = float(m_cfg.get("cost_cap_primary", 10.0))
    hop_cap_primary = float(m_cfg.get("hop_cap_primary", 5.0))
    seed = int(cfg.get("random", {}).get("global_seed", 7))
    rng = np.random.default_rng(seed)
    random.seed(seed)

    runtime_by_view: dict[str, dict[str, float]] = {}
    outputs_by_view: dict[str, dict[str, str]] = {}

    for view in views:
        print(f"[m6_4] starting view={view}", flush=True)
        include_any = [str(x) for x in cfg.get("views", {}).get(view, {}).get("include_any", [])]
        if not include_any:
            raise ValueError(f"view {view} include_any missing")

        pool_path = m6_1_dir / f"candidate_pool_{view}.csv"
        if not pool_path.exists():
            raise FileNotFoundError(f"M6.1 candidate pool missing for view={view}: {pool_path}")
        pool = pd.read_csv(pool_path)
        pool = pool.sort_values(
            ["candidate_rank", "aggregate_score", "analysis_uid"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        if len(pool) > candidate_pool_n:
            pool = pool.head(candidate_pool_n).copy()

        removable_set = set(removable_idx.tolist())
        pool = pool[pool["node_idx"].isin(removable_set)].copy()
        if pool.empty:
            raise RuntimeError(f"candidate pool empty after firm filtering for view={view}")

        mask = get_view_mask(edges, include_any)
        edge_view = build_view_edges(edges, mask)
        src = edge_view["src_idx"].to_numpy(np.int32, copy=False)
        dst = edge_view["dst_idx"].to_numpy(np.int32, copy=False)
        predicted_only = (
            edge_view["is_predicted"].astype(bool)
            & ~(edge_view["is_disclosed"].astype(bool) | edge_view["is_observed_ship"].astype(bool))
        ).to_numpy()
        edge_cost = np.where(predicted_only, predicted_only_edge_cost, 1.0).astype(
            np.float64, copy=False
        )

        t0 = time.perf_counter()
        baseline_metrics = evaluate_metrics(
            n_nodes=n_nodes,
            src=src,
            dst=dst,
            edge_cost=edge_cost,
            semi_idx=semi_idx,
            prime_idx=prime_idx,
            prime_w_log=prime_w_log,
            prime_w_unit=prime_w_unit,
            removed_nodes=np.array([], dtype=np.int32),
            cost_cap=cost_cap_primary,
            hop_cap=hop_cap_primary,
            baseline={
                "support_total_cost_log": 1.0,
                "support_total_hop_log": 1.0,
                "avg_cost_supported": 0.0,
                "avg_hop_supported": 0.0,
            },
        )
        baseline_metrics["avg_cost_supported"] = float(
            np.nan if baseline_metrics["support_total_cost_unit"] <= 0 else 0.0
        )
        baseline_metrics["avg_hop_supported"] = float(
            np.nan if baseline_metrics["support_total_hop_unit"] <= 0 else 0.0
        )

        # Recompute baseline with explicit averages and normalizers.
        # This second call keeps code simple and stable.
        baseline_base = {
            "support_total_cost_log": max(baseline_metrics["support_total_cost_log"], 1e-12),
            "support_total_hop_log": max(baseline_metrics["support_total_hop_log"], 1e-12),
            "avg_cost_supported": 0.0,
            "avg_hop_supported": 0.0,
        }
        baseline_eval = evaluate_metrics(
            n_nodes=n_nodes,
            src=src,
            dst=dst,
            edge_cost=edge_cost,
            semi_idx=semi_idx,
            prime_idx=prime_idx,
            prime_w_log=prime_w_log,
            prime_w_unit=prime_w_unit,
            removed_nodes=np.array([], dtype=np.int32),
            cost_cap=cost_cap_primary,
            hop_cap=hop_cap_primary,
            baseline=baseline_base,
        )
        baseline = {
            "support_total_cost_log": baseline_eval["support_total_cost_log"],
            "support_total_hop_log": baseline_eval["support_total_hop_log"],
            "avg_cost_supported": baseline_eval["h2_cost_path_growth"] + 0.0,
            "avg_hop_supported": baseline_eval["h2_hop_path_growth"] + 0.0,
        }
        # reset growth to true baseline semantics
        baseline["avg_cost_supported"] = 0.0
        baseline["avg_hop_supported"] = 0.0
        baseline_runtime = float(time.perf_counter() - t0)

        # Single-node deep evaluation over top deep_eval_n pool entries.
        t0 = time.perf_counter()
        deep_df = pool.head(deep_eval_n).copy()
        deep_rows: list[dict[str, Any]] = []
        for i, row in enumerate(deep_df.itertuples(index=False), start=1):
            idx = int(row.node_idx)
            metrics = evaluate_metrics(
                n_nodes=n_nodes,
                src=src,
                dst=dst,
                edge_cost=edge_cost,
                semi_idx=semi_idx,
                prime_idx=prime_idx,
                prime_w_log=prime_w_log,
                prime_w_unit=prime_w_unit,
                removed_nodes=np.array([idx], dtype=np.int32),
                cost_cap=cost_cap_primary,
                hop_cap=hop_cap_primary,
                baseline={
                    "support_total_cost_log": max(baseline_eval["support_total_cost_log"], 1e-12),
                    "support_total_hop_log": max(baseline_eval["support_total_hop_log"], 1e-12),
                    "avg_cost_supported": baseline_eval["h2_cost_path_growth"] + 0.0,
                    "avg_hop_supported": baseline_eval["h2_hop_path_growth"] + 0.0,
                },
            )
            rec = {
                "view": view,
                "analysis_uid": str(row.analysis_uid),
                "node_idx": idx,
                "name": row.name,
                "entity_role": row.entity_role,
            }
            rec.update(metrics)
            deep_rows.append(rec)
            if i % 100 == 0 or i == len(deep_df):
                print(f"[m6_4] single-node {i}/{len(deep_df)} view={view}", flush=True)
        deep_out = (
            pd.DataFrame(deep_rows)
            .sort_values(
                ["h1_cost_any_support_loss", "analysis_uid"],
                ascending=[False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        deep_out["rank_h1_cost_any_support_loss"] = (
            deep_out["h1_cost_any_support_loss"]
            .rank(method="min", ascending=False)
            .astype(np.int32)
        )
        deep_runtime = float(time.perf_counter() - t0)

        impacts_csv = m6_4_dir / f"node_single_removal_impacts_effective_{view}.csv"
        impacts_pq = m6_4_dir / f"node_single_removal_impacts_effective_{view}.parquet"
        top100_csv = m6_4_dir / f"top100_high_impact_effective_{view}.csv"
        deep_out.to_csv(impacts_csv, index=False)
        deep_out.to_parquet(impacts_pq, index=False)
        deep_out.head(100).to_csv(top100_csv, index=False)

        # Robustness curves.
        t0 = time.perf_counter()
        strategy_to_rank = {
            "target_pagerank": "signal_pagerank",
            "target_m4_bottleneck": "signal_m4_bottleneck",
            "target_m3_prime_reach": "signal_m3_prime_reach",
        }
        robust_rows: list[dict[str, Any]] = []

        for frac in removal_fracs:
            k = max(1, round(frac * len(removable_idx)))
            for strategy, col in strategy_to_rank.items():
                rank = (
                    pool.sort_values(
                        [col, "analysis_uid"], ascending=[False, True], kind="mergesort"
                    )
                    .head(k)["node_idx"]
                    .to_numpy(np.int32)
                )
                metrics = evaluate_metrics(
                    n_nodes=n_nodes,
                    src=src,
                    dst=dst,
                    edge_cost=edge_cost,
                    semi_idx=semi_idx,
                    prime_idx=prime_idx,
                    prime_w_log=prime_w_log,
                    prime_w_unit=prime_w_unit,
                    removed_nodes=rank,
                    cost_cap=cost_cap_primary,
                    hop_cap=hop_cap_primary,
                    baseline={
                        "support_total_cost_log": max(
                            baseline_eval["support_total_cost_log"], 1e-12
                        ),
                        "support_total_hop_log": max(baseline_eval["support_total_hop_log"], 1e-12),
                        "avg_cost_supported": baseline_eval["h2_cost_path_growth"] + 0.0,
                        "avg_hop_supported": baseline_eval["h2_hop_path_growth"] + 0.0,
                    },
                )
                robust_rows.append(
                    {
                        "view": view,
                        "strategy": strategy,
                        "frac_removed": float(frac),
                        "k_removed": int(k),
                        "repeat_id": 0,
                        "removed_nodes_count": len(rank),
                        **metrics,
                    }
                )
            for rep in range(1, random_repeats + 1):
                rand_nodes = rng.choice(removable_idx, size=k, replace=False).astype(
                    np.int32, copy=False
                )
                metrics = evaluate_metrics(
                    n_nodes=n_nodes,
                    src=src,
                    dst=dst,
                    edge_cost=edge_cost,
                    semi_idx=semi_idx,
                    prime_idx=prime_idx,
                    prime_w_log=prime_w_log,
                    prime_w_unit=prime_w_unit,
                    removed_nodes=rand_nodes,
                    cost_cap=cost_cap_primary,
                    hop_cap=hop_cap_primary,
                    baseline={
                        "support_total_cost_log": max(
                            baseline_eval["support_total_cost_log"], 1e-12
                        ),
                        "support_total_hop_log": max(baseline_eval["support_total_hop_log"], 1e-12),
                        "avg_cost_supported": baseline_eval["h2_cost_path_growth"] + 0.0,
                        "avg_hop_supported": baseline_eval["h2_hop_path_growth"] + 0.0,
                    },
                )
                robust_rows.append(
                    {
                        "view": view,
                        "strategy": "random",
                        "frac_removed": float(frac),
                        "k_removed": int(k),
                        "repeat_id": int(rep),
                        "removed_nodes_count": len(rand_nodes),
                        **metrics,
                    }
                )
                if rep % 10 == 0 or rep == random_repeats:
                    print(
                        f"[m6_4] random frac={frac:.4f} rep={rep}/{random_repeats} view={view}",
                        flush=True,
                    )
        robust_df = pd.DataFrame(robust_rows)
        robust_curves_csv = m6_4_dir / f"robustness_curves_effective_{view}.csv"
        robust_df.to_csv(robust_curves_csv, index=False)

        agg_cols = [
            "h1_cost_any_support_loss",
            "h1_hop_any_support_loss",
            "h2_cost_path_growth",
            "h2_hop_path_growth",
            "h2_cost_disconnect_share",
            "h2_hop_disconnect_share",
        ]
        robust_summary = (
            robust_df.groupby(["view", "strategy", "frac_removed", "k_removed"], as_index=False)[
                agg_cols
            ]
            .mean()
            .sort_values(["frac_removed", "strategy"], ascending=[True, True], kind="mergesort")
        )
        robust_summary_csv = m6_4_dir / f"robustness_summary_effective_{view}.csv"
        robust_summary.to_csv(robust_summary_csv, index=False)
        robust_runtime = float(time.perf_counter() - t0)

        # Greedy interdiction.
        t0 = time.perf_counter()
        shortlist = deep_out.head(interdiction_shortlist_n)["node_idx"].astype(np.int32).to_numpy()
        k_max = max(interdiction_k) if interdiction_k else 0

        inter_rows: list[dict[str, Any]] = []
        set_rows: list[dict[str, Any]] = []
        for objective in ["deny_h1_cost", "delay_h2_cost"]:
            selected: list[int] = []
            for step in range(1, k_max + 1):
                best_score = -np.inf
                best_idx = None
                best_metrics: dict[str, float] | None = None
                for cand in shortlist.tolist():
                    if cand in selected:
                        continue
                    removed = np.array([*selected, cand], dtype=np.int32)
                    metrics = evaluate_metrics(
                        n_nodes=n_nodes,
                        src=src,
                        dst=dst,
                        edge_cost=edge_cost,
                        semi_idx=semi_idx,
                        prime_idx=prime_idx,
                        prime_w_log=prime_w_log,
                        prime_w_unit=prime_w_unit,
                        removed_nodes=removed,
                        cost_cap=cost_cap_primary,
                        hop_cap=hop_cap_primary,
                        baseline={
                            "support_total_cost_log": max(
                                baseline_eval["support_total_cost_log"], 1e-12
                            ),
                            "support_total_hop_log": max(
                                baseline_eval["support_total_hop_log"], 1e-12
                            ),
                            "avg_cost_supported": baseline_eval["h2_cost_path_growth"] + 0.0,
                            "avg_hop_supported": baseline_eval["h2_hop_path_growth"] + 0.0,
                        },
                    )
                    score = (
                        metrics["h1_cost_any_support_loss"]
                        if objective == "deny_h1_cost"
                        else metrics["h2_cost_path_growth"]
                    )
                    if score > best_score:
                        best_score = score
                        best_idx = int(cand)
                        best_metrics = metrics
                if best_idx is None or best_metrics is None:
                    break
                selected.append(best_idx)
                uid = str(nodes.at[best_idx, "analysis_uid"])
                set_rows.append(
                    {
                        "view": view,
                        "objective": objective,
                        "step": int(step),
                        "selected_uid": uid,
                        "selected_idx": int(best_idx),
                    }
                )
                if step in interdiction_k:
                    inter_rows.append(
                        {
                            "view": view,
                            "objective": objective,
                            "k_target": int(step),
                            "objective_value": float(best_score),
                            "removed_nodes_count": len(selected),
                            **best_metrics,
                        }
                    )
                if step % 5 == 0 or step == k_max:
                    print(
                        f"[m6_4] interdiction objective={objective} step={step}/{k_max} view={view}",
                        flush=True,
                    )

        inter_df = pd.DataFrame(inter_rows)
        set_df = pd.DataFrame(set_rows)
        inter_csv = m6_4_dir / f"interdiction_performance_effective_{view}.csv"
        sets_csv = m6_4_dir / f"interdiction_sets_effective_{view}.csv"
        inter_df.to_csv(inter_csv, index=False)
        set_df.to_csv(sets_csv, index=False)
        inter_runtime = float(time.perf_counter() - t0)

        baseline_json = m6_4_dir / f"baseline_metrics_effective_{view}.json"
        baseline_payload = {
            "view": view,
            "support_total_cost_log": float(baseline_eval["support_total_cost_log"]),
            "support_total_hop_log": float(baseline_eval["support_total_hop_log"]),
            "support_total_cost_unit": float(baseline_eval["support_total_cost_unit"]),
            "support_total_hop_unit": float(baseline_eval["support_total_hop_unit"]),
            "cost_cap_primary": cost_cap_primary,
            "hop_cap_primary": hop_cap_primary,
        }
        baseline_json.write_text(json.dumps(baseline_payload, indent=2))

        runtime_by_view[view] = {
            "baseline": baseline_runtime,
            "single_node_deep": deep_runtime,
            "robustness": robust_runtime,
            "interdiction": inter_runtime,
        }
        outputs_by_view[view] = {
            "node_single_removal_impacts_csv": str(impacts_csv),
            "node_single_removal_impacts_parquet": str(impacts_pq),
            "top100_high_impact_csv": str(top100_csv),
            "robustness_curves_csv": str(robust_curves_csv),
            "robustness_summary_csv": str(robust_summary_csv),
            "interdiction_performance_csv": str(inter_csv),
            "interdiction_sets_csv": str(sets_csv),
            "baseline_metrics_json": str(baseline_json),
        }
        print(f"[m6_4] done view={view}", flush=True)

    run_metadata = {
        "module": "m6_4",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "settings": {
            "views": views,
            "candidate_pool_n": candidate_pool_n,
            "deep_eval_n": deep_eval_n,
            "random_repeats": random_repeats,
            "removal_fracs": removal_fracs,
            "interdiction_k": interdiction_k,
            "interdiction_shortlist_n": interdiction_shortlist_n,
            "predicted_only_edge_cost": predicted_only_edge_cost,
            "cost_cap_primary": cost_cap_primary,
            "hop_cap_primary": hop_cap_primary,
            "seed": seed,
        },
        "graph": {
            "n_nodes": int(n_nodes),
            "n_edges_total": len(edges),
            "n_primes": len(prime_idx),
            "n_semis": len(semi_idx),
            "n_removable_firms": len(removable_idx),
        },
        "runtime_seconds_by_view": runtime_by_view,
    }
    run_meta_path = m6_4_dir / "run_metadata.json"
    run_meta_path.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m6_4",
        "snapshot": snapshot,
        "run_id": run_id,
        "inputs": {
            "node_table_contract": str(node_path),
            "edge_table_contract": str(edge_path),
            "candidate_pools_root": str(m6_1_dir),
            "prime_weights": str(weights_path),
            "node_table_contract_sha256": file_sha256(node_path),
            "edge_table_contract_sha256": file_sha256(edge_path),
            "prime_weights_sha256": file_sha256(weights_path) if weights_path.exists() else None,
        },
        "outputs": outputs_by_view,
        "run_metadata": str(run_meta_path),
    }
    manifest_path = m6_4_dir / "manifest_m6_4.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
