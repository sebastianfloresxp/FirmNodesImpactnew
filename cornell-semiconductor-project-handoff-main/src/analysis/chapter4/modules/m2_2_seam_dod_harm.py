#!/usr/bin/env python3
"""Module 2.2: score seam candidates by DoD harm metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

THIS_FILE = Path(__file__).resolve()
MODULE_DIR = THIS_FILE.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from m6_1_weighted_disruption import (
    GraphContext,
    build_prime_incoming,
    build_view_edges,
    compute_source_reach_and_scc,
    evaluate_metrics,
    get_view_mask,
    load_prime_weight_table,
    multi_source_weighted_distance,
    parse_h1_profiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 2.2 seam DoD harm")
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


def build_context_for_view(
    cfg: dict[str, Any],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    view: str,
) -> tuple[GraphContext, dict[str, int], dict[str, Any], np.ndarray, np.ndarray]:
    m6_cfg = cfg.get("m6_1", cfg.get("m6", {}))
    geodesic_tol = float(
        m6_cfg.get("geodesic_tolerance", cfg.get("m5", {}).get("geodesic_tolerance", 1e-9))
    )
    pred_only_cost = float(
        m6_cfg.get(
            "predicted_only_edge_cost", cfg.get("m5", {}).get("predicted_only_edge_cost", 3.0)
        )
    )
    h1_profiles = parse_h1_profiles(m6_cfg)
    h1_primary_key = str(m6_cfg.get("primary_h1_key", "log_obligation_any_support"))
    if not any(f"{p['weight_mode']}_{p['support_form']}" == h1_primary_key for p in h1_profiles):
        h1_primary_key = f"{h1_profiles[0]['weight_mode']}_{h1_profiles[0]['support_form']}"

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
    if view not in views:
        raise ValueError(f"view {view} not found in config.views")
    include_any = views[view].get("include_any", [])
    view_mask = get_view_mask(edges, include_any)
    src, dst, is_disclosed, is_observed, is_predicted = build_view_edges(edges, view_mask)
    pred_only = is_predicted & (~is_disclosed) & (~is_observed)
    edge_cost = np.where(pred_only, pred_only_cost, 1.0).astype(np.float64, copy=False)

    role = nodes["entity_role"].astype(str).to_numpy()
    is_prime = role == "prime_vendor"
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
    if len(prime_indices) == 0:
        raise ValueError(f"No primes found for m2_2 view={view}")
    if len(semi_indices) == 0:
        raise ValueError(f"No semis found for m2_2 view={view}")

    prime_uids = node_uids[prime_indices]
    weights_path = Path(
        str(cfg.get("paths", {}).get("prime_weights", "artifacts/ch4/inputs/prime_weights.parquet"))
    )
    weight_table, _ = load_prime_weight_table(weights_path=weights_path, prime_uids=prime_uids)
    weight_map = weight_table.set_index("analysis_uid")

    h1_weights: dict[str, np.ndarray] = {}
    for profile in h1_profiles:
        weight_mode = profile["weight_mode"]
        support_form = profile["support_form"]
        key = f"{weight_mode}_{support_form}"
        if weight_mode == "unit":
            arr = np.ones(len(prime_indices), dtype=np.float64)
        elif weight_mode == "log_obligation":
            arr = (
                weight_map["weight_log_obligation"]
                .reindex(prime_uids)
                .fillna(0.0)
                .to_numpy(np.float64)
            )
        elif weight_mode == "raw_obligation":
            arr = (
                weight_map["weight_raw_obligation"]
                .reindex(prime_uids)
                .fillna(0.0)
                .to_numpy(np.float64)
            )
        else:
            arr = np.ones(len(prime_indices), dtype=np.float64)
        h1_weights[key] = arr

    baseline_support, baseline_scc_labels, _ = compute_source_reach_and_scc(
        n_nodes=n_nodes,
        src_idx=src,
        dst_idx=dst,
        source_indices=semi_indices,
    )
    baseline_prime_support_count = baseline_support[prime_indices].astype(np.float64, copy=False)
    baseline_prime_support_any = (baseline_prime_support_count > 0).astype(np.float64, copy=False)
    h1_baseline_total: dict[str, float] = {}
    for profile in h1_profiles:
        weight_mode = profile["weight_mode"]
        support_form = profile["support_form"]
        key = f"{weight_mode}_{support_form}"
        weights = h1_weights[key]
        support_vec = (
            baseline_prime_support_count
            if support_form == "support_count"
            else baseline_prime_support_any
        )
        h1_baseline_total[key] = float(np.dot(weights, support_vec))

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
        h1_profiles=h1_profiles,
        h1_weights=h1_weights,
        h1_baseline_total=h1_baseline_total,
        h1_primary_key=h1_primary_key,
        prime_incoming=prime_incoming,
        baseline_prime_dist=baseline_prime_dist,
        baseline_avg_len=baseline_avg_len,
        baseline_finite_mask=baseline_finite_mask,
        baseline_share_single_point_supported=baseline_share_single_point_supported,
        baseline_share_no_support=baseline_share_no_support,
        geodesic_tol=geodesic_tol,
    )
    baseline = evaluate_metrics(
        context=context, removed_nodes=np.array([], dtype=np.int32), need_h2=True, need_h3=True
    )
    return context, uid_to_idx, baseline, src, dst


def evaluate_metrics_with_edge_removal(
    context: GraphContext,
    edge_pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    if not edge_pairs:
        return evaluate_metrics(
            context=context, removed_nodes=np.array([], dtype=np.int32), need_h2=True, need_h3=True
        )
    edge_mask = np.ones(len(context.src), dtype=bool)
    for a, b in edge_pairs:
        edge_mask &= ~(
            ((context.src == a) & (context.dst == b)) | ((context.src == b) & (context.dst == a))
        )

    src_active = context.src[edge_mask]
    dst_active = context.dst[edge_mask]
    cost_active = context.edge_cost[edge_mask]
    semi_active = context.semi_indices
    semi_reach_count, scc_labels, _ = compute_source_reach_and_scc(
        n_nodes=context.n_nodes,
        src_idx=src_active,
        dst_idx=dst_active,
        source_indices=semi_active.astype(np.int32, copy=False),
    )

    prime_support_count = semi_reach_count[context.prime_indices].astype(np.float64, copy=False)
    prime_support_any = (prime_support_count > 0).astype(np.float64, copy=False)
    result: dict[str, Any] = {"removed_nodes_count": 0, "removed_edges_count": len(edge_pairs)}
    for profile in context.h1_profiles:
        weight_mode = profile["weight_mode"]
        support_form = profile["support_form"]
        key = f"{weight_mode}_{support_form}"
        weights = context.h1_weights[key]
        support_vec = prime_support_count if support_form == "support_count" else prime_support_any
        total_support = float(np.dot(weights, support_vec))
        baseline_total = float(context.h1_baseline_total.get(key, 0.0))
        h1_loss = (
            float((baseline_total - total_support) / baseline_total)
            if baseline_total > 0
            else np.nan
        )
        result[f"support_total_{key}"] = total_support
        result[f"h1_{key}"] = h1_loss

    result["h1_reach_loss"] = float(result.get(f"h1_{context.h1_primary_key}", np.nan))
    result["support_total"] = float(result.get(f"support_total_{context.h1_primary_key}", np.nan))

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
    avg_len = (
        float(np.mean(prime_dist[finite_overlap])) if int(finite_overlap.sum()) > 0 else np.nan
    )
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
    result["h2_avg_path_len"] = avg_len
    result["h2_path_growth"] = h2_growth
    result["h2_disconnect_share"] = disconnect_share

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
            if semi_reach_count[pred] <= 0:
                continue
            # Keep only entries still present after edge removals.
            keep = True
            for a, b in edge_pairs:
                if (pred == a and prime_node == b) or (pred == b and prime_node == a):
                    keep = False
                    break
            if keep:
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
    result["h3_share_no_support"] = share_no_support
    result["h3_share_single_point_supported"] = share_single_point_supported
    result["h3_redundancy_collapse"] = (
        float(share_single_point_supported - context.baseline_share_single_point_supported)
        if np.isfinite(share_single_point_supported)
        else np.nan
    )
    result["h3_no_support_increase"] = float(share_no_support - context.baseline_share_no_support)
    return result


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    m0_dir = out_root / snapshot / "m0"
    m2_dir = out_root / snapshot / "m2"
    out_dir = out_root / snapshot / "m2_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    views = [
        str(v)
        for v in cfg.get("m2_2", {}).get(
            "views", cfg.get("m6_1", {}).get("views", ["disclosed", "observed", "full"])
        )
    ]
    top_n_art = int(cfg.get("m2_2", {}).get("top_n_articulation", 500))
    top_n_bridge = int(cfg.get("m2_2", {}).get("top_n_bridge", 300))

    node_path = m0_dir / "node_table_contract.parquet"
    edge_path = m0_dir / "edge_table_contract.parquet"
    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)

    input_hashes: dict[str, str] = {
        str(node_path): file_sha256(node_path),
        str(edge_path): file_sha256(edge_path),
    }
    outputs: dict[str, dict[str, str]] = {}
    runtime_by_view: dict[str, float] = {}

    for view in views:
        t0 = time.perf_counter()
        seam_path = m2_dir / f"seam_nodes_{view}.csv"
        if not seam_path.exists():
            raise FileNotFoundError(seam_path)
        input_hashes[str(seam_path)] = file_sha256(seam_path)

        context, uid_to_idx, baseline_metrics, _, _ = build_context_for_view(
            cfg=cfg, nodes=nodes, edges=edges, view=view
        )
        seam_df = pd.read_csv(seam_path)

        art = seam_df[
            (seam_df["seam_type"] == "articulation") & seam_df["analysis_uid"].notna()
        ].copy()
        art = art.sort_values(["proxy_score", "analysis_uid"], ascending=[False, True]).head(
            top_n_art
        )

        bridge = seam_df[
            (seam_df["seam_type"] == "bridge")
            & seam_df["src_uid"].notna()
            & seam_df["dst_uid"].notna()
        ].copy()
        bridge = bridge.sort_values(
            ["proxy_score", "src_uid", "dst_uid"], ascending=[False, True, True]
        ).head(top_n_bridge)

        rows: list[dict[str, Any]] = []
        for i, row in enumerate(art.itertuples(index=False), start=1):
            uid = str(row.analysis_uid)
            idx = uid_to_idx.get(uid)
            if idx is None:
                continue
            metrics = evaluate_metrics(
                context=context,
                removed_nodes=np.array([idx], dtype=np.int32),
                need_h2=True,
                need_h3=True,
            )
            rec = {
                "view": view,
                "seam_type": "articulation",
                "analysis_uid": uid,
                "src_uid": "",
                "dst_uid": "",
            }
            rec.update(
                {
                    k: getattr(row, k)
                    for k in art.columns
                    if k in {"proxy_score", "is_touch_prime", "second_component", "split_imbalance"}
                }
            )
            rec.update(metrics)
            rows.append(rec)
            if i % 50 == 0:
                print(f"[m2_2] articulation evaluated {i}/{len(art)} view={view}")

        for i, row in enumerate(bridge.itertuples(index=False), start=1):
            src_uid = str(row.src_uid)
            dst_uid = str(row.dst_uid)
            a = uid_to_idx.get(src_uid)
            b = uid_to_idx.get(dst_uid)
            if a is None or b is None:
                continue
            metrics = evaluate_metrics_with_edge_removal(context=context, edge_pairs=[(a, b)])
            rec = {
                "view": view,
                "seam_type": "bridge",
                "analysis_uid": "",
                "src_uid": src_uid,
                "dst_uid": dst_uid,
            }
            rec.update(
                {
                    k: getattr(row, k)
                    for k in bridge.columns
                    if k
                    in {
                        "proxy_score",
                        "is_touch_prime",
                        "second_component",
                        "split_imbalance",
                        "component_u",
                        "component_v",
                    }
                }
            )
            rec.update(metrics)
            rows.append(rec)
            if i % 50 == 0:
                print(f"[m2_2] bridge evaluated {i}/{len(bridge)} view={view}")

        harm_df = pd.DataFrame(rows)
        harm_out = out_dir / f"seam_dod_harm_{view}.csv"
        harm_df.to_csv(harm_out, index=False)

        top_node_out = out_dir / f"top_seam_nodes_dod_harm_{view}.csv"
        harm_df[harm_df["seam_type"] == "articulation"].sort_values(
            ["h1_reach_loss", "analysis_uid"], ascending=[False, True]
        ).head(100).to_csv(top_node_out, index=False)
        top_edge_out = out_dir / f"top_seam_edges_dod_harm_{view}.csv"
        harm_df[harm_df["seam_type"] == "bridge"].sort_values(
            ["h1_reach_loss", "src_uid", "dst_uid"], ascending=[False, True, True]
        ).head(100).to_csv(top_edge_out, index=False)

        summary = (
            harm_df.groupby("seam_type", dropna=False)[
                ["h1_reach_loss", "h2_path_growth", "h2_disconnect_share", "h3_no_support_increase"]
            ]
            .agg(["count", "mean", "median", "max"])
            .reset_index()
        )
        summary.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            for col in summary.columns.to_list()
        ]
        summary_out = out_dir / f"seam_dod_harm_summary_{view}.csv"
        summary.to_csv(summary_out, index=False)

        baseline_out = out_dir / f"baseline_metrics_{view}.json"
        baseline_out.write_text(json.dumps(baseline_metrics, indent=2))

        runtime = time.perf_counter() - t0
        runtime_by_view[view] = float(runtime)
        outputs[view] = {
            "seam_dod_harm": str(harm_out),
            "top_seam_nodes_dod_harm": str(top_node_out),
            "top_seam_edges_dod_harm": str(top_edge_out),
            "seam_dod_harm_summary": str(summary_out),
            "baseline_metrics": str(baseline_out),
        }
        print(f"[m2_2] completed view={view} runtime_s={runtime:.2f}")

    run_metadata = {
        "module": "m2_2",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "platform": platform.platform(),
        "settings": {
            "views": views,
            "top_n_articulation": top_n_art,
            "top_n_bridge": top_n_bridge,
        },
        "runtime_seconds_by_view": runtime_by_view,
    }
    run_metadata_path = out_dir / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m2_2",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs_sha256": input_hashes,
        "outputs": outputs,
        "run_metadata": str(run_metadata_path),
    }
    manifest_path = out_dir / "manifest_m2_2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {manifest_path}")
    print(f"[done] wrote {run_metadata_path}")


if __name__ == "__main__":
    main()
