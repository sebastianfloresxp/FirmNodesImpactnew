#!/usr/bin/env python3
"""Module 5: prime-level redundancy/substitution proxy metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 5 redundancy/substitution proxy")
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


def rank_desc(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values.astype(float))
    return series.rank(method="min", ascending=False).astype(np.int32).to_numpy()


def build_view_edges(
    edges: pd.DataFrame,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    needed_cols = ["src_idx", "dst_idx", "is_predicted", "is_disclosed", "is_observed_ship"]
    ve = edges.loc[mask, needed_cols].copy()
    ve = ve[ve["src_idx"] != ve["dst_idx"]]
    ve = (
        ve.groupby(["src_idx", "dst_idx"], as_index=False)[
            ["is_predicted", "is_disclosed", "is_observed_ship"]
        ]
        .max()
        .sort_values(["src_idx", "dst_idx"], ascending=[True, True], kind="mergesort")
    )
    src = ve["src_idx"].to_numpy(np.int32, copy=False)
    dst = ve["dst_idx"].to_numpy(np.int32, copy=False)
    is_pred = ve["is_predicted"].astype(bool).to_numpy()
    is_disc = ve["is_disclosed"].astype(bool).to_numpy()
    is_obs = ve["is_observed_ship"].astype(bool).to_numpy()
    return src, dst, is_pred, is_disc, is_obs


def load_m4_refined(
    m4_refine_dir: Path,
    view_name: str,
    node_uids: np.ndarray,
) -> pd.DataFrame:
    path = m4_refine_dir / f"corridor_nodes_refined_{view_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"M5 requires M4.1 output missing: {path}")
    cols = [
        "analysis_uid",
        "is_corridor",
        "semi_reach_count",
        "dist_from_semi_weighted",
    ]
    m4r = pd.read_parquet(path, columns=cols)
    merged = pd.DataFrame({"analysis_uid": node_uids}).merge(m4r, on="analysis_uid", how="left")
    if merged.isna().any().any():
        missing = int(merged["semi_reach_count"].isna().sum())
        raise ValueError(f"M4.1 metrics missing for {missing} nodes in view {view_name}")
    return merged


def classify_redundancy(semi_support_count: int, redundancy_proxy: int) -> str:
    if semi_support_count <= 0:
        return "no_semi_support"
    if redundancy_proxy <= 1:
        return "single_point_exposed"
    if redundancy_proxy <= 3:
        return "low_redundancy"
    return "higher_redundancy"


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "unspecified_snapshot"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2")))
    m0_dir = out_root / snapshot / "m0"
    m4r_dir = out_root / snapshot / "m4_refine"
    m5_dir = out_root / snapshot / "m5"
    m5_dir.mkdir(parents=True, exist_ok=True)

    node_path = m0_dir / "node_table_contract.parquet"
    edge_path = m0_dir / "edge_table_contract.parquet"
    if not node_path.exists():
        raise FileNotFoundError(node_path)
    if not edge_path.exists():
        raise FileNotFoundError(edge_path)

    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)

    node_uids = nodes["analysis_uid"].astype(str).to_numpy()
    uid_to_idx = {uid: i for i, uid in enumerate(node_uids.tolist())}
    n_nodes = len(node_uids)

    src = edges["src_uid"].astype(str).map(uid_to_idx)
    dst = edges["dst_uid"].astype(str).map(uid_to_idx)
    valid = src.notna() & dst.notna()
    edges = edges.loc[valid].copy()
    edges["src_idx"] = src.loc[valid].astype(np.int32).to_numpy()
    edges["dst_idx"] = dst.loc[valid].astype(np.int32).to_numpy()

    role = (
        nodes["entity_role"].astype(str)
        if "entity_role" in nodes.columns
        else pd.Series("unknown", index=nodes.index)
    )
    is_prime = role.eq("prime_vendor").to_numpy(bool)
    is_supplier_firm = role.eq("firm").to_numpy(bool)
    name = nodes["name"] if "name" in nodes.columns else pd.Series([None] * n_nodes)
    is_semi = (
        nodes["is_semi_strict"].fillna(False).to_numpy(bool)
        if "is_semi_strict" in nodes.columns
        else np.zeros(n_nodes, dtype=bool)
    )

    prime_indices = np.flatnonzero(is_prime).astype(np.int32, copy=False)
    n_primes = len(prime_indices)
    n_semis = int(is_semi.sum())
    if n_primes == 0:
        raise ValueError("No prime nodes found (entity_role == 'prime_vendor')")
    if n_semis == 0:
        raise ValueError("No strict semiconductor nodes found (is_semi_strict == True)")

    views = cfg.get("views", {})
    if not isinstance(views, dict) or not views:
        raise ValueError("config.views must be a non-empty mapping")

    m5_cfg = cfg.get("m5", {})
    use_supplier_only_entry = bool(m5_cfg.get("use_supplier_only_entry", True))
    geodesic_tol = float(m5_cfg.get("geodesic_tolerance", 1e-9))
    pred_only_cost = float(
        m5_cfg.get(
            "predicted_only_edge_cost",
            cfg.get("m4_refine", {}).get("predicted_only_edge_cost", 3.0),
        )
    )
    if pred_only_cost < 1.0:
        raise ValueError("m5.predicted_only_edge_cost must be >= 1")
    primary_view = str(m5_cfg.get("primary_view", "full"))

    prime_outputs: dict[str, str] = {}
    single_outputs: dict[str, str] = {}
    view_stats: dict[str, Any] = {}
    all_view_tables: dict[str, pd.DataFrame] = {}

    for view_name, view_spec in views.items():
        t_view = time.perf_counter()
        include_any = view_spec.get("include_any", [])
        mask = get_view_mask(edges, include_any)

        t0 = time.perf_counter()
        src_idx, dst_idx, is_pred, is_disc, is_obs = build_view_edges(edges, mask)
        t_edges = time.perf_counter() - t0

        t0 = time.perf_counter()
        m4r = load_m4_refined(m4_refine_dir=m4r_dir, view_name=view_name, node_uids=node_uids)
        is_corridor = m4r["is_corridor"].astype(bool).to_numpy()
        semi_reach_count = m4r["semi_reach_count"].astype(np.int32).to_numpy()
        dist_from_semi_weighted = m4r["dist_from_semi_weighted"].astype(np.float64).to_numpy()
        t_m4 = time.perf_counter() - t0

        pred_only = is_pred & (~is_disc) & (~is_obs)
        edge_cost = np.where(pred_only, pred_only_cost, 1.0).astype(np.float64, copy=False)

        t0 = time.perf_counter()
        matrix = csr_matrix(
            (np.ones(len(src_idx), dtype=np.int8), (src_idx, dst_idx)), shape=(n_nodes, n_nodes)
        )
        _, scc_labels = connected_components(
            matrix, directed=True, connection="strong", return_labels=True
        )
        scc_labels = scc_labels.astype(np.int32, copy=False)
        t_scc = time.perf_counter() - t0

        t0 = time.perf_counter()
        incoming: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for u, v, c in zip(src_idx.tolist(), dst_idx.tolist(), edge_cost.tolist(), strict=False):
            if is_prime[v]:
                incoming[v].append((u, float(c)))
        t_incoming = time.perf_counter() - t0

        t0 = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for p_idx in prime_indices.tolist():
            pred_list = incoming.get(p_idx, [])
            if use_supplier_only_entry:
                candidates = [(u, c) for (u, c) in pred_list if is_supplier_firm[u]]
            else:
                candidates = pred_list

            corridor_candidates = [(u, c) for (u, c) in candidates if is_corridor[u]]
            entry_branch_count = len(corridor_candidates)
            entry_scc_count = len({int(scc_labels[u]) for (u, _) in corridor_candidates})

            prime_dist = float(dist_from_semi_weighted[p_idx])
            geodesic_entry_count = 0
            if np.isfinite(prime_dist):
                for u, c in corridor_candidates:
                    u_dist = float(dist_from_semi_weighted[u])
                    if np.isfinite(u_dist) and np.isclose(
                        u_dist + c, prime_dist, atol=geodesic_tol, rtol=0.0
                    ):
                        geodesic_entry_count += 1

            semi_support = int(semi_reach_count[p_idx])
            redundancy_proxy = int(
                min(semi_support, entry_branch_count, entry_scc_count, geodesic_entry_count)
            )
            redundancy_class = classify_redundancy(
                semi_support_count=semi_support, redundancy_proxy=redundancy_proxy
            )
            is_single_point_exposed = bool(semi_support > 0 and redundancy_proxy <= 1)
            is_no_semi_support = bool(semi_support == 0)

            rows.append(
                {
                    "view": view_name,
                    "analysis_uid": node_uids[p_idx],
                    "name": name.iloc[p_idx],
                    "entity_role": role.iloc[p_idx],
                    "is_tier1_prime": True,
                    "semi_support_count": semi_support,
                    "semi_support_share": float(semi_support / max(n_semis, 1)),
                    "entry_branch_count": entry_branch_count,
                    "entry_scc_count": entry_scc_count,
                    "geodesic_entry_count": int(geodesic_entry_count),
                    "redundancy_proxy": redundancy_proxy,
                    "redundancy_class": redundancy_class,
                    "is_single_point_exposed": is_single_point_exposed,
                    "is_no_semi_support": is_no_semi_support,
                }
            )

        prime_df = pd.DataFrame(rows)
        prime_df["rank_vulnerability_redundancy"] = rank_desc(
            -prime_df["redundancy_proxy"].to_numpy(np.float64)
        )
        prime_df["rank_semi_support_count"] = rank_desc(
            prime_df["semi_support_count"].to_numpy(np.float64)
        )
        prime_df = prime_df.sort_values(
            ["rank_vulnerability_redundancy", "semi_support_count", "analysis_uid"],
            ascending=[True, True, True],
        )

        single_df = prime_df.loc[prime_df["redundancy_proxy"] <= 1].sort_values(
            ["is_no_semi_support", "redundancy_proxy", "semi_support_count", "analysis_uid"],
            ascending=[False, True, True, True],
        )

        out_prime = m5_dir / f"prime_redundancy_{view_name}.csv"
        out_single = m5_dir / f"redundancy_single_point_primes_{view_name}.csv"
        prime_df.to_csv(out_prime, index=False)
        single_df.to_csv(out_single, index=False)
        prime_outputs[view_name] = str(out_prime)
        single_outputs[view_name] = str(out_single)
        all_view_tables[view_name] = prime_df

        view_stats[view_name] = {
            "edges_view": len(src_idx),
            "predicted_only_edges_view": int(pred_only.sum()),
            "n_primes": len(prime_df),
            "n_primes_single_point_exposed": int(prime_df["is_single_point_exposed"].sum()),
            "share_primes_single_point_exposed": float(prime_df["is_single_point_exposed"].mean()),
            "n_primes_no_semi_support": int(prime_df["is_no_semi_support"].sum()),
            "share_primes_no_semi_support": float(prime_df["is_no_semi_support"].mean()),
            "redundancy_proxy_median": float(prime_df["redundancy_proxy"].median()),
            "redundancy_proxy_p90": float(prime_df["redundancy_proxy"].quantile(0.9)),
            "runtime_seconds": {
                "build_view_edges": t_edges,
                "load_m4_refined": t_m4,
                "compute_scc": t_scc,
                "build_prime_incoming": t_incoming,
                "compute_prime_metrics": time.perf_counter() - t0,
                "view_total": time.perf_counter() - t_view,
            },
        }

    monotone_rows: list[dict[str, Any]] = []
    ordered = [k for k in ["disclosed", "observed", "full"] if k in all_view_tables]
    if len(ordered) >= 2:
        for metric in [
            "semi_support_count",
            "entry_branch_count",
            "entry_scc_count",
            "geodesic_entry_count",
            "redundancy_proxy",
        ]:
            base = all_view_tables[ordered[0]][["analysis_uid", metric]].rename(
                columns={metric: ordered[0]}
            )
            comp = base
            for view in ordered[1:]:
                comp = comp.merge(
                    all_view_tables[view][["analysis_uid", metric]].rename(columns={metric: view}),
                    on="analysis_uid",
                    how="inner",
                )
            monotone = np.ones(len(comp), dtype=bool)
            for i in range(1, len(ordered)):
                monotone &= comp[ordered[i]].to_numpy() >= comp[ordered[i - 1]].to_numpy()
            monotone_rows.append(
                {
                    "metric": metric,
                    "views_order": " <= ".join(ordered),
                    "share_primes_monotone_non_decreasing": float(monotone.mean())
                    if len(monotone)
                    else np.nan,
                    "n_primes_compared": len(monotone),
                }
            )
    monotone_df = pd.DataFrame(monotone_rows)
    monotone_out = m5_dir / "redundancy_monotonicity_checks.csv"
    monotone_df.to_csv(monotone_out, index=False)

    run_metadata = {
        "module": "m5",
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
            "use_supplier_only_entry": use_supplier_only_entry,
            "predicted_only_edge_cost": pred_only_cost,
            "geodesic_tolerance": geodesic_tol,
            "primary_view": primary_view,
        },
        "view_stats": view_stats,
    }
    run_metadata_out = m5_dir / "run_metadata.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m5",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs": {
            "node_table_contract": str(node_path),
            "edge_table_contract": str(edge_path),
            "node_sha256": file_sha256(node_path),
            "edge_sha256": file_sha256(edge_path),
            "m4_refine_disclosed": str(m4r_dir / "corridor_nodes_refined_disclosed.parquet"),
            "m4_refine_observed": str(m4r_dir / "corridor_nodes_refined_observed.parquet"),
            "m4_refine_full": str(m4r_dir / "corridor_nodes_refined_full.parquet"),
        },
        "params": {
            "views": views,
            "use_supplier_only_entry": use_supplier_only_entry,
            "predicted_only_edge_cost": pred_only_cost,
            "geodesic_tolerance": geodesic_tol,
            "primary_view": primary_view,
        },
        "outputs": {
            "prime_redundancy": prime_outputs,
            "redundancy_single_point_primes": single_outputs,
            "redundancy_monotonicity_checks": str(monotone_out),
            "run_metadata": str(run_metadata_out),
        },
    }
    manifest_out = m5_dir / "manifest_m5.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))

    for _view_name, out in prime_outputs.items():
        print(f"[done] wrote {out}")
    for _view_name, out in single_outputs.items():
        print(f"[done] wrote {out}")
    print(f"[done] wrote {monotone_out}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")


if __name__ == "__main__":
    main()
