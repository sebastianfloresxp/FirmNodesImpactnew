#!/usr/bin/env python3
"""Module 3.1: supplier-centric DoD exposure/common-mode refinement."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 3.1 supplier-centric refinement")
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


def get_view_mask(
    edges: pd.DataFrame, include_any: list[str], exclude_any: list[str]
) -> np.ndarray:
    if not include_any:
        raise ValueError("include_any must not be empty")
    missing_include = [col for col in include_any if col not in edges.columns]
    if missing_include:
        raise ValueError(f"missing include view flags: {missing_include}")

    mask = np.zeros(len(edges), dtype=bool)
    for col in include_any:
        mask |= edges[col].astype(bool).to_numpy()

    for col in exclude_any:
        if col in edges.columns:
            mask &= ~edges[col].astype(bool).to_numpy()
    return mask


def rank_desc(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values.astype(float))
    return series.rank(method="min", ascending=False).astype(np.int32).to_numpy()


def load_prime_weights(
    cfg: dict[str, Any],
    prime_uids: np.ndarray,
    key: str = "m3_refine",
) -> tuple[np.ndarray, str, dict[str, Any]]:
    module_cfg = cfg.get(key, {})
    mode = str(module_cfg.get("prime_weights_mode", "unit")).strip().lower()
    paths_cfg = cfg.get("paths", {})
    weights_path = (
        Path(str(paths_cfg.get("prime_weights", ""))) if paths_cfg.get("prime_weights") else None
    )
    diagnostics: dict[str, Any] = {
        "mode_requested": mode,
        "weights_path": str(weights_path) if weights_path else None,
        "matched_primes": 0,
        "missing_primes": len(prime_uids),
        "fallback_to_unit": False,
    }

    if mode == "unit":
        return np.ones(len(prime_uids), dtype=np.float64), "unit", diagnostics

    if weights_path is None or not weights_path.exists():
        diagnostics["fallback_to_unit"] = True
        return np.ones(len(prime_uids), dtype=np.float64), "unit_fallback_missing_file", diagnostics

    if weights_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(weights_path)
    else:
        df = pd.read_csv(weights_path)

    uid_candidates = ["analysis_uid", "prime_uid", "uid", "node_uid"]
    weight_candidates = ["obligation_weight", "weight", "w_p", "total_obligations"]
    uid_col = next((c for c in uid_candidates if c in df.columns), None)
    weight_col = next((c for c in weight_candidates if c in df.columns), None)

    if uid_col is None or weight_col is None:
        diagnostics["fallback_to_unit"] = True
        diagnostics["missing_columns"] = {"uid_col": uid_col, "weight_col": weight_col}
        return np.ones(len(prime_uids), dtype=np.float64), "unit_fallback_bad_schema", diagnostics

    map_df = df[[uid_col, weight_col]].copy()
    map_df[uid_col] = map_df[uid_col].astype(str)
    map_df[weight_col] = pd.to_numeric(map_df[weight_col], errors="coerce").fillna(0.0)
    weights_map = map_df.groupby(uid_col, as_index=True)[weight_col].sum()
    weights = weights_map.reindex(pd.Index(prime_uids.astype(str))).fillna(1.0).to_numpy(np.float64)

    diagnostics["matched_primes"] = int(
        weights_map.index.intersection(pd.Index(prime_uids.astype(str))).shape[0]
    )
    diagnostics["missing_primes"] = int(len(prime_uids) - diagnostics["matched_primes"])
    diagnostics["weights_min"] = float(weights.min()) if len(weights) else 0.0
    diagnostics["weights_max"] = float(weights.max()) if len(weights) else 0.0
    diagnostics["weights_sum"] = float(weights.sum()) if len(weights) else 0.0
    return weights, "external", diagnostics


def build_reversed_adjacency(
    n_nodes: int,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
) -> list[list[int]]:
    reverse_adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for src, dst in zip(src_idx.tolist(), dst_idx.tolist(), strict=False):
        if src == dst:
            continue
        reverse_adj[dst].append(src)
    return reverse_adj


def compute_reach_bfs_capped(
    n_nodes: int,
    reverse_adj: list[list[int]],
    prime_indices: np.ndarray,
    prime_weights: np.ndarray,
    hop_cap: int,
    exclude_self: bool,
    is_semi: np.ndarray,
    is_prime: np.ndarray,
    is_dod: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_prime_count = np.zeros(n_nodes, dtype=np.int32)
    node_prime_weight = np.zeros(n_nodes, dtype=np.float64)
    up_count = np.zeros(len(prime_indices), dtype=np.int64)
    up_semi_count = np.zeros(len(prime_indices), dtype=np.int64)
    up_prime_count = np.zeros(len(prime_indices), dtype=np.int64)
    up_dod_count = np.zeros(len(prime_indices), dtype=np.int64)

    visit_mark = np.zeros(n_nodes, dtype=np.int32)
    mark = 0

    for pos, src in enumerate(prime_indices.tolist()):
        mark += 1
        q: deque[tuple[int, int]] = deque()
        q.append((src, 0))
        visit_mark[src] = mark

        while q:
            node, depth = q.popleft()
            include = not (exclude_self and node == src)
            if include:
                node_prime_count[node] += 1
                node_prime_weight[node] += prime_weights[pos]
                up_count[pos] += 1
                if is_semi[node]:
                    up_semi_count[pos] += 1
                if is_prime[node]:
                    up_prime_count[pos] += 1
                if is_dod[node]:
                    up_dod_count[pos] += 1
            if depth >= hop_cap:
                continue
            for pred in reverse_adj[node]:
                if visit_mark[pred] != mark:
                    visit_mark[pred] = mark
                    q.append((pred, depth + 1))

    return (
        node_prime_count,
        node_prime_weight,
        up_count,
        up_semi_count,
        up_prime_count,
        up_dod_count,
    )


def compute_reach_full_scc(
    n_nodes: int,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    prime_indices: np.ndarray,
    prime_weights: np.ndarray,
    exclude_self: bool,
    is_semi: np.ndarray,
    is_prime: np.ndarray,
    is_dod: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    dedup = pd.DataFrame({"s": src_idx, "d": dst_idx}).drop_duplicates(ignore_index=True)
    dedup = dedup[dedup["s"] != dedup["d"]]
    s = dedup["s"].to_numpy(np.int32, copy=False)
    d = dedup["d"].to_numpy(np.int32, copy=False)

    matrix = csr_matrix((np.ones(len(s), dtype=np.int8), (s, d)), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(
        matrix, directed=True, connection="strong", return_labels=True
    )
    labels = labels.astype(np.int32, copy=False)
    comp_sizes = np.bincount(labels, minlength=n_comp).astype(np.int64, copy=False)

    comp_semi_sizes = np.bincount(
        labels, weights=is_semi.astype(np.int64), minlength=n_comp
    ).astype(np.int64, copy=False)
    comp_prime_sizes = np.bincount(
        labels, weights=is_prime.astype(np.int64), minlength=n_comp
    ).astype(np.int64, copy=False)
    comp_dod_sizes = np.bincount(labels, weights=is_dod.astype(np.int64), minlength=n_comp).astype(
        np.int64, copy=False
    )

    prime_pos = np.full(n_nodes, -1, dtype=np.int32)
    prime_pos[prime_indices] = np.arange(len(prime_indices), dtype=np.int32)

    comp_prime_bits: list[int] = [0] * n_comp
    for node_idx in prime_indices.tolist():
        comp_id = int(labels[node_idx])
        bit = 1 << int(prime_pos[node_idx])
        comp_prime_bits[comp_id] |= bit

    comp_src = labels[s]
    comp_dst = labels[d]
    keep = comp_src != comp_dst
    if int(keep.sum()) > 0:
        cond = np.unique(np.stack([comp_src[keep], comp_dst[keep]], axis=1), axis=0)
        cu = cond[:, 0].astype(np.int32, copy=False)
        cv = cond[:, 1].astype(np.int32, copy=False)
    else:
        cu = np.array([], dtype=np.int32)
        cv = np.array([], dtype=np.int32)

    succ: list[list[int]] = [[] for _ in range(n_comp)]
    indeg = np.zeros(n_comp, dtype=np.int32)
    for u, v in zip(cu.tolist(), cv.tolist(), strict=False):
        succ[u].append(v)
        indeg[v] += 1

    topo: list[int] = []
    q: deque[int] = deque(np.flatnonzero(indeg == 0).tolist())
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != n_comp:
        raise RuntimeError("condensation graph topological sort failed")

    comp_reach_bits: list[int] = [0] * n_comp
    for u in reversed(topo):
        bits = comp_prime_bits[u]
        for v in succ[u]:
            bits |= comp_reach_bits[v]
        comp_reach_bits[u] = bits

    comp_reach_count = np.fromiter(
        (int(bits.bit_count()) for bits in comp_reach_bits), dtype=np.int32, count=n_comp
    )
    node_prime_count = comp_reach_count[labels]

    if np.allclose(prime_weights, 1.0):
        comp_reach_weight = comp_reach_count.astype(np.float64, copy=False)
    else:
        comp_reach_weight = np.zeros(n_comp, dtype=np.float64)
        for comp_id in range(n_comp):
            bits = comp_reach_bits[comp_id]
            total = 0.0
            while bits:
                lsb = bits & -bits
                bit_pos = lsb.bit_length() - 1
                total += float(prime_weights[bit_pos])
                bits ^= lsb
            comp_reach_weight[comp_id] = total
    node_prime_weight = comp_reach_weight[labels]

    up_count = np.zeros(len(prime_indices), dtype=np.int64)
    up_semi_count = np.zeros(len(prime_indices), dtype=np.int64)
    up_prime_count = np.zeros(len(prime_indices), dtype=np.int64)
    up_dod_count = np.zeros(len(prime_indices), dtype=np.int64)
    for comp_id in range(n_comp):
        bits = comp_reach_bits[comp_id]
        if bits == 0:
            continue
        size_total = int(comp_sizes[comp_id])
        size_semi = int(comp_semi_sizes[comp_id])
        size_prime = int(comp_prime_sizes[comp_id])
        size_dod = int(comp_dod_sizes[comp_id])
        while bits:
            lsb = bits & -bits
            bit_pos = lsb.bit_length() - 1
            up_count[bit_pos] += size_total
            up_semi_count[bit_pos] += size_semi
            up_prime_count[bit_pos] += size_prime
            up_dod_count[bit_pos] += size_dod
            bits ^= lsb

    if exclude_self:
        up_count = np.maximum(up_count - 1, 0)
        up_prime_count = np.maximum(up_prime_count - 1, 0)

    diag = {
        "n_scc": int(n_comp),
        "condensation_edges": len(cu),
        "largest_scc_size": int(comp_sizes.max()) if len(comp_sizes) else 0,
    }
    return (
        node_prime_count,
        node_prime_weight,
        up_count,
        up_semi_count,
        up_prime_count,
        up_dod_count,
        diag,
    )


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
    m3r_dir = out_root / snapshot / "m3_refine"
    m3r_dir.mkdir(parents=True, exist_ok=True)

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
    is_dod = role.eq("dod_component").to_numpy(bool)
    is_semi = (
        nodes["is_semi_strict"].fillna(False).to_numpy(bool)
        if "is_semi_strict" in nodes.columns
        else np.zeros(n_nodes, dtype=bool)
    )
    name = nodes["name"] if "name" in nodes.columns else pd.Series([None] * n_nodes)

    prime_indices = np.flatnonzero(is_prime).astype(np.int32, copy=False)
    prime_uids = node_uids[prime_indices]
    n_primes = len(prime_indices)
    if n_primes == 0:
        raise ValueError("No prime nodes found (entity_role == 'prime_vendor')")

    prime_weights, prime_weight_source, prime_weight_diag = load_prime_weights(
        cfg, prime_uids, key="m3_refine"
    )
    total_prime_weight = float(prime_weights.sum())

    views = cfg.get("views", {})
    if not isinstance(views, dict) or not views:
        raise ValueError("config.views must be a non-empty mapping")

    m3r_cfg = cfg.get("m3_refine", {})
    exclude_flags = [str(x) for x in m3r_cfg.get("exclude_edge_flags", ["is_contract_seed"])]
    hop_cap_raw = m3r_cfg.get("hop_cap", None)
    if hop_cap_raw is None or str(hop_cap_raw).strip().lower() in {"full", "none", "null", ""}:
        hop_mode = "full"
        hop_cap = None
    else:
        hop_mode = "capped"
        hop_cap = int(hop_cap_raw)
        if hop_cap < 0:
            raise ValueError("m3_refine.hop_cap must be >= 0 when capped")
    exclude_self = bool(m3r_cfg.get("exclude_self_from_upstream", True))
    top_n = int(m3r_cfg.get("top_n_export", 100))
    top_n_firm = int(m3r_cfg.get("export_firm_top_n", top_n))
    top_n_semi = int(m3r_cfg.get("export_semi_top_n", top_n))

    prime_outputs: dict[str, str] = {}
    node_outputs: dict[str, str] = {}
    common_all_outputs: dict[str, str] = {}
    common_firm_outputs: dict[str, str] = {}
    common_semi_outputs: dict[str, str] = {}
    view_stats: dict[str, Any] = {}

    for view_name, view_spec in views.items():
        t_view = time.perf_counter()
        include_any = view_spec.get("include_any", [])
        mask = get_view_mask(edges, include_any, exclude_flags)
        ve = edges.loc[mask, ["src_idx", "dst_idx"]]
        src_idx = ve["src_idx"].to_numpy(np.int32, copy=False)
        dst_idx = ve["dst_idx"].to_numpy(np.int32, copy=False)

        t0 = time.perf_counter()
        scc_diag: dict[str, Any] = {}
        if hop_mode == "full":
            (
                node_prime_count,
                node_prime_weight,
                up_count,
                up_semi_count,
                up_prime_count,
                up_dod_count,
                scc_diag,
            ) = compute_reach_full_scc(
                n_nodes=n_nodes,
                src_idx=src_idx,
                dst_idx=dst_idx,
                prime_indices=prime_indices,
                prime_weights=prime_weights,
                exclude_self=exclude_self,
                is_semi=is_semi,
                is_prime=is_prime,
                is_dod=is_dod,
            )
            method_runtime = time.perf_counter() - t0
        else:
            reverse_adj = build_reversed_adjacency(
                n_nodes=n_nodes, src_idx=src_idx, dst_idx=dst_idx
            )
            (
                node_prime_count,
                node_prime_weight,
                up_count,
                up_semi_count,
                up_prime_count,
                up_dod_count,
            ) = compute_reach_bfs_capped(
                n_nodes=n_nodes,
                reverse_adj=reverse_adj,
                prime_indices=prime_indices,
                prime_weights=prime_weights,
                hop_cap=int(hop_cap),
                exclude_self=exclude_self,
                is_semi=is_semi,
                is_prime=is_prime,
                is_dod=is_dod,
            )
            method_runtime = time.perf_counter() - t0

        up_nonprime_count = np.maximum(up_count - up_prime_count, 0)
        denom_nodes = max(n_nodes - (1 if exclude_self else 0), 1)
        denom_semis = max(int(is_semi.sum()), 1)

        prime_df = pd.DataFrame(
            {
                "view": view_name,
                "analysis_uid": prime_uids,
                "name": name.iloc[prime_indices].to_numpy(),
                "entity_role": role.iloc[prime_indices].to_numpy(),
                "is_tier1_prime": True,
                "prime_weight": prime_weights,
                "upstream_nodes_count": up_count.astype(np.int64),
                "upstream_nodes_share": up_count.astype(np.float64) / float(denom_nodes),
                "upstream_semi_count": up_semi_count.astype(np.int64),
                "upstream_semi_share": up_semi_count.astype(np.float64) / float(denom_semis),
                "upstream_prime_count": up_prime_count.astype(np.int64),
                "upstream_nonprime_count": up_nonprime_count.astype(np.int64),
                "upstream_dod_count": up_dod_count.astype(np.int64),
                "rank_upstream_nodes_count": rank_desc(up_count.astype(np.float64)),
                "rank_upstream_semi_count": rank_desc(up_semi_count.astype(np.float64)),
            }
        ).sort_values(["rank_upstream_nodes_count", "analysis_uid"], ascending=[True, True])

        node_df = pd.DataFrame(
            {
                "view": view_name,
                "analysis_uid": node_uids,
                "name": name.to_numpy(),
                "entity_role": role.to_numpy(),
                "is_tier1_prime": is_prime,
                "is_dod_component": is_dod,
                "is_supplier_firm": role.eq("firm").to_numpy(bool),
                "is_semi_strict": is_semi,
                "prime_reach_count": node_prime_count.astype(np.int32),
                "prime_reach_share": node_prime_count.astype(np.float64) / float(max(n_primes, 1)),
                "prime_reach_weighted": node_prime_weight.astype(np.float64),
                "prime_reach_weighted_share": node_prime_weight.astype(np.float64)
                / float(max(total_prime_weight, 1.0)),
                "is_upstream_to_any_prime": node_prime_count > 0,
                "rank_prime_reach_count": rank_desc(node_prime_count.astype(np.float64)),
                "rank_prime_reach_weighted": rank_desc(node_prime_weight.astype(np.float64)),
            }
        ).sort_values(["rank_prime_reach_weighted", "analysis_uid"], ascending=[True, True])

        common_all = node_df.sort_values(
            ["prime_reach_weighted", "prime_reach_count", "analysis_uid"],
            ascending=[False, False, True],
        ).head(top_n)
        common_firm = (
            node_df.loc[node_df["is_supplier_firm"]]
            .sort_values(
                ["prime_reach_weighted", "prime_reach_count", "analysis_uid"],
                ascending=[False, False, True],
            )
            .head(top_n_firm)
        )
        common_semi = (
            node_df.loc[node_df["is_semi_strict"]]
            .sort_values(
                ["prime_reach_weighted", "prime_reach_count", "analysis_uid"],
                ascending=[False, False, True],
            )
            .head(top_n_semi)
        )

        prime_out = m3r_dir / f"prime_upstream_sizes_suppliercentric_{view_name}.csv"
        node_out = m3r_dir / f"node_prime_reach_suppliercentric_{view_name}.parquet"
        common_all_out = m3r_dir / f"common_mode_top100_suppliercentric_all_{view_name}.csv"
        common_firm_out = m3r_dir / f"common_mode_top100_suppliercentric_firm_{view_name}.csv"
        common_semi_out = m3r_dir / f"common_mode_top100_suppliercentric_semi_{view_name}.csv"

        prime_df.to_csv(prime_out, index=False)
        node_df.to_parquet(node_out, index=False)
        common_all.to_csv(common_all_out, index=False)
        common_firm.to_csv(common_firm_out, index=False)
        common_semi.to_csv(common_semi_out, index=False)

        prime_outputs[view_name] = str(prime_out)
        node_outputs[view_name] = str(node_out)
        common_all_outputs[view_name] = str(common_all_out)
        common_firm_outputs[view_name] = str(common_firm_out)
        common_semi_outputs[view_name] = str(common_semi_out)

        lhs_pairs = int(node_prime_count.sum())
        rhs_pairs = int(up_count.sum()) + (n_primes if exclude_self else 0)
        view_stats[view_name] = {
            "edges_view_after_exclusion": len(ve),
            "n_nodes": int(n_nodes),
            "n_primes": int(n_primes),
            "n_nodes_with_prime_reach": int((node_prime_count > 0).sum()),
            "n_nodes_full_prime_reach": int((node_prime_count == n_primes).sum()),
            "prime_upstream_median": float(np.median(up_count)) if len(up_count) else 0.0,
            "prime_upstream_p95": float(np.quantile(up_count, 0.95)) if len(up_count) else 0.0,
            "pair_reconcile_lhs_node_sum": lhs_pairs,
            "pair_reconcile_rhs_prime_sum": rhs_pairs,
            "pair_reconcile_delta": int(lhs_pairs - rhs_pairs),
            "hop_mode": hop_mode,
            "hop_cap": int(hop_cap) if hop_cap is not None else None,
            "runtime_seconds": {
                "view_total": time.perf_counter() - t_view,
                "reach_compute": method_runtime,
            },
            "scc_diagnostics": scc_diag,
        }

    run_metadata = {
        "module": "m3_refine",
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
        "weights": {
            "source": prime_weight_source,
            "diagnostics": prime_weight_diag,
            "n_primes": int(n_primes),
            "total_prime_weight": float(total_prime_weight),
        },
        "settings": {
            "exclude_edge_flags": exclude_flags,
            "hop_mode": hop_mode,
            "hop_cap": int(hop_cap) if hop_cap is not None else None,
            "exclude_self_from_upstream": exclude_self,
            "top_n_export": top_n,
            "top_n_firm_export": top_n_firm,
            "top_n_semi_export": top_n_semi,
        },
        "view_stats": view_stats,
    }
    run_metadata_out = m3r_dir / "run_metadata.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m3_refine",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs": {
            "node_table_contract": str(node_path),
            "edge_table_contract": str(edge_path),
            "node_sha256": file_sha256(node_path),
            "edge_sha256": file_sha256(edge_path),
        },
        "params": {
            "views": views,
            "exclude_edge_flags": exclude_flags,
            "hop_mode": hop_mode,
            "hop_cap": int(hop_cap) if hop_cap is not None else None,
            "exclude_self_from_upstream": exclude_self,
            "prime_weight_source": prime_weight_source,
            "top_n_export": top_n,
            "top_n_firm_export": top_n_firm,
            "top_n_semi_export": top_n_semi,
        },
        "outputs": {
            "prime_upstream_sizes_suppliercentric": prime_outputs,
            "node_prime_reach_suppliercentric": node_outputs,
            "common_mode_top100_suppliercentric_all": common_all_outputs,
            "common_mode_top100_suppliercentric_firm": common_firm_outputs,
            "common_mode_top100_suppliercentric_semi": common_semi_outputs,
            "run_metadata": str(run_metadata_out),
        },
    }
    manifest_out = m3r_dir / "manifest_m3_refine.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))

    for _view_name, out in prime_outputs.items():
        print(f"[done] wrote {out}")
    for _view_name, out in node_outputs.items():
        print(f"[done] wrote {out}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")


if __name__ == "__main__":
    main()
