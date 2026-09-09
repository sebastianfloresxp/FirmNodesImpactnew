#!/usr/bin/env python3
"""Module 4: semiconductor-to-prime corridor importance."""

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
    parser = argparse.ArgumentParser(description="Chapter 4 Module 4 corridor importance")
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


def build_adjacency_lists(
    n_nodes: int,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
) -> tuple[list[list[int]], list[list[int]], np.ndarray, np.ndarray]:
    dedup = pd.DataFrame({"s": src_idx, "d": dst_idx})
    dedup = dedup[dedup["s"] != dedup["d"]].drop_duplicates(ignore_index=True)
    s = dedup["s"].to_numpy(np.int32, copy=False)
    d = dedup["d"].to_numpy(np.int32, copy=False)

    forward: list[list[int]] = [[] for _ in range(n_nodes)]
    reverse: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in zip(s.tolist(), d.tolist(), strict=False):
        forward[u].append(v)
        reverse[v].append(u)
    return forward, reverse, s, d


def multi_source_bfs(
    adjacency: list[list[int]],
    sources: np.ndarray,
    hop_cap: int | None,
) -> np.ndarray:
    n_nodes = len(adjacency)
    dist = np.full(n_nodes, -1, dtype=np.int32)
    queue: deque[int] = deque()
    for src in sources.tolist():
        if dist[src] == -1:
            dist[src] = 0
            queue.append(src)
    while queue:
        node = queue.popleft()
        depth = int(dist[node])
        if hop_cap is not None and depth >= hop_cap:
            continue
        for nbr in adjacency[node]:
            if dist[nbr] == -1:
                dist[nbr] = depth + 1
                queue.append(nbr)
    return dist


def compute_source_reach_count_via_scc(
    n_nodes: int,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    source_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
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
    queue: deque[int] = deque(np.flatnonzero(indeg == 0).tolist())
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(topo) != n_comp:
        raise RuntimeError("condensation graph topological sort failed")

    source_pos = np.full(n_nodes, -1, dtype=np.int32)
    source_pos[source_indices] = np.arange(len(source_indices), dtype=np.int32)
    comp_source_bits: list[int] = [0] * n_comp
    for node_idx in source_indices.tolist():
        comp_id = int(labels[node_idx])
        bit = 1 << int(source_pos[node_idx])
        comp_source_bits[comp_id] |= bit

    comp_reach_bits = comp_source_bits.copy()
    for u in topo:
        bits = comp_reach_bits[u]
        if bits == 0:
            continue
        for v in succ[u]:
            comp_reach_bits[v] |= bits

    comp_reach_count = np.fromiter(
        (int(bits.bit_count()) for bits in comp_reach_bits), dtype=np.int32, count=n_comp
    )
    node_reach_count = comp_reach_count[labels]
    diag = {
        "n_scc": int(n_comp),
        "condensation_edges": len(cu),
        "largest_scc_size": int(comp_sizes.max()) if len(comp_sizes) else 0,
    }
    return node_reach_count, diag


def load_prime_reach_count(
    m3_dir: Path,
    view_name: str,
    node_uids: np.ndarray,
) -> np.ndarray:
    path = m3_dir / f"node_prime_reach_{view_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"M4 requires M3 output missing: {path}")
    df = pd.read_parquet(path, columns=["analysis_uid", "prime_reach_count"])
    merged = pd.DataFrame({"analysis_uid": node_uids}).merge(df, on="analysis_uid", how="left")
    if merged["prime_reach_count"].isna().any():
        missing = int(merged["prime_reach_count"].isna().sum())
        raise ValueError(f"M3 prime_reach_count missing for {missing} nodes in view {view_name}")
    return merged["prime_reach_count"].astype(np.int32).to_numpy()


def distance_validation_sample(
    rng: random.Random,
    node_uids: np.ndarray,
    forward: list[list[int]],
    reverse: list[list[int]],
    dist_from_semi: np.ndarray,
    dist_to_prime: np.ndarray,
    sample_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    corridor_idx = np.flatnonzero((dist_from_semi >= 0) & (dist_to_prime >= 0))
    if len(corridor_idx) == 0 or sample_n <= 0:
        empty = pd.DataFrame(
            columns=[
                "analysis_uid",
                "dist_from_semi",
                "dist_to_prime",
                "from_semi_step_check",
                "to_prime_step_check",
                "sample_pass",
            ]
        )
        summary = {
            "sampled_nodes": 0,
            "share_sample_pass": np.nan,
            "share_from_semi_step_check": np.nan,
            "share_to_prime_step_check": np.nan,
        }
        return empty, summary

    sample_n_eff = min(sample_n, len(corridor_idx))
    chosen = rng.sample(corridor_idx.tolist(), sample_n_eff)
    rows: list[dict[str, Any]] = []
    pass_count = 0
    from_ok_count = 0
    to_ok_count = 0
    for idx in chosen:
        dfs = int(dist_from_semi[idx])
        dtp = int(dist_to_prime[idx])
        if dfs == 0:
            from_ok = True
        else:
            from_ok = any(dist_from_semi[pred] == (dfs - 1) for pred in reverse[idx])
        if dtp == 0:
            to_ok = True
        else:
            to_ok = any(dist_to_prime[succ] == (dtp - 1) for succ in forward[idx])
        sample_pass = bool(from_ok and to_ok)
        if from_ok:
            from_ok_count += 1
        if to_ok:
            to_ok_count += 1
        if sample_pass:
            pass_count += 1
        rows.append(
            {
                "analysis_uid": node_uids[idx],
                "dist_from_semi": dfs,
                "dist_to_prime": dtp,
                "from_semi_step_check": bool(from_ok),
                "to_prime_step_check": bool(to_ok),
                "sample_pass": sample_pass,
            }
        )
    sample_df = pd.DataFrame(rows)
    summary = {
        "sampled_nodes": int(sample_n_eff),
        "share_sample_pass": float(pass_count / sample_n_eff),
        "share_from_semi_step_check": float(from_ok_count / sample_n_eff),
        "share_to_prime_step_check": float(to_ok_count / sample_n_eff),
    }
    return sample_df, summary


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "unspecified_snapshot"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    random_cfg = cfg.get("random", {})
    seed = int(random_cfg.get("global_seed", 7))
    rng = random.Random(seed)  # nosec B311 -- seeded for reproducible sampling, not cryptographic

    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2")))
    m0_dir = out_root / snapshot / "m0"
    m3_dir = out_root / snapshot / "m3"
    m4_dir = out_root / snapshot / "m4"
    m4_dir.mkdir(parents=True, exist_ok=True)

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

    semi_indices = np.flatnonzero(is_semi).astype(np.int32, copy=False)
    prime_indices = np.flatnonzero(is_prime).astype(np.int32, copy=False)
    if len(semi_indices) == 0:
        raise ValueError("No strict semiconductor nodes found for M4")
    if len(prime_indices) == 0:
        raise ValueError("No prime nodes found for M4")

    views = cfg.get("views", {})
    if not isinstance(views, dict) or not views:
        raise ValueError("config.views must be a non-empty mapping")

    m4_cfg = cfg.get("m4", {})
    hop_cap_raw = m4_cfg.get("hop_cap", None)
    if hop_cap_raw is None or str(hop_cap_raw).strip().lower() in {"full", "none", "null", ""}:
        hop_cap = None
        hop_mode = "full"
    else:
        hop_cap = int(hop_cap_raw)
        if hop_cap < 0:
            raise ValueError("m4.hop_cap must be >= 0 when capped")
        hop_mode = "capped"
    top_n = int(m4_cfg.get("top_n_export", 100))
    top_n_intermediary = int(m4_cfg.get("top_n_intermediary_export", 100))
    sample_n = int(m4_cfg.get("distance_validation_sample_n", 200))
    primary_view = str(m4_cfg.get("primary_view", "full"))

    corridor_outputs: dict[str, str] = {}
    top_outputs: dict[str, str] = {}
    top_intermediary_outputs: dict[str, str] = {}
    validation_outputs: dict[str, str] = {}
    view_stats: dict[str, Any] = {}

    for view_name, view_spec in views.items():
        t_view = time.perf_counter()
        include_any = view_spec.get("include_any", [])
        mask = get_view_mask(edges, include_any)
        ve = edges.loc[mask, ["src_idx", "dst_idx"]]
        src_idx = ve["src_idx"].to_numpy(np.int32, copy=False)
        dst_idx = ve["dst_idx"].to_numpy(np.int32, copy=False)

        t0 = time.perf_counter()
        forward, reverse, s, d = build_adjacency_lists(
            n_nodes=n_nodes, src_idx=src_idx, dst_idx=dst_idx
        )
        t_adj = time.perf_counter() - t0

        t0 = time.perf_counter()
        dist_from_semi = multi_source_bfs(adjacency=forward, sources=semi_indices, hop_cap=hop_cap)
        dist_to_prime = multi_source_bfs(adjacency=reverse, sources=prime_indices, hop_cap=hop_cap)
        t_dist = time.perf_counter() - t0

        t0 = time.perf_counter()
        semi_reach_count, semi_diag = compute_source_reach_count_via_scc(
            n_nodes=n_nodes,
            src_idx=s,
            dst_idx=d,
            source_indices=semi_indices,
        )
        t_semi_reach = time.perf_counter() - t0

        t0 = time.perf_counter()
        prime_reach_count = load_prime_reach_count(
            m3_dir=m3_dir,
            view_name=view_name,
            node_uids=node_uids,
        )
        t_prime_reach = time.perf_counter() - t0

        is_corridor = (dist_from_semi >= 0) & (dist_to_prime >= 0)
        is_intermediary_corridor = is_corridor & (~is_semi) & (~is_prime) & (~is_dod)

        semi_reach_share = semi_reach_count.astype(np.float64) / float(max(len(semi_indices), 1))
        prime_reach_share = prime_reach_count.astype(np.float64) / float(max(len(prime_indices), 1))
        corridor_score = semi_reach_count.astype(np.int64) * prime_reach_count.astype(np.int64)
        corridor_score_intermediary = corridor_score.copy()
        corridor_score_intermediary[~is_intermediary_corridor] = -1

        corridor_df = pd.DataFrame(
            {
                "view": view_name,
                "analysis_uid": node_uids,
                "name": name.to_numpy(),
                "entity_role": role.to_numpy(),
                "is_dod_component": is_dod,
                "is_tier1_prime": is_prime,
                "is_semi_strict": is_semi,
                "dist_from_semi": dist_from_semi.astype(np.int32),
                "dist_to_prime": dist_to_prime.astype(np.int32),
                "is_corridor": is_corridor,
                "is_intermediary_corridor": is_intermediary_corridor,
                "semi_reach_count": semi_reach_count.astype(np.int32),
                "semi_reach_share": semi_reach_share,
                "prime_reach_count": prime_reach_count.astype(np.int32),
                "prime_reach_share": prime_reach_share,
                "corridor_score": corridor_score.astype(np.int64),
                "rank_corridor_score": rank_desc(corridor_score.astype(np.float64)),
                "corridor_score_intermediary": corridor_score_intermediary.astype(np.int64),
                "rank_corridor_score_intermediary": rank_desc(
                    corridor_score_intermediary.astype(np.float64)
                ),
            }
        )

        top_all = (
            corridor_df[corridor_df["is_corridor"]]
            .sort_values(
                ["corridor_score", "prime_reach_count", "semi_reach_count", "analysis_uid"],
                ascending=[False, False, False, True],
            )
            .head(top_n)
        )
        top_intermediary = (
            corridor_df[corridor_df["is_intermediary_corridor"]]
            .sort_values(
                ["corridor_score", "prime_reach_count", "semi_reach_count", "analysis_uid"],
                ascending=[False, False, False, True],
            )
            .head(top_n_intermediary)
        )

        val_df, val_summary = distance_validation_sample(
            rng=rng,
            node_uids=node_uids,
            forward=forward,
            reverse=reverse,
            dist_from_semi=dist_from_semi,
            dist_to_prime=dist_to_prime,
            sample_n=sample_n,
        )

        corridor_out = m4_dir / f"corridor_nodes_{view_name}.parquet"
        top_out = m4_dir / f"top_corridor_nodes_{view_name}.csv"
        top_intermediary_out = m4_dir / f"top_corridor_nodes_intermediary_{view_name}.csv"
        val_out = m4_dir / f"distance_validation_{view_name}.csv"
        corridor_df.to_parquet(corridor_out, index=False)
        top_all.to_csv(top_out, index=False)
        top_intermediary.to_csv(top_intermediary_out, index=False)
        val_df.to_csv(val_out, index=False)

        corridor_outputs[view_name] = str(corridor_out)
        top_outputs[view_name] = str(top_out)
        top_intermediary_outputs[view_name] = str(top_intermediary_out)
        validation_outputs[view_name] = str(val_out)

        view_stats[view_name] = {
            "edges_view": len(ve),
            "corridor_nodes_count": int(is_corridor.sum()),
            "intermediary_corridor_count": int(is_intermediary_corridor.sum()),
            "corridor_share_nodes": float(is_corridor.mean()),
            "intermediary_corridor_share_nodes": float(is_intermediary_corridor.mean()),
            "distance_validation": val_summary,
            "semi_reach_scc_diagnostics": semi_diag,
            "runtime_seconds": {
                "build_adjacency": t_adj,
                "compute_distances": t_dist,
                "compute_semi_reach_count": t_semi_reach,
                "load_prime_reach_count": t_prime_reach,
                "view_total": time.perf_counter() - t_view,
            },
        }

    run_metadata = {
        "module": "m4",
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
            "hop_mode": hop_mode,
            "hop_cap": int(hop_cap) if hop_cap is not None else None,
            "top_n_export": top_n,
            "top_n_intermediary_export": top_n_intermediary,
            "distance_validation_sample_n": sample_n,
            "primary_view": primary_view,
        },
        "view_stats": view_stats,
    }
    run_metadata_out = m4_dir / "run_metadata.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m4",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs": {
            "node_table_contract": str(node_path),
            "edge_table_contract": str(edge_path),
            "node_sha256": file_sha256(node_path),
            "edge_sha256": file_sha256(edge_path),
            "m3_node_prime_reach_disclosed": str(m3_dir / "node_prime_reach_disclosed.parquet"),
            "m3_node_prime_reach_observed": str(m3_dir / "node_prime_reach_observed.parquet"),
            "m3_node_prime_reach_full": str(m3_dir / "node_prime_reach_full.parquet"),
        },
        "params": {
            "views": views,
            "hop_mode": hop_mode,
            "hop_cap": int(hop_cap) if hop_cap is not None else None,
            "top_n_export": top_n,
            "top_n_intermediary_export": top_n_intermediary,
            "distance_validation_sample_n": sample_n,
            "primary_view": primary_view,
        },
        "outputs": {
            "corridor_nodes": corridor_outputs,
            "top_corridor_nodes": top_outputs,
            "top_corridor_nodes_intermediary": top_intermediary_outputs,
            "distance_validation_samples": validation_outputs,
            "run_metadata": str(run_metadata_out),
        },
    }
    manifest_out = m4_dir / "manifest_m4.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))

    for _view_name, out in corridor_outputs.items():
        print(f"[done] wrote {out}")
    for _view_name, out in top_outputs.items():
        print(f"[done] wrote {out}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")


if __name__ == "__main__":
    main()
