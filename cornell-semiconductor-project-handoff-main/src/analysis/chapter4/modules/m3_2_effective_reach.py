#!/usr/bin/env python3
"""Module 3.2: effective reachability under hop and cost constraints."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
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
    parser = argparse.ArgumentParser(description="Chapter 4 Module 3.2 effective reach")
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


def build_view_edges(
    edges: pd.DataFrame, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    src = df["src_idx"].to_numpy(np.int32, copy=False)
    dst = df["dst_idx"].to_numpy(np.int32, copy=False)
    predicted_only = (
        df["is_predicted"].astype(bool)
        & ~(df["is_disclosed"].astype(bool) | df["is_observed_ship"].astype(bool))
    ).to_numpy()
    return src, dst, predicted_only


def load_prime_weight_table(
    weights_path: Path, prime_uids: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "weights_path": str(weights_path),
        "found_file": bool(weights_path.exists()),
        "matched_primes": 0,
        "missing_primes": len(prime_uids),
        "fallback_to_unit": False,
    }
    if not weights_path.exists():
        diagnostics["fallback_to_unit"] = True
        return pd.DataFrame(
            {
                "analysis_uid": prime_uids.astype(str),
                "weight_unit": np.ones(len(prime_uids), dtype=np.float64),
                "weight_log_obligation": np.ones(len(prime_uids), dtype=np.float64),
                "weight_raw_obligation": np.ones(len(prime_uids), dtype=np.float64),
            }
        ), diagnostics

    if weights_path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(weights_path)
    else:
        raw = pd.read_csv(weights_path)

    uid_col = next(
        (c for c in ["analysis_uid", "prime_uid", "uid", "node_uid"] if c in raw.columns), None
    )
    if uid_col is None:
        diagnostics["fallback_to_unit"] = True
        return pd.DataFrame(
            {
                "analysis_uid": prime_uids.astype(str),
                "weight_unit": np.ones(len(prime_uids), dtype=np.float64),
                "weight_log_obligation": np.ones(len(prime_uids), dtype=np.float64),
                "weight_raw_obligation": np.ones(len(prime_uids), dtype=np.float64),
            }
        ), diagnostics

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

    merged = merged[
        ["analysis_uid", "weight_unit", "weight_log_obligation", "weight_raw_obligation"]
    ]
    matched = int(merged["weight_log_obligation"].gt(0).sum())
    diagnostics["matched_primes"] = matched
    diagnostics["missing_primes"] = int(len(merged) - matched)
    diagnostics["weights_sum_log"] = float(merged["weight_log_obligation"].sum())
    diagnostics["weights_sum_raw"] = float(merged["weight_raw_obligation"].sum())
    return merged, diagnostics


def compute_support_by_caps(
    matrix: csr_matrix,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    caps: list[float],
    *,
    unweighted: bool,
    chunk_size: int,
) -> dict[float, np.ndarray]:
    if len(source_indices) == 0 or len(target_indices) == 0:
        return {float(cap): np.zeros(len(target_indices), dtype=np.int32) for cap in caps}

    caps_sorted = sorted(float(c) for c in caps)
    max_cap = caps_sorted[-1]
    out = {cap: np.zeros(len(target_indices), dtype=np.int32) for cap in caps_sorted}

    for start in range(0, len(source_indices), chunk_size):
        batch = source_indices[start : start + chunk_size]
        dist = dijkstra(
            matrix,
            directed=True,
            indices=batch.astype(np.int32, copy=False),
            unweighted=unweighted,
            limit=max_cap,
        )
        tgt_dist = dist[:, target_indices]
        finite = np.isfinite(tgt_dist)
        for cap in caps_sorted:
            mask = finite & (tgt_dist <= cap)
            out[cap] += mask.sum(axis=0).astype(np.int32, copy=False)
    return out


def summarize_support(
    prime_df: pd.DataFrame,
    prefix: str,
    caps: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_primes = len(prime_df)
    for cap in sorted(float(c) for c in caps):
        cap_label = str(int(cap)) if float(cap).is_integer() else str(cap).replace(".", "p")
        col_support = f"semi_support_{prefix}_{cap_label}"
        col_any = f"any_support_{prefix}_{cap_label}"
        any_flag = prime_df[col_any].astype(bool).to_numpy()
        rows.append(
            {
                "constraint_type": prefix,
                "cap": float(cap),
                "supported_primes_count": int(any_flag.sum()),
                "supported_primes_share": float(any_flag.mean() if n_primes else 0.0),
                "mean_support_count": float(prime_df[col_support].mean() if n_primes else 0.0),
                "median_support_count": float(prime_df[col_support].median() if n_primes else 0.0),
                "p90_support_count": float(
                    prime_df[col_support].quantile(0.9) if n_primes else 0.0
                ),
                "supported_log_obligation_share": float(
                    prime_df.loc[any_flag, "weight_log_obligation"].sum()
                    / max(prime_df["weight_log_obligation"].sum(), 1e-12)
                ),
                "supported_raw_obligation_share": float(
                    prime_df.loc[any_flag, "weight_raw_obligation"].sum()
                    / max(prime_df["weight_raw_obligation"].sum(), 1e-12)
                ),
            }
        )
    return pd.DataFrame(rows)


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
    m3_2_dir = out_root / snapshot / "m3_2"
    m3_2_dir.mkdir(parents=True, exist_ok=True)

    node_path = m0_dir / "node_table_contract.parquet"
    edge_path = m0_dir / "edge_table_contract.parquet"
    if not node_path.exists() or not edge_path.exists():
        raise FileNotFoundError("M0 contract outputs are required before M3.2")

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
    prime_idx = np.flatnonzero(is_prime).astype(np.int32)
    semi_idx = np.flatnonzero(is_semi).astype(np.int32)
    prime_uids = nodes.loc[prime_idx, "analysis_uid"].astype(str).to_numpy()

    weights_path = Path(
        str(paths_cfg.get("prime_weights", "artifacts/ch4/inputs/prime_weights.parquet"))
    )
    prime_weights_df, weight_diag = load_prime_weight_table(weights_path, prime_uids)
    prime_meta = nodes.loc[prime_idx, ["analysis_uid", "name", "entity_role"]].reset_index(
        drop=True
    )
    prime_meta = prime_meta.merge(prime_weights_df, on="analysis_uid", how="left")

    m_cfg = cfg.get("m3_2", {})
    views = [str(v) for v in m_cfg.get("views", list(cfg.get("views", {}).keys()))]
    hop_caps = [float(x) for x in m_cfg.get("hop_caps", [3, 5, 7, 10])]
    cost_caps = [float(x) for x in m_cfg.get("cost_caps", [5, 10, 15, 20])]
    predicted_only_edge_cost = float(m_cfg.get("predicted_only_edge_cost", 3.0))
    chunk_size = int(m_cfg.get("chunk_size", 128))

    runtime_by_view: dict[str, float] = {}
    output_manifest: dict[str, dict[str, str]] = {}

    for view in views:
        t0 = time.perf_counter()
        include_any = [str(x) for x in cfg.get("views", {}).get(view, {}).get("include_any", [])]
        if not include_any:
            raise ValueError(f"view {view} include_any missing")

        mask = get_view_mask(edges, include_any)
        src, dst, predicted_only = build_view_edges(edges, mask)
        n_nodes = len(nodes)

        matrix_unweighted = csr_matrix(
            (np.ones(len(src), dtype=np.int8), (src, dst)),
            shape=(n_nodes, n_nodes),
        )
        edge_cost = np.where(predicted_only, predicted_only_edge_cost, 1.0).astype(
            np.float64, copy=False
        )
        matrix_weighted = csr_matrix((edge_cost, (src, dst)), shape=(n_nodes, n_nodes))

        hop_support = compute_support_by_caps(
            matrix_unweighted,
            semi_idx,
            prime_idx,
            hop_caps,
            unweighted=True,
            chunk_size=chunk_size,
        )
        cost_support = compute_support_by_caps(
            matrix_weighted,
            semi_idx,
            prime_idx,
            cost_caps,
            unweighted=False,
            chunk_size=chunk_size,
        )

        prime_view = prime_meta.copy()
        for cap in sorted(hop_support.keys()):
            cap_label = str(int(cap)) if float(cap).is_integer() else str(cap).replace(".", "p")
            col = f"semi_support_hop_{cap_label}"
            prime_view[col] = hop_support[cap]
            prime_view[f"any_support_hop_{cap_label}"] = (prime_view[col] > 0).astype(np.int8)
        for cap in sorted(cost_support.keys()):
            cap_label = str(int(cap)) if float(cap).is_integer() else str(cap).replace(".", "p")
            col = f"semi_support_cost_{cap_label}"
            prime_view[col] = cost_support[cap]
            prime_view[f"any_support_cost_{cap_label}"] = (prime_view[col] > 0).astype(np.int8)

        prime_view.insert(0, "view", view)
        prime_out_csv = m3_2_dir / f"prime_effective_support_{view}.csv"
        prime_out_pq = m3_2_dir / f"prime_effective_support_{view}.parquet"
        prime_view.to_csv(prime_out_csv, index=False)
        prime_view.to_parquet(prime_out_pq, index=False)

        summary = pd.concat(
            [
                summarize_support(prime_view, "hop", hop_caps),
                summarize_support(prime_view, "cost", cost_caps),
            ],
            ignore_index=True,
        )
        summary.insert(0, "view", view)
        summary_out = m3_2_dir / f"effective_support_summary_{view}.csv"
        summary.to_csv(summary_out, index=False)

        output_manifest[view] = {
            "prime_effective_support_csv": str(prime_out_csv),
            "prime_effective_support_parquet": str(prime_out_pq),
            "effective_support_summary_csv": str(summary_out),
        }
        runtime_by_view[view] = float(time.perf_counter() - t0)
        print(f"[m3_2] view={view} runtime_s={runtime_by_view[view]:.2f}", flush=True)

    run_metadata = {
        "module": "m3_2",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "settings": {
            "views": views,
            "hop_caps": hop_caps,
            "cost_caps": cost_caps,
            "predicted_only_edge_cost": predicted_only_edge_cost,
            "chunk_size": chunk_size,
        },
        "graph": {
            "n_nodes": len(nodes),
            "n_edges_total": len(edges),
            "n_primes": len(prime_idx),
            "n_semis": len(semi_idx),
        },
        "weights": weight_diag,
        "runtime_seconds_by_view": runtime_by_view,
    }
    run_meta_path = m3_2_dir / "run_metadata.json"
    run_meta_path.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m3_2",
        "snapshot": snapshot,
        "run_id": run_id,
        "inputs": {
            "node_table_contract": str(node_path),
            "edge_table_contract": str(edge_path),
            "node_table_contract_sha256": file_sha256(node_path),
            "edge_table_contract_sha256": file_sha256(edge_path),
            "prime_weights": str(weights_path),
            "prime_weights_sha256": file_sha256(weights_path) if weights_path.exists() else None,
        },
        "outputs": output_manifest,
        "run_metadata": str(run_meta_path),
    }
    manifest_path = m3_2_dir / "manifest_m3_2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
