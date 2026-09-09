#!/usr/bin/env python3
"""
Build derived structural scalars for core_v1 (inductive, OON-safe).

Outputs a Parquet file with per-node structural signals computed from the
train-only adjacency at T0:
  - logdeg_in, logdeg_out (log1p of degrees)
  - pagerank (directed PageRank, alpha=0.85)
  - hits_auth, hits_hub (HITS scores)

Notes
- Does not modify existing node_features_T0.parquet. Models should concatenate
  this file at load time when --use-struct-feats is enabled.
- Normalizes PR/HITS via z-score across nodes at T0.
- Writes a small JSON sidecar with method metadata and a version stamp.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_adjacency(adj_path: Path) -> tuple[sp.csr_matrix, sp.csc_matrix]:
    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency not found: {adj_path}")
    csr = sp.load_npz(adj_path).tocsr(copy=False)
    csc = csr.tocsc(copy=False)
    return csr, csc


def compute_degrees(csr: sp.csr_matrix, csc: sp.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    out_deg = np.diff(csr.indptr).astype(np.int64, copy=False)
    in_deg = np.diff(csc.indptr).astype(np.int64, copy=False)
    return in_deg, out_deg


def pagerank_power(
    csr: sp.csr_matrix, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6
) -> np.ndarray:
    """Compute directed PageRank via power iteration on row-stochastic matrix.

    P is the row-normalized adjacency (outgoing). Handle dangling nodes by
    redistributing their weight uniformly each iteration.
    """
    n = csr.shape[0]  # type: ignore[index]
    # Row-normalize csr to get P (handle rows with zero outdeg separately)
    out_deg = np.diff(csr.indptr)
    data = csr.data.copy()
    rows = np.repeat(np.arange(n, dtype=np.int64), out_deg)
    # Safe divide
    with np.errstate(divide="ignore", invalid="ignore"):
        data = np.divide(data, out_deg[rows], where=out_deg[rows] != 0)
    P = sp.csr_matrix((data, csr.indices.copy(), csr.indptr.copy()), shape=csr.shape)

    r = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - alpha) / n
    for _ in range(max_iter):
        r_old = r
        # r_new = alpha * (P^T @ r_old) + teleport + alpha * (dangling_sum / n)
        dangling_sum = float(r_old[out_deg == 0].sum())
        r = alpha * (P.T @ r_old)
        r = np.asarray(r).ravel()
        r += teleport + alpha * (dangling_sum / n)
        # L1 convergence
        if np.linalg.norm(r - r_old, 1) <= tol:
            break
    # Normalize to sum 1
    s = r.sum()
    if s > 0:
        r = r / s
    return r.astype(np.float64, copy=False)


def hits_power(
    csr: sp.csr_matrix, max_iter: int = 100, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Compute HITS (authority, hub) via power iteration.

    a <- A^T h; h <- A a; with normalization each step.
    """
    n = csr.shape[0]  # type: ignore[index]
    csc = csr.tocsc(copy=False)
    a = np.ones(n, dtype=np.float64)
    h = np.ones(n, dtype=np.float64)
    for _ in range(max_iter):
        a_old = a
        h_old = h
        a = csc @ h_old
        na = np.linalg.norm(a, 2)
        if na > 0:
            a = a / na
        h = csr @ a
        nh = np.linalg.norm(h, 2)
        if nh > 0:
            h = h / nh
        if np.linalg.norm(a - a_old, 1) + np.linalg.norm(h - h_old, 1) <= tol:
            break
    return a.astype(np.float64, copy=False), h.astype(np.float64, copy=False)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / sd).astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build structural scalars parquet for core_v1")
    ap.add_argument("--adj", required=True, type=str, help="Path to train_adj_T0.npz")
    ap.add_argument(
        "--out",
        type=str,
        default="data/processed/core/releases/core_v1/features/node_structural_v1.parquet",
    )
    ap.add_argument(
        "--splits-root",
        type=str,
        default="data/processed/core/releases/core_v1/splits",
        help="To resolve T0 for metadata",
    )
    ap.add_argument("--pr-alpha", type=float, default=0.85)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument(
        "--skip-pr-hits", action="store_true", help="If set, only compute degrees (fast path)"
    )
    args = ap.parse_args()

    adj_path = Path(args.adj)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csr, csc = load_adjacency(adj_path)
    in_deg, out_deg = compute_degrees(csr, csc)
    logdeg_in = np.log1p(in_deg).astype(np.float32, copy=False)
    logdeg_out = np.log1p(out_deg).astype(np.float32, copy=False)

    n = csr.shape[0]  # type: ignore[index]
    cols = {
        "node_id": np.arange(n, dtype=np.int64),
        "logdeg_in": logdeg_in,
        "logdeg_out": logdeg_out,
    }
    if not args.skip_pr_hits:
        pr = pagerank_power(
            csr, alpha=float(args.pr_alpha), max_iter=int(args.max_iter), tol=float(args.tol)
        )
        a, h = hits_power(csr, max_iter=int(args.max_iter), tol=float(args.tol))
        # Normalize PR/HITS (z-score) as per Phase 0 guardrails
        cols["pagerank"] = zscore(pr)
        cols["hits_auth"] = zscore(a)
        cols["hits_hub"] = zscore(h)
    df = pd.DataFrame(cols)
    # Write Parquet
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, out_path)
    except Exception:
        # Fallback to pandas engine
        df.to_parquet(out_path, index=False)

    # Metadata sidecar
    meta = {
        "created_at": _now_iso(),
        "adjacency": str(adj_path),
        "version": "node_structural_v1",
        "columns": [c for c in df.columns if c != "node_id"],
        "transforms": {
            "logdeg": "log1p(in_deg/out_deg)",
            "pagerank": None
            if args.skip_pr_hits
            else {
                "alpha": float(args.pr_alpha),
                "max_iter": int(args.max_iter),
                "tol": float(args.tol),
                "normalized": "zscore",
            },
            "hits": None
            if args.skip_pr_hits
            else {"max_iter": int(args.max_iter), "tol": float(args.tol), "normalized": "zscore"},
        },
        "notes": "Concatenate with node_features_T0.parquet at load time when --use-struct-feats is set.",
    }
    sidecar = out_path.with_suffix("")
    sidecar = Path(str(sidecar) + "_meta.json")
    sidecar.write_text(json.dumps(meta, indent=2))

    print(f"[OK] Wrote structural scalars to {out_path} and metadata to {sidecar}")


if __name__ == "__main__":
    main()
