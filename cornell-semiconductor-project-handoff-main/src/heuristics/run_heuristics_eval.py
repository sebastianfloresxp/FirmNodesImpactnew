#!/usr/bin/env python3
"""
Heuristics Scorer & Evaluator (Dissertation Core)

Single-command, deterministic, leakage-safe baseline evaluation over fixed
candidate pools derived from the train-only snapshot (T0). Computes classic
heuristics using only the train CSR and evaluates per-source ranking with a
shared scorecard (Hit@K, Recall@K, Precision@K, MRR, MAP, nDCG@100).

CLI (example):
  python src/models/heuristics/run_heuristics_eval.py \
    --adj data/processed/core/adjacency/train_adj_T0.npz \
    --candidates-val data/processed/core/candidates/val_candidates.parquet \
    --candidates-test data/processed/core/candidates/test_candidates.parquet \
    --splits-root data/processed/core/splits \
    --out-dir results/heuristics \
    --artifacts-dir artifacts/heuristics \
    --Ks 1,10,50 \
    --include-pagerank false \
    --include-katz false \
    --include-svd false \
    --undirected false \
    --num-threads 0 \
    --batch-size 2000000

Notes:
  - Requires pyarrow for streaming parquet. Fails fast with a clear message if
    not available.
  - Defaults tuned for large servers; override via CLI.
  - Optional heavy extras (pagerank/katz/svd) are wired but disabled by default.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.horizons import assign_horizon_buckets, horizon_slice_names

# ------------------------------ Utilities --------------------------------- #


def _bool_arg(v: str) -> bool:
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")


def _set_num_threads(n: int) -> None:
    if n and n > 0:
        os.environ["OMP_NUM_THREADS"] = str(n)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n)
        os.environ["MKL_NUM_THREADS"] = str(n)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
        os.environ["NUMEXPR_NUM_THREADS"] = str(n)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_temporal_t0(splits_root: Path) -> int | None:
    # Try meta/temporal_splits.json first (preferred)
    meta = splits_root.parent / "meta" / "temporal_splits.json"
    try:
        if meta.exists():
            obj = json.loads(meta.read_text())
            return int(obj.get("boundaries", {}).get("T0_end"))
    except Exception:  # nosec B110 -- best-effort metadata read, pass is intentional
        pass
    # Fallback: max ts in train_edges.parquet
    train_path = splits_root / "train_edges.parquet"
    if train_path.exists():
        try:
            import pyarrow.parquet as pq  # type: ignore

            tbl = pq.read_table(train_path, columns=["ts"])
            arr = tbl.column(0).to_numpy(zero_copy_only=False)
            if len(arr):
                return int(np.nanmax(arr))
        except Exception:
            # Last resort with pandas
            df = pd.read_parquet(train_path, columns=["ts"])  # type: ignore
            if len(df):
                return int(df["ts"].max())
    return None


# ----------------------------- Calibration --------------------------------- #


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def fit_platt(
    scores: np.ndarray, labels: np.ndarray, l2: float = 1e-6, iters: int = 25
) -> tuple[float, float]:
    """Fit logistic regression with bias: p = sigmoid(A*s + B) via Newton steps.

    Returns (A, B). Adds tiny L2 to stabilize.
    """
    s = scores.astype(np.float64)
    y = labels.astype(np.float64)
    # Standardize scores to improve conditioning
    mu, sd = float(np.mean(s)), float(np.std(s) + 1e-12)
    s = (s - mu) / sd
    X = np.stack([s, np.ones_like(s)], axis=1)  # [N,2]
    w = np.zeros(2, dtype=np.float64)
    for _ in range(max(1, iters)):
        z = X @ w
        p = _sigmoid(z)
        # Gradient and Hessian
        g = X.T @ (p - y) + l2 * w
        r = p * (1.0 - p)
        # Hessian = X^T R X + l2 I
        XR = X * r[:, None]
        H = XR.T @ X + l2 * np.eye(2)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= delta
        if np.linalg.norm(delta) < 1e-6:
            break
    A, B = float(w[0] / sd), float(w[1] - w[0] * (mu / sd))
    return A, B


def calibrate_heuristic_on_val(
    key: str,
    g: GraphData,
    cand_val: Path,
    splits_root: Path,
    sample_per_src: int = 50,
    max_pairs: int = 1_000_000,
) -> dict[str, object]:
    # Stream val candidates and sample scores/labels for the chosen heuristic
    try:
        import pyarrow.parquet as pq  # type: ignore

        has_ts = "ts" in set(pq.ParquetFile(cand_val).schema.names)
    except Exception:
        has_ts = False
    cols = ["src_id", "dst_id", "label"] + (["ts"] if has_ts else [])
    streamer = CandidateStreamer(cand_val, columns=cols, batch_size=1_000_000)
    scores = []
    labels = []
    tmp_buf: dict[str, object] = {}
    total = 0
    for chunk in streamer:
        if chunk.empty:
            continue
        chunk["src_id"] = chunk["src_id"].astype(np.int64)
        chunk["dst_id"] = chunk["dst_id"].astype(np.int64)
        chunk["label"] = chunk["label"].astype(np.int8)
        src_vals = chunk["src_id"].to_numpy()
        change = np.where(np.diff(src_vals) != 0)[0] + 1
        bounds = np.concatenate(([0], change, [len(chunk)]))
        for i in range(len(bounds) - 1):
            a, b = int(bounds[i]), int(bounds[i + 1])
            sub = chunk.iloc[a:b]
            u = int(sub["src_id"].iloc[0])
            vs = sub["dst_id"].to_numpy(dtype=np.int64, copy=False)
            lbl = sub["label"].to_numpy(dtype=np.int8, copy=False)
            if vs.size == 0:
                continue
            # Sample per source
            if vs.size > sample_per_src:
                idx = np.random.choice(vs.size, size=sample_per_src, replace=False)
                vs_s = vs[idx]
                lbl_s = lbl[idx]
            else:
                vs_s = vs
                lbl_s = lbl
            hs = compute_heuristics_for_source(u, vs_s, g, tmp_buf)
            by = {
                "CN": hs.cn,
                "Jaccard": hs.jaccard,
                "AA": hs.aa,
                "RA": hs.ra,
                "COS": hs.cos,
                "PA": hs.pa,
            }
            if key not in by:
                continue
            scores.append(by[key].astype(np.float64))
            labels.append(lbl_s.astype(np.float64))
            total += len(vs_s)
            if total >= max_pairs:
                break
        if total >= max_pairs:
            break
    if not scores:
        raise RuntimeError(f"No calibration samples collected for heuristic {key}")
    s = np.concatenate(scores, axis=0)
    y = np.concatenate(labels, axis=0)
    A, B = fit_platt(s, y)
    return {"method": "platt", "A": A, "B": B, "samples": int(s.size)}


# ----------------------------- Data Loading -------------------------------- #


@dataclass
class GraphData:
    csr: sp.csr_matrix
    csc: sp.csc_matrix
    out_deg: np.ndarray  # int64
    in_deg: np.ndarray  # int64
    aa_w: np.ndarray  # float64, 1/log(max(deg,2)) for out-deg
    ra_w: np.ndarray  # float64, 1/max(deg,1) for out-deg


def load_graph(adj_path: Path, undirected: bool = False) -> GraphData:
    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency not found: {adj_path}")
    csr = sp.load_npz(adj_path).tocsr(copy=False)
    if undirected:
        # Symmetrize once: A_und = (A + A^T) > 0
        csc_tmp = csr.tocsc(copy=False)
        und = (csr + csc_tmp).astype(bool).astype(np.uint8)
        csr = und.tocsr(copy=False)
    csc = csr.tocsc(copy=False)
    out_deg = np.diff(csr.indptr).astype(np.int64, copy=False)
    in_deg = np.diff(csc.indptr).astype(np.int64, copy=False)
    # Weights for AA/RA over out-neighbor degrees
    od = out_deg.copy()
    aa_w = np.zeros_like(od, dtype=np.float64)
    # avoid div-by-zero; use max(deg, 2) per spec
    denom = np.log(np.maximum(od, 2))
    aa_w[denom > 0] = 1.0 / denom[denom > 0]
    ra_w = np.zeros_like(od, dtype=np.float64)
    ra_w = 1.0 / np.maximum(od, 1)
    return GraphData(csr=csr, csc=csc, out_deg=out_deg, in_deg=in_deg, aa_w=aa_w, ra_w=ra_w)


# ----------------------------- Parquet Reader ------------------------------- #


class CandidateStreamer:
    """Stream candidates grouped by contiguous src using pyarrow.

    Assumes the parquet file is written in per-source contiguous blocks (as
    produced by the candidate builder). If a source spans a row group boundary,
    this class buffers across batches to emit full groups.
    """

    def __init__(self, path: Path, columns: list[str] | None = None, batch_size: int = 2_000_000):
        self.path = path
        self.columns = columns if columns is not None else ["src_id", "dst_id", "label", "ts"]
        self.batch_size = int(max(100_000, batch_size))
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq
        except Exception as e:
            raise RuntimeError(
                "pyarrow is required for streaming candidate reading. "
                "Please install pyarrow or lower batch_size and use pandas (not recommended)."
            ) from e
        self.pq = pq  # type: ignore

    def __iter__(self) -> Iterator[pd.DataFrame]:
        tbl = self.pq.read_table(self.path, columns=self.columns)
        # Convert to pandas in chunks to bound memory
        # We will then group by contiguous src_id boundaries
        num_rows = tbl.num_rows
        start = 0
        remainder: pd.DataFrame | None = None
        while start < num_rows:
            end = min(start + self.batch_size, num_rows)
            batch = tbl.slice(start, end - start)
            df = batch.to_pandas(types_mapper={})
            # Normalize column names
            cols_map = {c: c for c in df.columns}
            for c in ["src", "src_id"]:
                if c in df.columns:
                    cols_map[c] = "src_id"
            for c in ["dst", "dst_id"]:
                if c in df.columns:
                    cols_map[c] = "dst_id"
            df = df.rename(columns=cols_map)

            if remainder is not None and len(remainder):
                df = pd.concat([remainder, df], ignore_index=True)
                remainder = None

            if df.empty:
                start = end
                continue

            # Find last run of src_id to buffer for next iteration
            last_src = int(df["src_id"].iloc[-1])
            cut = len(df)
            # Walk backwards to find first index where src_id changes from last_src
            i = cut - 2
            while i >= 0 and int(df["src_id"].iloc[i]) == last_src:
                i -= 1
            i += 1  # first index of the last_src run
            if i > 0 and i < cut:
                remainder = df.iloc[i:].copy()
                df = df.iloc[:i].copy()

            yield df
            start = end


# ------------------------------ Metrics ------------------------------------- #


@dataclass
class KMetrics:
    Ks: list[int]
    # Macro accumulators
    n_sources: int = 0
    sum_hit: dict[int, float] = field(default_factory=dict)
    sum_rec: dict[int, float] = field(default_factory=dict)
    sum_prec: dict[int, float] = field(default_factory=dict)
    sum_mrr: float = 0.0
    sum_map: float = 0.0
    sum_ndcg100: float = 0.0
    # Micro accumulators
    micro_pos_total: int = 0
    micro_pos_in_top: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        for k in self.Ks:
            self.sum_hit.setdefault(k, 0.0)
            self.sum_rec.setdefault(k, 0.0)
            self.sum_prec.setdefault(k, 0.0)
            self.micro_pos_in_top.setdefault(k, 0)

    def update_from_ranking(self, labels_sorted: np.ndarray, Ks: list[int]) -> None:
        # labels_sorted: 1D array of 0/1 sorted by (-score, dst)
        n = labels_sorted.size
        pos_total = int(labels_sorted.sum())
        # Prefix sums for efficient AP and Recall@K
        cumsum = labels_sorted.cumsum()

        # AP (Average Precision)
        if pos_total > 0:
            # indices of positives (1-based ranks)
            pos_idx = np.nonzero(labels_sorted)[0]
            # precision at each positive position i: cumsum[i] / (i+1)
            prec_at_pos = cumsum[pos_idx] / (pos_idx + 1)
            ap = float(prec_at_pos.sum() / pos_total)
        else:
            ap = 0.0

        # MRR
        if pos_total > 0:
            first_pos = int(np.argmax(labels_sorted > 0)) if (labels_sorted > 0).any() else -1
            mrr = 1.0 / (first_pos + 1) if first_pos >= 0 else 0.0
        else:
            mrr = 0.0

        # nDCG@100
        K_ndcg = 100
        upto = min(K_ndcg, n)
        gains = labels_sorted[:upto].astype(np.float64)
        if gains.any():
            discounts = 1.0 / np.log2(np.arange(2, 2 + upto))
            dcg = float((gains * discounts).sum())
            ideal = min(pos_total, K_ndcg)
            ideal_gains = np.ones(ideal, dtype=np.float64)
            idcg = float((ideal_gains * discounts[:ideal]).sum()) if ideal > 0 else 0.0
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
        else:
            ndcg = 0.0

        # Update macro accumulators
        self.n_sources += 1
        self.sum_mrr += mrr
        self.sum_map += ap
        self.sum_ndcg100 += ndcg

        for K in Ks:
            k = min(K, n)
            top_k_pos = int(cumsum[k - 1]) if k > 0 else 0
            # Hit@K: 1 if any positive in top K
            hit = 1.0 if top_k_pos > 0 else 0.0
            # Recall@K
            rec = (top_k_pos / pos_total) if pos_total > 0 else 0.0
            # Precision@K
            prec = (top_k_pos / K) if K > 0 else 0.0

            self.sum_hit[K] += hit
            self.sum_rec[K] += rec
            self.sum_prec[K] += prec

            # Micro accumulators: weight by totals across sources
            self.micro_pos_total += pos_total
            self.micro_pos_in_top[K] += top_k_pos

    def to_row(self, heuristic: str, macro: bool) -> dict[str, object]:
        row: dict[str, object] = {"heuristic": heuristic, "macro": macro}
        if macro:
            ns = max(1, self.n_sources)
            for K in self.Ks:
                row[f"hit@{K}"] = self.sum_hit[K] / ns
                row[f"recall@{K}"] = self.sum_rec[K] / ns
                row[f"precision@{K}"] = self.sum_prec[K] / ns
            row["mrr"] = self.sum_mrr / ns
            row["map"] = self.sum_map / ns
            row["ndcg@100"] = self.sum_ndcg100 / ns
        else:
            # micro: recall/precision via pooled counts; hit is same as macro proportion
            ns = max(1, self.n_sources)
            for K in self.Ks:
                row[f"hit@{K}"] = self.sum_hit[K] / ns
                row[f"recall@{K}"] = self.micro_pos_in_top[K] / max(1, self.micro_pos_total)
                row[f"precision@{K}"] = self.micro_pos_in_top[K] / (K * ns)
            row["mrr"] = self.sum_mrr / ns
            # micro-MAP as positive-weighted mean of per-source AP approximately equals macro here
            row["map"] = self.sum_map / ns
            row["ndcg@100"] = self.sum_ndcg100 / ns
        return row


# --------------------------- Heuristics Scoring ----------------------------- #


@dataclass
class HeuristicScores:
    cn: np.ndarray
    jaccard: np.ndarray
    aa: np.ndarray
    ra: np.ndarray
    cos: np.ndarray
    pa: np.ndarray


def _vector_from_row_indices(cols: np.ndarray, data: np.ndarray, size: int) -> sp.csr_matrix:
    # Build 1xN sparse row from given columns and data
    if cols.size == 0:
        return sp.csr_matrix((1, size), dtype=np.float64)
    rows = np.zeros_like(cols, dtype=np.int32)
    return sp.csr_matrix((data.astype(np.float64), (rows, cols.astype(np.int32))), shape=(1, size))


def _gather_from_sparse_row(row: sp.csr_matrix, cols: np.ndarray) -> np.ndarray:
    # row is 1xN CSR; fetch values at given column indices without Python loops
    idx = row.indices
    dat = row.data
    if idx.size == 0 or cols.size == 0:
        return np.zeros(cols.size, dtype=np.float64)
    order = np.argsort(idx)
    idx_sorted = idx[order]
    dat_sorted = dat[order]
    pos = np.searchsorted(idx_sorted, cols)
    # Fix: check bounds before accessing idx_sorted
    valid_mask = pos < idx_sorted.size
    # Only check equality for valid positions
    m = np.zeros_like(cols, dtype=bool)
    m[valid_mask] = idx_sorted[pos[valid_mask]] == cols[valid_mask]
    out = np.zeros(cols.size, dtype=np.float64)
    # Only use valid positions that are within bounds
    final_mask = m & valid_mask
    out[final_mask] = dat_sorted[pos[final_mask]]
    return out


def compute_heuristics_for_source(
    u: int,
    vs: np.ndarray,
    g: GraphData,
    tmp_buf: dict[str, object],
) -> HeuristicScores:
    """Compute default heuristics for one source u over candidates vs.

    Uses CSR-based sparse multiplications for CN/AA/RA and closed-forms for
    Jaccard/Cosine/PA. No Python loops over pairs.
    """
    csr = g.csr
    # Row slice for u
    start_u, end_u = csr.indptr[u], csr.indptr[u + 1]
    neigh_u = csr.indices[start_u:end_u]
    deg_u = g.out_deg[u]
    N = csr.shape[0]  # type: ignore[index]

    # CN via (row_u) @ CSR^T
    row_u = _vector_from_row_indices(neigh_u, np.ones_like(neigh_u, dtype=np.float64), N)
    cn_row = row_u.dot(csr.T)  # 1 x N
    cn = _gather_from_sparse_row(cn_row, vs.astype(np.int64))

    # AA via (row_u with aa weights) @ CSR^T
    if neigh_u.size:
        aa_weights_u = g.aa_w[neigh_u]
        row_u_aa = _vector_from_row_indices(neigh_u, aa_weights_u, N)
        aa_row = row_u_aa.dot(csr.T)
        aa = _gather_from_sparse_row(aa_row, vs.astype(np.int64))
    else:
        aa = np.zeros(vs.size, dtype=np.float64)

    # RA via (row_u with ra weights) @ CSR^T
    if neigh_u.size:
        ra_weights_u = g.ra_w[neigh_u]
        row_u_ra = _vector_from_row_indices(neigh_u, ra_weights_u, N)
        ra_row = row_u_ra.dot(csr.T)
        ra = _gather_from_sparse_row(ra_row, vs.astype(np.int64))
    else:
        ra = np.zeros(vs.size, dtype=np.float64)

    # Jaccard & Cosine via degs + CN
    deg_v = g.out_deg[vs]
    denom_j = deg_u + deg_v - cn
    jacc = np.divide(cn, denom_j, out=np.zeros_like(cn), where=denom_j > 0)
    denom_c = deg_u * deg_v
    denom_c = np.where(denom_c > 0, np.sqrt(denom_c, dtype=np.float64), 0.0)
    cos = np.divide(cn, denom_c, out=np.zeros_like(cn), where=denom_c > 0)

    # Preferential Attachment
    pa = float(deg_u) * g.in_deg[vs].astype(np.float64)

    return HeuristicScores(
        cn=cn.astype(np.float64),
        jaccard=jacc.astype(np.float64),
        aa=aa.astype(np.float64),
        ra=ra.astype(np.float64),
        cos=cos.astype(np.float64),
        pa=pa.astype(np.float64),
    )


# ------------------------------ Slices -------------------------------------- #


def classify_warm_cold(u: int, vs: np.ndarray, g: GraphData) -> np.ndarray:
    """Return slice labels for warm/cold combinations per pair.

    0=WW, 1=WC, 2=CW, 3=CC
    """
    warm_u = g.out_deg[u] > 0
    warm_v = g.in_deg[vs] > 0
    out = np.empty(vs.size, dtype=np.int8)
    if warm_u:
        out[warm_v] = 0  # WW
        out[~warm_v] = 1  # WC
    else:
        out[warm_v] = 2  # CW
        out[~warm_v] = 3  # CC
    return out


def classify_warm_cold_strict(
    u: int, vs: np.ndarray, g: GraphData, threshold: int = 3
) -> np.ndarray:
    """Strict warm/cold by degree threshold (default ≥3 edges).

    0=WW3, 1=WC3, 2=CW3, 3=CC3
    """
    warm_u = g.out_deg[u] >= int(threshold)
    warm_v = g.in_deg[vs] >= int(threshold)
    out = np.empty(vs.size, dtype=np.int8)
    if warm_u:
        out[warm_v] = 0  # WW3
        out[~warm_v] = 1  # WC3
    else:
        out[warm_v] = 2  # CW3
        out[~warm_v] = 3  # CC3
    return out


def classify_twohop(u: int, vs: np.ndarray, g: GraphData) -> np.ndarray:
    """Mark whether each v in vs is 2-hop reachable from u in undirected sense.

    Uses union of out/in neighbors for 1-hop and expands 2-hop similarly.
    """
    csr, csc = g.csr, g.csc
    out_u = csr.indices[csr.indptr[u] : csr.indptr[u + 1]]
    in_u = csc.indices[csc.indptr[u] : csc.indptr[u + 1]]
    one = (
        np.unique(np.concatenate([out_u, in_u]))
        if (out_u.size or in_u.size)
        else np.empty(0, dtype=np.int64)
    )
    if one.size == 0:
        return np.zeros(vs.size, dtype=bool)
    # Gather neighbors of neighbors (both directions)
    parts: list[np.ndarray] = []
    indptr_out, idx_out = csr.indptr, csr.indices
    indptr_in, idx_in = csc.indptr, csc.indices
    for w in one:
        parts.append(idx_out[indptr_out[w] : indptr_out[w + 1]])
        parts.append(idx_in[indptr_in[w] : indptr_in[w + 1]])
    two = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
    if two.size == 0:
        return np.zeros(vs.size, dtype=bool)
    # Membership test
    two_sorted = np.sort(two)
    pos = np.searchsorted(two_sorted, vs)
    # Fix: check bounds before accessing two_sorted
    valid_mask = pos < two_sorted.size
    m = np.zeros_like(vs, dtype=bool)
    m[valid_mask] = two_sorted[pos[valid_mask]] == vs[valid_mask]
    return m


# ------------------------------ Evaluator ----------------------------------- #


def compute_pagerank(
    g: GraphData, alpha: float = 0.85, max_iter: int = 50, tol: float = 1e-6
) -> np.ndarray:
    """Compute PageRank over the train graph using out-degree normalization.

    pr[v] = (1-alpha)/N + alpha * sum_{u in in(v)} pr[u]/out_deg[u] + alpha * (dangling_mass/N)
    """
    N = g.csr.shape[0]  # type: ignore[index]
    pr = np.full(N, 1.0 / N, dtype=np.float64)
    out_deg = g.out_deg.astype(np.float64)
    mask = out_deg > 0
    for _it in range(max_iter):
        dangling_mass = pr[~mask].sum() / N
        # pr_div[u] = pr[u]/out_deg[u] if out_deg[u] > 0 else 0
        pr_div = np.zeros(N, dtype=np.float64)
        pr_div[mask] = pr[mask] / out_deg[mask]
        contrib = g.csc.dot(pr_div)  # in-neighbor contributions
        pr_new = (1.0 - alpha) / N + alpha * (contrib + dangling_mass)
        # Convergence check
        if np.linalg.norm(pr_new - pr, 1) < tol:
            pr = pr_new
            break
        pr = pr_new
    # Normalize to sum=1
    s = pr.sum()
    if s > 0:
        pr /= s
    return pr


def compute_katz_for_source(
    u: int, vs: np.ndarray, g: GraphData, beta: float = 0.01, L: int = 3
) -> np.ndarray:
    """Compute Katz(u,v) for candidate vs up to length L using sparse row multiplications.

    Katz(u,v) = sum_{l=1..L} beta^l [A^l]_{u,v}
    """
    csr = g.csr
    N = csr.shape[0]  # type: ignore[index]
    # A^1 row: neighbors of u
    neigh_u = csr.indices[csr.indptr[u] : csr.indptr[u + 1]]
    row_l = _vector_from_row_indices(neigh_u, np.ones_like(neigh_u, dtype=np.float64), N)  # 1xN
    # Gather len1
    val = beta * _gather_from_sparse_row(row_l, vs.astype(np.int64))
    if L >= 2:
        row_l = row_l.dot(csr).tocsr()  # type: ignore[union-attr]  # A^2 row
        val += (beta**2) * _gather_from_sparse_row(row_l, vs.astype(np.int64))
    if L >= 3:
        row_l = row_l.dot(csr).tocsr()  # type: ignore[union-attr]  # A^3 row
        val += (beta**3) * _gather_from_sparse_row(row_l, vs.astype(np.int64))
    # For L>3, could iterate further, but default is 3
    return val.astype(np.float64)


def compute_svd_embeddings(csr: sp.csr_matrix, k: int = 64) -> np.ndarray:
    """Compute TruncatedSVD embeddings (N x k) from CSR adjacency."""
    try:
        from sklearn.decomposition import TruncatedSVD
    except Exception as e:
        raise RuntimeError("scikit-learn is required for --include-svd") from e
    svd = TruncatedSVD(n_components=int(k), random_state=42)
    Z = svd.fit_transform(csr)
    # optional: L2 normalize rows to keep dot products stable
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Z = Z / norms
    return Z.astype(np.float32)


def sort_by_score_tiebreak(scores: np.ndarray, dsts: np.ndarray) -> np.ndarray:
    # Primary: score desc; Tie-break: dst asc
    return np.lexsort((dsts, -scores))


def evaluate_split(
    name: str,
    cand_path: Path,
    g: GraphData,
    Ks: list[int],
    out_dir: Path,
    artifacts_dir: Path,
    batch_size: int,
    write_scores: bool = False,
    pr_vec: np.ndarray | None = None,
    katz_beta: float | None = None,
    katz_L: int | None = None,
    svd_embeddings: dict[int, np.ndarray] | None = None,
    svd_dims: list[int] | None = None,
    emit_strict_warm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Evaluate all heuristics on a candidate split, streaming per source.

    Returns: (global_macro_df, micro_df, slices_macro_df, counts_dict)
    """
    # Check if ts column exists in the parquet file
    try:
        import pyarrow.parquet as pq

        pq.read_table(cand_path, columns=["ts"])
        has_ts = True
    except Exception:
        has_ts = False

    columns = ["src_id", "dst_id", "label"]
    if has_ts:
        columns.append("ts")

    streamer = CandidateStreamer(cand_path, columns=columns, batch_size=batch_size)

    # Prepare sidecar ts map from split edges when ts is absent in candidates
    ts_sidecar: dict[tuple[int, int], int] | None = None
    if not has_ts:
        splits_root = cand_path.parent.parent / "splits"
        split_edges_path = splits_root / f"{name}_edges.parquet"
        if split_edges_path.exists():
            try:
                df_edges = pd.read_parquet(split_edges_path, columns=["src_id", "dst_id", "ts"])  # type: ignore
                ts_sidecar = {(int(s), int(d)): int(t) for s, d, t in df_edges.to_numpy()}
            except Exception:
                ts_sidecar = None
    # Accumulators per heuristic
    heuristics = ["CN", "Jaccard", "AA", "RA", "COS", "PA"]
    if pr_vec is not None:
        heuristics += ["PR_prod", "PR_sum", "PR_diff"]
    if katz_beta is not None and katz_L is not None:
        heuristics += ["Katz"]
    if svd_embeddings:
        for d in svd_dims or sorted(svd_embeddings.keys()):
            heuristics.append(f"SVD-{int(d)}")
    macro: dict[str, KMetrics] = {h: KMetrics(Ks=Ks) for h in heuristics}
    micro: dict[str, KMetrics] = {h: KMetrics(Ks=Ks) for h in heuristics}

    # Slice accumulators (macro only)
    slice_names = [
        "WW",
        "WC",
        "CW",
        "CC",  # warm/cold
        "twohop",
        ">2hop",
        *horizon_slice_names(),
        "deg_q1",
        "deg_q2",
        "deg_q3",
        "deg_q4",
    ]
    if emit_strict_warm:
        slice_names.extend(["WW3", "WC3", "CW3", "CC3"])  # strict warm (≥3)
    slices_macro: dict[str, dict[str, KMetrics]] = {
        h: {s: KMetrics(Ks=Ks) for s in slice_names} for h in heuristics
    }

    # Degree quartiles for src bins
    od = g.out_deg
    # Use non-zero degrees plus zeros; compute quartiles over all nodes
    try:
        q = np.quantile(od, [0.25, 0.5, 0.75])
    except Exception:
        q = np.array([0, 0, 0], dtype=np.float64)

    # T0 horizon boundaries (epoch-days), interpreted in evaluate_split scope for positives
    # We'll try to derive T0 from the path structure
    splits_root = cand_path.parent.parent / "splits"
    T0 = _load_temporal_t0(splits_root) or 0

    # Optional scores output
    if write_scores:
        out_scores_path = artifacts_dir / f"{name}_scores.parquet"
        _ensure_dir(out_scores_path.parent)
        try:
            import pyarrow.parquet as pq
        except Exception:
            write_scores = False
            out_scores_path = None  # type: ignore

    # Streaming iteration grouped by contiguous src
    total_rows = 0
    total_sources = 0
    tmp_buf: dict[str, object] = {}

    # For score dumps
    if write_scores:
        score_batches: list[pd.DataFrame] = []
        score_batch_rows = 0
        score_batch_limit = 5_000_000

    for chunk in streamer:
        if chunk.empty:
            continue
        # Basic validations
        for col in ("src_id", "dst_id", "label"):
            if col not in chunk.columns:
                raise ValueError(f"Missing required column '{col}' in {cand_path}")
        # Ensure types
        chunk["src_id"] = chunk["src_id"].astype(np.int64)
        chunk["dst_id"] = chunk["dst_id"].astype(np.int64)
        chunk["label"] = chunk["label"].astype(np.int8)
        if "ts" in chunk.columns:
            # positives have ts; negatives may be NaN
            chunk["ts"] = chunk["ts"].astype("float64")

        # Group by contiguous src runs in chunk
        src_values = chunk["src_id"].to_numpy()
        # Find boundaries where src changes
        change = np.where(np.diff(src_values) != 0)[0] + 1
        boundaries = np.concatenate(([0], change, [len(chunk)]))

        for i in range(len(boundaries) - 1):
            a, b = int(boundaries[i]), int(boundaries[i + 1])
            sub = chunk.iloc[a:b]
            u = int(sub["src_id"].iloc[0])

            vs = sub["dst_id"].to_numpy(dtype=np.int64, copy=False)
            labels = sub["label"].to_numpy(dtype=np.int8, copy=False)
            # Heuristic scores
            hs = compute_heuristics_for_source(u, vs, g, tmp_buf)

            # Rank and update metrics per heuristic
            dsts = vs

            # Warm/cold slice per pair
            wc = classify_warm_cold(u, vs, g)
            # Two-hop classification per pair
            th_mask = classify_twohop(u, vs, g)
            # Strict warm/cold per pair if requested
            if emit_strict_warm:
                wc3 = classify_warm_cold_strict(u, vs, g, threshold=3)
            # Horizon bins (positives only)
            if T0:
                if "ts" in sub.columns:
                    ts_vals = sub["ts"].to_numpy(dtype=np.float64, copy=False)
                elif ts_sidecar is not None:
                    ts_vals = np.full(len(sub), np.nan, dtype=np.float64)
                    pos_m = labels > 0
                    if np.any(pos_m):
                        src_arr = sub["src_id"].to_numpy()[pos_m]
                        dst_arr = sub["dst_id"].to_numpy()[pos_m]
                        looked = [
                            ts_sidecar.get((int(s), int(d)), np.nan)
                            for s, d in zip(src_arr, dst_arr, strict=False)
                        ]
                        ts_vals[pos_m] = np.array(looked, dtype=np.float64)
                else:
                    ts_vals = np.full(len(sub), np.nan, dtype=np.float64)
            else:
                ts_vals = np.full(len(sub), np.nan, dtype=np.float64)
            # bins in days
            horizon_label = np.zeros_like(labels, dtype=np.int8)
            if T0:
                pos_m = labels > 0
                if np.any(pos_m):
                    deltas = ts_vals[pos_m] - float(T0)
                    assignments = assign_horizon_buckets(deltas)
                    horizon_label[np.nonzero(pos_m)[0]] = assignments

            # Degree quartile slice for source (source-level slice)
            d = g.out_deg[u]
            if d <= q[0]:
                deg_bin = "deg_q1"
            elif d <= q[1]:
                deg_bin = "deg_q2"
            elif d <= q[2]:
                deg_bin = "deg_q3"
            else:
                deg_bin = "deg_q4"

            # Prepare per-heuristic arrays
            by_name: dict[str, np.ndarray] = {
                "CN": hs.cn,
                "Jaccard": hs.jaccard,
                "AA": hs.aa,
                "RA": hs.ra,
                "COS": hs.cos,
                "PA": hs.pa,
            }
            # PR-based features
            if pr_vec is not None:
                pr_u = float(pr_vec[u])
                pr_vs = pr_vec[vs]
                by_name["PR_prod"] = (pr_u * pr_vs).astype(np.float64)
                by_name["PR_sum"] = (pr_u + pr_vs).astype(np.float64)
                by_name["PR_diff"] = np.abs(pr_u - pr_vs).astype(np.float64)

            # Katz index (sum of path counts up to L)
            if katz_beta is not None and katz_L is not None:
                katz_scores = compute_katz_for_source(
                    u, vs, g, beta=float(katz_beta), L=int(katz_L)
                )
                by_name["Katz"] = katz_scores

            # SVD-based dot products
            if svd_embeddings:
                for d, Z in svd_embeddings.items():
                    zu = Z[u].astype(np.float64)
                    zv = Z[vs].astype(np.float64)
                    svd_scores = zv.dot(zu)
                    by_name[f"SVD-{int(d)}"] = svd_scores

            # Optional: append scores for dumping later
            if write_scores:
                s_df = pd.DataFrame(
                    {
                        "src": u,
                        "dst": vs,
                        "label": labels.astype(np.int8),
                        "CN": hs.cn,
                        "Jaccard": hs.jaccard,
                        "AA": hs.aa,
                        "RA": hs.ra,
                        "COS": hs.cos,
                        "PA": hs.pa,
                    }
                )
                if "ts" in sub.columns:
                    s_df["ts"] = ts_vals
                if pr_vec is not None:
                    s_df["PR_prod"] = by_name["PR_prod"]
                    s_df["PR_sum"] = by_name["PR_sum"]
                    s_df["PR_diff"] = by_name["PR_diff"]
                if katz_beta is not None and katz_L is not None:
                    s_df["Katz"] = by_name["Katz"]
                if svd_embeddings:
                    for d in svd_dims or sorted(svd_embeddings.keys()):
                        s_df[f"SVD-{int(d)}"] = by_name[f"SVD-{int(d)}"]
                score_batches.append(s_df)
                score_batch_rows += len(s_df)
                if score_batch_rows >= score_batch_limit:
                    # flush to parquet
                    out_scores_path = artifacts_dir / f"{name}_scores.parquet"
                    "wb" if not out_scores_path.exists() else "ab"
                    score_df = pd.concat(score_batches, ignore_index=True)
                    score_df.to_parquet(out_scores_path, index=False)
                    score_batches.clear()
                    score_batch_rows = 0

            # Update metrics
            for hname, scores in by_name.items():
                order = sort_by_score_tiebreak(scores, dsts)
                labs_sorted = labels[order].astype(np.int64)

                # Global accumulators
                macro[hname].update_from_ranking(labs_sorted, Ks)
                micro[hname].update_from_ranking(labs_sorted, Ks)

                # Slice metrics (macro only)
                # Warm/Cold slices use pair-level masks
                wc_sorted = wc[order]
                # two-hop vs >2
                th_sorted = th_mask[order]
                # horizon
                hz_sorted = horizon_label[order]

                # Warm/Cold classes
                for label_val, slice_tag in zip(
                    [0, 1, 2, 3], ["WW", "WC", "CW", "CC"], strict=False
                ):
                    m = wc_sorted == label_val
                    if m.any():
                        slices_macro[hname][slice_tag].update_from_ranking(labs_sorted[m], Ks)

                # Two-hop slices
                if th_sorted.any():
                    slices_macro[hname]["twohop"].update_from_ranking(labs_sorted[th_sorted], Ks)
                if (~th_sorted).any():
                    slices_macro[hname][">2hop"].update_from_ranking(labs_sorted[~th_sorted], Ks)

                # Strict warm/cold (WW3/WC3/CW3/CC3)
                if emit_strict_warm:
                    wc3_sorted = wc3[order]
                    for label_val, slice_tag in zip(
                        [0, 1, 2, 3], ["WW3", "WC3", "CW3", "CC3"], strict=False
                    ):
                        m3 = wc3_sorted == label_val
                        if m3.any():
                            slices_macro[hname][slice_tag].update_from_ranking(labs_sorted[m3], Ks)

                # Horizon slices (positives-only masks)
                for hval, stag in enumerate(horizon_slice_names(), start=1):
                    mpos = hz_sorted == hval
                    if mpos.any():
                        # Keep the same candidate list but mark positives outside slice as 0
                        # Equivalent to evaluating metrics for that subset of positives
                        labs_slice = labs_sorted.copy()
                        # Zero-out positives not in this slice
                        drop = (labs_slice > 0) & (~mpos)
                        labs_slice[drop] = 0
                        slices_macro[hname][stag].update_from_ranking(labs_slice, Ks)

                # Degree quartile slice (source-level): apply entire list to the bin
                slices_macro[hname][deg_bin].update_from_ranking(labs_sorted, Ks)

            total_rows += len(sub)
            total_sources += 1

    # Flush remaining score batches
    if write_scores and score_batches:
        out_scores_path = artifacts_dir / f"{name}_scores.parquet"
        score_df = pd.concat(score_batches, ignore_index=True)
        score_df.to_parquet(out_scores_path, index=False)

    # Build dataframes
    global_rows = []
    micro_rows = []
    slices_rows = []
    for h in ["CN", "Jaccard", "AA", "RA", "COS", "PA"]:
        global_rows.append(macro[h].to_row(h, macro=True))
        micro_rows.append(micro[h].to_row(h, macro=False))
        for s in slice_names:
            slices_rows.append({**slices_macro[h][s].to_row(h, macro=True), "slice_name": s})

    global_df = pd.DataFrame(global_rows)
    micro_df = pd.DataFrame(micro_rows)
    slices_df = pd.DataFrame(slices_rows)

    counts = {
        "rows": int(total_rows),
        "sources": int(total_sources),
    }
    return global_df, micro_df, slices_df, counts


# ---------------------------------- Main ------------------------------------ #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministic, leakage-safe heuristics evaluator over core candidate pools"
    )
    p.add_argument("--adj", required=True, type=str, help="Path to train CSR adjacency (.npz)")
    p.add_argument(
        "--candidates-val", required=True, type=str, help="Path to val candidates parquet"
    )
    p.add_argument(
        "--candidates-test", required=True, type=str, help="Path to test candidates parquet"
    )
    p.add_argument(
        "--splits-root",
        required=True,
        type=str,
        help="Root dir containing train_edges.parquet (and meta/temporal_splits.json)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results/heuristics",
        help="Output directory for results CSVs",
    )
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/heuristics",
        help="Artifacts directory for run summary and optional scores",
    )
    p.add_argument(
        "--Ks", type=str, default="1,10,50", help="Comma-separated K values (e.g., 1,10,50)"
    )
    p.add_argument(
        "--include-pagerank",
        type=_bool_arg,
        default=False,
        help="Include PageRank-based features (PR_prod, PR_sum, PR_diff)",
    )
    p.add_argument(
        "--pagerank-alpha", type=float, default=0.85, help="PageRank damping factor (default 0.85)"
    )
    p.add_argument(
        "--pagerank-iter", type=int, default=50, help="Max PageRank iterations (default 50)"
    )
    p.add_argument(
        "--pagerank-tol", type=float, default=1e-6, help="Convergence tolerance (default 1e-6)"
    )

    p.add_argument(
        "--include-katz",
        type=_bool_arg,
        default=False,
        help="Include Katz index (beta^1 A + beta^2 A^2 + ... up to L)",
    )
    p.add_argument(
        "--katz-beta", type=float, default=0.01, help="Katz attenuation beta (default 0.01)"
    )
    p.add_argument("--katz-L", type=int, default=3, help="Katz maximum path length L (default 3)")

    p.add_argument(
        "--include-svd",
        type=_bool_arg,
        default=False,
        help="Include TruncatedSVD dot-product baselines",
    )
    p.add_argument(
        "--svd-dims",
        type=str,
        default="64,128",
        help="Comma-separated SVD dimensions (e.g., 64,128)",
    )
    p.add_argument(
        "--undirected", type=_bool_arg, default=False, help="If true, symmetrize CSR before scoring"
    )
    p.add_argument(
        "--num-threads", type=int, default=0, help="Set BLAS threading env vars (0=leave default)"
    )
    p.add_argument(
        "--batch-size", type=int, default=2_000_000, help="Candidate rows per batch to bound memory"
    )
    p.add_argument(
        "--write-scores",
        action="store_true",
        help="Write per-pair scores parquet to artifacts (big)",
    )
    p.add_argument(
        "--emit-strict-warm",
        action="store_true",
        help="Emit additional WW3/WC3/CW3/CC3 slices (warm=≥3 edges)",
    )
    # Calibration
    p.add_argument(
        "--calibrate",
        type=str,
        default="none",
        choices=["none", "platt", "isotonic"],
        help="Fit a score calibrator on validation (saves artifacts; does not change ranking)",
    )
    p.add_argument(
        "--calibrate-key",
        type=str,
        default="PA",
        help="Heuristic key to calibrate (e.g., PA, CN, Jaccard, AA, RA, COS)",
    )
    p.add_argument(
        "--calibrate-sample-per-src",
        type=int,
        default=50,
        help="Pairs per source to sample for calibration",
    )
    p.add_argument(
        "--calibrate-max-pairs",
        type=int,
        default=1000000,
        help="Global cap on pairs for calibration",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_threads and args.num_threads > 0:
        _set_num_threads(args.num_threads)

    Ks = [int(x.strip()) for x in str(args.Ks).split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    artifacts_dir = Path(args.artifacts_dir)
    _ensure_dir(out_dir)
    _ensure_dir(artifacts_dir)

    # Load graph (CSR/CSC, degs, weights)
    g = load_graph(Path(args.adj), undirected=args.undirected)

    # Sanity counts from candidates meta if available
    # Try to infer meta path from candidates path
    try:
        meta_path = Path(args.candidates_val).parents[1] / "meta" / "candidate_pools_meta.json"
        cand_meta = json.loads(Path(meta_path).read_text()) if meta_path.exists() else {}
    except Exception:
        cand_meta = {}

    # Optional extras precompute
    pr_vec: np.ndarray | None = None
    if args.include_pagerank:
        print("[INFO] Computing PageRank vector...", flush=True)
        pr_vec = compute_pagerank(
            g,
            alpha=float(args.pagerank_alpha),
            max_iter=int(args.pagerank_iter),
            tol=float(args.pagerank_tol),
        )
        print("[INFO] PageRank computed.", flush=True)

    svd_embeddings: dict[int, np.ndarray] = {}
    svd_dims: list[int] = []
    if args.include_svd:
        try:
            svd_dims = [int(x.strip()) for x in str(args.svd_dims).split(",") if x.strip()]
        except Exception:
            svd_dims = [64, 128]
        # Deduplicate and sort small to large
        svd_dims = sorted(set(svd_dims))
        for d in svd_dims:
            print(f"[INFO] Computing TruncatedSVD (k={d}) on adjacency...", flush=True)
            Z = compute_svd_embeddings(g.csr, k=d)
            svd_embeddings[d] = Z
            print(f"[INFO] SVD-{d} embeddings ready (shape={Z.shape}).", flush=True)

    # Evaluate splits
    splits = {
        "val": Path(args.candidates_val),
        "test": Path(args.candidates_test),
    }

    summary = {
        "adjacency": str(Path(args.adj)),
        "candidates": {k: str(v) for k, v in splits.items()},
        "Ks": Ks,
        "directed": (not args.undirected),
        "include_pagerank": bool(args.include_pagerank),
        "pagerank": {
            "alpha": float(args.pagerank_alpha),
            "iter": int(args.pagerank_iter),
            "tol": float(args.pagerank_tol),
        }
        if args.include_pagerank
        else None,
        "include_katz": bool(args.include_katz),
        "katz": {"beta": float(args.katz_beta), "L": int(args.katz_L)}
        if args.include_katz
        else None,
        "include_svd": bool(args.include_svd),
        "svd_dims": [
            int(x) for x in (str(args.svd_dims).split(",") if args.include_svd else []) if x.strip()
        ],
        "num_threads": int(args.num_threads),
        "batch_size": int(args.batch_size),
        "deg": {
            "in_min": int(np.min(g.in_deg)) if g.in_deg.size else 0,
            "in_max": int(np.max(g.in_deg)) if g.in_deg.size else 0,
            "out_min": int(np.min(g.out_deg)) if g.out_deg.size else 0,
            "out_max": int(np.max(g.out_deg)) if g.out_deg.size else 0,
        },
        "counts": {},
        "timestamp": _now_iso(),
    }

    for split_name, path in splits.items():
        print(f"[INFO] Evaluating split: {split_name} ({path})", flush=True)
        t0 = time.time()
        gdf, mdf, sdf, counts = evaluate_split(
            name=split_name,
            cand_path=path,
            g=g,
            Ks=Ks,
            out_dir=out_dir,
            artifacts_dir=artifacts_dir,
            batch_size=int(args.batch_size),
            write_scores=bool(args.write_scores),
            pr_vec=pr_vec,
            katz_beta=float(args.katz_beta) if args.include_katz else None,
            katz_L=int(args.katz_L) if args.include_katz else None,
            svd_embeddings=svd_embeddings if args.include_svd else None,
            svd_dims=svd_dims if args.include_svd else None,
            emit_strict_warm=bool(args.emit_strict_warm),
        )
        # Write results
        gdf.to_csv(out_dir / f"global_{split_name}.csv", index=False)
        mdf.to_csv(out_dir / f"micro_{split_name}.csv", index=False)
        sdf.to_csv(out_dir / f"slices_{split_name}.csv", index=False)
        print(
            f"[INFO] Done {split_name} in {time.time() - t0:.1f}s: {counts['sources']} sources, {counts['rows']:,} rows",
            flush=True,
        )
        summary["counts"][split_name] = counts

    # Augment with candidate meta if present
    if cand_meta:
        with contextlib.suppress(Exception):
            summary["candidate_meta"] = cand_meta

    # Write summary JSON
    (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote summary to {artifacts_dir / 'summary.json'}")

    # Optional: fit calibrator on validation
    if args.calibrate and args.calibrate != "none":
        print(
            f"[CAL] Fitting {args.calibrate} calibrator on validation for key={args.calibrate_key} ...",
            flush=True,
        )
        try:
            cal = calibrate_heuristic_on_val(
                key=str(args.calibrate_key),
                g=g,
                cand_val=Path(args.candidates_val),
                splits_root=Path(args.splits_root),
                sample_per_src=int(args.calibrate_sample_per_src),
                max_pairs=int(args.calibrate_max_pairs),
            )
            cal_out = {
                "created_at": _now_iso(),
                "model_tag": f"heuristics:{args.calibrate_key}",
                "features_version": "node_structural_v1",  # for ensemble bookkeeping; heuristics do not use features
                "T0": int(_load_temporal_t0(Path(args.splits_root)) or 0),
                "candidate_pool": str(Path(args.candidates_val)),
                "method": cal.get("method"),
                "params": {k: v for k, v in cal.items() if k not in {"method"}},
            }
            (artifacts_dir / "calibration.json").write_text(json.dumps(cal_out, indent=2))
            print(f"[CAL] Saved calibrator to {artifacts_dir / 'calibration.json'}")
        except Exception as e:
            print(f"[CAL][WARN] Calibration failed: {e}", flush=True)


if __name__ == "__main__":
    main()
