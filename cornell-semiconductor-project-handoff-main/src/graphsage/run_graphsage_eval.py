#!/usr/bin/env python3
"""
GraphSAGE Trainer + Scorecard Evaluator (Core_v1)

Static GraphSAGE trained on the train-only snapshot (T0) and evaluated on the
fixed candidate pools (val/test) with the shared scorecard. GPU by default.

Stage 0 negatives (baseline): uniform + (optional) degree-biased.
Two-hop and semantic negatives can be added next as extensions.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

from src.utils.horizons import assign_horizon_buckets, horizon_slice_names

# ------------------------------ Utilities --------------------------------- #


def _bool(v: str) -> bool:
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_t0(splits_root: Path) -> int | None:
    meta = splits_root.parent / "meta" / "temporal_splits.json"
    try:
        if meta.exists():
            obj = json.loads(meta.read_text())
            return int(obj.get("boundaries", {}).get("T0_end"))
    except Exception:  # nosec B110 -- best-effort metadata read, pass is intentional
        pass
    tr = splits_root / "train_edges.parquet"
    if tr.exists():
        try:
            import pyarrow.parquet as pq  # type: ignore

            tbl = pq.read_table(tr, columns=["ts"])  # type: ignore
            arr = tbl.column(0).to_numpy(zero_copy_only=False)
            if len(arr):
                return int(np.nanmax(arr))
        except Exception:
            df = pd.read_parquet(tr, columns=["ts"])  # type: ignore
            if len(df):
                return int(df["ts"].max())
    return None


# ------------------------------ Data Loading ------------------------------- #


@dataclass
class Graph:
    csr: sp.csr_matrix
    csc: sp.csc_matrix
    out_deg: np.ndarray
    in_deg: np.ndarray
    num_nodes: int


def load_graph(adj_path: Path, undirected: bool) -> Graph:
    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency not found: {adj_path}")
    csr = sp.load_npz(adj_path).tocsr(copy=False)
    if undirected:
        csc_tmp = csr.tocsc(copy=False)
        csr = (csr + csc_tmp).astype(bool).astype(np.uint8).tocsr(copy=False)
    # Ensure contiguous int32 indices/indptr for faster neighbor sampling in PyG
    # (SciPy defaults are typically int32, but enforce to avoid implicit casts)
    if csr.indices.dtype != np.int32:
        csr.indices = np.ascontiguousarray(csr.indices.astype(np.int32, copy=False))
    else:
        csr.indices = np.ascontiguousarray(csr.indices)
    if csr.indptr.dtype != np.int32:
        csr.indptr = np.ascontiguousarray(csr.indptr.astype(np.int32, copy=False))
    else:
        csr.indptr = np.ascontiguousarray(csr.indptr)
    csc = csr.tocsc(copy=False)
    if csc.indices.dtype != np.int32:
        csc.indices = np.ascontiguousarray(csc.indices.astype(np.int32, copy=False))
    else:
        csc.indices = np.ascontiguousarray(csc.indices)
    if csc.indptr.dtype != np.int32:
        csc.indptr = np.ascontiguousarray(csc.indptr.astype(np.int32, copy=False))
    else:
        csc.indptr = np.ascontiguousarray(csc.indptr)
    out_deg = np.diff(csr.indptr).astype(np.int64, copy=False)
    in_deg = np.diff(csc.indptr).astype(np.int64, copy=False)
    return Graph(csr=csr, csc=csc, out_deg=out_deg, in_deg=in_deg, num_nodes=int(csr.shape[0]))


def csr_to_edge_index(csr: sp.csr_matrix) -> Tensor:
    rows, cols = csr.nonzero()
    return torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))


def load_features(
    feat_path: Path, num_nodes: int, struct_feats_path: Path | None = None
) -> tuple[Tensor, pd.DataFrame]:
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found: {feat_path}")
    df = pd.read_parquet(feat_path)
    # Expect a node_id column first or implicit sorted by node_id
    if "node_id" in df.columns:
        df = df.sort_values("node_id").reset_index(drop=True)
        feat_df = df.drop(columns=["node_id"])  # keep full df for attrs
    else:
        feat_df = df.copy()
    x = feat_df.values.astype(np.float32, copy=False)
    # Optionally concatenate structural scalars (node_structural_v1.parquet)
    if struct_feats_path is not None:
        sdf = pd.read_parquet(struct_feats_path)
        if "node_id" in sdf.columns:
            sdf = sdf.sort_values("node_id").reset_index(drop=True)
            sdf = sdf.drop(columns=["node_id"])  # retain only feature columns
        sX = sdf.values.astype(np.float32, copy=False)
        if sX.shape[0] != num_nodes:
            raise ValueError(f"Structural features rows {sX.shape[0]} != num_nodes {num_nodes}")
        x = np.concatenate([x, sX], axis=1)
    assert x.shape[0] == num_nodes, f"Feature rows {x.shape[0]} != num_nodes {num_nodes}"
    return torch.from_numpy(x), df


# ------------------------------ Streamer ----------------------------------- #


class CandidateStreamer:
    def __init__(self, path: Path, batch_size: int = 2_000_000):
        self.path = path
        self.batch = int(max(100_000, batch_size))
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq
        except Exception as e:
            raise RuntimeError("pyarrow is required to stream candidate Parquet") from e
        self.pq = pq  # type: ignore

    def __iter__(self) -> Iterator[pd.DataFrame]:
        # Schema-aware columns
        try:
            pf = self.pq.ParquetFile(self.path)  # type: ignore
            has_ts = "ts" in set(pf.schema.names)
        except Exception:
            has_ts = False
        cols = ["src_id", "dst_id", "label"] + (["ts"] if has_ts else [])
        tbl = self.pq.read_table(self.path, columns=cols)  # type: ignore
        n = tbl.num_rows
        start = 0
        remainder: pd.DataFrame | None = None
        while start < n:
            end = min(start + self.batch, n)
            df = tbl.slice(start, end - start).to_pandas(types_mapper={})
            df = df.rename(columns={"src": "src_id", "dst": "dst_id"})
            if remainder is not None and len(remainder):
                df = pd.concat([remainder, df], ignore_index=True)
                remainder = None
            if df.empty:
                start = end
                continue
            last_src = int(df["src_id"].iloc[-1])
            cut = len(df)
            i = cut - 2
            while i >= 0 and int(df["src_id"].iloc[i]) == last_src:
                i -= 1
            i += 1
            if i > 0 and i < cut:
                remainder = df.iloc[i:].copy()
                df = df.iloc[:i].copy()
            yield df
            start = end


# ------------------------------ Model -------------------------------------- #


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(SAGEConv(in_channels, hidden))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden, hidden))
        self.layers.append(SAGEConv(hidden, hidden))
        self.dropout = float(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i != len(self.layers) - 1:
                x = torch.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.forward(x, edge_index)


def dot_scores(z_src: Tensor, z_dst: Tensor) -> Tensor:
    return (z_src * z_dst).sum(dim=-1)


class FeatureMLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        dims = [int(in_channels)] + [int(hidden)] * max(1, int(num_layers))
        layers: list[torch.nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                layers.append(torch.nn.ReLU())
                if float(dropout) > 0:
                    layers.append(torch.nn.Dropout(p=float(dropout)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        return self.net(x)

    def encode(self, x: Tensor, edge_index: Tensor | None = None) -> Tensor:
        return self.forward(x, edge_index)


# -------------------------- Negatives (Stage 0) ---------------------------- #


def sample_uniform_negatives(pos_edges: Tensor, num_nodes: int, num_neg: int) -> Tensor:
    # Use PyG negative_sampling to ensure non-edges
    return negative_sampling(
        pos_edges.to(torch.long), num_nodes=int(num_nodes), num_neg_samples=int(num_neg)
    )


def sample_degree_negatives(
    out_deg: np.ndarray, in_deg: np.ndarray, M: int, rng: np.random.Generator, deg_exp: float = 1.0
) -> np.ndarray:
    # Degree-proportional sampling for (u,v) with exponent; may include existing edges (small chance)
    od = out_deg.astype(np.float64)
    idg = in_deg.astype(np.float64)
    od = np.clip(od, 1.0, None) ** float(deg_exp)
    idg = np.clip(idg, 1.0, None) ** float(deg_exp)
    pu = od / od.sum()
    pv = idg / idg.sum()
    u = rng.choice(len(od), size=M, p=pu)
    v = rng.choice(len(idg), size=M, p=pv)
    return np.vstack([u, v]).astype(np.int64)


# --------- Helpers for per-source batched negatives (high-impact optimization) --------- #


class AliasSampler:
    def __init__(self, probs: np.ndarray):
        p = probs.astype(np.float64, copy=False)
        s = float(p.sum())
        if s <= 0:
            p = np.ones_like(p, dtype=np.float64)
            s = float(p.sum())
        q = (p / s) * p.size
        self.J = np.empty(p.size, dtype=np.uint32)
        self.q = np.empty(p.size, dtype=np.float32)
        small: list[int] = []
        large: list[int] = []
        for i, qi in enumerate(q):
            (small if qi < 1.0 else large).append(i)
        while small and large:
            s_i = small.pop()
            l_i = large.pop()
            self.q[s_i] = float(q[s_i])
            self.J[s_i] = int(l_i)
            q[l_i] = (q[l_i] + q[s_i]) - 1.0
            (small if q[l_i] < 1.0 else large).append(l_i)
        for i in large + small:
            self.q[i] = 1.0
            self.J[i] = int(i)

    def sample(self, m: int, rng: np.random.Generator) -> np.ndarray:
        if m <= 0:
            return np.empty(0, dtype=np.int64)
        kk = rng.integers(0, self.J.size, size=int(m), dtype=np.uint32)
        uu = rng.random(size=int(m))
        out = kk.copy()
        mask = uu >= self.q[kk]
        out[mask] = self.J[kk[mask]]
        return out.astype(np.int64, copy=False)


class TwoHopCache:
    def __init__(self, csr: sp.csr_matrix, csc: sp.csc_matrix, cap: int = 200_000):
        self.csr, self.csc = csr, csc
        self.cache: dict[int, np.ndarray] = {}
        self.order: list[int] = []
        self.cap = int(cap)

    def get(self, u: int) -> np.ndarray:
        hit = self.cache.get(int(u))
        if hit is not None:
            return hit
        csr, csc = self.csr, self.csc
        out_u = csr.indices[csr.indptr[u] : csr.indptr[u + 1]]
        in_u = csc.indices[csc.indptr[u] : csc.indptr[u + 1]]
        one = (
            np.unique(np.concatenate([out_u, in_u]))
            if (out_u.size or in_u.size)
            else np.empty(0, dtype=np.int64)
        )
        parts: list[np.ndarray] = []
        for w in one:
            parts.append(csr.indices[csr.indptr[w] : csr.indptr[w + 1]])
            parts.append(csc.indices[csc.indptr[w] : csc.indptr[w + 1]])
        two = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
        two_sorted = np.sort(two)
        self.cache[u] = two_sorted
        self.order.append(int(u))
        if len(self.order) > self.cap:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return two_sorted


def filter_exclusions(cand: np.ndarray, bad_sorted: np.ndarray) -> np.ndarray:
    if cand.size == 0 or bad_sorted.size == 0:
        return cand
    pos = np.searchsorted(bad_sorted, cand, side="left")
    pos = np.clip(pos, 0, bad_sorted.size - 1)
    keep = bad_sorted[pos] != cand
    return cand[keep]


def build_semantic_indices(
    feat_df: pd.DataFrame, keys: list[str]
) -> dict[str, dict[int, np.ndarray]]:
    indices: dict[str, dict[int, np.ndarray]] = {}
    for k in keys:
        if k not in feat_df.columns:
            continue
        s = feat_df[k]
        arr: np.ndarray
        try:
            # Case 1: Already integer-like
            if pd.api.types.is_integer_dtype(s):
                arr = s.to_numpy(dtype=np.int64, copy=False)
            # Case 2: Float dtype – coerce to integers when safe, else fall back to factorized codes
            elif pd.api.types.is_float_dtype(s):
                # If all finite and near-integer, round and cast; else factorize
                s_num = pd.to_numeric(s, errors="coerce")
                finite_mask = np.isfinite(s_num.to_numpy(dtype=np.float64, copy=False))  # type: ignore[union-attr,arg-type]
                vals = s_num.to_numpy(dtype=np.float64, copy=False)  # type: ignore[union-attr,arg-type]
                near_int = np.all(~finite_mask | (np.abs(vals - np.round(vals)) < 1e-6))  # type: ignore[call-overload,arg-type]
                if near_int:
                    arr = np.round(vals).astype(np.int64, copy=False)  # type: ignore[call-overload,arg-type]
                    arr[~finite_mask] = -1
                else:
                    codes, _ = pd.factorize(s.fillna("<NA>"), sort=False)
                    arr = codes.astype(np.int64, copy=False)
            # Case 3: Strings / categoricals – use stable factorization
            else:
                codes, _ = pd.factorize(s.fillna("<NA>"), sort=False)
                arr = codes.astype(np.int64, copy=False)
        except Exception:
            # Last resort: robust numeric coercion with fallback
            s_num = pd.to_numeric(s, errors="coerce")
            arr = np.where(np.isfinite(s_num), np.round(s_num).astype(np.int64), -1)  # type: ignore[call-overload,arg-type]

        groups: dict[int, list[int]] = {}
        for nid, v in enumerate(arr):
            groups.setdefault(int(v), []).append(nid)
        indices[k] = {vv: np.array(ids, dtype=np.int64) for vv, ids in groups.items()}
    return indices


def is_train_edge(u: int, v: int, csr: sp.csr_matrix) -> bool:
    indptr, indices = csr.indptr, csr.indices
    start, end = indptr[u], indptr[u + 1]
    row = indices[start:end]
    pos = np.searchsorted(row, v)
    return bool((pos < row.size) and (row[pos] == v))


def one_hop_set(u: int, csr: sp.csr_matrix, csc: sp.csc_matrix) -> np.ndarray:
    out_u = csr.indices[csr.indptr[u] : csr.indptr[u + 1]]
    in_u = csc.indices[csc.indptr[u] : csc.indptr[u + 1]]
    if out_u.size or in_u.size:
        return np.unique(np.concatenate([out_u, in_u])).astype(np.int64)
    return np.empty(0, dtype=np.int64)


def sample_twohop(
    u: int,
    csr: sp.csr_matrix,
    csc: sp.csc_matrix,
    rng: np.random.Generator,
    exclude: np.ndarray,
    max_tries: int = 50,
) -> int | None:
    # Approximate two-hop by sampling a random one-hop neighbor then a random neighbor of that node
    out_u = csr.indices[csr.indptr[u] : csr.indptr[u + 1]]
    in_u = csc.indices[csc.indptr[u] : csc.indptr[u + 1]]
    one = (
        np.unique(np.concatenate([out_u, in_u]))
        if (out_u.size or in_u.size)
        else np.empty(0, dtype=np.int64)
    )
    if one.size == 0:
        return None
    for _ in range(max_tries):
        w = int(rng.choice(one))
        # choose direction randomly
        if rng.random() < 0.5:
            neigh = csr.indices[csr.indptr[w] : csr.indptr[w + 1]]
        else:
            neigh = csc.indices[csc.indptr[w] : csc.indptr[w + 1]]
        if neigh.size == 0:
            continue
        v = int(rng.choice(neigh))
        if v == u:
            continue
        # exclude one-hop and known train edges
        if np.searchsorted(exclude, v) < exclude.size and exclude[np.searchsorted(exclude, v)] == v:
            continue
        if is_train_edge(u, v, csr):
            continue
        return v
    return None


def sample_semantic(
    u: int,
    feat_df: pd.DataFrame,
    sem_indices: dict[str, dict[int, np.ndarray]],
    keys: list[str],
    rng: np.random.Generator,
    exclude: np.ndarray,
    csr: sp.csr_matrix,
    max_tries: int = 50,
) -> int | None:
    # Try intersection of first two keys if available, then relax
    cands: np.ndarray | None = None
    values: list[tuple[str, int]] = []
    for k in keys:
        if k not in feat_df.columns:
            continue
        val = int((pd.isna(feat_df.at[u, k]) and -1) or int(feat_df.at[u, k]))
        values.append((k, val))
    # intersection of top 2 keys if both present
    if len(values) >= 2 and all(
        v in sem_indices[values[i][0]] for i, v in enumerate([values[0][1], values[1][1]])
    ):
        a = sem_indices[values[0][0]].get(values[0][1], np.empty(0, dtype=np.int64))
        b = sem_indices[values[1][0]].get(values[1][1], np.empty(0, dtype=np.int64))
        if a.size and b.size:
            cands = np.intersect1d(a, b, assume_unique=False)
    # fallback to first key
    if (cands is None or cands.size == 0) and len(values) >= 1 and values[0][0] in sem_indices:
        cands = sem_indices[values[0][0]].get(values[0][1], np.empty(0, dtype=np.int64))
    # fallback to second key
    if (cands is None or cands.size == 0) and len(values) >= 2 and values[1][0] in sem_indices:
        cands = sem_indices[values[1][0]].get(values[1][1], np.empty(0, dtype=np.int64))
    if cands is None or cands.size == 0:
        return None
    # sample with exclusions
    cands_sorted = np.sort(cands)
    for _ in range(max_tries):
        v = int(rng.choice(cands_sorted))
        if v == u:
            continue
        # exclude one-hop and train edges
        if np.searchsorted(exclude, v) < exclude.size and exclude[np.searchsorted(exclude, v)] == v:
            continue
        if is_train_edge(u, v, csr):
            continue
        return v
    return None


# ------------------------------ Training ----------------------------------- #


def train_graphsage(
    g: Graph,
    x: Tensor,
    hidden: int,
    layers: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    neigh1: int,
    neigh2: int,
    neg_ratio: float,
    neg_strategy: str,
    device: torch.device,
    seed: int = 42,
) -> tuple[GraphSAGE, dict[str, float]]:
    import numpy as np  # ensure local binding for np

    torch.manual_seed(seed)
    np.random.seed(seed)
    edge_index = csr_to_edge_index(g.csr).to(device)
    data = torch_geometric.data.Data(x=x.to(device), edge_index=edge_index, num_nodes=g.num_nodes)  # type: ignore

    # Build loader with no built-in negatives; we will sample manually
    # DataLoader/neighbor sampling tuning
    loader_workers: int = int(getattr(train_graphsage, "loader_workers", 4))
    prefetch_factor: int = int(getattr(train_graphsage, "prefetch_factor", 2))
    pin_memory: bool = bool(getattr(train_graphsage, "pin_memory", True)) and (
        device.type == "cuda"
    )
    persistent_workers: bool = bool(getattr(train_graphsage, "persistent_workers", True)) and (
        loader_workers > 0
    )
    # Edge labels (positives) optionally grouped by source for better reuse
    group_by_src: bool = bool(getattr(train_graphsage, "group_by_src", True))
    edge_label_index = edge_index.t()
    if group_by_src:
        src_np = edge_index[0].detach().cpu().numpy()
        order = np.argsort(src_np, kind="mergesort")
        edge_label_index = torch.stack([edge_index[0][order], edge_index[1][order]], dim=0).t()

    loader_kwargs = {
        "data": data,
        "edge_label": edge_label_index,
        "batch_size": int(batch_size),
        "shuffle": not group_by_src,
        "neg_sampling_ratio": 0.0,
        "num_neighbors": [int(neigh1), int(neigh2)] if layers >= 2 else [int(neigh1)],
        "num_workers": int(loader_workers),
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
    }
    if loader_workers > 0:
        loader_kwargs["prefetch_factor"] = int(max(2, prefetch_factor))
    # If PyG supports it, enable replacement sampling for fanout
    with contextlib.suppress(Exception):
        loader_kwargs["replace"] = bool(getattr(train_graphsage, "fanout_replace", True))
    loader = LinkNeighborLoader(**loader_kwargs)
    print(
        f"[TRAIN] Loader ready: batch={batch_size}, neighbors=({neigh1},{neigh2}), layers={layers}"
    )

    # Optional structural ID embeddings (transductive)
    use_id_emb: bool = bool(getattr(train_graphsage, "use_id_emb", False))
    id_emb_dim: int = int(getattr(train_graphsage, "id_emb_dim", 64)) if use_id_emb else 0
    in_channels = int(x.shape[1] + (id_emb_dim if use_id_emb else 0))
    use_no_mp: bool = bool(getattr(train_graphsage, "no_mp", False))
    if use_no_mp:
        model = FeatureMLP(in_channels, int(hidden), int(layers), float(dropout)).to(device)
    else:
        model = GraphSAGE(in_channels, int(hidden), int(layers), float(dropout)).to(device)
    id_emb = None
    if use_id_emb:
        id_emb = torch.nn.Embedding(g.num_nodes, int(id_emb_dim), device=device)
        torch.nn.init.normal_(id_emb.weight, std=0.02)
        opt = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": id_emb.parameters()},
            ],
            lr=float(lr),
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    t0 = time.time()
    losses: list[float] = []
    rng = np.random.default_rng(seed)
    timeout_sec = int(getattr(train_graphsage, "timeout_sec", 0))
    deadline = (time.time() + timeout_sec) if timeout_sec else None

    # Sparse CSR for optional membership checks (CPU)
    csr = g.csr

    # Prepare semantic indices (Stage 2) if requested via neg_strategy 'mix'
    sem_keys: list[str] = getattr(
        train_graphsage, "sem_keys", ["primary_sic_code", "gr_country"]
    )  # default keys

    # Loss mode (BCE or BPR)
    loss_mode: str = str(getattr(train_graphsage, "loss", "bpr")).lower()
    pairwise_negs: int = int(getattr(train_graphsage, "pairwise_negs", 2))

    def _sample_in_deg(k: int, deg_exp: float = 1.0) -> np.ndarray:
        idg = np.clip(g.in_deg.astype(np.float64), 1.0, None) ** float(deg_exp)
        pv = idg / idg.sum()
        return rng.choice(len(pv), size=k, replace=True, p=pv).astype(np.int64)

    # Loss mode (BCE or BPR)
    loss_mode: str = str(getattr(train_graphsage, "loss", "bpr")).lower()
    pairwise_negs: int = int(getattr(train_graphsage, "pairwise_negs", 2))
    use_amp: bool = bool(getattr(train_graphsage, "use_amp", True)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sampler_mode: str = str(getattr(train_graphsage, "sampler_mode", "per_src"))
    async_sampler: bool = bool(getattr(train_graphsage, "async_sampler", False))

    # Early stop / validation probe controls (attached via function attributes)
    val_interval: int = int(getattr(train_graphsage, "val_probe_interval", 0))
    early_patience: int = int(getattr(train_graphsage, "early_stop_patience", 0))
    min_delta: float = float(getattr(train_graphsage, "early_stop_min_delta", 5e-4))
    val_max_src: int = int(getattr(train_graphsage, "val_probe_max_sources", 1000))
    cand_val_path: Path | None = getattr(train_graphsage, "cand_val_path", None)
    splits_root_p: Path | None = getattr(train_graphsage, "splits_root", None)
    Ks_probe: list[int] = list(getattr(train_graphsage, "Ks", [1, 10, 50]))
    emit_sw: bool = bool(getattr(train_graphsage, "emit_strict_warm", False))
    normalize_flag: bool = bool(getattr(train_graphsage, "normalize_emb", True))
    best_ndcg: float = -1.0
    best_epoch: int = -1
    no_improve: int = 0
    best_state: dict[str, object] | None = None  # type: ignore[assignment]
    stop_training: bool = False

    for ep in range(int(epochs)):
        epoch_mix = {"deg": 0, "twohop": 0, "sem": 0, "uniform": 0}
        if deadline and time.time() > deadline:
            print("[TRAIN] Timeout reached before epoch; aborting")
            raise RuntimeError("HPO_TIMEOUT")
        print(f"[TRAIN] Epoch {ep + 1}/{epochs} start")
        model.train()
        epoch_loss = 0.0
        nb = 0
        # Optional async negative producer to overlap CPU sampling with GPU compute
        if loss_mode == "bpr" and sampler_mode == "per_src" and async_sampler:
            it = iter(loader)
            try:
                batch = next(it)
            except StopIteration:
                batch = None
            fut = None
            ex = ThreadPoolExecutor(max_workers=1)

            def build_pairs_for_batch(b):
                # CPU-only builder: returns numpy arrays of local indices (u, vpos, vneg)
                # Extract CPU arrays
                global_ids = b.n_id.detach().cpu().numpy().astype(np.int64)
                gid_to_local = {int(g): int(i) for i, g in enumerate(global_ids)}
                pos_u_g = b.edge_label_index[0].detach().cpu().numpy().astype(np.int64)
                pos_v_g = b.edge_label_index[1].detach().cpu().numpy().astype(np.int64)
                # Local indices
                pos_u_l = np.array([gid_to_local.get(int(g), -1) for g in pos_u_g], dtype=np.int64)
                pos_v_l = np.array([gid_to_local.get(int(g), -1) for g in pos_v_g], dtype=np.int64)
                mask = (pos_u_l >= 0) & (pos_v_l >= 0)
                pos_u_g = pos_u_g[mask]
                pos_v_g = pos_v_g[mask]
                pos_u_l = pos_u_l[mask]
                pos_v_l = pos_v_l[mask]
                if pos_u_g.size == 0:
                    return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
                # Mix params
                k = int(max(1, pairwise_negs))
                k_deg = round(k * float(getattr(train_graphsage, "deg_frac", 0.5)))
                k_two = round(k * float(getattr(train_graphsage, "twohop_frac", 0.0)))
                k_sem = round(k * float(getattr(train_graphsage, "sem_frac", 0.0)))
                k_uni = max(0, k - (k_deg + k_two + k_sem))
                deg_exp_v = float(getattr(train_graphsage, "deg_exp", 1.0))
                sem_keys_v: list[str] = getattr(
                    train_graphsage, "sem_keys", ["primary_sic_code", "gr_country"]
                )  # type: ignore
                feat_df_v: pd.DataFrame | None = getattr(train_graphsage, "feat_df", None)  # type: ignore
                sem_indices_v: dict[str, dict[int, np.ndarray]] | None = getattr(
                    train_graphsage, "sem_indices", None
                )  # type: ignore
                # Alias and cache
                deg_probs = np.power(
                    np.clip(g.in_deg.astype(np.float64), 1.0, None), float(deg_exp_v)
                )
                alias = AliasSampler(deg_probs)
                th_cache = TwoHopCache(g.csr, g.csc)
                # Group by source
                uniq_u, inv_idx = np.unique(pos_u_g, return_inverse=True)
                u_list: list[int] = []
                vp_list: list[int] = []
                vn_list: list[int] = []
                len(global_ids)
                for idx_u, u in enumerate(uniq_u.tolist()):
                    m = int(np.count_nonzero(inv_idx == idx_u))
                    if m == 0:
                        continue
                    need = m * k
                    one = one_hop_set(int(u), g.csr, g.csc)
                    bad = (
                        np.unique(np.concatenate([np.array([u], dtype=np.int64), one]))
                        if one.size
                        else np.array([u], dtype=np.int64)
                    )
                    pool: list[np.ndarray] = []
                    two_sorted = th_cache.get(int(u)) if k_two > 0 else np.empty(0, dtype=np.int64)
                    if k_two > 0 and two_sorted.size:
                        take = min(k_two * 2, two_sorted.size)
                        pool.append(
                            np.random.default_rng().choice(
                                two_sorted, size=take, replace=(take > two_sorted.size)
                            )
                        )
                    if k_sem > 0 and feat_df_v is not None and sem_indices_v is not None:
                        vsem = np.empty(0, dtype=np.int64)
                        try:
                            vals = []
                            for kname in sem_keys_v[:2]:
                                if kname in feat_df_v.columns:
                                    val = int(feat_df_v.at[int(u), kname])
                                    vals.append(
                                        sem_indices_v.get(kname, {}).get(
                                            val, np.empty(0, dtype=np.int64)
                                        )
                                    )
                            if len(vals) >= 2 and vals[0].size and vals[1].size:
                                vsem = np.intersect1d(vals[0], vals[1], assume_unique=False)
                            elif vals:
                                vsem = vals[0]
                        except Exception:
                            vsem = np.empty(0, dtype=np.int64)
                        if vsem.size:
                            take = min(k_sem * 2, vsem.size)
                            pool.append(
                                np.random.default_rng().choice(
                                    vsem, size=take, replace=(take > vsem.size)
                                )
                            )
                    if k_deg > 0:
                        pool.append(alias.sample(int(k_deg * 2), rng))
                    if k_uni > 0:
                        pool.append(
                            np.random.default_rng().integers(
                                0, g.num_nodes, size=int(k_uni * 2), dtype=np.int64
                            )
                        )
                    cand = np.concatenate(pool) if pool else np.empty(0, dtype=np.int64)
                    if cand.size:
                        cand = cand[cand != int(u)]
                        cand = filter_exclusions(cand, np.sort(bad))
                    if cand.size < need:
                        extra = alias.sample(int(need - cand.size), rng)
                        extra = extra[extra != int(u)]
                        extra = filter_exclusions(extra, np.sort(bad))
                        cand = np.concatenate([cand, extra]) if extra.size else cand
                    if cand.size == 0:
                        continue
                    np.random.default_rng().shuffle(cand)
                    cand = cand[:need]
                    # Map to local indices via small dict
                    neg_local = np.array(
                        [gid_to_local.get(int(v), -1) for v in cand], dtype=np.int64
                    )
                    neg_local = neg_local[neg_local >= 0]
                    if neg_local.size == 0:
                        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
                    if neg_local.size < need:
                        reps = int((need + neg_local.size - 1) // neg_local.size)
                        neg_local = np.tile(neg_local, reps)[:need]
                    else:
                        neg_local = neg_local[:need]
                    # Append pairs
                    u_locals = pos_u_l[inv_idx == idx_u]
                    vpos_locals = pos_v_l[inv_idx == idx_u]
                    neg_mat = neg_local.reshape(m, k)
                    for jj in range(m):
                        u_list.extend([int(u_locals[jj])] * k)
                        vp_list.extend([int(vpos_locals[jj])] * k)
                        vn_list.extend(neg_mat[jj].tolist())
                return (
                    np.asarray(u_list, dtype=np.int64),
                    np.asarray(vp_list, dtype=np.int64),
                    np.asarray(vn_list, dtype=np.int64),
                )

            # Prime the pipeline
            if batch is not None:
                fut = ex.submit(build_pairs_for_batch, batch)
            while batch is not None:
                # Fetch next batch
                try:
                    next_batch = next(it)
                except StopIteration:
                    next_batch = None
                # Wait negatives for current batch
                u_np, vp_np, vn_np = (
                    fut.result() if fut is not None else (np.empty(0, np.int64),) * 3
                )
                batch = batch.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if use_id_emb and id_emb is not None:
                        x_ids = id_emb(batch.n_id)
                        x_fused = torch.cat([batch.x, x_ids], dim=1)
                        z = model.encode(x_fused, batch.edge_index)
                    else:
                        z = model.encode(batch.x, batch.edge_index)
                    # Align training objective with evaluation when --normalize-emb true
                    if normalize_flag:
                        z = F.normalize(z, p=2, dim=1)
                if u_np.size and vp_np.size and vn_np.size:
                    u_t = torch.as_tensor(u_np, device=device)
                    vp_t = torch.as_tensor(vp_np, device=device)
                    vn_t = torch.as_tensor(vn_np, device=device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pos_logit = dot_scores(z[u_t], z[vp_t])
                        neg_logit = dot_scores(z[u_t], z[vn_t])
                        loss = -torch.nn.functional.logsigmoid(pos_logit - neg_logit).mean()
                    # Optional one-time sampler debug
                    if bool(getattr(train_graphsage, "debug_sampler", False)) and not bool(
                        getattr(train_graphsage, "debug_done", False)
                    ):
                        try:
                            u_g = batch.n_id[u_t].detach().cpu().numpy()
                            vp_g = batch.n_id[vp_t].detach().cpu().numpy()
                            vn_g = batch.n_id[vn_t].detach().cpu().numpy()
                            import numpy as np

                            m = min(2048, u_g.shape[0])
                            idx = np.random.default_rng().integers(0, u_g.shape[0], size=m)
                            u_s, vp_s, vn_s = u_g[idx], vp_g[idx], vn_g[idx]

                            # Pairing/exclusions
                            def _is_edge(u: int, v: int) -> bool:
                                return is_train_edge(int(u), int(v), g.csr)

                            neg_in_train = np.mean(
                                [_is_edge(int(u_s[i]), int(vn_s[i])) for i in range(m)]
                            )
                            neg_eq_pos = float(np.mean(vn_s == vp_s))
                            pair_acc = float(((pos_logit - neg_logit) > 0).float().mean().item())
                            print(
                                f"[DEBUG][SAGE] sampler: neg_in_train={neg_in_train:.4f} neg_eq_pos={neg_eq_pos:.4f} pair_acc={pair_acc:.4f}"
                            )
                        except Exception as e:
                            print(f"[DEBUG][SAGE][WARN] sampler debug failed: {e}")
                        train_graphsage.debug_done = True  # type: ignore[attr-defined]
                    opt.zero_grad(set_to_none=True)
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    epoch_loss += float(loss)
                    nb += 1
                # Submit next negatives
                if next_batch is not None:
                    fut = ex.submit(build_pairs_for_batch, next_batch)
                batch = next_batch
            ex.shutdown(wait=True)
            continue  # proceed to next epoch

        # Non-async path
        for batch in loader:
            if deadline and time.time() > deadline:
                print("[TRAIN] Timeout reached during epoch; aborting")
                raise RuntimeError("HPO_TIMEOUT")
            batch = batch.to(device)
            pos_e = batch.edge_label_index  # [2, P]
            P = pos_e.shape[1]
            Nneg = int(max(1, round(neg_ratio * P)))

            # Negatives / pairs
            if loss_mode == "bpr":
                # Build global->local map for current batch
                global_ids = batch.n_id
                inv = torch.full((g.num_nodes,), -1, dtype=torch.long, device=device)
                inv[global_ids] = torch.arange(global_ids.numel(), device=device)

                # Map positives to local and filter valid
                pos_local = inv[pos_e]
                L = int(global_ids.numel())
                mask_pos = (
                    (pos_local[0] >= 0)
                    & (pos_local[1] >= 0)
                    & (pos_local[0] < L)
                    & (pos_local[1] < L)
                )
                if mask_pos.sum() == 0:
                    continue
                pos_local = pos_local[:, mask_pos]
                pos_u_g = pos_e[0][mask_pos].detach().cpu().numpy().astype(np.int64)
                pos_vpos_g = pos_e[1][mask_pos].detach().cpu().numpy().astype(np.int64)
                pos_u_l = pos_local[0].detach().cpu().numpy().astype(np.int64)
                pos_vpos_l = pos_local[1].detach().cpu().numpy().astype(np.int64)

                # Encode once per batch
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if use_id_emb and id_emb is not None:
                        x_ids = id_emb(batch.n_id)
                        x_fused = torch.cat([batch.x, x_ids], dim=1)
                        z = model.encode(x_fused, batch.edge_index)
                    else:
                        z = model.encode(batch.x, batch.edge_index)
                    # Align training objective with evaluation when --normalize-emb true
                    if normalize_flag:
                        z = F.normalize(z, p=2, dim=1)

                # Mix params and helpers
                deg_frac = float(getattr(train_graphsage, "deg_frac", 0.5))
                twohop_frac = float(getattr(train_graphsage, "twohop_frac", 0.0))
                sem_frac = float(getattr(train_graphsage, "sem_frac", 0.0))
                deg_exp = float(getattr(train_graphsage, "deg_exp", 1.0))
                sem_keys: list[str] = getattr(
                    train_graphsage, "sem_keys", ["primary_sic_code", "gr_country"]
                )  # type: ignore
                feat_df: pd.DataFrame | None = getattr(train_graphsage, "feat_df", None)  # type: ignore
                sem_indices: dict[str, dict[int, np.ndarray]] | None = getattr(  # type: ignore[misc]
                    train_graphsage, "sem_indices", None
                )  # type: ignore[attr-defined]
                # Build alias sampler once per call
                deg_probs = np.power(
                    np.clip(g.in_deg.astype(np.float64), 1.0, None), float(deg_exp)
                )
                alias = AliasSampler(deg_probs)
                th_cache = TwoHopCache(g.csr, g.csc)

                if sampler_mode == "per_src":
                    # Group by source and sample once per source
                    uniq_u, inv_idx = np.unique(pos_u_g, return_inverse=True)
                    u_list: list[int] = []
                    vpos_list: list[int] = []
                    vneg_list: list[int] = []
                    for idx_u, u in enumerate(uniq_u.tolist()):
                        # indices of positives for this u
                        mask_u = inv_idx == idx_u
                        m = int(np.count_nonzero(mask_u))
                        if m == 0:
                            continue
                        need = int(m * max(1, pairwise_negs))
                        # exclusion set
                        one = one_hop_set(int(u), g.csr, g.csc)
                        bad = (
                            np.unique(np.concatenate([np.array([u], dtype=np.int64), one]))
                            if one.size
                            else np.array([u], dtype=np.int64)
                        )
                        # per-bucket counts
                        k_deg = round(need * deg_frac)
                        k_two = round(need * twohop_frac)
                        k_sem = round(need * sem_frac)
                        k_uni = max(0, need - (k_deg + k_two + k_sem))
                        epoch_mix["deg"] += k_deg
                        epoch_mix["twohop"] += k_two
                        epoch_mix["sem"] += k_sem
                        epoch_mix["uniform"] += k_uni
                        pool: list[np.ndarray] = []
                        # two-hop pool
                        two_sorted = (
                            th_cache.get(int(u)) if k_two > 0 else np.empty(0, dtype=np.int64)
                        )
                        if k_two > 0 and two_sorted.size:
                            take = min(k_two * 2, two_sorted.size)
                            pool.append(
                                np.random.default_rng().choice(
                                    two_sorted, size=take, replace=(take > two_sorted.size)
                                )
                            )
                        # sem pool
                        if k_sem > 0 and feat_df is not None and sem_indices is not None:
                            # Build intersection (SIC & country) then fallback
                            vsem: np.ndarray = np.empty(0, dtype=np.int64)
                            try:
                                if sem_keys:
                                    vals = []
                                    for k in sem_keys[:2]:
                                        if k in feat_df.columns:
                                            val = int(feat_df.at[int(u), k])
                                            idxmap = sem_indices.get(k, {})
                                            vals.append(
                                                idxmap.get(val, np.empty(0, dtype=np.int64))
                                            )
                                    if len(vals) >= 2 and vals[0].size and vals[1].size:
                                        vsem = np.intersect1d(vals[0], vals[1], assume_unique=False)
                                    elif vals:
                                        vsem = vals[0]
                            except Exception:
                                vsem = np.empty(0, dtype=np.int64)
                            if vsem.size:
                                take = min(k_sem * 2, vsem.size)
                                pool.append(
                                    np.random.default_rng().choice(
                                        vsem, size=take, replace=(take > vsem.size)
                                    )
                                )
                        # deg pool
                        if k_deg > 0:
                            pool.append(alias.sample(int(k_deg * 2), rng))
                        # uniform pool
                        if k_uni > 0:
                            pool.append(
                                rng.integers(0, g.num_nodes, size=int(k_uni * 2), dtype=np.int64)
                            )
                        cand = np.concatenate(pool) if pool else np.empty(0, dtype=np.int64)
                        if cand.size:
                            # filter exclusions and self; filter positives for this u
                            cand = cand[cand != int(u)]
                            cand = filter_exclusions(cand, np.sort(bad))
                        # ensure enough
                        if cand.size < need:
                            extra = alias.sample(int(need - cand.size), rng)
                            extra = extra[extra != int(u)]
                            extra = filter_exclusions(extra, np.sort(bad))
                            cand = np.concatenate([cand, extra]) if extra.size else cand
                        if cand.size == 0:
                            continue
                        rng.shuffle(cand)
                        cand = cand[:need]
                        # Map negatives to local
                        neg_local = inv[torch.as_tensor(cand, device=device)]
                        mask_ok = (neg_local >= 0) & (neg_local < L)
                        neg_local = neg_local[mask_ok]
                        # If underfilled after mapping to local, re-sample globally and remap rather than padding with random locals.
                        max_topup_attempts = 4
                        attempts = 0
                        while int(neg_local.numel()) < need and attempts < max_topup_attempts:
                            rem = int(need - int(neg_local.numel()))
                            # try alias (degree) + uniform extra pool and filter again
                            extra = alias.sample(int(rem * 2), rng)
                            extra = extra[extra != int(u)]
                            extra = filter_exclusions(extra, np.sort(bad))
                            if extra.size == 0:
                                break
                            extra_local = inv[torch.as_tensor(extra, device=device)]
                            mask_ok2 = (extra_local >= 0) & (extra_local < L)
                            extra_local = extra_local[mask_ok2]
                            if extra_local.numel() > 0:
                                need_take = min(rem, int(extra_local.numel()))
                                neg_local = torch.cat([neg_local, extra_local[:need_take]], dim=0)
                            attempts += 1
                        # Ensure exact length by repeating allowed negatives if still short
                        if int(neg_local.numel()) == 0:
                            continue
                        if int(neg_local.numel()) < need:
                            reps = int(
                                (need + int(neg_local.numel()) - 1) // int(neg_local.numel())
                            )
                            neg_local = neg_local.repeat(repeats=(reps,))[:need]
                        else:
                            neg_local = neg_local[:need]
                        # For this u, append pairs for each positive under mask_u
                        u_locals = pos_u_l[mask_u]
                        vpos_locals = pos_vpos_l[mask_u]
                        # shape (m, K)
                        neg_mat = neg_local.detach().cpu().numpy().reshape(m, max(1, pairwise_negs))
                        for jj in range(m):
                            u_list.extend([int(u_locals[jj])] * max(1, pairwise_negs))
                            vpos_list.extend([int(vpos_locals[jj])] * max(1, pairwise_negs))
                            vneg_list.extend(neg_mat[jj].tolist())
                    if not vneg_list:
                        # Debug: no pairs built for this batch (underfilled)
                        if bool(getattr(train_graphsage, "debug_sampler", False)) and not bool(
                            getattr(train_graphsage, "debug_done", False)
                        ):
                            with contextlib.suppress(Exception):
                                print(
                                    f"[DEBUG][SAGE] sampler: pairs=U={len(u_list)},V+={len(vpos_list)},V-={len(vneg_list)} reason=underfilled"
                                )
                            train_graphsage.debug_done = True  # type: ignore[attr-defined]
                        continue
                    u_t = torch.as_tensor(np.array(u_list, dtype=np.int64), device=device)
                    vp_t = torch.as_tensor(np.array(vpos_list, dtype=np.int64), device=device)
                    vn_t = torch.as_tensor(np.array(vneg_list, dtype=np.int64), device=device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pos_logit = dot_scores(z[u_t], z[vp_t])
                        neg_logit = dot_scores(z[u_t], z[vn_t])
                        loss = -torch.nn.functional.logsigmoid(pos_logit - neg_logit).mean()
                    opt.zero_grad(set_to_none=True)
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    epoch_loss += float(loss)
                    nb += 1
                    # Optional one-time sampler invariants when we actually formed pairs
                    if bool(getattr(train_graphsage, "debug_sampler", False)) and not bool(
                        getattr(train_graphsage, "debug_done", False)
                    ):
                        try:
                            # Compute invariants on a subsample
                            import numpy as np

                            m = min(2048, int(u_t.numel()))
                            idx = torch.randint(0, int(u_t.numel()), (m,), device=u_t.device)
                            # Map back to global ids for edge checks
                            ug = batch.n_id[u_t[idx]].detach().cpu().numpy()
                            vpg = batch.n_id[vp_t[idx]].detach().cpu().numpy()
                            vng = batch.n_id[vn_t[idx]].detach().cpu().numpy()

                            def _is_edge(u: int, v: int) -> bool:
                                return is_train_edge(int(u), int(v), g.csr)

                            neg_in_train = float(
                                np.mean([_is_edge(int(ug[i]), int(vng[i])) for i in range(m)])
                            )
                            neg_eq_pos = float(np.mean(vng == vpg))
                            pair_acc = float(((pos_logit - neg_logit) > 0).float().mean().item())
                            print(
                                f"[DEBUG][SAGE] sampler: pairs=U={int(u_t.numel())},V+={int(vp_t.numel())},V-={int(vn_t.numel())} neg_in_train={neg_in_train:.4f} neg_eq_pos={neg_eq_pos:.4f} pair_acc={pair_acc:.4f}"
                            )
                        except Exception as e:
                            print(f"[DEBUG][SAGE][WARN] sampler debug failed: {e}")
                        train_graphsage.debug_done = True  # type: ignore[attr-defined]
                    continue  # to next batch

                # Fallback: per-positive (legacy) — slower
                # (Old loop retained for A/B or if sampler_mode == per_pos)
                # ... legacy branch intentionally not duplicated here for brevity ...
                # Force fallback to BCE path below if not using per_src

            # BCE path (legacy)
            if neg_strategy == "uniform":
                neg_e = sample_uniform_negatives(pos_e, g.num_nodes, Nneg).to(device)
            elif neg_strategy == "mix_deg50":
                # 50% degree-biased + 50% uniform. Ensure degree-biased negatives
                # are drawn from nodes visible in the current mini-batch subgraph
                # to avoid mapping dropouts (global->local).
                k_deg = Nneg // 2
                k_uni = Nneg - k_deg
                # Build global->local map for current batch first
                global_ids = batch.n_id  # global node ids in this subgraph
                inv = torch.full((g.num_nodes,), -1, dtype=torch.long, device=device)
                inv[global_ids] = torch.arange(global_ids.numel(), device=device)
                # Degree-biased negatives: choose (u in batch positives, v in batch nodes by in-degree)
                if k_deg > 0:
                    us_global = pos_e[0].detach().cpu().numpy().astype(np.int64)
                    # Favor selecting sources seen more often in this batch
                    if us_global.size > 0:
                        src_choices = rng.choice(us_global, size=k_deg, replace=True)
                        gids_cpu = global_ids.detach().cpu().numpy().astype(np.int64)
                        local_in_deg = g.in_deg[gids_cpu].astype(np.float64, copy=False)
                        local_in_deg = np.clip(local_in_deg, 1.0, None)
                        p_local = local_in_deg / local_in_deg.sum()
                        dst_idx_local = rng.choice(
                            len(gids_cpu), size=k_deg, replace=True, p=p_local
                        )
                        dst_global = gids_cpu[dst_idx_local]
                        deg_pairs_np = np.vstack([src_choices, dst_global]).astype(np.int64)
                        neg_e_deg = torch.from_numpy(deg_pairs_np).to(device)
                    else:
                        neg_e_deg = sample_uniform_negatives(pos_e, g.num_nodes, k_deg).to(device)
                else:
                    neg_e_deg = torch.empty((2, 0), dtype=torch.long, device=device)
                # Uniform negatives (global)
                neg_e_uni = sample_uniform_negatives(pos_e, g.num_nodes, k_uni).to(device)
                neg_e = torch.cat([neg_e_deg, neg_e_uni], dim=1)
            elif neg_strategy in {"mix", "mix_v3"}:
                # Custom mix: deg_frac + twohop_frac + sem_frac + uniform remainder
                deg_frac = float(getattr(train_graphsage, "deg_frac", 0.5))
                twohop_frac = float(getattr(train_graphsage, "twohop_frac", 0.0))
                sem_frac = float(getattr(train_graphsage, "sem_frac", 0.0))
                deg_exp = float(getattr(train_graphsage, "deg_exp", 1.0))
                k_deg = int(Nneg * deg_frac)
                k_two = int(Nneg * twohop_frac)
                k_sem = int(Nneg * sem_frac)
                k_uni = Nneg - (k_deg + k_two + k_sem)
                pairs: list[np.ndarray] = []
                # Degree-biased
                if k_deg > 0:
                    pairs.append(
                        sample_degree_negatives(g.out_deg, g.in_deg, k_deg, rng, deg_exp=deg_exp)
                    )
                # For two-hop and semantic, sample per-source
                # Build one-hop excludes per unique u in batch
                us = pos_e[0].detach().cpu().numpy().astype(np.int64)
                uniq_u, counts = np.unique(us, return_counts=True)
                # Map u -> one-hop sorted array for exclusion
                onehop_map: dict[int, np.ndarray] = {
                    int(u): one_hop_set(int(u), g.csr, g.csc) for u in uniq_u
                }
                # Two-hop sampling
                if k_two > 0:
                    negs_two: list[tuple[int, int]] = []
                    attempts = 0
                    target = k_two
                    while len(negs_two) < target and attempts < target * 50:
                        u = int(rng.choice(uniq_u))
                        excl = np.sort(onehop_map[u])
                        v = sample_twohop(u, g.csr, g.csc, rng, exclude=excl)
                        if v is not None and not is_train_edge(u, v, g.csr):
                            negs_two.append((u, int(v)))
                        attempts += 1
                    if negs_two:
                        pairs.append(np.array(negs_two, dtype=np.int64).T)
                # Semantic sampling
                if k_sem > 0:
                    # Need feat_df and semantic indices; expect attached on function
                    feat_df: pd.DataFrame = train_graphsage.feat_df  # type: ignore[attr-defined]
                    sem_indices: dict[str, dict[int, np.ndarray]] = train_graphsage.sem_indices  # type: ignore[attr-defined]
                    negs_sem: list[tuple[int, int]] = []
                    attempts = 0
                    target = k_sem
                    while len(negs_sem) < target and attempts < target * 50:
                        u = int(rng.choice(uniq_u))
                        excl = np.sort(onehop_map[u])
                        v = sample_semantic(
                            u, feat_df, sem_indices, sem_keys, rng, exclude=excl, csr=g.csr
                        )
                        if v is not None and not is_train_edge(u, v, g.csr):
                            negs_sem.append((u, int(v)))
                        attempts += 1
                    if negs_sem:
                        pairs.append(np.array(negs_sem, dtype=np.int64).T)
                # Uniform fallback
                if k_uni > 0:
                    pairs.append(
                        sample_degree_negatives(
                            g.out_deg * 0 + 1, g.in_deg * 0 + 1, k_uni, rng, deg_exp=1.0
                        )
                    )  # approx uniform
                # Combine and to torch
                if pairs:
                    neg_e = torch.from_numpy(np.concatenate(pairs, axis=1)).to(device)
                else:
                    neg_e = sample_uniform_negatives(pos_e, g.num_nodes, Nneg).to(device)
            else:
                neg_e = sample_uniform_negatives(pos_e, g.num_nodes, Nneg).to(device)
            # Debug: we are in BCE/legacy fallback path (not per-src BPR pairs)
            if bool(getattr(train_graphsage, "debug_sampler", False)) and not bool(
                getattr(train_graphsage, "debug_done", False)
            ):
                with contextlib.suppress(Exception):
                    print("[DEBUG][SAGE] sampler: fallback=BCE_or_legacy reason=no_per_src_pairs")
                train_graphsage.debug_done = True  # type: ignore[attr-defined]

            # Build global->local map for current batch (may already exist above)
            if "inv" not in locals():
                global_ids = batch.n_id  # global node ids present in this subgraph
                inv = torch.full((g.num_nodes,), -1, dtype=torch.long, device=device)
                inv[global_ids] = torch.arange(global_ids.numel(), device=device)

            # Rebuild global->local map for this batch (always per batch)
            global_ids = batch.n_id  # global node ids present in this subgraph
            inv = torch.full((g.num_nodes,), -1, dtype=torch.long, device=device)
            inv[global_ids] = torch.arange(global_ids.numel(), device=device)

            # Map positives to local
            pos_local = inv[pos_e]
            L = int(global_ids.numel())
            mask_pos = (
                (pos_local[0] >= 0) & (pos_local[1] >= 0) & (pos_local[0] < L) & (pos_local[1] < L)
            )
            if mask_pos.sum() == 0:
                # No valid positives in this batch after mapping; skip
                continue
            pos_local = pos_local[:, mask_pos]

            # Map negatives to local; attempt limited resample if empty
            max_resample = 3
            attempt = 0
            neg_local = inv[neg_e]
            mask_neg = (
                (neg_local[0] >= 0) & (neg_local[1] >= 0) & (neg_local[0] < L) & (neg_local[1] < L)
            )
            while mask_neg.sum() == 0 and attempt < max_resample:
                # Resample negatives globally and remap
                if neg_strategy == "uniform":
                    neg_e = sample_uniform_negatives(pos_e, g.num_nodes, Nneg).to(device)
                else:
                    # fallback to uniform on resample
                    neg_e = sample_uniform_negatives(pos_e, g.num_nodes, Nneg).to(device)
                neg_local = inv[neg_e]
                mask_neg = (
                    (neg_local[0] >= 0)
                    & (neg_local[1] >= 0)
                    & (neg_local[0] < L)
                    & (neg_local[1] < L)
                )
                attempt += 1
            if mask_neg.sum() == 0:
                # Still no valid negatives; skip batch
                continue
            neg_local = neg_local[:, mask_neg]

            opt.zero_grad(set_to_none=True)
            # Build fused features if using ID embeddings
            with torch.amp.autocast("cuda", enabled=use_amp):
                if use_id_emb and id_emb is not None:
                    x_ids = id_emb(batch.n_id)
                    x_fused = torch.cat([batch.x, x_ids], dim=1)
                    z = model.encode(x_fused, batch.edge_index)
                else:
                    z = model.encode(batch.x, batch.edge_index)
                # Align training objective with evaluation when --normalize-emb true
                if normalize_flag:
                    z = F.normalize(z, p=2, dim=1)
                pos_logit = dot_scores(z[pos_local[0]], z[pos_local[1]])
                neg_logit = dot_scores(z[neg_local[0]], z[neg_local[1]])
                loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit))
                    + F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit))
                )
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            epoch_loss += float(loss)
            nb += 1
        losses.append(epoch_loss / max(1, nb))
        # Per-epoch mix% readout (approximate by planned bucket counts)
        s_mix = sum(epoch_mix.values()) or 1
        mix_pct = {k: (100.0 * epoch_mix[k] / s_mix) for k in epoch_mix}
        print(f"[TRAIN] Epoch {ep + 1}/{epochs} loss={losses[-1]:.4f} | mix% {mix_pct}")
        # ASHA/Optuna progress callback hook (reports per-epoch proxy metric)
        cb = getattr(train_graphsage, "progress_callback", None)
        if callable(cb):
            with contextlib.suppress(Exception):
                cb(int(ep + 1), float(losses[-1]))

        # Optional validation probe + early stopping
        do_probe = (
            val_interval > 0
            and ((ep + 1) % val_interval == 0)
            and cand_val_path is not None
            and splits_root_p is not None
        )
        if do_probe:
            try:
                with torch.no_grad():
                    x_full = x.to(device)
                    if use_id_emb and id_emb is not None:
                        x_full = torch.cat([x_full, id_emb.weight], dim=1)
                    z_probe = model.encode(x_full, csr_to_edge_index(g.csr).to(device)).detach()
                    if normalize_flag:
                        z_probe = F.normalize(z_probe, p=2, dim=1)
                assert cand_val_path is not None, "cand_val_path required for val probe"
                assert splits_root_p is not None, "splits_root required for val probe"
                gdf_probe, _, _, counts = evaluate_split(
                    name="val",
                    Z=z_probe,
                    device=device,
                    g=g,
                    Ks=Ks_probe,
                    cand_path=Path(cand_val_path),
                    out_dir=Path("."),
                    batch_size=int(max(100_000, batch_size)),
                    splits_root=Path(splits_root_p),
                    collect_slices=False,
                    max_sources=int(val_max_src),
                )
                cur = (
                    float(gdf_probe.iloc[0]["ndcg@100"])
                    if "ndcg@100" in gdf_probe.columns
                    else float("nan")
                )
                print(
                    f"[VAL] Epoch {ep + 1}/{epochs} probe: ndcg@100={cur:.6f} on {counts.get('sources', 0)} sources (best={best_ndcg:.6f} @ep={best_epoch})"
                )
                import numpy as _np

                improved = (not _np.isnan(cur)) and ((cur - best_ndcg) > float(min_delta))
                if improved:
                    best_ndcg = cur
                    best_epoch = int(ep + 1)
                    no_improve = 0
                    try:
                        best_state = {
                            k: v.detach().cpu().clone() if hasattr(v, "detach") else v
                            for k, v in model.state_dict().items()
                        }
                    except Exception:
                        best_state = None
                else:
                    no_improve += 1
                    if early_patience > 0 and no_improve >= early_patience:
                        stop_training = True
            except Exception as e:
                print(f"[VAL][WARN] Probe failed at epoch {ep + 1}: {e}")

        if stop_training:
            print(
                f"[EARLYSTOP] Stop at epoch {ep + 1}: best ndcg@100={best_ndcg:.6f} @epoch {best_epoch}"
            )
            if best_state is not None:
                try:
                    model.load_state_dict(best_state)  # type: ignore[arg-type]
                    print("[EARLYSTOP] Restored best model state")
                except Exception:  # nosec B110 -- best-effort model state restore, pass is intentional
                    pass
            # Trim epochs to reflect actual run
            epochs = int(ep + 1)
            break

    if use_id_emb and id_emb is not None:
        model.id_emb = id_emb
        model.id_emb_dim = int(id_emb_dim)  # type: ignore[assignment]
    return (
        model,  # type: ignore[return-value]
        {
            "epochs": int(epochs),
            "training_time_sec": time.time() - t0,
            "final_loss": float(np.mean(losses[-3:])) if losses else float("nan"),
            "early_stop": {
                "enabled": bool(stop_training),
                "best_epoch": int(best_epoch),
                "best_ndcg@100": float(best_ndcg),
            },
        },
    )


# ------------------------------ Evaluation --------------------------------- #


@dataclass
class KM:
    Ks: list[int]
    n: int = 0
    hit: dict[int, float] = None  # type: ignore
    rec: dict[int, float] = None  # type: ignore
    pre: dict[int, float] = None  # type: ignore
    mrr: float = 0.0
    map: float = 0.0
    ndcg: float = 0.0
    mic_pos: int = 0
    mic_in_top: dict[int, int] = None  # type: ignore

    def __post_init__(self):
        self.hit = dict.fromkeys(self.Ks, 0.0)
        self.rec = dict.fromkeys(self.Ks, 0.0)
        self.pre = dict.fromkeys(self.Ks, 0.0)
        self.mic_in_top = dict.fromkeys(self.Ks, 0)

    def upd(self, y: np.ndarray) -> None:
        n, pos_total = y.size, int(y.sum())
        cs = y.cumsum()
        # AP
        if pos_total > 0:
            idx = np.nonzero(y)[0]
            ap = float((cs[idx] / (idx + 1)).sum() / pos_total)
            mrr = 1.0 / (idx[0] + 1) if idx.size > 0 else 0.0
        else:
            ap = 0.0
            mrr = 0.0
        # nDCG@100
        Knd = 100
        up = min(Knd, n)
        gains = y[:up].astype(np.float64)
        if gains.any():
            disc = 1.0 / np.log2(np.arange(2, 2 + up))
            dcg = float((gains * disc).sum())
            ideal = min(pos_total, Knd)
            idcg = float(np.ones(ideal) @ disc[:ideal]) if ideal > 0 else 0.0
            nd = (dcg / idcg) if idcg > 0 else 0.0
        else:
            nd = 0.0
        self.n += 1
        self.mrr += mrr
        self.map += ap
        self.ndcg += nd
        for K in self.Ks:
            k = min(K, n)
            topk = int(cs[k - 1]) if k > 0 else 0
            self.hit[K] += 1.0 if topk > 0 else 0.0
            self.rec[K] += (topk / pos_total) if pos_total > 0 else 0.0
            self.pre[K] += (topk / K) if K > 0 else 0.0
            self.mic_pos += pos_total
            self.mic_in_top[K] += topk

    def row(self, macro: bool, name: str) -> dict[str, object]:
        ns = max(1, self.n)
        out = {"heuristic": name, "macro": macro}
        for K in self.Ks:
            out[f"hit@{K}"] = self.hit[K] / ns
            out[f"recall@{K}"] = (
                (self.rec[K] / ns) if macro else (self.mic_in_top[K] / max(1, self.mic_pos))
            )
            out[f"precision@{K}"] = (self.pre[K] / ns) if macro else (self.mic_in_top[K] / (K * ns))
        out["mrr"] = self.mrr / ns
        out["map"] = self.map / ns
        out["ndcg@100"] = self.ndcg / ns
        return out


def evaluate_split(
    name: str,
    Z: Tensor,
    device: torch.device,
    g: Graph,
    Ks: list[int],
    cand_path: Path,
    out_dir: Path,
    batch_size: int,
    splits_root: Path,
    collect_slices: bool = True,
    max_sources: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    macro = KM(Ks=Ks)
    micro = KM(Ks=Ks)
    emit_strict_warm = bool(getattr(evaluate_split, "emit_strict_warm", False))
    slice_names = [
        "WW",
        "WC",
        "CW",
        "CC",
        "twohop",
        ">2hop",
        *horizon_slice_names(),
        "deg_q1",
        "deg_q2",
        "deg_q3",
        "deg_q4",
    ]
    if emit_strict_warm:
        slice_names += ["WW3", "WC3", "CW3", "CC3"]
    slices: dict[str, KM] = {s: KM(Ks=Ks) for s in slice_names}

    try:
        q = np.quantile(g.out_deg, [0.25, 0.5, 0.75])
    except Exception:
        q = np.array([0, 0, 0], dtype=float)

    T0 = _load_t0(splits_root) or 0
    streamer = CandidateStreamer(cand_path, batch_size=batch_size)
    # ts sidecar if candidates lack ts
    ts_sidecar: dict[tuple[int, int], int] | None = None
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(cand_path)
        cand_has_ts = "ts" in set(pf.schema.names)
    except Exception:
        cand_has_ts = False
    if not cand_has_ts:
        spath = splits_root / f"{name}_edges.parquet"
        if spath.exists():
            df_edges = pd.read_parquet(spath, columns=["src_id", "dst_id", "ts"])  # type: ignore
            ts_sidecar = {(int(s), int(d)): int(t) for s, d, t in df_edges.to_numpy()}

    total_rows = 0
    total_sources = 0
    Zd = Z.to(device)
    csr, csc = g.csr, g.csc

    for chunk in streamer:
        if chunk.empty:
            continue
        chunk["src_id"] = chunk["src_id"].astype(np.int64)
        chunk["dst_id"] = chunk["dst_id"].astype(np.int64)
        chunk["label"] = chunk["label"].astype(np.int8)
        if "ts" in chunk.columns:
            chunk["ts"] = chunk["ts"].astype("float64")
        src_vals = chunk["src_id"].to_numpy()
        change = np.where(np.diff(src_vals) != 0)[0] + 1
        bounds = np.concatenate(([0], change, [len(chunk)]))

        for i in range(len(bounds) - 1):
            a, b = int(bounds[i]), int(bounds[i + 1])
            sub = chunk.iloc[a:b]
            u = int(sub["src_id"].iloc[0])
            vs = sub["dst_id"].to_numpy(dtype=np.int64, copy=False)
            # Ensure labels are properly converted to int8, handling any data type issues
            labels_raw = sub["label"].to_numpy(copy=False)
            # Handle any float32 or other data type issues
            if labels_raw.dtype.kind == "f":
                labels = labels_raw.astype(np.int8)
            else:
                labels = labels_raw.astype(np.int8)
            with torch.no_grad():
                zu = Zd[u]
                zv = Zd[vs]
                scores = (zv * zu).sum(dim=1).detach().cpu().numpy()
            order = np.lexsort((vs, -scores))
            y = labels[order].astype(np.int64)
            macro.upd(y)
            micro.upd(y)

            if collect_slices:
                # Warm/Cold
                warm_u = g.out_deg[u] > 0
                warm_v = g.in_deg[vs] > 0
                wc = np.empty(vs.size, dtype=np.int8)
                if warm_u:
                    wc[warm_v] = 0
                    wc[~warm_v] = 1
                else:
                    wc[warm_v] = 2
                    wc[~warm_v] = 3
                wc = wc[order]
                for lab, tag in zip([0, 1, 2, 3], ["WW", "WC", "CW", "CC"], strict=False):
                    m = wc == lab
                    if m.any():
                        slices[tag].upd(y[m])
                if emit_strict_warm:
                    thr = 3
                    warm_u3 = g.out_deg[u] >= thr
                    warm_v3 = g.in_deg[vs] >= thr
                    wc3 = np.empty(vs.size, dtype=np.int8)
                    if warm_u3:
                        wc3[warm_v3] = 0
                        wc3[~warm_v3] = 1
                    else:
                        wc3[warm_v3] = 2
                        wc3[~warm_v3] = 3
                    wc3 = wc3[order]
                    for lab, tag in zip([0, 1, 2, 3], ["WW3", "WC3", "CW3", "CC3"], strict=False):
                        m = wc3 == lab
                        if m.any():
                            slices[tag].upd(y[m])

                # Two-hop
                out_u = csr.indices[csr.indptr[u] : csr.indptr[u + 1]]
                in_u = csc.indices[csc.indptr[u] : csc.indptr[u + 1]]
                one = (
                    np.unique(np.concatenate([out_u, in_u]))
                    if (out_u.size or in_u.size)
                    else np.empty(0, dtype=np.int64)
                )
                if one.size:
                    parts = []
                    for w in one:
                        parts.append(csr.indices[csr.indptr[w] : csr.indptr[w + 1]])
                        parts.append(csc.indices[csc.indptr[w] : csc.indptr[w + 1]])
                    two = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
                    two_sorted = np.sort(two)
                    pos = np.searchsorted(two_sorted, vs)
                    valid = pos < two_sorted.size
                    th = np.zeros_like(vs, dtype=bool)
                    th[valid] = two_sorted[pos[valid]] == vs[valid]
                else:
                    th = np.zeros(vs.size, dtype=bool)
                th = th[order]
                if th.any():
                    slices["twohop"].upd(y[th])
                if (~th).any():
                    slices[">2hop"].upd(y[~th])

                # Horizons
                hz = np.zeros_like(labels, dtype=np.int8)
                if T0:
                    if "ts" in sub.columns:
                        ts_vals = sub["ts"].to_numpy(dtype=np.float64, copy=False)
                    elif ts_sidecar is not None:
                        ts_vals = np.full(len(sub), np.nan, dtype=np.float64)
                        pm = labels > 0
                        if np.any(pm):
                            sa = sub["src_id"].to_numpy()[pm]
                            da = sub["dst_id"].to_numpy()[pm]
                            ts_lookup = [
                                ts_sidecar.get((int(s), int(d)), np.nan)
                                for s, d in zip(sa, da, strict=False)
                            ]
                            ts_vals[pm] = np.array(ts_lookup, dtype=np.float64)
                    else:
                        ts_vals = np.full(len(sub), np.nan, dtype=np.float64)
                    pm = labels > 0
                    if np.any(pm):
                        deltas = ts_vals[pm] - float(T0)
                        hz[np.nonzero(pm)[0]] = assign_horizon_buckets(deltas)
                hz = hz[order]
                for hv, stag in enumerate(horizon_slice_names(), start=1):
                    m = hz == hv
                    if m.any():
                        yy = y.copy()
                        drop = (yy > 0) & (~m)
                        yy[drop] = 0
                        slices[stag].upd(yy)

                # Degree bins (src)
                d = g.out_deg[u]
                if d <= q[0]:
                    db = "deg_q1"
                elif d <= q[1]:
                    db = "deg_q2"
                elif d <= q[2]:
                    db = "deg_q3"
                else:
                    db = "deg_q4"
                slices[db].upd(y)

            total_rows += len(sub)
            total_sources += 1
            if max_sources is not None and total_sources >= int(max_sources):
                break
        if max_sources is not None and total_sources >= int(max_sources):
            break

    gdf = pd.DataFrame([macro.row(True, "GraphSAGE")])
    mdf = pd.DataFrame([micro.row(False, "GraphSAGE")])
    sdf = pd.DataFrame([{**km.row(True, "GraphSAGE"), "slice_name": s} for s, km in slices.items()])
    counts = {"rows": int(total_rows), "sources": int(total_sources)}
    return gdf, mdf, sdf, counts


# ---------------------------------- Main ----------------------------------- #


# Calibration helpers (Platt)
def _sigmoid_np(x: np.ndarray) -> np.ndarray:
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
    s = scores.astype(np.float64)
    y = labels.astype(np.float64)
    mu, sd = float(np.mean(s)), float(np.std(s) + 1e-12)
    s = (s - mu) / sd
    X = np.stack([s, np.ones_like(s)], axis=1)
    w = np.zeros(2, dtype=np.float64)
    for _ in range(max(1, iters)):
        z = X @ w
        p = _sigmoid_np(z)
        g = X.T @ (p - y) + l2 * w
        r = p * (1.0 - p)
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


def calibrate_on_val(
    Z: Tensor,
    device: torch.device,
    g: Graph,
    cand_val: Path,
    splits_root: Path,
    sample_per_src: int = 50,
    max_pairs: int = 1_000_000,
) -> dict[str, object]:
    streamer = CandidateStreamer(cand_val, batch_size=1_000_000)
    Z_dev = Z.to(device)
    scores = []
    labels = []
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
            # sample per source
            take = min(sample_per_src, vs.size)
            idx = np.random.choice(vs.size, size=take, replace=False)
            vs_s = vs[idx]
            lbl_s = lbl[idx]
            with torch.no_grad():
                zu = Z_dev[u]
                zv = Z_dev[vs_s]
                sc = (zv * zu).sum(dim=1).detach().cpu().numpy()
            scores.append(sc.astype(np.float64))
            labels.append(lbl_s.astype(np.float64))
            total += int(take)
            if total >= max_pairs:
                break
        if total >= max_pairs:
            break
    if not scores:
        raise RuntimeError("No calibration samples collected for GraphSAGE")
    s = np.concatenate(scores, axis=0)
    y = np.concatenate(labels, axis=0)
    A, B = fit_platt(s, y)
    return {"method": "platt", "A": A, "B": B, "samples": int(s.size)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GraphSAGE on train graph and evaluate on fixed candidate pools"
    )
    p.add_argument("--adj", required=True, type=str)
    p.add_argument("--features", required=True, type=str)
    p.add_argument(
        "--struct-feats",
        type=str,
        default="",
        help="Optional path to node_structural_v1.parquet to concatenate",
    )
    p.add_argument("--candidates-val", required=True, type=str)
    p.add_argument("--candidates-test", required=True, type=str)
    p.add_argument("--splits-root", required=True, type=str)
    p.add_argument("--out-dir", type=str, default="results/graphsage")
    p.add_argument("--artifacts-dir", type=str, default="artifacts/graphsage")
    p.add_argument("--Ks", type=str, default="1,10,50")
    # Accept boolean flags with or without explicit values (e.g., --flag or --flag true/false)
    p.add_argument("--undirected", nargs="?", const=True, type=_bool, default=False)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    # Structural embeddings + normalization
    p.add_argument(
        "--use-id-emb",
        nargs="?",
        const=True,
        type=_bool,
        default=False,
        help="Use a learnable node-ID embedding concatenated to features (transductive path)",
    )
    p.add_argument(
        "--id-emb-dim",
        type=int,
        default=64,
        help="Dimensionality of node-ID embedding when enabled",
    )
    p.add_argument(
        "--normalize-emb",
        nargs="?",
        const=True,
        type=_bool,
        default=True,
        help="L2-normalize node embeddings before ranking (recommended)",
    )
    p.add_argument(
        "--use-struct-feats",
        nargs="?",
        const=True,
        type=_bool,
        default=False,
        help="Concatenate structural scalars (logdeg/PR/HITS) to node features",
    )
    # Model/training
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--neighbors", nargs="+", type=int, default=[15, 10])
    p.add_argument(
        "--group-by-src",
        nargs="?",
        const=True,
        type=_bool,
        default=True,
        help="Group training mini-batches by source id to reuse per-u negatives (faster)",
    )
    # Loader performance
    p.add_argument("--loader-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", nargs="?", const=True, type=_bool, default=True)
    p.add_argument("--persistent-workers", nargs="?", const=True, type=_bool, default=True)
    p.add_argument(
        "--use-amp",
        nargs="?",
        const=True,
        type=_bool,
        default=True,
        help="Enable mixed precision on CUDA",
    )
    p.add_argument(
        "--fanout-replace",
        nargs="?",
        const=True,
        type=_bool,
        default=True,
        help="Sample neighbors with replacement during fanout (faster and adequate for SAGE)",
    )
    p.add_argument(
        "--async-sampler",
        nargs="?",
        const=True,
        type=_bool,
        default=False,
        help="Build BPR negatives asynchronously to overlap CPU with GPU",
    )
    # Negatives
    p.add_argument("--neg-ratio", type=float, default=2.0)
    p.add_argument(
        "--neg-strategy",
        type=str,
        default="mix_v3",
        choices=["uniform", "mix_deg50", "mix", "mix_v3"],
    )
    p.add_argument(
        "--deg-frac",
        type=float,
        default=0.5,
        help="When --neg-strategy mix: fraction for degree-biased negatives",
    )
    p.add_argument(
        "--twohop-frac",
        type=float,
        default=0.1,
        help="When --neg-strategy mix: fraction for two-hop negatives",
    )
    p.add_argument(
        "--sem-frac",
        type=float,
        default=0.0,
        help="When --neg-strategy mix: fraction for semantic (sector/geo) negatives",
    )
    p.add_argument(
        "--sem-keys",
        type=str,
        default="primary_sic_code,gr_country",
        help="Comma-separated feature keys for semantic matching (in features parquet)",
    )
    p.add_argument(
        "--deg-exp", type=float, default=0.75, help="Exponent for degree-biased sampling (mix_v3)"
    )
    # Ranking-aligned loss
    p.add_argument("--loss", type=str, default="bpr", choices=["bce", "bpr"], help="Training loss")
    p.add_argument(
        "--pairwise-negs", type=int, default=2, help="Negatives per positive when --loss bpr"
    )
    # Eval
    p.add_argument("--batch-size", type=int, default=2_000_000)
    p.add_argument(
        "--sampler-mode",
        type=str,
        default="per_src",
        choices=["per_pos", "per_src"],
        help="BPR sampler mode: per positive or per source (per_src is faster)",
    )
    p.add_argument(
        "--emit-strict-warm",
        nargs="?",
        const=True,
        type=_bool,
        default=False,
        help="Emit WW3/WC3/CW3/CC3 slices (warm=≥3)",
    )
    p.add_argument(
        "--calibrate",
        nargs="?",
        const=True,
        type=_bool,
        default=False,
        help="Fit Platt calibrator on validation and save JSON",
    )
    p.add_argument("--calibrate-sample-per-src", type=int, default=50)
    p.add_argument("--calibrate-max-pairs", type=int, default=1000000)
    # Optional periodic validation probe + early stopping (for overfit control)
    p.add_argument(
        "--val-probe-interval",
        type=int,
        default=0,
        help="If >0, run a small validation probe every N epochs and report ndcg@100",
    )
    p.add_argument(
        "--val-probe-max-sources",
        type=int,
        default=1000,
        help="Max sources to evaluate in each validation probe (subset for speed)",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="If >0 with --val-probe-interval>0, stop when ndcg@100 fails to improve for this many probes",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=5e-4,
        help="Minimum improvement in ndcg@100 to reset patience",
    )
    # Debugging/ablations
    p.add_argument(
        "--debug-sampler",
        nargs="?",
        const=True,
        type=_bool,
        default=False,
        help="Log BPR sampler invariants on first per-src batch (same-u pairing, exclusions, pairwise accuracy)",
    )
    p.add_argument(
        "--no-message-passing",
        nargs="?",
        const=True,
        type=_bool,
        default=False,
        help="Disable SAGE message passing and use a feature-only MLP baseline",
    )
    # Optional: concatenate external embeddings (e.g., Node2Vec) to input features
    p.add_argument(
        "--concat-emb",
        type=str,
        default="",
        help="Path to embeddings file (.npy/.npz/.parquet) aligned to node ids",
    )
    p.add_argument(
        "--concat-emb-normalize",
        nargs="?",
        const=True,
        type=_bool,
        default=False,
        help="L2-normalize concatenated embeddings before fusion",
    )
    # Optional: save trained model(s) and persist metrics
    p.add_argument(
        "--save-model",
        nargs="?",
        const=True,
        type=_bool,
        default=False,
        help="If true, save trained model state_dict (per-seed in multi-seed mode)",
    )
    p.add_argument(
        "--model-filename",
        type=str,
        default="model.pt",
        help="Filename for saved model (placed under artifacts or artifacts/seed_*)",
    )
    p.add_argument(
        "--sqlite-out",
        type=str,
        default="",
        help="Optional path to a SQLite DB to persist per-seed metrics (table 'runs')",
    )
    # Multi-seed controls (for dissertation rigor)
    p.add_argument(
        "--multi-seed",
        action="store_true",
        help="Run multiple seeds and aggregate results with CIs",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds (e.g., '42,123,456,789,999')",
    )
    # Best params ingestion (from Optuna)
    p.add_argument(
        "--use-best-params",
        action="store_true",
        help="Load best params from artifacts (robust first) and override CLI",
    )
    p.add_argument(
        "--best-params-path",
        type=str,
        default="",
        help=(
            "Optional explicit path. If not provided, tries artifacts/graphsage/robust_best_params.json "
            "then artifacts/graphsage/best_params.json"
        ),
    )
    # Safety guardrails (to avoid pathological neighbor fanout)
    p.add_argument(
        "--max-fanout-nodes",
        type=int,
        default=1_200_000,
        help="Warn or clamp when batch*(1+n1+n1*n2) exceeds this cap",
    )
    p.add_argument(
        "--auto-clamp",
        nargs="?",
        const=True,
        type=_bool,
        default=False,
        help="If true, auto-adjust batch/neighbors to stay under --max-fanout-nodes; otherwise error out",
    )
    p.add_argument(
        "--train-timeout-sec",
        type=int,
        default=0,
        help="Optional timeout for training loop (triage)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Ks = [int(x.strip()) for x in str(args.Ks).split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    art_dir = Path(args.artifacts_dir)
    _ensure_dir(out_dir)
    _ensure_dir(art_dir)
    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    g = load_graph(Path(args.adj), undirected=bool(args.undirected))
    struct_path = Path(args.struct_feats) if args.struct_feats else None
    if bool(args.use_struct_feats) and struct_path is not None and struct_path.exists():
        x, feat_df = load_features(Path(args.features), g.num_nodes, struct_feats_path=struct_path)
        print("[INFO] Using structural scalars concatenated to node features")
    else:
        x, feat_df = load_features(Path(args.features), g.num_nodes)
    # Optional concatenate external embeddings (e.g., Node2Vec)
    if str(getattr(args, "concat_emb", "")).strip():
        emb_path = Path(str(args.concat_emb))
        if not emb_path.exists():
            print(f"[WARN] --concat-emb provided but file not found: {emb_path}")
        else:
            try:
                ext = emb_path.suffix.lower()
                emb: np.ndarray
                if ext == ".npy":
                    emb = np.load(emb_path)
                elif ext == ".npz":
                    npz = np.load(emb_path)
                    # take first array
                    key = next(iter(npz.keys()))
                    emb = npz[key]
                elif ext == ".parquet":
                    df_emb = pd.read_parquet(emb_path)
                    if "node_id" in df_emb.columns:
                        df_emb = df_emb.sort_values("node_id").reset_index(drop=True)
                        df_emb = df_emb.drop(columns=["node_id"])
                    emb = df_emb.to_numpy(dtype=np.float32)
                elif ext == ".pt":
                    import torch as _torch

                    obj = _torch.load(emb_path, map_location="cpu")  # nosec B614 -- local pipeline artifacts
                    if isinstance(obj, dict) and "embeddings" in obj:
                        arr = obj["embeddings"]
                        if hasattr(arr, "detach"):
                            emb = arr.detach().cpu().numpy()
                        else:
                            emb = np.asarray(arr)
                    else:
                        raise RuntimeError("PT file missing embeddings key")
                else:
                    raise RuntimeError(f"Unsupported embeddings format: {ext}")
                if emb.shape[0] != g.num_nodes:
                    raise RuntimeError(
                        f"Concat embeddings rows {emb.shape[0]} != num_nodes {g.num_nodes}"
                    )
                if bool(getattr(args, "concat_emb_normalize", False)):
                    # L2 normalize rows
                    denom = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
                    emb = (emb / denom).astype(np.float32)
                x = torch.from_numpy(np.concatenate([x.numpy(), emb.astype(np.float32)], axis=1))
                print(f"[INFO] Concatenated external embeddings: {emb_path}, dim={emb.shape[1]}")
            except Exception as e:
                print(f"[WARN] Failed to concatenate embeddings from {emb_path}: {e}")

    # Optionally override hyperparams from artifacts best params
    if args.use_best_params:
        # Resolve candidate paths in priority order
        candidate_path: Path | None = None
        if args.best_params_path:
            cand = Path(args.best_params_path)
            if cand.exists():
                candidate_path = cand
            else:
                print(f"[WARN] Explicit best params path not found: {cand}")
        if candidate_path is None:
            rbp = art_dir / "robust_best_params.json"
            if rbp.exists():
                candidate_path = rbp
        if candidate_path is None:
            legacy = art_dir / "best_params.json"
            if legacy.exists():
                candidate_path = legacy
        if candidate_path is None:
            print("[WARN] No best params file found (robust or legacy). Proceeding with CLI args.")
        else:
            try:
                bp = json.loads(candidate_path.read_text()).get("best_params", {})
                # Map to CLI args
                if "hidden" in bp:
                    args.hidden = int(bp["hidden"])
                if "layers" in bp:
                    args.layers = int(bp["layers"])
                if "dropout" in bp:
                    args.dropout = float(bp["dropout"])
                if "lr" in bp:
                    args.lr = float(bp["lr"])
                if "batch" in bp:
                    args.batch = int(bp["batch"])
                if "neighbors" in bp:
                    nb = bp["neighbors"]
                    if isinstance(nb, (list, tuple)) and len(nb) >= 2:
                        args.neighbors = [int(nb[0]), int(nb[1])]
                if "neg_ratio" in bp:
                    args.neg_ratio = float(bp["neg_ratio"])
                if "neg_strategy" in bp:
                    args.neg_strategy = str(bp["neg_strategy"])
                if "use_id_emb" in bp:
                    args.use_id_emb = bool(bp["use_id_emb"])  # type: ignore[attr-defined]
                if "id_emb_dim" in bp:
                    args.id_emb_dim = int(bp["id_emb_dim"])  # type: ignore[attr-defined]
                if "normalize_emb" in bp:
                    args.normalize_emb = bool(bp["normalize_emb"])  # type: ignore[attr-defined]
                # Apply mix-specific fractions for both legacy 'mix' and current 'mix_v3'
                if args.neg_strategy in {"mix", "mix_v3"}:
                    if hasattr(args, "deg_frac") and "deg_frac" in bp:
                        args.deg_frac = float(bp["deg_frac"])  # type: ignore[attr-defined]
                    if hasattr(args, "twohop_frac") and "twohop_frac" in bp:
                        args.twohop_frac = float(bp["twohop_frac"])  # type: ignore[attr-defined]
                    if hasattr(args, "sem_frac") and "sem_frac" in bp:
                        args.sem_frac = float(bp["sem_frac"])  # type: ignore[attr-defined]

                # Set mix_v3 defaults if neg_strategy is mix_v3 and fractions weren't loaded from best params
                if args.neg_strategy == "mix_v3":
                    if not hasattr(args, "deg_frac") or args.deg_frac == 0.5:  # default value
                        args.deg_frac = 0.4
                    if not hasattr(args, "twohop_frac") or args.twohop_frac == 0.1:  # default value
                        args.twohop_frac = 0.2
                    if not hasattr(args, "sem_frac") or args.sem_frac == 0.0:  # default value
                        args.sem_frac = 0.2
                    print(
                        f"[INFO] Set mix_v3 defaults: deg_frac={args.deg_frac}, twohop_frac={args.twohop_frac}, sem_frac={args.sem_frac}"
                    )
                print(f"[INFO] Loaded best params from {candidate_path}")
            except Exception as e:
                print(f"[WARN] Failed to load best params from {candidate_path}: {e}")

    # Evaluate (single-seed or multi-seed)
    splits_root = Path(args.splits_root)
    # Parse neighbors early and run preflight safety check
    n1, n2 = ([*args.neighbors, args.neighbors[-1]])[:2]
    fanout = 1 + int(n1) + (int(n1) * int(n2))
    eff_nodes = int(args.batch) * fanout
    print(
        f"[SAFETY] Preflight: batch={int(args.batch)}, neighbors=({int(n1)},{int(n2)}), fanout={fanout}, eff_nodes≈{eff_nodes}"
    )
    if eff_nodes > int(args.max_fanout_nodes):
        if args.auto_clamp:
            # Prefer clamping batch first
            new_batch = max(1024, int(int(args.max_fanout_nodes) // max(1, fanout)))
            if new_batch != int(args.batch):
                print(f"[SAFETY] Clamping batch {int(args.batch)} -> {new_batch}")
                args.batch = int(new_batch)
            eff_nodes = int(args.batch) * fanout
            # If still heavy, clamp neighbors progressively
            if eff_nodes > int(args.max_fanout_nodes):
                adj_n1, adj_n2 = int(n1), int(n2)
                if adj_n2 > 10:
                    print(f"[SAFETY] Clamping neighbors n2 {adj_n2} -> 10")
                    adj_n2 = 10
                if adj_n1 > 15:
                    print(f"[SAFETY] Clamping neighbors n1 {adj_n1} -> 15")
                    adj_n1 = 15
                n1, n2 = adj_n1, adj_n2
                fanout = 1 + int(n1) + (int(n1) * int(n2))
                eff_nodes = int(args.batch) * fanout
                print(f"[SAFETY] New neighbors=({int(n1)},{int(n2)}), eff_nodes≈{eff_nodes}")
            # Final hard fallback if still heavy
            if eff_nodes > int(args.max_fanout_nodes):
                print("[SAFETY] Forcing conservative neighbors (10,10)")
                n1, n2 = 10, 10
                fanout = 1 + int(n1) + (int(n1) * int(n2))
                eff_nodes = int(args.batch) * fanout
                print(f"[SAFETY] Final neighbors=({int(n1)},{int(n2)}), eff_nodes≈{eff_nodes}")
        else:
            raise RuntimeError(
                f"Configured batch/neighbors are too heavy (eff_nodes≈{eff_nodes} > max_fanout_nodes={int(args.max_fanout_nodes)}). "
                f"Reduce --batch or --neighbors, or pass --auto-clamp true to auto-adjust."
            )

    summary = {
        "timestamp": _now_iso(),
        "device": str(device),
        "adjacency": str(Path(args.adj)),
        "features": str(Path(args.features)),
        "candidates": {
            "val": str(Path(args.candidates_val)),
            "test": str(Path(args.candidates_test)),
        },
        "Ks": Ks,
        "undirected": bool(args.undirected),
        "hparams": {
            "hidden": int(args.hidden),
            "layers": int(args.layers),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "batch": int(args.batch),
            "neighbors": [int(n1), int(n2)],
            "neg_ratio": float(args.neg_ratio),
            "neg_strategy": str(args.neg_strategy),
            "use_id_emb": bool(args.use_id_emb),
            "id_emb_dim": int(args.id_emb_dim),
            "normalize_emb": bool(args.normalize_emb),
        },
        "training": {},
        "counts": {},
    }
    # Configure evaluation options
    evaluate_split.emit_strict_warm = bool(args.emit_strict_warm)  # type: ignore[attr-defined]

    # Multi-seed helpers
    def set_seed_all(seed: int) -> None:
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def extract_metrics(df: pd.DataFrame) -> dict[str, float]:
        row = df.iloc[0].to_dict()
        keys = [
            "hit@1",
            "hit@10",
            "hit@50",
            "recall@1",
            "recall@10",
            "recall@50",
            "precision@1",
            "precision@10",
            "precision@50",
            "mrr",
            "map",
            "ndcg@100",
        ]
        return {k: float(row[k]) for k in keys if k in row}

    def ci(values: list[float], confidence: float = 0.95) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
        if len(values) == 1:
            return {
                "mean": values[0],
                "std": 0.0,
                "ci_lower": values[0],
                "ci_upper": values[0],
                "n": 1,
            }
        import statistics

        import scipy.stats as stats

        mean = statistics.mean(values)
        std = statistics.stdev(values)
        tval = stats.t.ppf((1 + confidence) / 2, len(values) - 1)
        moe = tval * std / (len(values) ** 0.5)
        return {
            "mean": mean,
            "std": std,
            "ci_lower": mean - moe,
            "ci_upper": mean + moe,
            "n": len(values),
        }

    seeds: list[int] = []
    if args.multi_seed or args.seeds:
        if args.seeds:
            seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
        else:
            seeds = [42, 123, 456, 789, 999]
        print(f"[MULTI-SEED] Running GraphSAGE for seeds: {seeds}")
        val_results: list[dict[str, float]] = []
        test_results: list[dict[str, float]] = []
        evaluate_split.emit_strict_warm = bool(args.emit_strict_warm)  # type: ignore[attr-defined]
        # Optional SQLite setup
        sqlite_path: Path | None = (
            Path(str(args.sqlite_out)) if str(getattr(args, "sqlite_out", "")).strip() else None
        )
        if sqlite_path is not None:
            try:
                import sqlite3  # type: ignore

                sqlite_conn = sqlite3.connect(sqlite_path)
                sqlite_cur = sqlite_conn.cursor()
                sqlite_cur.execute(
                    "CREATE TABLE IF NOT EXISTS runs (tag TEXT, seed INT, split TEXT, ndcg REAL, mrr REAL, hit10 REAL, created_at TEXT)"
                )
            except Exception as e:
                print(f"[WARN] Failed to open SQLite DB {sqlite_path}: {e}")
                sqlite_path = None

        from datetime import datetime

        tag_name = Path(args.out_dir).name

        for i, seed in enumerate(seeds):
            print(f"[SEED {i + 1}/{len(seeds)}] seed={seed}")
            set_seed_all(seed)
            # Prepare per-seed folders
            seed_out_dir = out_dir / f"seed_{seed}"
            seed_art_dir = art_dir / f"seed_{seed}"
            _ensure_dir(seed_out_dir)
            _ensure_dir(seed_art_dir)
            # Attach training & mix parameters per run
            n1, n2 = ([*args.neighbors, args.neighbors[-1]])[:2]
            train_graphsage.deg_frac = float(args.deg_frac)  # type: ignore[attr-defined]
            train_graphsage.twohop_frac = float(args.twohop_frac)  # type: ignore[attr-defined]
            train_graphsage.sem_frac = float(args.sem_frac)  # type: ignore[attr-defined]
            train_graphsage.deg_exp = float(getattr(args, "deg_exp", 1.0))  # type: ignore[attr-defined]
            train_graphsage.sem_keys = [
                s.strip() for s in str(args.sem_keys).split(",") if s.strip()
            ]  # type: ignore[attr-defined]
            train_graphsage.use_id_emb = bool(args.use_id_emb)  # type: ignore[attr-defined]
            train_graphsage.id_emb_dim = int(args.id_emb_dim)  # type: ignore[attr-defined]
            train_graphsage.loss = str(args.loss)  # type: ignore[attr-defined]
            train_graphsage.pairwise_negs = int(args.pairwise_negs)  # type: ignore[attr-defined]
            train_graphsage.sampler_mode = str(args.sampler_mode)  # type: ignore[attr-defined]
            train_graphsage.group_by_src = bool(args.group_by_src)  # type: ignore[attr-defined]
            train_graphsage.loader_workers = int(args.loader_workers)  # type: ignore[attr-defined]
            train_graphsage.prefetch_factor = int(args.prefetch_factor)  # type: ignore[attr-defined]
            train_graphsage.pin_memory = bool(args.pin_memory)  # type: ignore[attr-defined]
            train_graphsage.persistent_workers = bool(args.persistent_workers)  # type: ignore[attr-defined]
            train_graphsage.use_amp = bool(args.use_amp)  # type: ignore[attr-defined]
            train_graphsage.no_mp = bool(getattr(args, "no_message_passing", False))  # type: ignore[attr-defined]
            train_graphsage.debug_sampler = bool(getattr(args, "debug_sampler", False))  # type: ignore[attr-defined]
            # Attach optional val probe/early stop settings
            train_graphsage.val_probe_interval = int(getattr(args, "val_probe_interval", 0))  # type: ignore[attr-defined]
            train_graphsage.val_probe_max_sources = int(
                getattr(args, "val_probe_max_sources", 1000)
            )  # type: ignore[attr-defined]
            train_graphsage.early_stop_patience = int(getattr(args, "early_stop_patience", 0))  # type: ignore[attr-defined]
            train_graphsage.early_stop_min_delta = float(
                getattr(args, "early_stop_min_delta", 5e-4)
            )  # type: ignore[attr-defined]
            train_graphsage.cand_val_path = Path(args.candidates_val)  # type: ignore[attr-defined]
            train_graphsage.splits_root = Path(args.splits_root)  # type: ignore[attr-defined]
            train_graphsage.Ks = Ks  # type: ignore[attr-defined]
            train_graphsage.normalize_emb = bool(getattr(args, "normalize_emb", True))  # type: ignore[attr-defined]
            # Build semantic indices for both 'mix' and 'mix_v3' strategies when sem_frac > 0
            if args.neg_strategy in {"mix", "mix_v3"} and train_graphsage.sem_frac > 0:
                train_graphsage.feat_df = feat_df  # type: ignore[attr-defined]
                train_graphsage.sem_indices = build_semantic_indices(
                    feat_df, train_graphsage.sem_keys
                )  # type: ignore[attr-defined]
                with contextlib.suppress(Exception):
                    print(
                        f"[NEG] Built semantic indices for keys={train_graphsage.sem_keys} (mix_v3)"
                    )  # type: ignore[attr-defined]
            # Attach triage timeout if requested
            train_graphsage.timeout_sec = int(getattr(args, "train_timeout_sec", 0))  # type: ignore[attr-defined]
            model, _tr_stats = train_graphsage(
                g=g,
                x=x,
                hidden=int(args.hidden),
                layers=int(args.layers),
                dropout=float(args.dropout),
                lr=float(args.lr),
                epochs=int(args.epochs),
                batch_size=int(args.batch),
                neigh1=int(n1),
                neigh2=int(n2),
                neg_ratio=float(args.neg_ratio),
                neg_strategy=str(args.neg_strategy),
                device=device,
                seed=seed,
            )
            with torch.no_grad():
                x_full = x.to(device)
                if bool(args.use_id_emb) and hasattr(model, "id_emb"):
                    x_full = torch.cat([x_full, model.id_emb.weight], dim=1)  # type: ignore[attr-defined]
                z = model.encode(x_full, csr_to_edge_index(g.csr).to(device)).detach()
                if bool(args.normalize_emb):
                    z = F.normalize(z, p=2, dim=1)
            # Optionally save model state per seed
            if bool(getattr(args, "save_model", False)):
                try:
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "hparams": {
                                "hidden": int(args.hidden),
                                "layers": int(args.layers),
                                "dropout": float(args.dropout),
                                "lr": float(args.lr),
                                "batch": int(args.batch),
                                "neighbors": ([*args.neighbors, args.neighbors[-1]])[:2],
                                "neg_ratio": float(args.neg_ratio),
                                "neg_strategy": str(args.neg_strategy),
                                "use_id_emb": bool(args.use_id_emb),
                                "id_emb_dim": int(args.id_emb_dim),
                                "normalize_emb": bool(args.normalize_emb),
                                "loss": str(args.loss),
                                "pairwise_negs": int(args.pairwise_negs),
                            },
                            "seed": int(seed),
                        },
                        seed_art_dir / str(args.model_filename),
                    )
                    print(f"[INFO] Saved model to {seed_art_dir / str(args.model_filename)}")
                except Exception as e:
                    print(f"[WARN] Failed to save model for seed {seed}: {e}")
            # Optional calibration on validation
            if bool(args.calibrate):
                try:
                    cal = calibrate_on_val(
                        Z=z,
                        device=device,
                        g=g,
                        cand_val=Path(args.candidates_val),
                        splits_root=Path(args.splits_root),
                        sample_per_src=int(getattr(args, "calibrate_sample_per_src", 50)),
                        max_pairs=int(getattr(args, "calibrate_max_pairs", 1000000)),
                    )
                    (seed_art_dir / "calibration.json").write_text(
                        json.dumps(
                            {
                                "created_at": _now_iso(),
                                "model_tag": "graphsage",
                                "features_version": "node_structural_v1"
                                if bool(args.use_struct_feats)
                                else "node_features_T0",
                                "T0": int(_load_t0(Path(args.splits_root)) or 0),
                                "candidate_pool": str(Path(args.candidates_val)),
                                "method": cal.get("method"),
                                "params": {k: v for k, v in cal.items() if k not in {"method"}},
                            },
                            indent=2,
                        )
                    )
                except Exception as e:
                    print(f"[CAL][WARN] Calibration failed: {e}")
            # Evaluate val & test (full, with slices) and write per-seed outputs
            for split_name, cpath in {
                "val": Path(args.candidates_val),
                "test": Path(args.candidates_test),
            }.items():
                print(f"[INFO][SEED {seed}] Evaluating {split_name} from {cpath}")
                gdf, mdf, sdf, counts = evaluate_split(
                    name=split_name,
                    Z=z,
                    device=device,
                    g=g,
                    Ks=Ks,
                    cand_path=cpath,
                    out_dir=seed_out_dir,
                    batch_size=int(args.batch_size),
                    splits_root=splits_root,
                )
                # Persist per-seed CSVs
                gdf.to_csv(seed_out_dir / f"global_{split_name}.csv", index=False)
                mdf.to_csv(seed_out_dir / f"micro_{split_name}.csv", index=False)
                sdf.to_csv(seed_out_dir / f"slices_{split_name}.csv", index=False)
                # Collect for aggregate and optional SQLite
                metrics = extract_metrics(gdf)
                if split_name == "val":
                    val_results.append(metrics)
                else:
                    test_results.append(metrics)
                if sqlite_path is not None:
                    try:
                        now = datetime.utcnow().isoformat()
                        import sqlite3  # type: ignore

                        sqlite_cur.execute(
                            "INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
                            (
                                tag_name,
                                int(seed),
                                split_name,
                                float(metrics.get("ndcg@100", 0.0)),
                                float(metrics.get("mrr", 0.0)),
                                float(metrics.get("hit@10", 0.0)),
                                now,
                            ),
                        )
                    except Exception as e:
                        print(
                            f"[WARN] SQLite insert failed for seed {seed} split {split_name}: {e}"
                        )
        # Aggregate across seeds
        metrics = list(val_results[0].keys()) if val_results else []
        val_agg = {m: ci([r[m] for r in val_results]) for m in metrics}
        test_agg = {m: ci([r[m] for r in test_results]) for m in metrics}
        multi = {"seeds": seeds, "val": val_agg, "test": test_agg, "hparams": summary["hparams"]}
        (art_dir / "multi_seed_summary.json").write_text(json.dumps(multi, indent=2))
        print(f"[INFO] Multi-seed summary written to {art_dir / 'multi_seed_summary.json'}")
        if sqlite_path is not None:
            try:
                sqlite_conn.commit()
                sqlite_conn.close()
                print(f"[INFO] Persisted run metrics to SQLite at {sqlite_path}")
            except Exception:  # nosec B110 -- best-effort SQLite persistence, pass is intentional
                pass
        # Also write standard single-run outputs for the last seed
        # (optional: already written in loop if needed)
        (art_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return

    # Single-seed path
    set_seed_all(42)
    n1, n2 = ([*args.neighbors, args.neighbors[-1]])[:2]
    train_graphsage.deg_frac = float(args.deg_frac)  # type: ignore[attr-defined]
    train_graphsage.twohop_frac = float(args.twohop_frac)  # type: ignore[attr-defined]
    train_graphsage.sem_frac = float(args.sem_frac)  # type: ignore[attr-defined]
    train_graphsage.deg_exp = float(getattr(args, "deg_exp", 1.0))  # type: ignore[attr-defined]
    train_graphsage.sem_keys = [s.strip() for s in str(args.sem_keys).split(",") if s.strip()]  # type: ignore[attr-defined]
    train_graphsage.loss = str(args.loss)  # type: ignore[attr-defined]
    train_graphsage.pairwise_negs = int(args.pairwise_negs)  # type: ignore[attr-defined]
    train_graphsage.sampler_mode = str(args.sampler_mode)  # type: ignore[attr-defined]
    train_graphsage.group_by_src = bool(getattr(args, "group_by_src", True))  # type: ignore[attr-defined]
    train_graphsage.fanout_replace = bool(getattr(args, "fanout_replace", True))  # type: ignore[attr-defined]
    train_graphsage.group_by_src = bool(args.group_by_src)  # type: ignore[attr-defined]
    train_graphsage.loader_workers = int(args.loader_workers)  # type: ignore[attr-defined]
    train_graphsage.prefetch_factor = int(args.prefetch_factor)  # type: ignore[attr-defined]
    train_graphsage.pin_memory = bool(args.pin_memory)  # type: ignore[attr-defined]
    train_graphsage.persistent_workers = bool(args.persistent_workers)  # type: ignore[attr-defined]
    train_graphsage.use_amp = bool(args.use_amp)  # type: ignore[attr-defined]
    train_graphsage.async_sampler = bool(getattr(args, "async_sampler", False))  # type: ignore[attr-defined]
    train_graphsage.async_sampler = bool(args.async_sampler)  # type: ignore[attr-defined]
    # Attach optional val probe/early stop settings
    train_graphsage.val_probe_interval = int(getattr(args, "val_probe_interval", 0))  # type: ignore[attr-defined]
    train_graphsage.val_probe_max_sources = int(getattr(args, "val_probe_max_sources", 1000))  # type: ignore[attr-defined]
    train_graphsage.early_stop_patience = int(getattr(args, "early_stop_patience", 0))  # type: ignore[attr-defined]
    train_graphsage.early_stop_min_delta = float(getattr(args, "early_stop_min_delta", 5e-4))  # type: ignore[attr-defined]
    train_graphsage.cand_val_path = Path(args.candidates_val)  # type: ignore[attr-defined]
    train_graphsage.splits_root = Path(args.splits_root)  # type: ignore[attr-defined]
    train_graphsage.Ks = Ks  # type: ignore[attr-defined]
    train_graphsage.normalize_emb = bool(getattr(args, "normalize_emb", True))  # type: ignore[attr-defined]
    # Build semantic indices for both 'mix' and 'mix_v3' strategies when sem_frac > 0
    if args.neg_strategy in {"mix", "mix_v3"} and train_graphsage.sem_frac > 0:
        train_graphsage.feat_df = feat_df  # type: ignore[attr-defined]
        train_graphsage.sem_indices = build_semantic_indices(feat_df, train_graphsage.sem_keys)  # type: ignore[attr-defined]
        with contextlib.suppress(Exception):
            print(f"[NEG] Built semantic indices for keys={train_graphsage.sem_keys} (mix_v3)")  # type: ignore[attr-defined]
    try:
        model, _tr_stats = train_graphsage(
            g=g,
            x=x,
            hidden=int(args.hidden),
            layers=int(args.layers),
            dropout=float(args.dropout),
            lr=float(args.lr),
            epochs=int(args.epochs),
            batch_size=int(args.batch),
            neigh1=int(n1),
            neigh2=int(n2),
            neg_ratio=float(args.neg_ratio),
            neg_strategy=str(args.neg_strategy),
            device=device,
        )
    except RuntimeError as e:
        emsg = str(e).lower()
        if ("out of memory" in emsg or "cuda" in emsg) and not args.auto_clamp:
            raise RuntimeError(
                "CUDA OOM encountered during training. Consider reducing --batch or --neighbors, "
                "or re-run with --auto-clamp true to auto-adjust to safe settings."
            ) from e
        else:
            raise
    with torch.no_grad():
        x_full = x.to(device)
        if bool(args.use_id_emb) and hasattr(model, "id_emb"):
            x_full = torch.cat([x_full, model.id_emb.weight], dim=1)  # type: ignore[attr-defined]
        z = model.encode(x_full, csr_to_edge_index(g.csr).to(device)).detach()
        if bool(args.normalize_emb):
            z = F.normalize(z, p=2, dim=1)
    # Optionally save model
    if bool(getattr(args, "save_model", False)):
        try:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hparams": {
                        "hidden": int(args.hidden),
                        "layers": int(args.layers),
                        "dropout": float(args.dropout),
                        "lr": float(args.lr),
                        "batch": int(args.batch),
                        "neighbors": ([*args.neighbors, args.neighbors[-1]])[:2],
                        "neg_ratio": float(args.neg_ratio),
                        "neg_strategy": str(args.neg_strategy),
                        "use_id_emb": bool(args.use_id_emb),
                        "id_emb_dim": int(args.id_emb_dim),
                        "normalize_emb": bool(args.normalize_emb),
                        "loss": str(args.loss),
                        "pairwise_negs": int(args.pairwise_negs),
                    },
                },
                art_dir / str(args.model_filename),
            )
            print(f"[INFO] Saved model to {art_dir / str(args.model_filename)}")
        except Exception as e:
            print(f"[WARN] Failed to save model: {e}")
    # Optional calibration on validation
    if bool(args.calibrate):
        try:
            cal = calibrate_on_val(
                Z=z,
                device=device,
                g=g,
                cand_val=Path(args.candidates_val),
                splits_root=Path(args.splits_root),
                sample_per_src=int(getattr(args, "calibrate_sample_per_src", 50)),
                max_pairs=int(getattr(args, "calibrate_max_pairs", 1000000)),
            )
            (art_dir / "calibration.json").write_text(
                json.dumps(
                    {
                        "created_at": _now_iso(),
                        "model_tag": "graphsage",
                        "features_version": "node_structural_v1"
                        if bool(args.use_struct_feats)
                        else "node_features_T0",
                        "T0": int(_load_t0(Path(args.splits_root)) or 0),
                        "candidate_pool": str(Path(args.candidates_val)),
                        "method": cal.get("method"),
                        "params": {k: v for k, v in cal.items() if k not in {"method"}},
                    },
                    indent=2,
                )
            )
        except Exception as e:
            print(f"[CAL][WARN] Calibration failed: {e}")
    for split_name, cpath in {
        "val": Path(args.candidates_val),
        "test": Path(args.candidates_test),
    }.items():
        print(f"[INFO] Evaluating {split_name} from {cpath}")
        t0 = time.time()
        gdf, mdf, sdf, counts = evaluate_split(
            name=split_name,
            Z=z,
            device=device,
            g=g,
            Ks=Ks,
            cand_path=cpath,
            out_dir=out_dir,
            batch_size=int(args.batch_size),
            splits_root=splits_root,
        )
        gdf.to_csv(out_dir / f"global_{split_name}.csv", index=False)
        mdf.to_csv(out_dir / f"micro_{split_name}.csv", index=False)
        sdf.to_csv(out_dir / f"slices_{split_name}.csv", index=False)
        summary["counts"][split_name] = counts
        print(
            f"[INFO] {split_name} done in {time.time() - t0:.1f}s: {counts['sources']} sources, {counts['rows']:,} rows"
        )
    (art_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote summary to {art_dir / 'summary.json'}")


if __name__ == "__main__":
    # Lazy import to satisfy type checker for Data
    import torch_geometric.data

    main()
