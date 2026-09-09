#!/usr/bin/env python3
"""
Two-Tower (Inductive) Trainer + Scorecard Evaluator

Trains a two-tower model on train-only edges (T0) with BPR loss and a
mix_v3 negative sampler, then evaluates on fixed candidate pools with the
shared scorecard. Supports strict warm/cold slices and optional calibration.

Inputs:
  - Adjacency (CSR/CSC) for degrees and two-hop logic
  - node_features_T0.parquet (categoricals) + node_structural_v1.parquet (scalars)
  - val/test candidate pools

Outputs:
  - results/twotower/<tag>/{global,micro,slices}_{val,test}.csv
  - artifacts/twotower/<tag>/{training_summary.json, calibration.json}

Notes:
  - Inductive: no node-ID embeddings. Uses only attributes + structural scalars.
  - Ranking: dot-product of L2-normalized tower outputs by default.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch import Tensor

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


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _zscore_cols(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def calculate_confidence_intervals(
    values: list[float], confidence: float = 0.95
) -> dict[str, float]:
    if len(values) < 2:
        return {
            "mean": values[0] if values else 0.0,
            "std": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n": len(values),
        }
    import statistics

    mean = statistics.mean(values)
    std = statistics.stdev(values)
    import scipy.stats as stats  # type: ignore

    t_value = stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    margin = t_value * std / (len(values) ** 0.5)
    return {
        "mean": mean,
        "std": std,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
        "n": len(values),
    }


def aggregate_multi_seed_results(
    seed_results: list[dict[str, object]], metrics: list[str]
) -> dict[str, dict[str, float]]:
    agg: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [float(r.get(metric, 0.0)) for r in seed_results]  # type: ignore[arg-type]
        if values:
            agg[metric] = calculate_confidence_intervals(values)
    return agg


# ------------------------------ Graph ------------------------------------- #


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
        csc_temp = csr.tocsc(copy=False)
        csr = (csr + csc_temp).astype(bool).astype(np.uint8).tocsr(copy=False)
    # Ensure contiguous int32 indices/indptr for CSR/CSC to avoid implicit casts
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


# ------------------------------ Candidates -------------------------------- #


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
        # Determine ts availability
        try:
            pf = self.pq.ParquetFile(self.path)
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


# ------------------------------ Metrics ----------------------------------- #


@dataclass
class KMetrics:
    Ks: list[int]
    n_sources: int = 0
    sum_hit: dict[int, float] = field(default_factory=dict)
    sum_rec: dict[int, float] = field(default_factory=dict)
    sum_prec: dict[int, float] = field(default_factory=dict)
    sum_mrr: float = 0.0
    sum_map: float = 0.0
    sum_ndcg100: float = 0.0
    micro_pos_total: int = 0
    micro_pos_in_top: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        for k in self.Ks:
            self.sum_hit.setdefault(k, 0.0)
            self.sum_rec.setdefault(k, 0.0)
            self.sum_prec.setdefault(k, 0.0)
            self.micro_pos_in_top.setdefault(k, 0)

    def update(self, labels_sorted: np.ndarray):
        n = labels_sorted.size
        pos_total = int(labels_sorted.sum())
        cumsum = labels_sorted.cumsum()
        # AP
        if pos_total > 0:
            pos_idx = np.nonzero(labels_sorted)[0]
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
            idcg = float((np.ones(ideal) * discounts[:ideal]).sum()) if ideal > 0 else 0.0
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
        else:
            ndcg = 0.0
        # aggregate
        self.n_sources += 1
        self.sum_mrr += mrr
        self.sum_map += ap
        self.sum_ndcg100 += ndcg
        for K in self.Ks:
            k = min(K, n)
            top_k_pos = int(cumsum[k - 1]) if k > 0 else 0
            self.sum_hit[K] += 1.0 if top_k_pos > 0 else 0.0
            self.sum_rec[K] += (top_k_pos / pos_total) if pos_total > 0 else 0.0
            self.sum_prec[K] += (top_k_pos / K) if K > 0 else 0.0
            self.micro_pos_total += pos_total
            self.micro_pos_in_top[K] += top_k_pos

    def to_row(self, macro: bool, heuristic: str = "TwoTower") -> dict[str, object]:
        row: dict[str, object] = {"heuristic": heuristic, "macro": macro}
        ns = max(1, self.n_sources)
        for K in self.Ks:
            row[f"hit@{K}"] = self.sum_hit[K] / ns
            if macro:
                row[f"recall@{K}"] = self.sum_rec[K] / ns
                row[f"precision@{K}"] = self.sum_prec[K] / ns
            else:
                row[f"recall@{K}"] = self.micro_pos_in_top[K] / max(1, self.micro_pos_total)
                row[f"precision@{K}"] = self.micro_pos_in_top[K] / (K * ns)
        row["mrr"] = self.sum_mrr / ns
        row["map"] = self.sum_map / ns
        row["ndcg@100"] = self.sum_ndcg100 / ns
        return row


def classify_warm_cold(u: int, vs: np.ndarray, g: Graph) -> np.ndarray:
    warm_u = g.out_deg[u] > 0
    warm_v = g.in_deg[vs] > 0
    out = np.empty(vs.size, dtype=np.int8)
    if warm_u:
        out[warm_v] = 0
        out[~warm_v] = 1
    else:
        out[warm_v] = 2
        out[~warm_v] = 3
    return out


def classify_warm_cold_strict(u: int, vs: np.ndarray, g: Graph, threshold: int = 3) -> np.ndarray:
    thr = int(max(1, threshold))
    warm_u = g.out_deg[u] >= thr
    warm_v = g.in_deg[vs] >= thr
    out = np.empty(vs.size, dtype=np.int8)
    if warm_u:
        out[warm_v] = 0
        out[~warm_v] = 1
    else:
        out[warm_v] = 2
        out[~warm_v] = 3
    return out


def classify_twohop(u: int, vs: np.ndarray, g: Graph) -> np.ndarray:
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
    parts = []
    indptr_out, idx_out = csr.indptr, csr.indices
    indptr_in, idx_in = csc.indptr, csc.indices
    for w in one:
        parts.append(idx_out[indptr_out[w] : indptr_out[w + 1]])
        parts.append(idx_in[indptr_in[w] : indptr_in[w + 1]])
    two = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
    if two.size == 0:
        return np.zeros(vs.size, dtype=bool)
    two_sorted = np.sort(two)
    pos = np.searchsorted(two_sorted, vs)
    valid_mask = pos < two_sorted.size
    m = np.zeros_like(vs, dtype=bool)
    m[valid_mask] = two_sorted[pos[valid_mask]] == vs[valid_mask]
    return m


# ------------------------------ Features ---------------------------------- #


@dataclass
class Features:
    struct: np.ndarray  # [N, Ds]
    attr_codes: dict[str, np.ndarray]  # name -> [N] int64 codes in [0..C-1]
    num_cats: dict[str, int]  # name -> C
    node_count: int


def load_features(features_path: Path, struct_path: Path, attr_keys: list[str]) -> Features:
    # Load structural scalars (small)
    try:
        import pyarrow.parquet as pq  # type: ignore

        tbl_s = pq.read_table(struct_path)  # type: ignore
        df_s = tbl_s.to_pandas()
    except Exception:
        df_s = pd.read_parquet(struct_path)  # type: ignore
    df_s = df_s.sort_values("node_id").reset_index(drop=True)
    struct_cols = [c for c in df_s.columns if c != "node_id"]
    S = df_s[struct_cols].to_numpy(dtype=np.float32)
    # Standardize all scalars (PR/HITS already standardized; this is idempotent)
    S = _zscore_cols(S)

    # Load attribute features
    try:
        import pyarrow.parquet as pq  # type: ignore

        tbl = pq.read_table(features_path)  # type: ignore
        df = tbl.to_pandas()
    except Exception:
        df = pd.read_parquet(features_path)  # type: ignore
    df = df.sort_values("node_id").reset_index(drop=True)
    N = len(df)

    # Map convenient aliases to actual column names if present
    alias_map = {
        # Coarse mappings and fallbacks (OON‑friendly)
        "country": "gr_country",
        "region": "gr_region",
        "continent": "gr_continent",
        # Sector-like fields available in this dataset
        "sector": "sector_code",
        # User-preferred, well-covered industry fields
        "industry_code": "industry_code",
        "primary_sic_code": "primary_sic_code",
        # Do not default to l-codes; they can still be explicitly passed
        # "cluster": "l3_id",
    }
    cols_to_use: list[str] = []
    for k in attr_keys:
        col = k
        if k in alias_map and alias_map[k] in df.columns:
            col = alias_map[k]
        if col in df.columns:
            cols_to_use.append(col)
        else:
            print(f"[WARN] Attribute column not found and skipped: {k}")

    attr_codes: dict[str, np.ndarray] = {}
    num_cats: dict[str, int] = {}
    for col in cols_to_use:
        vals = df[col]
        if pd.api.types.is_integer_dtype(vals):
            x = vals.to_numpy()
            x = np.where(np.isnan(x), -1, x) if np.issubdtype(x.dtype, np.floating) else x
            # Shift to start at 0 and reserve 0 for unknown if negatives present
            minv = int(x.min())
            if minv < 0:
                x = x - minv
            codes = x.astype(np.int64)
            C = int(codes.max()) + 2  # reserve last for unknown
            codes = np.clip(codes, 0, C - 1)
        else:
            # Factorize strings/categoricals
            cat = pd.Categorical(vals.astype("string").fillna("<UNK>"))
            codes = cat.codes.astype(np.int64)
            C = int(codes.max()) + 1
        attr_codes[col] = codes
        num_cats[col] = C

    return Features(struct=S, attr_codes=attr_codes, num_cats=num_cats, node_count=N)


# ------------------------------ Model ------------------------------------- #


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(p=dropout)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TwoTowerEncoder(nn.Module):
    def __init__(
        self,
        struct_in: int,
        struct_hidden: Sequence[int],
        attr_num_cats: dict[str, int],
        attr_emb_dim: int,
        attr_hidden: Sequence[int],
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.struct_net = MLP(struct_in, struct_hidden, out_dim, dropout=dropout)
        # Build one embedding per attribute and a small MLP on concatenated embeddings
        self.attr_names = list(attr_num_cats.keys())
        self.attr_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_embeddings=int(C), embedding_dim=attr_emb_dim)
                for name, C in attr_num_cats.items()
            }
        )
        attr_in = attr_emb_dim * max(1, len(self.attr_names))
        self.attr_net = MLP(attr_in, attr_hidden, out_dim, dropout=dropout)

    def forward(self, struct_x: Tensor, attr_codes: dict[str, Tensor]) -> Tensor:
        s = self.struct_net(struct_x)
        if self.attr_names:
            embs: list[Tensor] = []
            for name in self.attr_names:
                embs.append(self.attr_embeddings[name](attr_codes[name]))
            a_in = torch.cat(embs, dim=1)
            a = self.attr_net(a_in)
        else:
            a = torch.zeros_like(s)
        return torch.cat([s, a], dim=1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        struct_in: int,
        struct_hidden: Sequence[int],
        attr_num_cats: dict[str, int],
        attr_emb_dim: int,
        attr_hidden: Sequence[int],
        tower_out_dim: int,
        final_dim: int,
        dropout: float = 0.0,
        normalize: bool = True,
    ):
        super().__init__()
        # Each side concatenates [S_out, F_out] then projects to final_dim
        concat_dim = tower_out_dim * 2
        self.enc_u = TwoTowerEncoder(
            struct_in,
            struct_hidden,
            attr_num_cats,
            attr_emb_dim,
            attr_hidden,
            tower_out_dim,
            dropout,
        )
        self.enc_v = TwoTowerEncoder(
            struct_in,
            struct_hidden,
            attr_num_cats,
            attr_emb_dim,
            attr_hidden,
            tower_out_dim,
            dropout,
        )
        self.proj_u = nn.Linear(concat_dim, final_dim)
        self.proj_v = nn.Linear(concat_dim, final_dim)
        self.normalize = bool(normalize)

    def encode_nodes(self, side: str, struct_x: Tensor, attr_codes: dict[str, Tensor]) -> Tensor:
        if side == "u":
            h = self.enc_u(struct_x, attr_codes)
            z = self.proj_u(h)
        else:
            h = self.enc_v(struct_x, attr_codes)
            z = self.proj_v(h)
        if self.normalize:
            z = torch.nn.functional.normalize(z, p=2, dim=1)
        return z

    def score(self, zu: Tensor, zv: Tensor) -> Tensor:
        return (zu * zv).sum(dim=1)


# ------------------------------ Sampler ----------------------------------- #


class AliasSampler:
    """O(1) amortized sampling from a fixed discrete distribution using alias tables.

    Accepts raw non-negative weights; normalizes internally. Compatible with
    numpy Generator for RNG (passed at sample time).
    """

    def __init__(self, weights: np.ndarray):
        w = np.asarray(weights, dtype=np.float64)
        if w.size == 0:
            self.J = np.zeros(0, dtype=np.uint32)
            self.q = np.zeros(0, dtype=np.float32)
            return
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0.0:
            # fallback to uniform
            w = np.ones_like(w, dtype=np.float64)
            s = float(w.sum())
        q = (w / s) * w.size
        small: list[int] = []
        large: list[int] = []
        for i, qi in enumerate(q):
            (small if qi < 1.0 else large).append(i)
        self.J = np.empty(w.size, dtype=np.uint32)
        self.q = np.empty(w.size, dtype=np.float32)
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
        if m <= 0 or self.J.size == 0:
            return np.empty(0, dtype=np.int64)
        kk = rng.integers(0, self.J.size, size=int(m), dtype=np.uint32)
        uu = rng.random(size=int(m))
        out = kk.copy()
        mask = uu >= self.q[kk]
        out[mask] = self.J[kk[mask]]
        return out.astype(np.int64, copy=False)


class MixV3Sampler:
    def __init__(
        self,
        g: Graph,
        features: Features,
        deg_frac: float = 0.4,
        twohop_frac: float = 0.2,
        attr_frac: float = 0.2,
        deg_exp: float = 0.75,
        attr_order: Sequence[str] = ("sector", "country", "cluster"),
        seed: int | None = None,
    ):
        self.g = g
        self.features = features
        self.deg_frac = float(deg_frac)
        self.twohop_frac = float(twohop_frac)
        self.attr_frac = float(attr_frac)
        self.deg_exp = float(deg_exp)
        self.attr_order = list(attr_order)
        # Degree weights and alias sampler for fast draws
        w = np.power(np.maximum(1, g.in_deg).astype(np.float64), self.deg_exp)
        self._deg_weights = w
        self._alias = AliasSampler(w)
        # Attr indices: value -> node ids per attribute
        self.attr_indices: dict[str, dict[int, np.ndarray]] = {}
        for name, codes in features.attr_codes.items():
            idx: dict[int, list[int]] = {}
            for nid, code in enumerate(codes.tolist()):
                idx.setdefault(int(code), []).append(nid)
            self.attr_indices[name] = {k: np.array(v, dtype=np.int64) for k, v in idx.items()}
        # LRU cache for one-hop neighbors (sorted arrays)
        self._onehop_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._onehop_cache_cap: int = 200_000
        # LRU cache for full two-hop neighborhoods (sorted arrays)
        self._twohop_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._twohop_cache_cap: int = 200_000
        # Seeded RNG for reproducibility across runs/seeds
        self._rng = (
            np.random.default_rng(int(seed)) if (seed is not None) else np.random.default_rng()
        )

    def _cache_onehop_sorted(self, u: int) -> np.ndarray:
        arr = self._onehop_cache.get(u)
        if arr is not None:
            # mark as recently used
            self._onehop_cache.move_to_end(u)
            return arr
        csr, csc = self.g.csr, self.g.csc
        out_u = csr.indices[csr.indptr[u] : csr.indptr[u + 1]]
        in_u = csc.indices[csc.indptr[u] : csc.indptr[u + 1]]
        if out_u.size or in_u.size:
            one = np.unique(np.concatenate([out_u, in_u]))
        else:
            one = np.empty(0, dtype=np.int64)
        # store sorted unique (np.unique already sorted)
        self._onehop_cache[u] = one
        self._onehop_cache.move_to_end(u)
        if len(self._onehop_cache) > self._onehop_cache_cap:
            self._onehop_cache.popitem(last=False)
        return one

    def _cache_twohop_sorted(self, u: int) -> np.ndarray:
        arr = self._twohop_cache.get(u)
        if arr is not None:
            self._twohop_cache.move_to_end(u)
            return arr
        csr, csc = self.g.csr, self.g.csc
        one = self._cache_onehop_sorted(u)
        if one.size == 0:
            two_sorted = np.empty(0, dtype=np.int64)
        else:
            parts = []
            for w in one:
                parts.append(csr.indices[csr.indptr[w] : csr.indptr[w + 1]])
                parts.append(csc.indices[csc.indptr[w] : csc.indptr[w + 1]])
            two = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
            two_sorted = two  # already sorted by unique
        self._twohop_cache[u] = two_sorted
        self._twohop_cache.move_to_end(u)
        if len(self._twohop_cache) > self._twohop_cache_cap:
            self._twohop_cache.popitem(last=False)
        return two_sorted

    def _onehop_neighbors(self, u: int) -> np.ndarray:
        return self._cache_onehop_sorted(u)

    def _is_onehop(self, u: int, vs: np.ndarray) -> np.ndarray:
        one = self._onehop_neighbors(u)
        if one.size == 0:
            return np.zeros(vs.size, dtype=bool)
        pos = np.searchsorted(one, vs)
        m = np.zeros_like(vs, dtype=bool)
        valid = pos < one.size
        m[valid] = one[pos[valid]] == vs[valid]
        return m

    def sample_twohop(self, u: int, k: int) -> np.ndarray:
        two = self._cache_twohop_sorted(u)
        if two.size == 0:
            return np.empty(0, dtype=np.int64)
        if two.size <= k:
            return self._rng.choice(two, size=min(k, two.size), replace=False)
        return self._rng.choice(two, size=k, replace=False)

    def sample_deg(self, k: int) -> np.ndarray:
        # Use alias table for O(1) amortized draws
        return self._alias.sample(int(k), self._rng)

    def sample_uniform(self, k: int) -> np.ndarray:
        return self._rng.integers(0, self.g.num_nodes, size=int(k))

    def sample_attr(self, u: int, k: int) -> np.ndarray:
        """Attribute-similar negatives with SIC-first precedence (OON-safe).

        Precedence:
          1) (primary_sic_code & country)
          2) primary_sic_code
          3) country
          4) region
          5) continent
        """
        feats = self.features
        sic = feats.attr_codes.get("primary_sic_code")
        country = (
            feats.attr_codes.get("gr_country")
            if feats.attr_codes.get("gr_country") is not None
            else feats.attr_codes.get("country")
        )
        region = (
            feats.attr_codes.get("gr_region")
            if feats.attr_codes.get("gr_region") is not None
            else feats.attr_codes.get("region")
        )
        continent = (
            feats.attr_codes.get("gr_continent")
            if feats.attr_codes.get("gr_continent") is not None
            else feats.attr_codes.get("continent")
        )

        pools: list[np.ndarray] = []
        if sic is not None and country is not None:
            s = int(sic[u])
            c = int(country[u])
            S = self.attr_indices.get("primary_sic_code", {}).get(s, np.empty(0, dtype=np.int64))
            C = self.attr_indices.get("gr_country", self.attr_indices.get("country", {})).get(
                c, np.empty(0, dtype=np.int64)
            )
            if S.size and C.size:
                pools.append(np.intersect1d(S, C, assume_unique=False))
        if not pools and sic is not None:
            s = int(sic[u])
            pools.append(
                self.attr_indices.get("primary_sic_code", {}).get(s, np.empty(0, dtype=np.int64))
            )
        if not pools and country is not None:
            c = int(country[u])
            pools.append(
                self.attr_indices.get("gr_country", self.attr_indices.get("country", {})).get(
                    c, np.empty(0, dtype=np.int64)
                )
            )
        if not pools and region is not None:
            r = int(region[u])
            pools.append(
                self.attr_indices.get("gr_region", self.attr_indices.get("region", {})).get(
                    r, np.empty(0, dtype=np.int64)
                )
            )
        if not pools and continent is not None:
            co = int(continent[u])
            pools.append(
                self.attr_indices.get("gr_continent", self.attr_indices.get("continent", {})).get(
                    co, np.empty(0, dtype=np.int64)
                )
            )
        if not pools:
            return np.empty(0, dtype=np.int64)
        pool = pools[0]
        if pool.size == 0:
            return np.empty(0, dtype=np.int64)
        if pool.size <= k:
            return self._rng.choice(pool, size=min(k, pool.size), replace=False)
        return self._rng.choice(pool, size=k, replace=False)

    def _build_attr_pool(self, u: int) -> np.ndarray:
        feats = self.features
        sic = feats.attr_codes.get("primary_sic_code")
        country = (
            feats.attr_codes.get("gr_country")
            if feats.attr_codes.get("gr_country") is not None
            else feats.attr_codes.get("country")
        )
        region = (
            feats.attr_codes.get("gr_region")
            if feats.attr_codes.get("gr_region") is not None
            else feats.attr_codes.get("region")
        )
        continent = (
            feats.attr_codes.get("gr_continent")
            if feats.attr_codes.get("gr_continent") is not None
            else feats.attr_codes.get("continent")
        )
        pools: list[np.ndarray] = []
        if sic is not None and country is not None:
            s = int(sic[u])
            c = int(country[u])
            S = self.attr_indices.get("primary_sic_code", {}).get(s, np.empty(0, dtype=np.int64))
            C = self.attr_indices.get("gr_country", self.attr_indices.get("country", {})).get(
                c, np.empty(0, dtype=np.int64)
            )
            if S.size and C.size:
                pools.append(np.intersect1d(S, C, assume_unique=False))
        if not pools and sic is not None:
            s = int(sic[u])
            pools.append(
                self.attr_indices.get("primary_sic_code", {}).get(s, np.empty(0, dtype=np.int64))
            )
        if not pools and country is not None:
            c = int(country[u])
            pools.append(
                self.attr_indices.get("gr_country", self.attr_indices.get("country", {})).get(
                    c, np.empty(0, dtype=np.int64)
                )
            )
        if not pools and region is not None:
            r = int(region[u])
            pools.append(
                self.attr_indices.get("gr_region", self.attr_indices.get("region", {})).get(
                    r, np.empty(0, dtype=np.int64)
                )
            )
        if not pools and continent is not None:
            co = int(continent[u])
            pools.append(
                self.attr_indices.get("gr_continent", self.attr_indices.get("continent", {})).get(
                    co, np.empty(0, dtype=np.int64)
                )
            )
        pool = pools[0] if pools else np.empty(0, dtype=np.int64)
        return pool

    def _filter_exclusions(self, cand: np.ndarray, bad_sorted: np.ndarray) -> np.ndarray:
        if cand.size == 0 or bad_sorted.size == 0:
            return cand
        pos = np.searchsorted(bad_sorted, cand)
        valid = pos < bad_sorted.size
        keep = np.ones(cand.size, dtype=bool)
        keep[valid] = bad_sorted[pos[valid]] != cand[valid]
        return cand[keep]

    def sample_batch(
        self, us: np.ndarray, vs_pos: np.ndarray, K: int
    ) -> tuple[list[int], list[int], list[int], dict[str, int]]:
        # Group indices by source
        realized = {"deg": 0, "twohop": 0, "attr": 0, "uniform": 0}
        neg_us: list[int] = []
        neg_vs: list[int] = []
        counts_per_pos: list[int] = []
        # Build map from u -> indices in this batch
        from collections import defaultdict

        u_to_idxs: dict[int, list[int]] = defaultdict(list)
        for j, u in enumerate(us.tolist()):
            u_to_idxs[int(u)].append(j)
        for u, idxs in u_to_idxs.items():
            P = len(idxs)
            need = P * int(K)
            n_deg = round(self.deg_frac * need)
            n_two = round(self.twohop_frac * need)
            n_attr = round(self.attr_frac * need)
            used = n_deg + n_two + n_attr
            n_uni = max(0, need - used)
            # Build exclusion set for this u
            one = self._onehop_neighbors(u)
            pos_vs_u = vs_pos[idxs]
            bad = np.unique(
                np.concatenate(
                    [np.array([u], dtype=np.int64), one, pos_vs_u.astype(np.int64, copy=False)]
                )
            )
            picks: list[np.ndarray] = []
            # Degree-biased (oversample factor)
            if n_deg:
                cand = self._alias.sample(int(n_deg * 2), self._rng)
                cand = self._filter_exclusions(cand.astype(np.int64, copy=False), bad)
                picks.append(cand[:n_deg])
                realized["deg"] += min(n_deg, cand.size)
            # Two-hop (cached)
            if n_two:
                two = self._cache_twohop_sorted(u)
                if two.size:
                    # sample without replacement if possible
                    sz = min(int(n_two * 2), two.size)
                    cand = self._rng.choice(two, size=sz, replace=False)
                    cand = self._filter_exclusions(cand.astype(np.int64, copy=False), bad)
                    picks.append(cand[:n_two])
                    realized["twohop"] += min(n_two, cand.size)
            # Attr-sim
            if n_attr:
                pool = self._build_attr_pool(u)
                if pool.size:
                    sz = min(int(n_attr * 2), pool.size)
                    cand = self._rng.choice(pool, size=sz, replace=False)
                    cand = self._filter_exclusions(cand.astype(np.int64, copy=False), bad)
                    picks.append(cand[:n_attr])
                    realized["attr"] += min(n_attr, cand.size)
            # Uniform
            if n_uni:
                cand = self._rng.integers(0, self.g.num_nodes, size=int(n_uni * 2), dtype=np.int64)
                cand = self._filter_exclusions(cand, bad)
                picks.append(cand[:n_uni])
                realized["uniform"] += min(n_uni, cand.size)
            vs = np.concatenate(picks) if picks else np.empty(0, dtype=np.int64)
            # If short, top up from uniform
            short = need - vs.size
            if short > 0:
                top = self._rng.integers(0, self.g.num_nodes, size=int(short * 2), dtype=np.int64)
                top = self._filter_exclusions(top, bad)
                vs = np.concatenate([vs, top[:short]])
            if vs.size < need:
                # final fallback: pad by repeating allowed samples
                if vs.size > 0:
                    reps = np.resize(vs, need)
                    vs = reps
                else:
                    vs = np.zeros(need, dtype=np.int64)
            # Assign per-positive
            # Shuffle to avoid bias
            self._rng.shuffle(vs)
            # Take exactly need
            vs = vs[:need]
            # Emit for each pos
            for t, j in enumerate(idxs):
                start = t * K
                neg_chunk = vs[start : start + K]
                neg_us.extend([int(us[j])] * int(K))
                neg_vs.extend(neg_chunk.tolist())
                counts_per_pos.append(int(K))
        return neg_us, neg_vs, counts_per_pos, realized

    def sample(self, u: int, v_pos: int, K: int) -> tuple[np.ndarray, dict[str, int]]:
        # Determine counts per bucket
        n_deg = round(self.deg_frac * K)
        n_two = round(self.twohop_frac * K)
        n_attr = round(self.attr_frac * K)
        used = n_deg + n_two + n_attr
        n_uni = max(0, K - used)
        picks: list[int] = []
        # Degree-biased
        if n_deg:
            picks.extend(self.sample_deg(n_deg).tolist())
        # Two-hop strict
        if n_two:
            two = self.sample_twohop(u, n_two)
            picks.extend(two.tolist())
        # Attr-sim
        if n_attr:
            at = self.sample_attr(u, n_attr)
            picks.extend(at.tolist())
        # Uniform remainder
        if n_uni:
            picks.extend(self.sample_uniform(n_uni).tolist())

        vs = np.array(picks, dtype=np.int64)
        # Exclusions: self, positive, one-hop
        m_bad = (vs == u) | (vs == v_pos) | self._is_onehop(u, vs)
        vs = vs[~m_bad]
        # If we underfilled, top up with uniform
        short = max(0, K - vs.size)
        if short > 0:
            top = self.sample_uniform(short)
            m_bad2 = (top == u) | (top == v_pos) | self._is_onehop(u, top)
            top = top[~m_bad2]
            # Avoid costly np.unique; duplicates are acceptable for BPR
            vs = np.concatenate([vs, top.astype(np.int64, copy=False)])
        # Trim if too many
        if vs.size > K:
            vs = np.random.default_rng().choice(vs, size=K, replace=False)
        # Realized bucket counts (approximate, post-filter)
        realized = {
            "deg": int(n_deg),
            "twohop": int(n_two),
            "attr": int(n_attr),
            "uniform": int(n_uni),
        }
        return vs, realized


# ------------------------------ Training ---------------------------------- #


def bpr_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    # pos_scores, neg_scores: [B*K]
    x = pos_scores - neg_scores
    return -torch.nn.functional.logsigmoid(x).mean()


def train_twotower(
    model: TwoTowerModel,
    g: Graph,
    feats: Features,
    train_edges_path: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    K: int,
    sampler: MixV3Sampler,
) -> dict[str, object]:
    # Load train edges (positive pairs)
    try:
        import pyarrow.parquet as pq  # type: ignore

        tbl = pq.read_table(train_edges_path, columns=["src_id", "dst_id"])  # type: ignore
        df_tr = tbl.to_pandas()
    except Exception:
        df_tr = pd.read_parquet(train_edges_path, columns=["src_id", "dst_id"])  # type: ignore
    src = df_tr["src_id"].to_numpy(dtype=np.int64)
    dst = df_tr["dst_id"].to_numpy(dtype=np.int64)
    n_pos = src.size

    # Torch views of features (move per batch)
    S = torch.from_numpy(feats.struct).to(device)
    attr_codes_t: dict[str, torch.Tensor] = {
        k: torch.from_numpy(v).to(device) for k, v in feats.attr_codes.items()
    }

    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    model.train()
    t0_all = time.time()

    realized_mix_total = {"deg": 0, "twohop": 0, "attr": 0, "uniform": 0}
    timings_total = {"neg": 0.0, "forward": 0.0, "backward": 0.0}

    order = np.arange(n_pos)
    # Optional group-by-source for better reuse and cache locality
    group_by_src = bool(getattr(train_twotower, "group_by_src", True))
    if group_by_src:
        order = np.argsort(src, kind="mergesort")
    use_amp = bool(getattr(train_twotower, "use_amp", True)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    async_sampler = bool(getattr(train_twotower, "async_sampler", False))

    # Early stop / validation probe controls (attached via function attributes)
    val_interval: int = int(getattr(train_twotower, "val_probe_interval", 0))
    early_patience: int = int(getattr(train_twotower, "early_stop_patience", 0))
    min_delta: float = float(getattr(train_twotower, "early_stop_min_delta", 5e-4))
    val_max_src: int = int(getattr(train_twotower, "val_probe_max_sources", 1000))
    cand_val_path: Path | None = getattr(train_twotower, "cand_val_path", None)
    splits_root: Path | None = getattr(train_twotower, "splits_root", None)
    Ks_probe: list[int] = list(getattr(train_twotower, "Ks", [1, 10, 50]))
    emit_sw: bool = bool(getattr(train_twotower, "emit_strict_warm", False))
    best_ndcg: float = -1.0
    best_epoch: int = -1
    no_improve: int = 0
    best_state: dict[str, object] | None = None  # type: ignore[assignment]
    stop_training: bool = False

    for ep in range(epochs):
        if not group_by_src:
            np.random.shuffle(order)
        total_loss = 0.0
        nb = 0
        epoch_mix = {"deg": 0, "twohop": 0, "attr": 0, "uniform": 0}
        t_neg = 0.0
        t_fwd = 0.0
        t_bwd = 0.0
        batch_num = 0
        if async_sampler:
            ex = ThreadPoolExecutor(max_workers=1)
            # Prime first batch
            start = 0
            fut = None
            if start < n_pos:
                idx0 = order[start : start + batch_size]
                us0 = src[idx0]
                vs0 = dst[idx0]
                t0n = time.time()
                fut = ex.submit(sampler.sample_batch, us0, vs0, K)
                t_neg += time.time() - t0n
            while start < n_pos:
                idx = order[start : start + batch_size]
                us = src[idx]
                vs_pos = dst[idx]
                batch_num += 1
                if (batch_num % 200) == 0:
                    print(
                        f"[TRAIN] Epoch {ep + 1}/{epochs} batch {batch_num} (pos {start}/{n_pos})"
                    )
                # Launch next batch build
                next_start = start + batch_size
                next_fut = None
                if next_start < n_pos:
                    idxn = order[next_start : next_start + batch_size]
                    t0n = time.time()
                    next_fut = ex.submit(sampler.sample_batch, src[idxn], dst[idxn], K)
                    t_neg += time.time() - t0n
                # Wait current
                neg_us, neg_vs, counts_per_pos, realized = (
                    fut.result() if fut is not None else ([], [], [], dict.fromkeys(epoch_mix, 0))
                )
                for k in epoch_mix:
                    epoch_mix[k] += realized.get(k, 0)
                if neg_vs:
                    neg_pos_idx: list[int] = []
                    for j, c in enumerate(counts_per_pos):
                        neg_pos_idx.extend([j] * c)
                    neg_us_t = torch.tensor(neg_us, dtype=torch.long, device=device)
                    neg_vs_t = torch.tensor(neg_vs, dtype=torch.long, device=device)
                    pos_us_t = torch.from_numpy(us).to(device)
                    pos_vs_t = torch.from_numpy(vs_pos).to(device)
                    t0f = time.time()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        zu_pos = model.encode_nodes(
                            "u", S[pos_us_t], {k: v[pos_us_t] for k, v in attr_codes_t.items()}
                        )
                        zv_pos = model.encode_nodes(
                            "v", S[pos_vs_t], {k: v[pos_vs_t] for k, v in attr_codes_t.items()}
                        )
                        zu_neg = model.encode_nodes(
                            "u", S[neg_us_t], {k: v[neg_us_t] for k, v in attr_codes_t.items()}
                        )
                        zv_neg = model.encode_nodes(
                            "v", S[neg_vs_t], {k: v[neg_vs_t] for k, v in attr_codes_t.items()}
                        )
                        pos_scores = model.score(zu_pos, zv_pos)
                        neg_scores = model.score(zu_neg, zv_neg)
                        pos_for_negs = pos_scores[
                            torch.tensor(neg_pos_idx, dtype=torch.long, device=device)
                        ]
                        loss = bpr_loss(pos_for_negs, neg_scores)
                    t_fwd += time.time() - t0f
                    opt.zero_grad(set_to_none=True)
                    t0b = time.time()
                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                    t_bwd += time.time() - t0b
                    total_loss += float(loss)
                    nb += 1
                # Move to next
                fut = next_fut
                start = next_start
            ex.shutdown(wait=True)
            # Aggregate epoch stats
            for k in realized_mix_total:
                realized_mix_total[k] += epoch_mix.get(k, 0)
            timings_total["neg"] += t_neg
            timings_total["forward"] += t_fwd
            timings_total["backward"] += t_bwd
            # Epoch log
            s = sum(epoch_mix.values()) or 1
            mix_pct = {k: (100.0 * epoch_mix[k] / s) for k in epoch_mix}
            print(
                f"[TRAIN] Epoch {ep + 1}/{epochs} loss={total_loss / max(1, nb):.4f} | mix% {mix_pct} | t_neg={t_neg:.1f}s t_fwd={t_fwd:.1f}s t_bwd={t_bwd:.1f}s"
            )
            # Progress callback (for HPO/ASHA)
            try:
                cb = getattr(train_twotower, "progress_callback", None)
                if cb is not None:
                    cb(int(ep + 1), float(total_loss / max(1, nb)))
            except Exception:  # nosec B110 -- best-effort progress callback, pass is intentional
                pass

            # Optional validation probe + early stopping (async path)
            do_probe = (
                val_interval > 0
                and ((ep + 1) % val_interval == 0)
                and cand_val_path is not None
                and splits_root is not None
            )
            if do_probe:
                try:
                    model.eval()
                    ZU, ZV = compute_all_embeddings(model, feats, device=device)
                    gdf, _, _, counts = evaluate_split(
                        name="val",
                        ZU=ZU,
                        ZV=ZV,
                        device=device,
                        g=g,
                        Ks=Ks_probe,
                        cand_path=Path(cand_val_path),  # type: ignore[arg-type]
                        out_dir=Path("."),
                        batch_size=max(100_000, batch_size),
                        splits_root=Path(splits_root),  # type: ignore[arg-type]
                        collect_slices=False,
                        max_sources=int(val_max_src),
                        emit_strict_warm=emit_sw,
                    )
                    cur = (
                        float(gdf.iloc[0]["ndcg@100"])
                        if "ndcg@100" in gdf.columns
                        else float("nan")
                    )
                    print(
                        f"[VAL] Epoch {ep + 1}/{epochs} probe: ndcg@100={cur:.6f} on {counts.get('sources', 0)} sources (best={best_ndcg:.6f} @ep={best_epoch})"
                    )
                    import numpy as _np

                    improved = not _np.isnan(cur) and ((cur - best_ndcg) > float(min_delta))
                    if improved:
                        best_ndcg = cur
                        best_epoch = ep + 1
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
                            print(
                                f"[EARLYSTOP] Stop at epoch {ep + 1}: best ndcg@100={best_ndcg:.6f} @epoch {best_epoch}"
                            )
                            if best_state is not None:
                                try:
                                    model.load_state_dict(best_state)  # type: ignore[arg-type]
                                    print("[EARLYSTOP] Restored best model state")
                                except Exception:  # nosec B110 -- best-effort model state restore, pass is intentional
                                    pass
                            epochs = ep + 1
                            t1 = time.time()
                            stats = {
                                "epochs": int(epochs),
                                "training_time_sec": float(t1 - t0_all),
                                "realized_negative_mix": realized_mix_total,
                                "timings": timings_total,
                                "n_pos": int(n_pos),
                                "K": int(K),
                                "early_stop": {
                                    "enabled": True,
                                    "best_epoch": int(best_epoch),
                                    "best_ndcg@100": float(best_ndcg),
                                },
                            }
                            return stats
                except Exception as e:
                    print(f"[VAL][WARN] Probe failed at epoch {ep + 1}: {e}")
                finally:
                    model.train()
            # proceed to next epoch without entering non-async path
            continue

        for start in range(0, n_pos, batch_size):
            batch_num += 1
            if (batch_num % 200) == 0:
                print(f"[TRAIN] Epoch {ep + 1}/{epochs} batch {batch_num} (pos {start}/{n_pos})")
            idx = order[start : start + batch_size]
            us = src[idx]
            vs_pos = dst[idx]
            # Sample negatives per positive (batched by source)
            t0n = time.time()
            neg_us, neg_vs, counts_per_pos, realized = sampler.sample_batch(us, vs_pos, K)
            t_neg += time.time() - t0n
            for k in epoch_mix:
                epoch_mix[k] += realized[k]
            if not neg_vs:
                continue
            # Build mapping from each negative to its pos index (0..B-1)
            neg_pos_idx: list[int] = []
            for j, c in enumerate(counts_per_pos):
                neg_pos_idx.extend([j] * c)
            neg_us_t = torch.tensor(neg_us, dtype=torch.long, device=device)
            neg_vs_t = torch.tensor(neg_vs, dtype=torch.long, device=device)
            pos_us_t = torch.from_numpy(us).to(device)
            pos_vs_t = torch.from_numpy(vs_pos).to(device)

            # Encode
            t0f = time.time()
            with torch.cuda.amp.autocast(enabled=use_amp):
                zu_pos = model.encode_nodes(
                    "u", S[pos_us_t], {k: v[pos_us_t] for k, v in attr_codes_t.items()}
                )
                zv_pos = model.encode_nodes(
                    "v", S[pos_vs_t], {k: v[pos_vs_t] for k, v in attr_codes_t.items()}
                )
                zu_neg = model.encode_nodes(
                    "u", S[neg_us_t], {k: v[neg_us_t] for k, v in attr_codes_t.items()}
                )
                zv_neg = model.encode_nodes(
                    "v", S[neg_vs_t], {k: v[neg_vs_t] for k, v in attr_codes_t.items()}
                )
                pos_scores = model.score(zu_pos, zv_pos)  # [B]
                neg_scores = model.score(zu_neg, zv_neg)  # [M]
                pos_for_negs = pos_scores[
                    torch.tensor(neg_pos_idx, dtype=torch.long, device=device)
                ]  # [M]
                loss = bpr_loss(pos_for_negs, neg_scores)
            t_fwd += time.time() - t0f
            opt.zero_grad(set_to_none=True)
            t0b = time.time()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            t_bwd += time.time() - t0b
            total_loss += float(loss)
            nb += 1
        # Aggregate epoch stats
        for k in realized_mix_total:
            realized_mix_total[k] += epoch_mix.get(k, 0)
        timings_total["neg"] += t_neg
        timings_total["forward"] += t_fwd
        timings_total["backward"] += t_bwd
        s = sum(epoch_mix.values()) or 1
        mix_pct = {k: (100.0 * epoch_mix[k] / s) for k in epoch_mix}
        print(
            f"[TRAIN] Epoch {ep + 1}/{epochs} loss={total_loss / max(1, nb):.4f} | mix% {mix_pct} | t_neg={t_neg:.1f}s t_fwd={t_fwd:.1f}s t_bwd={t_bwd:.1f}s"
        )
        # Progress callback (for HPO/ASHA)
        try:
            cb = getattr(train_twotower, "progress_callback", None)
            if cb is not None:
                cb(int(ep + 1), float(total_loss / max(1, nb)))
        except Exception:  # nosec B110 -- best-effort progress callback, pass is intentional
            pass

        # Optional validation probe + early stopping
        do_probe = (
            val_interval > 0
            and ((ep + 1) % val_interval == 0)
            and cand_val_path is not None
            and splits_root is not None
        )
        if do_probe:
            try:
                model.eval()
                ZU, ZV = compute_all_embeddings(model, feats, device=device)
                gdf, _, _, counts = evaluate_split(
                    name="val",
                    ZU=ZU,
                    ZV=ZV,
                    device=device,
                    g=g,
                    Ks=Ks_probe,
                    cand_path=Path(cand_val_path),  # type: ignore[arg-type]
                    out_dir=Path("."),
                    batch_size=max(100_000, batch_size),
                    splits_root=Path(splits_root),  # type: ignore[arg-type]
                    collect_slices=False,
                    max_sources=int(val_max_src),
                    emit_strict_warm=emit_sw,
                )
                cur = float(gdf.iloc[0]["ndcg@100"]) if "ndcg@100" in gdf.columns else float("nan")
                print(
                    f"[VAL] Epoch {ep + 1}/{epochs} probe: ndcg@100={cur:.6f} on {counts.get('sources', 0)} sources (best={best_ndcg:.6f} @ep={best_epoch})"
                )
                improved = (cur - best_ndcg) > float(min_delta)
                import numpy as _np  # local guard

                if _np.isnan(cur):
                    improved = False
                if improved:
                    best_ndcg = cur
                    best_epoch = ep + 1
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
            finally:
                model.train()

        # Honor early stop outside probe block (return safely without raw 'break')
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
            epochs = ep + 1
            t1 = time.time()
            stats = {
                "epochs": int(epochs),
                "training_time_sec": float(t1 - t0_all),
                "realized_negative_mix": realized_mix_total,
                "timings": timings_total,
                "n_pos": int(n_pos),
                "K": int(K),
                "early_stop": {
                    "enabled": True,
                    "best_epoch": int(best_epoch),
                    "best_ndcg@100": float(best_ndcg),
                },
            }
            return stats

    t1 = time.time()
    stats = {
        "epochs": int(epochs),
        "training_time_sec": float(t1 - t0_all),
        "realized_negative_mix": realized_mix_total,
        "timings": timings_total,
        "n_pos": int(n_pos),
        "K": int(K),
    }
    stats["early_stop"] = {
        "enabled": bool(stop_training),
        "best_epoch": int(best_epoch),
        "best_ndcg@100": float(best_ndcg),
    }
    return stats


@torch.no_grad()
def compute_all_embeddings(
    model: TwoTowerModel, feats: Features, device: torch.device
) -> tuple[Tensor, Tensor]:
    model.eval()
    S = torch.from_numpy(feats.struct).to(device)
    attr_codes_t: dict[str, torch.Tensor] = {
        k: torch.from_numpy(v).to(device) for k, v in feats.attr_codes.items()
    }
    # Batch over nodes if needed
    N = feats.node_count
    bs = 131072
    ZU = []
    ZV = []
    for start in range(0, N, bs):
        end = min(N, start + bs)
        idx = torch.arange(start, end, device=device)
        zu = model.encode_nodes("u", S[idx], {k: v[idx] for k, v in attr_codes_t.items()})
        zv = model.encode_nodes("v", S[idx], {k: v[idx] for k, v in attr_codes_t.items()})
        ZU.append(zu)
        ZV.append(zv)
    ZU = torch.cat(ZU, dim=0)
    ZV = torch.cat(ZV, dim=0)
    return ZU, ZV


# ------------------------------ Evaluation -------------------------------- #


def evaluate_split(
    name: str,
    ZU: Tensor,
    ZV: Tensor,
    device: torch.device,
    g: Graph,
    Ks: list[int],
    cand_path: Path,
    out_dir: Path,
    batch_size: int,
    splits_root: Path,
    collect_slices: bool = True,
    max_sources: int | None = None,
    emit_strict_warm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    macro = KMetrics(Ks=Ks)
    micro = KMetrics(Ks=Ks)
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
    slices: dict[str, KMetrics] = {s: KMetrics(Ks=Ks) for s in slice_names}

    try:
        q = np.quantile(g.out_deg, [0.25, 0.5, 0.75])
    except Exception:
        q = np.array([0, 0, 0], dtype=float)

    T0 = _load_t0(splits_root) or 0
    streamer = CandidateStreamer(cand_path, batch_size=batch_size)
    # Prepare ts sidecar mapping if candidates lack ts
    ts_sidecar: dict[tuple[int, int], int] | None = None
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(cand_path)
        cand_has_ts = "ts" in set(pf.schema.names)
    except Exception:
        cand_has_ts = False
    if not cand_has_ts:
        split_edges_path = splits_root / f"{name}_edges.parquet"
        if split_edges_path.exists():
            try:
                df_edges = pd.read_parquet(split_edges_path, columns=["src_id", "dst_id", "ts"])  # type: ignore
                ts_sidecar = {(int(s), int(d)): int(t) for s, d, t in df_edges.to_numpy()}
            except Exception:
                ts_sidecar = None

    total_rows = 0
    total_sources = 0
    ZU_dev = ZU.to(device)
    ZV_dev = ZV.to(device)

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
            labels = sub["label"].to_numpy(dtype=np.int8, copy=False)

            with torch.no_grad():
                zu = ZU_dev[u]
                zv = ZV_dev[vs]
                scores_t = (zv * zu).sum(dim=1)
                scores = scores_t.detach().cpu().numpy()

            order = np.lexsort((vs, -scores))
            labs_sorted = labels[order].astype(np.int64)
            macro.update(labs_sorted)
            micro.update(labs_sorted)

            if collect_slices:
                wc = classify_warm_cold(u, vs, g)[order]
                if emit_strict_warm:
                    wc3 = classify_warm_cold_strict(u, vs, g, threshold=3)[order]
                th = classify_twohop(u, vs, g)[order]
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
                hz = np.zeros_like(labels, dtype=np.int8)
                if T0:
                    pos_m = labels > 0
                    if np.any(pos_m):
                        deltas = ts_vals[pos_m] - float(T0)
                        hz[np.nonzero(pos_m)[0]] = assign_horizon_buckets(deltas)
                hz = hz[order]

                d = g.out_deg[u]
                if d <= q[0]:
                    db = "deg_q1"
                elif d <= q[1]:
                    db = "deg_q2"
                elif d <= q[2]:
                    db = "deg_q3"
                else:
                    db = "deg_q4"

                for lab, tag in zip([0, 1, 2, 3], ["WW", "WC", "CW", "CC"], strict=False):
                    m = wc == lab
                    if m.any():
                        slices[tag].update(labs_sorted[m])
                if emit_strict_warm:
                    for lab, tag in zip([0, 1, 2, 3], ["WW3", "WC3", "CW3", "CC3"], strict=False):
                        m = wc3 == lab
                        if m.any():  # type: ignore[union-attr]
                            slices[tag].update(labs_sorted[m])
                if th.any():
                    slices["twohop"].update(labs_sorted[th])
                if (~th).any():
                    slices[">2hop"].update(labs_sorted[~th])
                for hv, stag in enumerate(horizon_slice_names(), start=1):
                    m = hz == hv
                    if m.any():
                        ls = labs_sorted.copy()
                        drop = (ls > 0) & (~m)
                        ls[drop] = 0
                        slices[stag].update(ls)
                slices[db].update(labs_sorted)

            total_rows += len(sub)
            total_sources += 1
            if max_sources is not None and total_sources >= int(max_sources):
                break
        if max_sources is not None and total_sources >= int(max_sources):
            break

    global_df = pd.DataFrame([macro.to_row(True, "TwoTower")])
    micro_df = pd.DataFrame([micro.to_row(False, "TwoTower")])
    rows = []
    if collect_slices:
        for name_s, km in slices.items():
            r = km.to_row(True, "TwoTower")
            r["slice_name"] = name_s
            rows.append(r)
    slices_df = pd.DataFrame(rows)
    counts = {"rows": int(total_rows), "sources": int(total_sources)}
    return global_df, micro_df, slices_df, counts


# ------------------------------ Calibration ------------------------------- #


def _sigmoid(x: np.ndarray) -> np.ndarray:
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
        p = _sigmoid(z)
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
    ZU: Tensor,
    ZV: Tensor,
    device: torch.device,
    g: Graph,
    cand_val: Path,
    splits_root: Path,
    sample_per_src: int = 50,
    max_pairs: int = 1_000_000,
) -> dict[str, object]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        has_ts = "ts" in set(pq.ParquetFile(cand_val).schema.names)
    except Exception:
        has_ts = False
    _ = ["src_id", "dst_id", "label"] + (["ts"] if has_ts else [])
    streamer = CandidateStreamer(cand_val, batch_size=1_000_000)
    ZU_dev = ZU.to(device)
    ZV_dev = ZV.to(device)
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
            if vs.size > sample_per_src:
                idx = np.random.choice(vs.size, size=sample_per_src, replace=False)
                vs_s = vs[idx]
                lbl_s = lbl[idx]
            else:
                vs_s = vs
                lbl_s = lbl
            with torch.no_grad():
                zu = ZU_dev[u]
                zv = ZV_dev[vs_s]
                sc = (zv * zu).sum(dim=1).detach().cpu().numpy()
            scores.append(sc.astype(np.float64))
            labels.append(lbl_s.astype(np.float64))
            total += len(vs_s)
            if total >= max_pairs:
                break
        if total >= max_pairs:
            break
    if not scores:
        raise RuntimeError("No calibration samples collected for TwoTower")
    s = np.concatenate(scores, axis=0)
    y = np.concatenate(labels, axis=0)
    A, B = fit_platt(s, y)
    return {"method": "platt", "A": A, "B": B, "samples": int(s.size)}


# ---------------------------------- Main ---------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Two-Tower on train graph and evaluate on fixed candidate pools"
    )
    # Data
    p.add_argument("--adj", required=True, type=str)
    p.add_argument("--features", required=True, type=str)
    p.add_argument("--struct-feats", required=True, type=str)
    p.add_argument("--candidates-val", required=True, type=str)
    p.add_argument("--candidates-test", required=True, type=str)
    p.add_argument("--splits-root", required=True, type=str)
    # Outputs
    p.add_argument("--out-dir", type=str, default="results/twotower/twotower_inductive_v1")
    p.add_argument("--artifacts-dir", type=str, default="artifacts/twotower/twotower_inductive_v1")
    p.add_argument("--version-tag", type=str, default="twotower_inductive_v1")
    # Model
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--struct-hidden", type=str, default="64,64")
    p.add_argument("--attr-emb-dim", type=int, default=32)
    p.add_argument("--attr-hidden", type=str, default="64,64")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--normalize-emb", type=_bool, default=True)
    # Best params ingestion (from HPO finalize)
    p.add_argument(
        "--use-best-params", action="store_true", help="Load best params JSON and override CLI"
    )
    p.add_argument(
        "--best-params-path", type=str, default="", help="Path to robust_best_params.json"
    )
    # Attributes
    p.add_argument(
        "--attr-keys", type=str, default="country,region,continent,entity_type,primary_sic_code"
    )
    # Training
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pairwise-negs", type=int, default=4)
    # Negatives mix
    p.add_argument("--neg-strategy", type=str, default="mix_v3", choices=["mix_v3"])
    p.add_argument("--mix-deg-frac", type=float, default=0.4)
    p.add_argument("--mix-twohop-frac", type=float, default=0.2)
    p.add_argument("--mix-attr-frac", type=float, default=0.2)
    p.add_argument("--mix-deg-exp", type=float, default=0.75)
    # Eval
    p.add_argument("--Ks", type=str, default="1,10,50")
    p.add_argument("--emit-strict-warm", type=_bool, default=False)
    p.add_argument("--calibrate", type=_bool, default=False)
    p.add_argument("--calibrate-sample-per-src", type=int, default=50)
    p.add_argument("--calibrate-max-pairs", type=int, default=1000000)
    # Optional periodic validation probe + early stopping
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
    # Persistence
    p.add_argument(
        "--save-model",
        nargs="?",
        const=True,
        type=_bool,
        default=False,
        help="Save trained model state per seed",
    )
    p.add_argument("--model-filename", type=str, default="model.pt")
    p.add_argument(
        "--sqlite-out", type=str, default="", help="Optional SQLite DB path for per-seed metrics"
    )
    # System
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated list of seeds for multi-seed evaluation (e.g., '42,123,456,789,999')",
    )
    p.add_argument(
        "--multi-seed",
        action="store_true",
        help="Run multi-seed evaluation and report confidence intervals",
    )
    p.add_argument("--undirected", type=_bool, default=False)
    # Performance toggles
    p.add_argument(
        "--group-by-src",
        type=_bool,
        default=True,
        help="Group training batches by source for negative reuse",
    )
    p.add_argument(
        "--async-sampler",
        type=_bool,
        default=True,
        help="Overlap CPU negative sampling with GPU compute",
    )
    p.add_argument(
        "--use-amp", type=_bool, default=True, help="Enable CUDA AMP for mixed-precision training"
    )
    return p.parse_args()


def run_single_seed(
    args: argparse.Namespace,
    g: Graph,
    feats: Features,
    device: torch.device,
    Ks: list[int],
    out_dir: Path,
    art_dir: Path,
    seed: int,
) -> dict[str, object]:
    print(f"[SEED {seed}] Starting Two-Tower seed run...")
    set_seed(int(seed))
    # Build a fresh model per seed
    struct_in = int(feats.struct.shape[1])
    struct_hidden = [int(x.strip()) for x in str(args.struct_hidden).split(",") if x.strip()]
    attr_hidden = [int(x.strip()) for x in str(args.attr_hidden).split(",") if x.strip()]
    model = TwoTowerModel(
        struct_in=struct_in,
        struct_hidden=struct_hidden,
        attr_num_cats=feats.num_cats,
        attr_emb_dim=int(args.attr_emb_dim),
        attr_hidden=attr_hidden,
        tower_out_dim=int(args.embed_dim // 2),
        final_dim=int(args.embed_dim),
        dropout=float(args.dropout),
        normalize=bool(args.normalize_emb),
    ).to(device)

    sampler = MixV3Sampler(
        g=g,
        features=feats,
        deg_frac=float(args.mix_deg_frac),
        twohop_frac=float(args.mix_twohop_frac),
        attr_frac=float(args.mix_attr_frac),
        deg_exp=float(args.mix_deg_exp),
        seed=int(seed),
    )

    train_edges_path = Path(args.splits_root) / "train_edges.parquet"
    print(f"[SEED {seed}] Training (K={int(args.pairwise_negs)})...")
    # Attach training toggles
    try:
        train_twotower.group_by_src = bool(args.group_by_src)  # type: ignore[attr-defined]
        train_twotower.async_sampler = bool(args.async_sampler)  # type: ignore[attr-defined]
        train_twotower.use_amp = bool(args.use_amp)  # type: ignore[attr-defined]
    except Exception:  # nosec B110 -- best-effort attribute injection, pass is intentional
        pass
    # Attach optional periodic validation probe/early stop settings
    try:
        train_twotower.val_probe_interval = int(getattr(args, "val_probe_interval", 0))  # type: ignore[attr-defined]
        train_twotower.val_probe_max_sources = int(getattr(args, "val_probe_max_sources", 1000))  # type: ignore[attr-defined]
        train_twotower.early_stop_patience = int(getattr(args, "early_stop_patience", 0))  # type: ignore[attr-defined]
        train_twotower.early_stop_min_delta = float(getattr(args, "early_stop_min_delta", 5e-4))  # type: ignore[attr-defined]
        train_twotower.cand_val_path = Path(args.candidates_val)  # type: ignore[attr-defined]
        train_twotower.splits_root = Path(args.splits_root)  # type: ignore[attr-defined]
        train_twotower.Ks = Ks  # type: ignore[attr-defined]
        train_twotower.emit_strict_warm = bool(getattr(args, "emit_strict_warm", False))  # type: ignore[attr-defined]
    except Exception:  # nosec B110 -- best-effort attribute injection, pass is intentional
        pass

    tr_stats = train_twotower(
        model=model,
        g=g,
        feats=feats,
        train_edges_path=train_edges_path,
        device=device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        K=int(args.pairwise_negs),
        sampler=sampler,
    )

    # Save model if requested
    if bool(getattr(args, "save_model", False)):
        try:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hparams": {
                        "embed_dim": int(args.embed_dim),
                        "struct_hidden": str(args.struct_hidden),
                        "attr_emb_dim": int(args.attr_emb_dim),
                        "attr_hidden": str(args.attr_hidden),
                        "dropout": float(args.dropout),
                        "lr": float(args.lr),
                        "batch_size": int(args.batch_size),
                        "pairwise_negs": int(args.pairwise_negs),
                        "normalize_emb": bool(args.normalize_emb),
                    },
                    "seed": int(seed),
                },
                art_dir / str(getattr(args, "model_filename", "model.pt")),
            )
            print(
                f"[SEED {seed}] Saved model to {art_dir / str(getattr(args, 'model_filename', 'model.pt'))}"
            )
        except Exception as e:
            print(f"[SEED {seed}][WARN] Failed to save model: {e}")

    print(f"[SEED {seed}] Computing embeddings...")
    ZU, ZV = compute_all_embeddings(model, feats, device=device)

    if bool(getattr(args, "calibrate", False)):
        try:
            print(f"[SEED {seed}] Calibrating on validation sample...")
            cal = calibrate_on_val(
                ZU=ZU,
                ZV=ZV,
                device=device,
                g=g,
                cand_val=Path(args.candidates_val),
                splits_root=Path(args.splits_root),
                sample_per_src=int(getattr(args, "calibrate_sample_per_src", 50)),
                max_pairs=int(getattr(args, "calibrate_max_pairs", 1000000)),
            )
            cal_out = {
                "created_at": _now_iso(),
                "model_tag": str(args.version_tag),
                "features_version": "node_structural_v1",
                "T0": int(_load_t0(Path(args.splits_root)) or 0),
                "candidate_pool": str(Path(args.candidates_val)),
                "method": cal.get("method"),
                "params": {k: v for k, v in cal.items() if k not in {"method"}},
            }
            with open(art_dir / "calibration.json", "w") as f:
                json.dump(cal_out, f, indent=2)
        except Exception as e:
            print(f"[SEED {seed}][CAL][WARN] Calibration failed: {e}")

    print(f"[SEED {seed}] Evaluating val/test...")
    val_g, val_m, val_s, _ = evaluate_split(
        name="val",
        ZU=ZU,
        ZV=ZV,
        device=device,
        g=g,
        Ks=Ks,
        cand_path=Path(args.candidates_val),
        out_dir=out_dir,
        batch_size=2_000_000,
        splits_root=Path(args.splits_root),
        collect_slices=True,
        max_sources=None,
        emit_strict_warm=bool(args.emit_strict_warm),
    )
    test_g, test_m, test_s, _ = evaluate_split(
        name="test",
        ZU=ZU,
        ZV=ZV,
        device=device,
        g=g,
        Ks=Ks,
        cand_path=Path(args.candidates_test),
        out_dir=out_dir,
        batch_size=2_000_000,
        splits_root=Path(args.splits_root),
        collect_slices=True,
        max_sources=None,
        emit_strict_warm=bool(args.emit_strict_warm),
    )

    val_g.to_csv(out_dir / "global_val.csv", index=False)
    val_m.to_csv(out_dir / "micro_val.csv", index=False)
    val_s.to_csv(out_dir / "slices_val.csv", index=False)
    test_g.to_csv(out_dir / "global_test.csv", index=False)
    test_m.to_csv(out_dir / "micro_test.csv", index=False)
    test_s.to_csv(out_dir / "slices_test.csv", index=False)

    return {
        "val": val_g.iloc[0].to_dict(),
        "test": test_g.iloc[0].to_dict(),
        "training_stats": tr_stats,
    }


def main() -> None:
    print("[MAIN] Starting Two-Tower training + evaluation...")
    args = parse_args()
    Ks = [int(x.strip()) for x in str(args.Ks).split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    art_dir = Path(args.artifacts_dir)
    _ensure_dir(out_dir)
    _ensure_dir(art_dir)

    # Device
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[MAIN] Using device: {device}")

    # Load graph/features once
    print("[MAIN] Loading graph and features...")
    g = load_graph(Path(args.adj), undirected=bool(args.undirected))
    attr_keys = [k.strip() for k in str(args.attr_keys).split(",") if k.strip()]
    feats = load_features(Path(args.features), Path(args.struct_feats), attr_keys)
    print(
        f"[MAIN] Graph: {g.num_nodes:,} nodes, {g.csr.nnz:,} edges; Features: struct={feats.struct.shape[1]} attrs={list(feats.attr_codes.keys())}"
    )

    # Optionally override hyperparams from best params JSON
    if bool(getattr(args, "use_best_params", False)):
        bp_path = Path(getattr(args, "best_params_path", "") or "")
        if not bp_path:
            print("[WARN] --use-best-params set but --best-params-path not provided")
        elif not bp_path.exists():
            print(f"[WARN] Best params path not found: {bp_path}")
        else:
            try:
                obj = json.loads(bp_path.read_text())
                bp = obj.get("params", obj)
                if "embed_dim" in bp:
                    args.embed_dim = int(bp["embed_dim"])
                if "struct_hidden" in bp:
                    args.struct_hidden = str(bp["struct_hidden"])
                if "attr_emb_dim" in bp:
                    args.attr_emb_dim = int(bp["attr_emb_dim"])
                if "attr_hidden" in bp:
                    args.attr_hidden = str(bp["attr_hidden"])
                if "dropout" in bp:
                    args.dropout = float(bp["dropout"])
                if "lr" in bp:
                    args.lr = float(bp["lr"])
                if "batch_size" in bp:
                    args.batch_size = int(bp["batch_size"])
                if "pairwise_negs" in bp:
                    args.pairwise_negs = int(bp["pairwise_negs"])
                print(f"[MAIN] Loaded best params from {bp_path}")
            except Exception as e:
                print(f"[WARN] Failed to parse best params from {bp_path}: {e}")

    # Multi-seed path
    if args.multi_seed or args.seeds:
        if args.seeds:
            seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
        else:
            seeds = [42, 123, 456, 789, 999]
        print(f"[MULTI-SEED] Running seeds: {seeds}")
        all_results = []
        # Optional SQLite persistence
        sqlite_path = (
            Path(str(args.sqlite_out)) if str(getattr(args, "sqlite_out", "")).strip() else None
        )
        if sqlite_path is not None:
            try:
                import sqlite3

                conn = sqlite3.connect(sqlite_path)
                cur = conn.cursor()
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS runs (tag TEXT, seed INT, split TEXT, ndcg REAL, mrr REAL, hit10 REAL, created_at TEXT)"
                )
            except Exception as e:
                print(f"[WARN] Failed to open SQLite {sqlite_path}: {e}")
                sqlite_path = None
        from datetime import datetime

        tag = out_dir.name
        for s in seeds:
            seed_out = out_dir / f"seed_{s}"
            seed_art = art_dir / f"seed_{s}"
            _ensure_dir(seed_out)
            _ensure_dir(seed_art)
            res = run_single_seed(args, g, feats, device, Ks, seed_out, seed_art, s)
            all_results.append(res)
            # Insert per-seed metrics
            if sqlite_path is not None:
                try:
                    import pandas as pd

                    now = datetime.utcnow().isoformat()
                    for split in ["val", "test"]:
                        gdf = pd.read_csv(seed_out / f"global_{split}.csv")
                        row = gdf.iloc[0]
                        conn.execute(
                            "INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
                            (
                                tag,
                                int(s),
                                split,
                                float(row.get("ndcg@100", 0.0)),
                                float(row.get("mrr", 0.0)),
                                float(row.get("hit@10", 0.0)),
                                now,
                            ),
                        )
                    conn.commit()
                except Exception as e:
                    print(f"[WARN] SQLite insert failed for seed {s}: {e}")
        # Aggregate
        metrics = ["hit@1", "hit@10", "hit@50", "mrr", "map", "ndcg@100"]
        val_agg = aggregate_multi_seed_results([r["val"] for r in all_results], metrics)
        test_agg = aggregate_multi_seed_results([r["test"] for r in all_results], metrics)
        pd.DataFrame([{"metric": m, **val_agg[m]} for m in val_agg]).to_csv(
            out_dir / "aggregate_val.csv", index=False
        )
        pd.DataFrame([{"metric": m, **test_agg[m]} for m in test_agg]).to_csv(
            out_dir / "aggregate_test.csv", index=False
        )
        multi = {
            "timestamp": _now_iso(),
            "device": str(device),
            "seeds": seeds,
            "validation": val_agg,
            "test": test_agg,
            "individual_results": all_results,
        }
        (art_dir / "multi_seed_summary.json").write_text(json.dumps(multi, indent=2))
        print(f"[MULTI-SEED] Wrote aggregates to {out_dir}")
        if "conn" in locals():
            try:
                conn.close()
                print(f"[INFO] Persisted run metrics to SQLite at {sqlite_path}")
            except Exception:  # nosec B110 -- best-effort SQLite cleanup, pass is intentional
                pass
        return

    # Single-seed path
    set_seed(int(args.seed))
    res = run_single_seed(args, g, feats, device, Ks, out_dir, art_dir, int(args.seed))
    # Basic summary
    summary = {
        "timestamp": _now_iso(),
        "device": str(device),
        "seed": int(args.seed),
        "val": res["val"],
        "test": res["test"],
        "training_stats": res.get("training_stats", {}),
    }
    (art_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[MAIN] Done. Wrote summary to {art_dir / 'training_summary.json'}")
    # Optional SQLite persistence (single-seed)
    sqlite_path = (
        Path(str(getattr(args, "sqlite_out", "")).strip())
        if str(getattr(args, "sqlite_out", "")).strip()
        else None
    )
    if sqlite_path is not None:
        try:
            import sqlite3

            import pandas as pd

            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(sqlite_path)
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS runs (tag TEXT, seed INT, split TEXT, ndcg REAL, mrr REAL, hit10 REAL, created_at TEXT)"
            )
            from datetime import datetime

            now = datetime.utcnow().isoformat()
            tag = out_dir.name
            for split in ["val", "test"]:
                gdf = pd.read_csv(out_dir / f"global_{split}.csv")
                row = gdf.iloc[0]
                conn.execute(
                    "INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
                    (
                        tag,
                        int(args.seed),
                        split,
                        float(row.get("ndcg@100", 0.0)),
                        float(row.get("mrr", 0.0)),
                        float(row.get("hit@10", 0.0)),
                        now,
                    ),
                )
            conn.commit()
            conn.close()
            print(f"[INFO] Persisted run metrics to SQLite at {sqlite_path}")
        except Exception as e:
            print(f"[WARN] SQLite insert failed: {e}")


if __name__ == "__main__":
    main()
