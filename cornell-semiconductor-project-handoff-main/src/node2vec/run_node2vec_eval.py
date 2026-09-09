#!/usr/bin/env python3
"""
Node2Vec Trainer + Scorecard Evaluator (Core_v1)

Trains Node2Vec on the train-only adjacency (T0) and evaluates on fixed
candidate pools (val, test) using the shared scorecard:
  - Per-source metrics: Hit@K, Recall@K, Precision@K (K in {1,10,50}), MRR, MAP, nDCG@100
  - Macro primary, micro secondary
  - Slices: warm/cold (WW/WC/CW/CC), two-hop vs >2-hop, time horizons, degree bins

GPU is used by default when available.

CLI example:
  python src/node2vec/run_node2vec_eval.py \
    --adj data/processed/core/releases/core_v1/adjacency/train_adj_T0.npz \
    --candidates-val data/processed/core/releases/core_v1/candidates/val_candidates.parquet \
    --candidates-test data/processed/core/releases/core_v1/candidates/test_candidates.parquet \
    --splits-root data/processed/core/releases/core_v1/splits \
    --out-dir results/node2vec \
    --artifacts-dir artifacts/node2vec \
    --Ks 1,10,50 \
    --embedding-dim 128 --p 1.0 --q 0.5 --walks-per-node 10 --walk-length 20 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import Tensor
from torch_geometric.nn import Node2Vec

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


def apply_best_params_override(args: argparse.Namespace) -> None:
    """Override CLI hyperparameters from a best_params.json artifact if requested."""
    if not bool(getattr(args, "use_best_params", False)):
        return

    path_str = str(getattr(args, "best_params_path", "") or "").strip()
    if not path_str:
        print("[WARN] --use-best-params set but --best-params-path not provided; ignoring override")
        return

    best_path = Path(path_str)
    if not best_path.exists():
        print(f"[WARN] Best params path not found: {best_path}; ignoring override")
        return

    try:
        obj = json.loads(best_path.read_text())
    except Exception as exc:
        print(f"[WARN] Failed to parse best params from {best_path}: {exc}; ignoring override")
        return

    params = obj.get("params", obj)
    mapping = {
        "embedding_dim": int,
        "walk_length": int,
        "walks_per_node": int,
        "p": float,
        "q": float,
        "lr": float,
        "epochs": int,
    }
    for key, caster in mapping.items():
        if key in params:
            try:
                setattr(args, key if key != "epochs" else "epochs", caster(params[key]))
            except Exception:
                print(
                    f"[WARN] Unable to cast best param '{key}'={params[key]!r}; keeping CLI value"
                )

    print(f"[INFO] Loaded best params from {best_path}")


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # NOTE: CUDA strict determinism is intentionally disabled.
    # torch.use_deterministic_algorithms(True) forces deterministic cuDNN
    # kernels, which incur a 15-30% training time penalty and raise errors
    # for operations without a deterministic implementation.  Results are
    # statistically reproducible across seeds (see Appendix A.2, seed
    # robustness table) but are NOT bit-for-bit identical across GPU runs.
    # This is standard practice for deep learning research.
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def calculate_confidence_intervals(
    values: list[float], confidence: float = 0.95
) -> dict[str, float]:
    """Calculate mean, std, and confidence intervals for a list of values."""
    if len(values) < 2:
        return {"mean": values[0] if values else 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    import statistics

    mean = statistics.mean(values)
    std = statistics.stdev(values)

    # Calculate confidence interval using t-distribution
    import scipy.stats as stats

    if len(values) > 1:
        t_value = stats.t.ppf((1 + confidence) / 2, len(values) - 1)
        margin_of_error = t_value * std / (len(values) ** 0.5)
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error
    else:
        ci_lower = ci_upper = mean

    return {"mean": mean, "std": std, "ci_lower": ci_lower, "ci_upper": ci_upper, "n": len(values)}


def aggregate_multi_seed_results(
    seed_results: list[dict[str, object]], metrics: list[str]
) -> dict[str, dict[str, float]]:
    """Aggregate results across multiple seeds with confidence intervals."""
    aggregated = {}

    for metric in metrics:
        values = [float(result[metric]) for result in seed_results if metric in result]  # type: ignore[arg-type]
        if values:
            aggregated[metric] = calculate_confidence_intervals(values)

    return aggregated


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
        # Symmetrize: (A + A^T) > 0
        csc_temp = csr.tocsc(copy=False)
        csr = (csr + csc_temp).astype(bool).astype(np.uint8).tocsr(copy=False)
    csc = csr.tocsc(copy=False)
    out_deg = np.diff(csr.indptr).astype(np.int64, copy=False)
    in_deg = np.diff(csc.indptr).astype(np.int64, copy=False)
    return Graph(csr=csr, csc=csc, out_deg=out_deg, in_deg=in_deg, num_nodes=int(csr.shape[0]))


def csr_to_edge_index(csr: sp.csr_matrix) -> Tensor:
    rows, cols = csr.nonzero()
    ei = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))
    return ei


# ------------------------------ Streaming IO ------------------------------- #


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
        # Check if ts column exists in the parquet file
        try:
            self.pq.read_table(self.path, columns=["ts"])
            has_ts = True
        except Exception:
            has_ts = False

        columns = ["src_id", "dst_id", "label"]
        if has_ts:
            columns.append("ts")

        tbl = self.pq.read_table(self.path, columns=columns)  # type: ignore
        n = tbl.num_rows
        start = 0
        remainder: pd.DataFrame | None = None
        while start < n:
            end = min(start + self.batch, n)
            df = tbl.slice(start, end - start).to_pandas(types_mapper={})
            # Normalize column names if needed
            df = df.rename(columns={"src": "src_id", "dst": "dst_id"})
            if remainder is not None and len(remainder):
                df = pd.concat([remainder, df], ignore_index=True)
                remainder = None
            if df.empty:
                start = end
                continue
            # Cut at last contiguous src run
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


# ------------------------------ Metrics ------------------------------------ #


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

    def to_row(self, macro: bool, heuristic: str = "node2vec") -> dict[str, object]:
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


def sort_by_score(scores: np.ndarray, dsts: np.ndarray) -> np.ndarray:
    return np.lexsort((dsts, -scores))


# ------------------------------ Slices ------------------------------------- #


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
    """Strict warm/cold classification where warm means degree ≥ threshold.

    Uses train-time degrees at T0: source warm if out_deg[u] ≥ threshold;
    destination warm if in_deg[v] ≥ threshold.
    Returns labels in {0:WW3, 1:WC3, 2:CW3, 3:CC3} matching non-strict mapping.
    """
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
    # Fix: check bounds before accessing two_sorted
    valid_mask = pos < two_sorted.size
    m = np.zeros_like(vs, dtype=bool)
    m[valid_mask] = two_sorted[pos[valid_mask]] == vs[valid_mask]
    return m


# ------------------------------ Training ----------------------------------- #


@torch.no_grad()
def get_embeddings(model: Node2Vec, device: torch.device) -> Tensor:
    model.eval()
    Z = model()
    return Z.to(device)


def train_node2vec(
    g: Graph,
    embedding_dim: int,
    p: float,
    q: float,
    walks_per_node: int,
    walk_length: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[Node2Vec, dict[str, float]]:
    edge_index = csr_to_edge_index(g.csr).to(device)
    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=int(embedding_dim),
        walk_length=int(walk_length),
        context_size=10,
        walks_per_node=int(walks_per_node),
        p=float(p),
        q=float(q),
        num_negative_samples=1,
        num_nodes=int(g.num_nodes),
        sparse=True,
    ).to(device)
    opt = torch.optim.SparseAdam(model.parameters(), lr=float(lr))
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)

    model.train()
    t0 = time.time()
    total_loss = 0.0
    print(f"[TRAIN] Starting Node2Vec training: {g.num_nodes:,} nodes, {g.csr.nnz:,} edges")
    print(
        f"[TRAIN] Config: dim={embedding_dim}, p={p}, q={q}, walks={walks_per_node}, length={walk_length}"
    )
    for ep in range(epochs):
        epoch_loss = 0.0
        for pos_rw, neg_rw in loader:
            pos_rw = pos_rw.to(device)
            neg_rw = neg_rw.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            opt.step()
            epoch_loss += float(loss)
        total_loss += epoch_loss
        if (ep + 1) % 5 == 0 or ep == 0:  # Log every 5 epochs
            print(f"[TRAIN] Epoch {ep + 1}/{epochs}, loss: {epoch_loss:.4f}")
    t1 = time.time()
    print(f"[TRAIN] Training completed in {t1 - t0:.1f}s")
    stats = {
        "epochs": int(epochs),
        "final_loss": total_loss / max(1, epochs),
        "training_time_sec": t1 - t0,
    }
    return model, stats


# ------------------------------ Evaluation --------------------------------- #


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
    emit_strict_warm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    # Prepare accumulators
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

    # Degree quartiles for src bins
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

    # Embeddings on device for fast dot products
    Z_dev = Z.to(device)

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

            # Scores: dot(z[u], z[v])
            with torch.no_grad():
                zu = Z_dev[u]
                zv = Z_dev[vs]
                scores_t = (zv * zu).sum(dim=1)
                scores = scores_t.detach().cpu().numpy()

            # Ranking
            order = np.lexsort((vs, -scores))
            labs_sorted = labels[order].astype(np.int64)
            macro.update(labs_sorted)
            micro.update(labs_sorted)

            # Slices (optional for speed during sweeps)
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
                        assignments = assign_horizon_buckets(deltas)
                        hz[np.nonzero(pos_m)[0]] = assignments
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

    # Make dataframes
    global_df = pd.DataFrame([macro.to_row(True, "Node2Vec")])
    micro_df = pd.DataFrame([micro.to_row(False, "Node2Vec")])
    rows = []
    if collect_slices:
        for name_s, km in slices.items():
            r = km.to_row(True, "Node2Vec")
            r["slice_name"] = name_s
            rows.append(r)
    slices_df = pd.DataFrame(rows)
    counts = {"rows": int(total_rows), "sources": int(total_sources)}
    return global_df, micro_df, slices_df, counts


# ---------------------------------- Main ----------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Node2Vec on train graph and evaluate on fixed candidate pools"
    )
    p.add_argument("--adj", required=True, type=str)
    p.add_argument("--candidates-val", required=True, type=str)
    p.add_argument("--candidates-test", required=True, type=str)
    p.add_argument("--splits-root", required=True, type=str)
    p.add_argument("--out-dir", type=str, default="results/node2vec")
    p.add_argument("--artifacts-dir", type=str, default="artifacts/node2vec")
    p.add_argument("--Ks", type=str, default="1,10,50")
    p.add_argument("--undirected", type=_bool, default=False)
    # Node2Vec hyperparams
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--p", type=float, default=1.0)
    p.add_argument("--q", type=float, default=0.5)
    p.add_argument("--walks-per-node", type=int, default=10)
    p.add_argument("--walk-length", type=int, default=20)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)
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
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--save-embeddings", action="store_true")
    p.add_argument(
        "--exit-after-save",
        action="store_true",
        help="If set with --save-embeddings, exits after saving embeddings (skips eval)",
    )
    p.add_argument(
        "--use-best-params",
        action="store_true",
        help="Load hyperparameters from best_params.json (from HPO) and override CLI values",
    )
    p.add_argument(
        "--best-params-path",
        type=str,
        default="",
        help="Path to best_params.json emitted by 01_node2vec_hpo_optuna.py",
    )
    # Eval controls
    p.add_argument(
        "--batch-size",
        type=int,
        default=2_000_000,
        help="Candidate rows per batch for streaming eval",
    )
    # Sweep controls
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Run a small hyperparameter sweep on val, pick best, then evaluate test",
    )
    p.add_argument(
        "--dims", type=str, default="64,128", help="Embedding dims to try when --sweep is set"
    )
    p.add_argument(
        "--p-grid", type=str, default="0.5,1.0,2.0", help="p values to try (comma-separated)"
    )
    p.add_argument(
        "--q-grid", type=str, default="0.5,1.0,2.0", help="q values to try (comma-separated)"
    )
    p.add_argument(
        "--sweep-epochs",
        type=int,
        default=10,
        help="Epochs per combo during sweep (lighter than final)",
    )
    p.add_argument(
        "--select-metric",
        type=str,
        default="mrr",
        choices=["mrr", "ndcg@100", "map", "hit@10"],
        help="Metric to select best combo",
    )
    p.add_argument(
        "--retrain-best",
        type=_bool,
        default=True,
        help="Retrain best config with --epochs before final eval",
    )
    p.add_argument(
        "--sweep-max-sources",
        type=int,
        default=5000,
        help="Limit number of val sources per combo to speed up sweep",
    )
    p.add_argument(
        "--sweep-disable-slices",
        type=_bool,
        default=True,
        help="Skip slice metrics during sweep for speed",
    )
    p.add_argument(
        "--emit-strict-warm",
        type=_bool,
        default=False,
        help="Emit additional WW3/WC3/CW3/CC3 slices (warm=≥3 edges)",
    )
    # Calibration
    p.add_argument(
        "--calibrate",
        type=_bool,
        default=False,
        help="Fit a Platt calibrator on validation and save artifacts",
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


def run_single_seed_evaluation(
    args: argparse.Namespace,
    g: Graph,
    device: torch.device,
    Ks: list[int],
    out_dir: Path,
    art_dir: Path,
    seed: int,
) -> dict[str, object]:
    """Run a single seed evaluation and return results."""
    print(f"[SEED {seed}] Starting evaluation...")

    # Set all seeds properly
    set_seed(seed)

    # Configuration
    best_cfg = {
        "embedding_dim": int(args.embedding_dim),
        "p": float(args.p),
        "q": float(args.q),
        "walks_per_node": int(args.walks_per_node),
        "walk_length": int(args.walk_length),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
    }

    # Create seed-specific directories
    seed_out_dir = out_dir / f"seed_{seed}"
    seed_art_dir = art_dir / f"seed_{seed}"
    _ensure_dir(seed_out_dir)
    _ensure_dir(seed_art_dir)

    # Train model
    print(f"[SEED {seed}] Training Node2Vec...")
    model, tr_stats = train_node2vec(
        g=g,
        embedding_dim=int(best_cfg["embedding_dim"]),
        p=float(best_cfg["p"]),
        q=float(best_cfg["q"]),
        walks_per_node=int(best_cfg["walks_per_node"]),
        walk_length=int(best_cfg["walk_length"]),
        epochs=int(best_cfg["epochs"]),
        lr=float(best_cfg["lr"]),
        device=device,
    )

    # Get embeddings
    Z = get_embeddings(model, device=device)

    # Optionally save embeddings for this seed
    if bool(getattr(args, "save_embeddings", False)):
        emb_path = seed_art_dir / f"embeddings_dim{int(best_cfg['embedding_dim'])}.pt"
        try:
            torch.save(
                {
                    "embeddings": Z.detach().cpu(),
                    "config": {
                        "dim": int(best_cfg["embedding_dim"]),
                        "p": float(best_cfg["p"]),
                        "q": float(best_cfg["q"]),
                        "walks_per_node": int(best_cfg["walks_per_node"]),
                        "walk_length": int(best_cfg["walk_length"]),
                        "epochs": int(best_cfg["epochs"]),
                        "lr": float(best_cfg["lr"]),
                        "seed": int(seed),
                    },
                },
                emb_path,
            )
            print(f"[SEED {seed}] Saved embeddings to {emb_path}")
        except Exception as e:
            print(f"[SEED {seed}][WARN] Failed to save embeddings: {e}")

    # Optional calibration on validation (once per seed run)
    if bool(getattr(args, "calibrate", False)):
        try:
            print(f"[SEED {seed}] Calibrating (Platt) on validation sample...")
            cal = calibrate_node2vec_on_val(
                Z=Z,
                device=device,
                g=g,
                cand_val=Path(args.candidates_val),
                splits_root=Path(args.splits_root),
                sample_per_src=int(getattr(args, "calibrate_sample_per_src", 50)),
                max_pairs=int(getattr(args, "calibrate_max_pairs", 1000000)),
            )
            cal_out = {
                "created_at": _now_iso(),
                "model_tag": "node2vec",
                "features_version": "node_structural_v1",
                "T0": int(_load_t0(Path(args.splits_root)) or 0),
                "candidate_pool": str(Path(args.candidates_val)),
                "method": cal.get("method"),
                "params": {k: v for k, v in cal.items() if k not in {"method"}},
            }
            cal_path = seed_art_dir / "calibration.json"
            with open(cal_path, "w") as f:
                json.dump(cal_out, f, indent=2)
            print(f"[SEED {seed}] Saved calibrator to {cal_path}")
        except Exception as e:
            print(f"[SEED {seed}][CAL][WARN] Calibration failed: {e}")

    # Evaluate validation
    print(f"[SEED {seed}] Evaluating validation...")
    val_global, val_micro, val_slices, val_counts = evaluate_split(
        name="val",
        Z=Z,
        device=device,
        g=g,
        Ks=Ks,
        cand_path=Path(args.candidates_val),
        out_dir=seed_out_dir,
        batch_size=int(args.batch_size),
        splits_root=Path(args.splits_root),
        collect_slices=True,
        max_sources=None,
        emit_strict_warm=bool(args.emit_strict_warm),
    )

    # Evaluate test
    print(f"[SEED {seed}] Evaluating test...")
    test_global, test_micro, test_slices, test_counts = evaluate_split(
        name="test",
        Z=Z,
        device=device,
        g=g,
        Ks=Ks,
        cand_path=Path(args.candidates_test),
        out_dir=seed_out_dir,
        batch_size=int(args.batch_size),
        splits_root=Path(args.splits_root),
        collect_slices=True,
        max_sources=None,
        emit_strict_warm=bool(args.emit_strict_warm),
    )

    # Save seed-specific results
    val_global.to_csv(seed_out_dir / "global_val.csv", index=False)
    val_micro.to_csv(seed_out_dir / "micro_val.csv", index=False)
    val_slices.to_csv(seed_out_dir / "slices_val.csv", index=False)

    test_global.to_csv(seed_out_dir / "global_test.csv", index=False)
    test_micro.to_csv(seed_out_dir / "micro_test.csv", index=False)
    test_slices.to_csv(seed_out_dir / "slices_test.csv", index=False)

    # Save seed-specific summary
    seed_summary = {
        "timestamp": _now_iso(),
        "seed": seed,
        "device": str(device),
        "configuration": best_cfg,
        "training_stats": tr_stats,
        "counts": {"val": val_counts, "test": test_counts},
    }

    with open(seed_art_dir / "run_summary.json", "w") as f:
        json.dump(seed_summary, f, indent=2)

    # Extract results for aggregation
    val_results = val_global.iloc[0].to_dict()
    test_results = test_global.iloc[0].to_dict()

    results = {
        "seed": seed,
        "val": val_results,
        "test": test_results,
        "training_stats": tr_stats,
        "counts": {"val": val_counts, "test": test_counts},
    }

    print(f"[SEED {seed}] Completed evaluation")
    return results


# ----------------------------- Calibration --------------------------------- #


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


def calibrate_node2vec_on_val(
    Z: Tensor,
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
            if vs.size > sample_per_src:
                idx = np.random.choice(vs.size, size=sample_per_src, replace=False)
                vs_s = vs[idx]
                lbl_s = lbl[idx]
            else:
                vs_s = vs
                lbl_s = lbl
            with torch.no_grad():
                zu = Z_dev[u]
                zv = Z_dev[vs_s]
                sc = (zv * zu).sum(dim=1).detach().cpu().numpy()
            scores.append(sc.astype(np.float64))
            labels.append(lbl_s.astype(np.float64))
            total += len(vs_s)
            if total >= max_pairs:
                break
        if total >= max_pairs:
            break
    if not scores:
        raise RuntimeError("No calibration samples collected for Node2Vec")
    s = np.concatenate(scores, axis=0)
    y = np.concatenate(labels, axis=0)
    A, B = fit_platt(s, y)
    return {"method": "platt", "A": A, "B": B, "samples": int(s.size)}


def main() -> None:
    print("[MAIN] Starting Node2Vec evaluation...")
    args = parse_args()
    apply_best_params_override(args)
    print(f"[MAIN] Arguments parsed, loading graph from {args.adj}...")
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

    # Load graph
    print("[MAIN] Loading graph (this may take a while for large graphs)...")
    g = load_graph(Path(args.adj), undirected=bool(args.undirected))
    print(f"[MAIN] Graph loaded: {g.num_nodes:,} nodes, {g.csr.nnz:,} edges")

    # Multi-seed evaluation
    if args.multi_seed or args.seeds:
        if args.seeds:
            seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
        else:
            # Default seeds for academic rigor
            seeds = [42, 123, 456, 789, 999]

        print(f"[MULTI-SEED] Running evaluation with {len(seeds)} seeds: {seeds}")

        all_results = []
        for i, seed in enumerate(seeds):
            print(f"[MULTI-SEED] [{i + 1}/{len(seeds)}] Running seed {seed}")
            result = run_single_seed_evaluation(args, g, device, Ks, out_dir, art_dir, seed)
            all_results.append(result)

        # Aggregate results with confidence intervals
        print("[MULTI-SEED] Aggregating results with confidence intervals...")
        metrics = ["hit@1", "hit@10", "hit@50", "mrr", "map", "ndcg@100"]

        val_aggregated = aggregate_multi_seed_results([r["val"] for r in all_results], metrics)
        test_aggregated = aggregate_multi_seed_results([r["test"] for r in all_results], metrics)

        # Create aggregated CSV files
        val_agg_data = []
        test_agg_data = []

        for metric in metrics:
            if metric in val_aggregated:
                stats = val_aggregated[metric]
                val_agg_data.append(
                    {
                        "metric": metric,
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "ci_lower": stats["ci_lower"],
                        "ci_upper": stats["ci_upper"],
                        "n": stats["n"],
                    }
                )

        for metric in metrics:
            if metric in test_aggregated:
                stats = test_aggregated[metric]
                test_agg_data.append(
                    {
                        "metric": metric,
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "ci_lower": stats["ci_lower"],
                        "ci_upper": stats["ci_upper"],
                        "n": stats["n"],
                    }
                )

        # Save aggregated results
        val_agg_df = pd.DataFrame(val_agg_data)
        test_agg_df = pd.DataFrame(test_agg_data)

        val_agg_df.to_csv(out_dir / "aggregate_val.csv", index=False)
        test_agg_df.to_csv(out_dir / "aggregate_test.csv", index=False)

        # Print results
        print("\n" + "=" * 60)
        print("MULTI-SEED EVALUATION RESULTS")
        print("=" * 60)
        print(
            f"Configuration: dim={args.embedding_dim}, p={args.p}, q={args.q}, epochs={args.epochs}"
        )
        print(f"Seeds: {seeds}")
        print("Confidence level: 95%")
        print()

        print("VALIDATION SET (Mean ± Std, [CI_lower, CI_upper]):")
        for metric in metrics:
            if metric in val_aggregated:
                stats = val_aggregated[metric]
                print(
                    f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
                )

        print("\nTEST SET (Mean ± Std, [CI_lower, CI_upper]):")
        for metric in metrics:
            if metric in test_aggregated:
                stats = test_aggregated[metric]
                print(
                    f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
                )

        # Save aggregated results
        multi_seed_summary = {
            "timestamp": _now_iso(),
            "device": str(device),
            "seeds": seeds,
            "configuration": {
                "embedding_dim": int(args.embedding_dim),
                "p": float(args.p),
                "q": float(args.q),
                "walks_per_node": int(args.walks_per_node),
                "walk_length": int(args.walk_length),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
            },
            "validation": val_aggregated,
            "test": test_aggregated,
            "individual_results": all_results,
        }

        with open(art_dir / "multi_seed_summary.json", "w") as f:
            json.dump(multi_seed_summary, f, indent=2)

        print("\n[MULTI-SEED] Results saved to:")
        print(f"  - {art_dir / 'multi_seed_summary.json'}")
        print(f"  - {out_dir / 'aggregate_val.csv'}")
        print(f"  - {out_dir / 'aggregate_test.csv'}")
        print(f"  - Individual seed results: {out_dir}/seed_*/")
        print("=" * 60)
        return

    # Single seed evaluation (original logic)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    # Sweep (optional) or single run
    best_cfg = {
        "embedding_dim": int(args.embedding_dim),
        "p": float(args.p),
        "q": float(args.q),
        "walks_per_node": int(args.walks_per_node),
        "walk_length": int(args.walk_length),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
    }
    sweep_results: list[dict[str, object]] = []
    model = None
    Z = None
    tr_stats = {}

    if args.sweep:
        # Build grid
        dims = [int(x.strip()) for x in str(args.dims).split(",") if x.strip()]
        p_grid = [float(x.strip()) for x in str(args.p_grid).split(",") if x.strip()]
        q_grid = [float(x.strip()) for x in str(args.q_grid).split(",") if x.strip()]
        print(f"[INFO] Sweep grid: dims={dims}, p={p_grid}, q={q_grid}; epochs={args.sweep_epochs}")
        best_score = -1.0
        total_combinations = len(dims) * len(p_grid) * len(q_grid)
        current_combo = 0
        for d in dims:
            for pval in p_grid:
                for qval in q_grid:
                    current_combo += 1
                    print(
                        f"[SWEEP] [{current_combo}/{total_combinations}] dim={d} p={pval} q={qval}"
                    )
                    print(f"[SWEEP] Starting training for combination {current_combo}...")
                    m, stats = train_node2vec(
                        g=g,
                        embedding_dim=int(d),
                        p=float(pval),
                        q=float(qval),
                        walks_per_node=int(args.walks_per_node),
                        walk_length=int(args.walk_length),
                        epochs=int(args.sweep_epochs),
                        lr=float(args.lr),
                        device=device,
                    )
                    print("[SWEEP] Training completed, getting embeddings...")
                    Z_tmp = get_embeddings(m, device=device)
                    print(f"[SWEEP] Starting evaluation for combination {current_combo}...")
                    gdf, _, _, counts = evaluate_split(
                        name="val",
                        Z=Z_tmp,
                        device=device,
                        g=g,
                        Ks=Ks,
                        cand_path=Path(args.candidates_val),
                        out_dir=out_dir,
                        batch_size=int(args.batch_size),
                        splits_root=Path(args.splits_root),
                        collect_slices=not bool(args.sweep_disable_slices),
                        max_sources=int(args.sweep_max_sources),
                        emit_strict_warm=bool(args.emit_strict_warm),
                    )
                    print(f"[SWEEP] Evaluation completed for combination {current_combo}")
                    # Selection metric
                    metric = args.select_metric
                    val = (
                        float(gdf.iloc[0][metric])
                        if metric in gdf.columns
                        else float(gdf.iloc[0]["mrr"])
                    )
                    sweep_results.append(
                        {
                            "dim": int(d),
                            "p": float(pval),
                            "q": float(qval),
                            "metric": metric,
                            "score": val,
                            "counts": counts,
                        }
                    )
                    if val > best_score:
                        best_score = val
                        best_cfg.update(
                            {
                                "embedding_dim": int(d),
                                "p": float(pval),
                                "q": float(qval),
                            }
                        )
                        # Optionally keep best model/embeddings when not retraining
                        model = m
                        Z = Z_tmp
                        tr_stats = stats
        print(
            f"[INFO] Best sweep config: {best_cfg} (score={best_score:.5f} by {args.select_metric})"
        )

        if bool(args.retrain_best):
            print("[INFO] Retraining best config with final epochs for evaluation...")
            model, tr_stats = train_node2vec(
                g=g,
                embedding_dim=int(best_cfg["embedding_dim"]),
                p=float(best_cfg["p"]),
                q=float(best_cfg["q"]),
                walks_per_node=int(args.walks_per_node),
                walk_length=int(args.walk_length),
                epochs=int(best_cfg["epochs"]),
                lr=float(best_cfg["lr"]),
                device=device,
            )
            Z = get_embeddings(model, device=device)
    else:
        # Single run with provided hyperparameters
        print(f"[INFO] Training Node2Vec on {g.num_nodes:,} nodes (device={device})")
        model, tr_stats = train_node2vec(
            g=g,
            embedding_dim=int(best_cfg["embedding_dim"]),
            p=float(best_cfg["p"]),
            q=float(best_cfg["q"]),
            walks_per_node=int(best_cfg["walks_per_node"]),
            walk_length=int(best_cfg["walk_length"]),
            epochs=int(best_cfg["epochs"]),
            lr=float(best_cfg["lr"]),
            device=device,
        )
        Z = get_embeddings(model, device=device)

    # Optionally save embeddings
    if args.save_embeddings and Z is not None:
        emb_path = art_dir / f"embeddings_dim{int(best_cfg['embedding_dim'])}.pt"
        torch.save(
            {
                "embeddings": Z.detach().cpu(),
                "config": {
                    "dim": int(best_cfg["embedding_dim"]),
                    "p": float(best_cfg["p"]),
                    "q": float(best_cfg["q"]),
                    "walks_per_node": int(best_cfg["walks_per_node"]),
                    "walk_length": int(best_cfg["walk_length"]),
                },
            },
            emb_path,
        )
        print(f"[INFO] Saved embeddings to {emb_path}")
        if args.exit_after_save:
            # Emit a lightweight summary and exit before evaluation
            summary = {
                "timestamp": _now_iso(),
                "device": str(device),
                "seed": int(args.seed),
                "adjacency": str(Path(args.adj)),
                "saved_embeddings": str(emb_path),
                "config": {
                    "embedding_dim": int(best_cfg["embedding_dim"]),
                    "p": float(best_cfg["p"]),
                    "q": float(best_cfg["q"]),
                    "walks_per_node": int(best_cfg["walks_per_node"]),
                    "walk_length": int(best_cfg["walk_length"]),
                    "epochs": int(best_cfg["epochs"]),
                    "lr": float(best_cfg["lr"]),
                },
            }
            try:
                (art_dir / "summary_save_only.json").write_text(json.dumps(summary, indent=2))
                print(f"[INFO] Wrote summary to {art_dir / 'summary_save_only.json'}")
            except Exception:  # nosec B110 -- best-effort artifact save, pass is intentional
                pass
            return

    # Evaluate val and test
    splits_root = Path(args.splits_root)
    summary = {
        "timestamp": _now_iso(),
        "device": str(device),
        "seed": int(args.seed),
        "adjacency": str(Path(args.adj)),
        "candidates": {
            "val": str(Path(args.candidates_val)),
            "test": str(Path(args.candidates_test)),
        },
        "Ks": Ks,
        "undirected": bool(args.undirected),
        "hparams": best_cfg,
        "training": tr_stats,
        "counts": {},
    }
    if args.sweep:
        summary["sweep"] = {
            "select_metric": args.select_metric,
            "max_sources": int(args.sweep_max_sources),
            "disable_slices": bool(args.sweep_disable_slices),
            "results": sweep_results,
        }

    # Optional calibration on validation (single-run path)
    if bool(getattr(args, "calibrate", False)):
        try:
            print("[MAIN] Calibrating (Platt) on validation sample...")
            assert Z is not None, "embeddings must be computed before calibration"
            cal = calibrate_node2vec_on_val(
                Z=Z,
                device=device,
                g=g,
                cand_val=Path(args.candidates_val),
                splits_root=Path(args.splits_root),
                sample_per_src=int(getattr(args, "calibrate_sample_per_src", 50)),
                max_pairs=int(getattr(args, "calibrate_max_pairs", 1000000)),
            )
            cal_out = {
                "created_at": _now_iso(),
                "model_tag": "node2vec",
                "features_version": "node_structural_v1",
                "T0": int(_load_t0(Path(args.splits_root)) or 0),
                "candidate_pool": str(Path(args.candidates_val)),
                "method": cal.get("method"),
                "params": {k: v for k, v in cal.items() if k not in {"method"}},
            }
            with open(art_dir / "calibration.json", "w") as f:
                json.dump(cal_out, f, indent=2)
            print(f"[MAIN] Saved calibrator to {art_dir / 'calibration.json'}")
        except Exception as e:
            print(f"[MAIN][CAL][WARN] Calibration failed: {e}")

    assert Z is not None, "embeddings must be computed before evaluation"
    for split_name, cpath in {
        "val": Path(args.candidates_val),
        "test": Path(args.candidates_test),
    }.items():
        print(f"[INFO] Evaluating {split_name} from {cpath}")
        t0 = time.time()
        gdf, mdf, sdf, counts = evaluate_split(
            name=split_name,
            Z=Z,
            device=device,
            g=g,
            Ks=Ks,
            cand_path=cpath,
            out_dir=out_dir,
            batch_size=int(args.batch_size),
            splits_root=splits_root,
            collect_slices=True,
            max_sources=None,
            emit_strict_warm=bool(args.emit_strict_warm),
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
    main()
