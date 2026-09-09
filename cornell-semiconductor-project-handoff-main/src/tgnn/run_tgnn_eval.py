#!/usr/bin/env python3
"""
TGNN Trainer + Scorecard Evaluator (Core_v1)
==============================================

Temporal Graph Neural Network trained on quarterly/annual snapshots and evaluated
on fixed candidate pools with comprehensive scorecard.

Key features:
- Temporal snapshot processing (quarterly/annual granularity)
- Attention-based aggregation across snapshots
- Platt calibration for score calibration
- Per-seed outputs (matching other models)
- Time horizon, warm/cold, degree quartile slices
- Robust logging for monitoring
- Model checkpointing and saving

Usage:
python src/tgnn/run_tgnn_eval.py \
  --adj data/.../train_adj_T0.npz \
  --features data/.../node_features_T0.parquet \
  --candidates-val data/.../val_candidates.parquet \
  --candidates-test data/.../test_candidates.parquet \
  --splits-root data/.../splits \
  --granularity quarter \
  --max-snapshots 62 \
  --epochs-per-snapshot 12 \
  --out-dir results/tgnn/test \
  --artifacts-dir artifacts/tgnn/test \
  --device auto
"""

from __future__ import annotations

import argparse
import json
import logging
import random

# Setup imports
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import Tensor

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Import TGNN model
from tgnn.model import TemporalGNN

# Import utilities
from utils.horizons import assign_horizon_buckets, horizon_slice_names

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


# ============================= Utilities ============================= #


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
    """Load T0_end timestamp from metadata."""
    meta = splits_root.parent / "meta" / "temporal_splits.json"
    try:
        if meta.exists():
            obj = json.loads(meta.read_text())
            return int(obj.get("boundaries", {}).get("T0_end"))
    except Exception:  # nosec B110 -- best-effort metadata read, pass is intentional
        pass
    return None


# ============================= Data Loading ============================= #


@dataclass
class Graph:
    csr: sp.csr_matrix
    csc: sp.csc_matrix
    out_deg: np.ndarray
    in_deg: np.ndarray
    num_nodes: int


def load_graph(adj_path: Path, undirected: bool) -> Graph:
    """Load graph from adjacency matrix."""
    LOGGER.info(f"Loading graph from {adj_path}")
    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency not found: {adj_path}")

    csr = sp.load_npz(adj_path).tocsr(copy=False)
    if undirected:
        csc_tmp = csr.tocsc(copy=False)
        csr = (csr + csc_tmp).astype(bool).astype(np.uint8).tocsr(copy=False)

    csr.indices = np.ascontiguousarray(csr.indices.astype(np.int32, copy=False))
    csr.indptr = np.ascontiguousarray(csr.indptr.astype(np.int32, copy=False))
    csc = csr.tocsc(copy=False)

    out_deg = np.diff(csr.indptr).astype(np.int64)
    in_deg = np.diff(csc.indptr).astype(np.int64)

    LOGGER.info(f"  Nodes: {csr.shape[0]:,}, Edges: {csr.nnz:,}")
    LOGGER.info(f"  Avg degree: {csr.nnz / csr.shape[0]:.2f}")

    return Graph(csr=csr, csc=csc, out_deg=out_deg, in_deg=in_deg, num_nodes=csr.shape[0])


def _load_node2vec_embeddings(path: Path, expected_rows: int) -> np.ndarray:
    """Load precomputed node2vec embeddings and return float32 numpy array."""
    if not path.exists():
        raise FileNotFoundError(f"Node2Vec embeddings not found: {path}")

    suffix = path.suffix.lower()
    emb: np.ndarray | None = None

    if suffix == ".pt":
        obj = torch.load(path, map_location="cpu")  # nosec B614 -- local pipeline artifacts
        if isinstance(obj, torch.Tensor):
            emb = obj.detach().cpu().numpy()
        elif isinstance(obj, dict):
            tensor = None
            for key in ("embeddings", "node_embeddings", "tensor", "values"):
                val = obj.get(key)
                if isinstance(val, torch.Tensor):
                    tensor = val
                    break
            if tensor is None:
                raise ValueError(f"Unsupported Node2Vec .pt format: keys={list(obj.keys())}")
            emb = tensor.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported object in {path}: {type(obj)}")
    elif suffix == ".npy":
        emb = np.load(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        if "node_id" in df.columns:
            df = df.sort_values("node_id").reset_index(drop=True)
            df = df.drop(columns=["node_id"])
        emb = df.values
    else:
        raise ValueError(f"Unsupported Node2Vec embedding format: {path.suffix}")

    if emb is None:
        raise ValueError(f"Failed to load Node2Vec embeddings from {path}")

    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Node2Vec embeddings should be 2-D, got shape {emb.shape}")
    if emb.shape[0] != expected_rows:
        raise ValueError(
            f"Node2Vec embeddings row count mismatch: expected {expected_rows}, got {emb.shape[0]}"
        )
    LOGGER.info(f"  Loaded Node2Vec embeddings from {path} with shape {emb.shape}")
    return emb


def load_features(
    feat_path: Path, node2vec_path: Path | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load node features (optionally concatenating Node2Vec embeddings)."""
    LOGGER.info(f"Loading features from {feat_path}")
    df = pd.read_parquet(feat_path)
    if "node_id" in df.columns:
        df = df.sort_values("node_id").reset_index(drop=True)
        feat_df = df.drop(columns=["node_id"])
    else:
        feat_df = df.copy()

    x = feat_df.values.astype(np.float32)

    if node2vec_path is not None:
        node2vec_emb = _load_node2vec_embeddings(node2vec_path, x.shape[0])
        x = np.concatenate([x, node2vec_emb], axis=1)

    LOGGER.info(f"  Features shape: {x.shape}")
    return x, df


def build_snapshots(
    splits_root: Path,
    granularity: str,
    max_snapshots: int | None = None,
) -> tuple[list[tuple[int, int]], list[np.ndarray], int]:
    """
    Build temporal snapshots from train edges.

    Args:
        splits_root: Path to splits directory
        granularity: "quarter" or "annual"
        max_snapshots: Maximum number of snapshots to process (None = all)

    Returns:
        windows: List of (start_ts, end_ts) tuples
        edge_lists: List of edge arrays [2, num_edges]
        num_nodes: Total number of nodes
    """
    LOGGER.info(f"Building {granularity} snapshots from {splits_root}")

    # Load temporal metadata
    meta_path = splits_root.parent / "meta" / "temporal_splits.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Temporal metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    bounds = meta["boundaries"]
    min_ts, T0_end = int(bounds["min_ts"]), int(bounds["T0_end"])

    # Define snapshot windows
    if granularity == "quarter":
        step = 91  # ~3 months
    elif granularity == "annual":
        step = 365  # ~1 year
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    windows = []
    start = min_ts
    while start <= T0_end:
        end = min(start + step - 1, T0_end)
        windows.append((start, end))
        start = end + 1

    if max_snapshots is not None and len(windows) > max_snapshots:
        windows = windows[-max_snapshots:]
        LOGGER.info(f"  Restricted to the most recent {len(windows)} snapshot windows")

    LOGGER.info(f"  Created {len(windows)} snapshot windows")

    # Load train edges
    train_path = splits_root / "train_edges.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train edges not found: {train_path}")

    LOGGER.info(f"  Loading train edges from {train_path}")
    df = pd.read_parquet(train_path, columns=["src_id", "dst_id", "ts"])
    df = df.astype({"src_id": np.int64, "dst_id": np.int64, "ts": np.int64})
    num_nodes = int(max(df.src_id.max(), df.dst_id.max()) + 1)

    LOGGER.info(f"  Train edges: {len(df):,}, Nodes: {num_nodes:,}")

    # Build edge lists per snapshot
    edge_lists = []
    for i, (a, b) in enumerate(windows):
        sub = df[(df.ts >= a) & (df.ts <= b)]
        if len(sub) < 10:
            LOGGER.warning(
                f"  Snapshot {i + 1}/{len(windows)} [{a},{b}]: Only {len(sub)} edges (skipping)"
            )
            continue

        edges = np.vstack([sub.src_id.values, sub.dst_id.values]).astype(np.int64)
        edge_lists.append(edges)

        if (i + 1) % 10 == 0 or i == len(windows) - 1:
            LOGGER.info(f"  Snapshot {i + 1}/{len(windows)} [{a},{b}]: {len(sub):,} edges")

    LOGGER.info(f"  Built {len(edge_lists)} valid snapshots")
    return windows, edge_lists, num_nodes


# ============================= Negative Sampling ============================= #


def sample_negatives_uniform(
    pos_edges: np.ndarray,
    num_nodes: int,
    neg_ratio: float,
    seed: int,
) -> np.ndarray:
    """Sample uniform negative edges."""
    num_pos = pos_edges.shape[1]
    num_neg = int(num_pos * neg_ratio)

    rng = np.random.default_rng(seed)
    src = rng.integers(0, num_nodes, size=num_neg, dtype=np.int64)
    dst = rng.integers(0, num_nodes, size=num_neg, dtype=np.int64)

    return np.vstack([src, dst])


# ============================= Training ============================= #


class ScoreWriter:
    """Stream raw logits to Parquet without loading everything into memory."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: pq.ParquetWriter | None = None

    def write(
        self,
        src_ids: np.ndarray,
        dst_ids: np.ndarray,
        labels: np.ndarray,
        logits_by_name: dict[str, np.ndarray],
    ) -> None:
        data = {
            "src_id": pa.array(src_ids.astype(np.int64, copy=False)),
            "dst_id": pa.array(dst_ids.astype(np.int64, copy=False)),
            "label": pa.array(labels.astype(np.int8, copy=False)),
        }
        for name, values in logits_by_name.items():
            data[name] = pa.array(values.astype(np.float32, copy=False))
        table = pa.table(data)
        if self._writer is None:
            self._writer = pq.ParquetWriter(str(self.path), table.schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def sample_degree_negatives(
    out_deg: np.ndarray,
    in_deg: np.ndarray,
    M: int,
    rng: np.random.Generator,
    deg_exp: float = 1.0,
) -> np.ndarray:
    od = out_deg.astype(np.float64, copy=False)
    idg = in_deg.astype(np.float64, copy=False)
    od = np.clip(od, 1.0, None) ** float(deg_exp)
    idg = np.clip(idg, 1.0, None) ** float(deg_exp)
    pu = od / od.sum()
    pv = idg / idg.sum()
    u = rng.choice(len(od), size=int(M), p=pu)
    v = rng.choice(len(idg), size=int(M), p=pv)
    return np.vstack([u.astype(np.int64), v.astype(np.int64)])


class TwoHopCache:
    def __init__(self, csr: sp.csr_matrix, csc: sp.csc_matrix, cap: int = 200_000):
        self.csr = csr
        self.csc = csc
        self.cap = int(cap)
        self.cache: dict[int, np.ndarray] = {}
        self.order: list[int] = []

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
        two_hop: set[int] = set()
        for w in one:
            out_w = csr.indices[csr.indptr[w] : csr.indptr[w + 1]]
            in_w = csc.indices[csc.indptr[w] : csc.indptr[w + 1]]
            if out_w.size:
                two_hop.update(int(v) for v in out_w)
            if in_w.size:
                two_hop.update(int(v) for v in in_w)
        two_hop.difference_update(int(v) for v in one)
        two_hop.discard(int(u))
        arr = np.array(sorted(two_hop), dtype=np.int64)
        self.cache[int(u)] = arr
        self.order.append(int(u))
        if len(self.order) > self.cap:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return arr


def _is_edge(u: int, v: int, csr: sp.csr_matrix) -> bool:
    start, end = csr.indptr[u], csr.indptr[u + 1]
    neigh = csr.indices[start:end]
    return bool(np.any(neigh == v))


def sample_negatives_mix_v3(
    pos_edges: np.ndarray,
    num_nodes: int,
    neg_ratio: float,
    rng: np.random.Generator,
    graph: Graph,
    twohop_cache: TwoHopCache,
    bucket_ids: np.ndarray,
    bucket_to_nodes: dict[int, np.ndarray],
) -> np.ndarray:
    num_pos = pos_edges.shape[1]
    total_neg = max(int(num_pos * neg_ratio), 1)
    counts_target = {
        "deg": int(total_neg * 0.4),
        "twohop": int(total_neg * 0.2),
        "bucket": int(total_neg * 0.2),
    }
    counts_target["uniform"] = total_neg - sum(counts_target.values())

    pairs: list[np.ndarray] = []
    actual_counts = dict.fromkeys(counts_target, 0)

    # Degree-based
    if counts_target["deg"] > 0:
        deg_arr = sample_degree_negatives(
            graph.out_deg,
            graph.in_deg,
            counts_target["deg"],
            rng,
            deg_exp=1.0,
        )
        pairs.append(deg_arr)
        actual_counts["deg"] = deg_arr.shape[1]

    # Two-hop negatives
    if counts_target["twohop"] > 0:
        negs_two: list[tuple[int, int]] = []
        attempts = 0
        target = counts_target["twohop"]
        src_nodes = pos_edges[0]
        while len(negs_two) < target and attempts < target * 50:
            u = int(rng.choice(src_nodes))
            candidates = twohop_cache.get(u)
            if candidates.size == 0:
                attempts += 1
                continue
            v = int(rng.choice(candidates))
            if v == u or _is_edge(u, v, graph.csr):
                attempts += 1
                continue
            negs_two.append((u, v))
            attempts += 1
        if negs_two:
            arr = np.array(negs_two, dtype=np.int64).T
            pairs.append(arr)
            actual_counts["twohop"] = arr.shape[1]

    # Degree-bucket negatives
    if counts_target["bucket"] > 0:
        negs_bucket: list[tuple[int, int]] = []
        attempts = 0
        target = counts_target["bucket"]
        tgt_nodes = pos_edges[1]
        while len(negs_bucket) < target and attempts < target * 50:
            idx = int(rng.integers(0, tgt_nodes.size))
            u = int(pos_edges[0, idx])
            target_node = int(tgt_nodes[idx])
            bucket = int(bucket_ids[target_node])
            candidates = bucket_to_nodes.get(bucket)
            if candidates is None or candidates.size == 0:
                attempts += 1
                continue
            v = int(rng.choice(candidates))
            if v == u or _is_edge(u, v, graph.csr):
                attempts += 1
                continue
            negs_bucket.append((u, v))
            attempts += 1
        if negs_bucket:
            arr = np.array(negs_bucket, dtype=np.int64).T
            pairs.append(arr)
            actual_counts["bucket"] = arr.shape[1]

    # Uniform fallback (includes filling any deficit)
    produced = sum(arr.shape[1] for arr in pairs) if pairs else 0
    remaining = max(total_neg - produced, 0)
    if remaining > 0:
        src = rng.integers(0, num_nodes, size=remaining, dtype=np.int64)
        dst = rng.integers(0, num_nodes, size=remaining, dtype=np.int64)
        arr = np.vstack([src, dst])
        pairs.append(arr)
        actual_counts["uniform"] = arr.shape[1]

    if not pairs:
        return np.vstack(
            [
                rng.integers(0, num_nodes, size=total_neg, dtype=np.int64),
                rng.integers(0, num_nodes, size=total_neg, dtype=np.int64),
            ]
        )

    negs = np.concatenate(pairs, axis=1)
    if negs.shape[1] > total_neg:
        idx = rng.choice(negs.shape[1], size=total_neg, replace=False)
        negs = negs[:, idx]

    if not getattr(sample_negatives_mix_v3, "_logged", False):
        LOGGER.info(
            "Negative mix_v3 actual counts: deg=%d twohop=%d bucket=%d uniform=%d (target total=%d)",
            actual_counts["deg"],
            actual_counts["twohop"],
            actual_counts["bucket"],
            actual_counts["uniform"],
            total_neg,
        )
        sample_negatives_mix_v3._logged = True

    return negs.astype(np.int64, copy=False)


def _encode_edges(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    return (edges[0].astype(np.int64) * num_nodes) + edges[1].astype(np.int64)


def _prepare_temporal_tasks(
    edge_lists: list[np.ndarray],
    num_nodes: int,
    forecast_horizon: int,
    min_new_edges: int,
) -> list[tuple[int, np.ndarray]]:
    if len(edge_lists) <= forecast_horizon:
        return []

    tasks: list[tuple[int, np.ndarray]] = []
    history: set[int] = set()

    # Seed history with the very first snapshot so we can detect new edges afterwards.
    history.update(_encode_edges(edge_lists[0], num_nodes).tolist())

    for t in range(len(edge_lists) - forecast_horizon):
        target_idx = t + forecast_horizon
        target_edges = edge_lists[target_idx]
        encoded_target = _encode_edges(target_edges, num_nodes)

        new_mask = np.array([edge not in history for edge in encoded_target], dtype=bool)
        new_edges = target_edges[:, new_mask]

        if new_edges.shape[1] >= min_new_edges:
            history_idx = target_idx - 1
            tasks.append((history_idx, new_edges))

        # Update history with edges from the target snapshot regardless of whether they were
        # considered positives; this ensures future iterations treat them as known structure.
        history.update(encoded_target.tolist())

    return tasks


def train_temporal(
    model: TemporalGNN,
    x: Tensor,
    graph: Graph,
    edge_lists: list[np.ndarray],
    epochs: int,
    lr: float,
    neg_ratio: float,
    device: torch.device,
    seed: int = 42,
    weight_decay: float = 1e-5,
    use_amp: bool = True,
    forecast_horizons: list[int] | tuple[int, ...] = (1,),
    min_new_edges: int = 10,
    bucket_ids: np.ndarray | None = None,
    bucket_to_nodes: dict[int, np.ndarray] | None = None,
    twohop_cache: TwoHopCache | None = None,
    head_loss_weights: list[float] | None = None,
    short_head_window: int = 4,
) -> list[float]:
    """
    Train TGNN model on temporal snapshots with stability improvements.

    Args:
        model: TGNN model
        x: Node features [num_nodes, in_channels]
        edge_lists: List of edge arrays for each snapshot
        epochs: Number of training epochs
        lr: Learning rate (max, will use warmup)
        neg_ratio: Negative sampling ratio
        device: Device to train on
        seed: Random seed
        weight_decay: L2 regularization
        use_amp: Use automatic mixed precision (float16) to save memory
        forecast_horizon: Number of snapshots ahead to predict
        min_new_edges: Minimum number of new edges required for a training task

    Returns:
        losses: Training loss per epoch
    """
    amp_enabled = use_amp and torch.cuda.is_available() and device.type == "cuda"
    if amp_enabled:
        LOGGER.info(
            f"Training TGNN for {epochs} epochs (lr={lr:.6f}, neg_ratio={neg_ratio:.1f}, wd={weight_decay:.1e}, AMP=ON)"
        )
    else:
        LOGGER.info(
            f"Training TGNN for {epochs} epochs (lr={lr:.6f}, neg_ratio={neg_ratio:.1f}, wd={weight_decay:.1e})"
        )

    # Optimizer with weight decay for stability
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # GradScaler for mixed precision
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    # Learning rate scheduler: warmup (20%) + cosine decay
    warmup_epochs = max(1, int(0.2 * epochs))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from 0.1x to 1.0x
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            # Cosine decay from 1.0x to 0.1x
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.1 + 0.45 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    LOGGER.info(f"  LR schedule: warmup {warmup_epochs} epochs → cosine decay to {lr * 0.1:.6f}")

    model.train()
    x = x.to(device)
    num_nodes = x.size(0)

    horizons = [int(h) for h in forecast_horizons]
    tasks_per_head: list[list[tuple[int, np.ndarray]]] = []
    for horizon in horizons:
        t = _prepare_temporal_tasks(
            edge_lists,
            num_nodes=num_nodes,
            forecast_horizon=horizon,
            min_new_edges=min_new_edges,
        )
        if not t:
            raise ValueError(
                f"No temporal training tasks generated for forecast horizon={horizon}. "
                "Lower --min-new-edges or adjust the horizon list."
            )
        LOGGER.info(
            "  Prepared %d temporal tasks (forecast_horizon=%d, min_new_edges=%d)",
            len(t),
            horizon,
            min_new_edges,
        )
        tasks_per_head.append(t)

    if model.num_score_heads != len(horizons):
        raise ValueError(
            f"Model score head count ({model.num_score_heads}) does not match number of horizons ({len(horizons)})."
        )

    if bucket_ids is None or bucket_to_nodes is None:
        bucket_ids = np.zeros(num_nodes, dtype=np.int64)
        bucket_to_nodes = {0: np.arange(num_nodes, dtype=np.int64)}

    if twohop_cache is None:
        twohop_cache = TwoHopCache(graph.csr, graph.csc)

    edge_tensor_cache = [torch.from_numpy(edges).long() for edges in edge_lists]
    adj_cache: dict[int, torch.sparse.Tensor] = {}

    def _get_normalized_adj(idx: int) -> torch.sparse.Tensor:
        adj = adj_cache.get(idx)
        if adj is not None and adj.device == device:
            return adj

        edge_tensor = edge_tensor_cache[idx]
        if edge_tensor.device != device:
            edge_tensor = edge_tensor.to(device=device, non_blocking=True)
            edge_tensor_cache[idx] = edge_tensor

        adj = TemporalGNN.normalize_adjacency(edge_tensor, num_nodes)
        adj_cache[idx] = adj
        return adj

    short_head_window = max(1, int(short_head_window))
    num_heads = len(horizons)
    if head_loss_weights is None:
        head_loss_weights = [1.5] + [1.0] * (num_heads - 1)
    else:
        if len(head_loss_weights) != num_heads:
            raise ValueError(
                f"head_loss_weights length ({len(head_loss_weights)}) does not match number of horizons ({num_heads})"
            )

    losses = []

    def _select_indices(history_idx: int, head_idx: int) -> list[int]:
        if head_idx == 0:
            start = max(0, history_idx + 1 - short_head_window)
        else:
            start = 0
        return list(range(start, history_idx + 1))

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses: list[float] = []

        task_pool: list[tuple[int, int, np.ndarray]] = []
        for head_idx, task_list in enumerate(tasks_per_head):
            for history_idx, pos_edges_np in task_list:
                task_pool.append((head_idx, history_idx, pos_edges_np))

        random.shuffle(task_pool)

        for head_idx, history_idx, pos_edges_np in task_pool:
            optimizer.zero_grad()
            idx_range = _select_indices(history_idx, head_idx)
            history_adjs = [_get_normalized_adj(i) for i in idx_range]

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                z_base = model(x, history_adjs)
                skip_features = model.skip_proj(x)
                z_proj = model.head_projections[head_idx](z_base)
                gate = torch.sigmoid(model.head_skip_gates[head_idx])
                z_head = z_proj + gate * skip_features

                num_pos = pos_edges_np.shape[1]
                rng = np.random.default_rng(seed + epoch + history_idx + head_idx * 13)
                neg_edges_np = sample_negatives_mix_v3(
                    pos_edges_np,
                    num_nodes,
                    neg_ratio,
                    rng,
                    graph,
                    twohop_cache,
                    bucket_ids,
                    bucket_to_nodes,
                )

                pos_edges = torch.from_numpy(pos_edges_np).long().to(device)
                neg_edges = torch.from_numpy(neg_edges_np).long().to(device)

                pos_score = (z_head[pos_edges[0]] * z_head[pos_edges[1]]).sum(dim=1)
                neg_score = (z_head[neg_edges[0]] * z_head[neg_edges[1]]).sum(dim=1)

                pairs_per_pos = 2
                total_pairs = num_pos * pairs_per_pos

                if total_pairs <= neg_score.numel():
                    idx = torch.randperm(neg_score.numel(), device=device)[:total_pairs]
                else:
                    idx = torch.randint(0, neg_score.numel(), (total_pairs,), device=device)

                neg_score_sampled = neg_score[idx]
                pos_score_expanded = pos_score.repeat_interleave(pairs_per_pos)

                diff = pos_score_expanded - neg_score_sampled
                loss = -F.logsigmoid(diff).mean()
                loss = loss * float(head_loss_weights[head_idx])

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_losses.append(float(loss.item()))

            # Free per-task tensors
            del history_adjs
            if device.type == "cuda":
                torch.cuda.empty_cache()

        scheduler.step()

        epoch_loss_val = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        losses.append(epoch_loss_val)

        elapsed = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        loss_str = f"{epoch_loss_val:.4f}" if not np.isnan(epoch_loss_val) else "nan"
        LOGGER.info(
            "  Epoch %d/%d: loss=%s, lr=%.6f (%0.1fs)",
            epoch + 1,
            epochs,
            loss_str,
            current_lr,
            elapsed,
        )

    return losses


# ============================= Evaluation (Streaming, like GraphSAGE) ============================= #


class CandidateStreamer:
    """Stream candidate pairs from parquet file in batches (adapted from GraphSAGE)."""

    def __init__(self, path: Path, batch_size: int = 2_000_000):
        self.path = path
        self.batch = int(max(100_000, batch_size))
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.parquet as pq
        except Exception as e:
            raise RuntimeError("pyarrow is required to stream candidate Parquet") from e
        self.pq = pq  # type: ignore

    def __iter__(self):
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
        remainder = None
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
            # Group by source: yield complete source groups
            src_vals = df["src_id"].to_numpy()
            change = np.where(np.diff(src_vals) != 0)[0] + 1
            if change.size == 0:
                # All same source - check if more data
                if end < n:
                    remainder = df
                    start = end
                    continue
                else:
                    yield df
                    break
            last_boundary = int(change[-1])
            if end < n:
                # Keep incomplete group for next batch
                yield df.iloc[:last_boundary]
                remainder = df.iloc[last_boundary:].reset_index(drop=True)
            else:
                yield df
            start = end


# ============================= Metrics Computation (from GraphSAGE) ============================= #

from dataclasses import dataclass, field


@dataclass
class KM:
    """K-Metrics accumulator for ranking evaluation (adapted from GraphSAGE)."""

    Ks: list[int]
    n: int = 0
    hit: dict[int, float] = field(default_factory=dict)
    rec: dict[int, float] = field(default_factory=dict)
    pre: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map: float = 0.0
    ndcg: float = 0.0
    mic_pos: int = 0
    mic_in_top: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        self.hit = dict.fromkeys(self.Ks, 0.0)
        self.rec = dict.fromkeys(self.Ks, 0.0)
        self.pre = dict.fromkeys(self.Ks, 0.0)
        self.mic_in_top = dict.fromkeys(self.Ks, 0)

    def upd(self, y: np.ndarray) -> None:
        """Update metrics with one ranking (y: binary array, 1=relevant, sorted by score)."""
        n, pos_total = y.size, int(y.sum())
        cs = y.cumsum()
        # AP & MRR
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
        """Return metrics as a dict row."""
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
    batch_size: int,
    splits_root: Path,
    collect_slices: bool = True,
    max_sources: int | None = None,
    calibrate: bool = False,
    score_writer: ScoreWriter | None = None,
    head_embeddings: list[Tensor] | None = None,
    head_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Evaluate TGNN on candidates with comprehensive slice metrics (adapted from GraphSAGE).

    Streams candidates, computes scores on-the-fly, and calculates all slice metrics:
    - WW/WC/CW/CC (warm/cold start combinations)
    - twohop/>2hop (graph distance slices)
    - horizon_0_6, horizon_6_12, etc. (time horizon slices)
    - deg_q1, deg_q2, deg_q3, deg_q4 (degree quartile slices)

    Args:
        calibrate: If True, apply sigmoid transformation to scores for probability-like outputs
    """
    LOGGER.info(f"Evaluating {name} split with comprehensive slices...")
    if calibrate:
        LOGGER.info("  Applying sigmoid calibration to scores")

    if head_embeddings is None:
        head_embeddings = [Z]
    head_embeddings = [emb.to(device) for emb in head_embeddings]
    if head_names is None:
        head_names = ["logit"] + [f"logit_head{i + 1}" for i in range(1, len(head_embeddings))]
    primary_head_name = head_names[0]

    macro = KM(Ks=Ks)
    micro = KM(Ks=Ks)
    emit_strict_warm = False
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
            LOGGER.info(f"  Loading timestamp sidecar from {spath.name}")
            df_edges = pd.read_parquet(spath, columns=["src_id", "dst_id", "ts"])  # type: ignore
            ts_sidecar = {(int(s), int(d)): int(t) for s, d, t in df_edges.to_numpy()}

    total_rows = 0
    total_sources = 0
    Z.to(device)
    csr, csc = g.csr, g.csc

    for chunk in streamer:
        if chunk.empty:
            continue
        # Make explicit copy to avoid SettingWithCopyWarning
        chunk = chunk.copy()
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
            labels_raw = sub["label"].to_numpy(copy=False)
            if labels_raw.dtype.kind == "f":
                labels = labels_raw.astype(np.int8)
            else:
                labels = labels_raw.astype(np.int8)

            # Score candidates
            with torch.no_grad():
                raw_logits: dict[str, np.ndarray] = {}
                for idx, emb in enumerate(head_embeddings):
                    zu = emb[u]
                    zv = emb[vs]
                    logits = (zv * zu).sum(dim=1).detach().cpu().numpy()
                    raw_logits[head_names[idx]] = logits

            primary_scores = raw_logits[primary_head_name].copy()

            # Apply Platt calibration if requested
            if calibrate:
                # Apply sigmoid transformation for probability-like scores
                scores = 1 / (1 + np.exp(-primary_scores))
            else:
                scores = primary_scores

            if score_writer is not None:
                logits_payload = dict(raw_logits)
                logits_payload["logit"] = raw_logits.get(primary_head_name, primary_scores)  # type: ignore[assignment]
                score_writer.write(
                    sub["src_id"].to_numpy(dtype=np.int64, copy=False),
                    sub["dst_id"].to_numpy(dtype=np.int64, copy=False),
                    labels.astype(np.int8, copy=False),
                    logits_payload,
                )

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

    LOGGER.info(f"  Evaluated {total_sources:,} sources, {total_rows:,} pairs")
    gdf = pd.DataFrame([macro.row(True, "TGNN")])
    mdf = pd.DataFrame([micro.row(False, "TGNN")])
    sdf = pd.DataFrame([{**km.row(True, "TGNN"), "slice_name": s} for s, km in slices.items()])
    counts = {"rows": int(total_rows), "sources": int(total_sources)}
    return gdf, mdf, sdf, counts


# ============================= Main ============================= #


def main():
    parser = argparse.ArgumentParser(description="TGNN Training and Evaluation")

    # Data paths
    parser.add_argument("--adj", type=str, required=True, help="Path to train adjacency .npz")
    parser.add_argument(
        "--features", type=str, required=True, help="Path to node features .parquet"
    )
    parser.add_argument(
        "--node2vec-emb",
        type=str,
        default=None,
        help="Optional path to precomputed Node2Vec embeddings to concatenate with features",
    )
    parser.add_argument("--candidates-val", type=str, required=True)
    parser.add_argument("--candidates-test", type=str, required=True)
    parser.add_argument("--splits-root", type=str, required=True)
    parser.add_argument(
        "--head-loss-weights",
        type=str,
        default=None,
        help="Comma-separated loss weights for each forecast horizon (defaults to emphasising the shortest)",
    )
    parser.add_argument(
        "--short-head-window",
        type=int,
        default=4,
        help="Number of most recent snapshots to use for the shortest forecast horizon",
    )

    # Model architecture
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--out-channels", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--temporal-mode", type=str, default="attention", choices=["attention", "mean", "last"]
    )

    # Temporal settings
    parser.add_argument("--granularity", type=str, default="quarter", choices=["quarter", "annual"])
    parser.add_argument("--max-snapshots", type=int, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per snapshot")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg-ratio", type=float, default=2.0)

    # Multi-seed
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (overrides multi-seed)")

    # Output
    parser.add_argument("--out-dir", type=str, default="results/tgnn/test")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/tgnn/test")
    parser.add_argument("--best-params-path", type=str, default=None)

    # Evaluation
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--undirected", action="store_true", default=True)
    parser.add_argument(
        "--normalize-emb",
        action="store_true",
        default=False,
        help="L2-normalize embeddings before scoring (disabled by default; only enable if training also uses cosine scores)",
    )

    # Memory management
    parser.add_argument(
        "--snapshot-batch-size",
        type=int,
        default=2,
        help="Max snapshots to process at once (lower = less GPU memory, default: 2)",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision (float16) to reduce memory (default: True)",
    )
    parser.add_argument(
        "--gcn-checkpoint",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing per snapshot to shrink activation memory",
    )
    parser.add_argument(
        "--temporal-decay",
        type=float,
        default=0.08,
        help="Exponential decay rate for temporal recency bias (0=no bias, default: 0.08)",
    )
    parser.add_argument(
        "--checkpoint-threshold",
        type=int,
        default=16000,
        help="Auto-enable checkpointing when hidden*out ≥ threshold (≤0 disables)",
    )
    parser.add_argument(
        "--forecast-horizons",
        type=str,
        default="1,4",
        help="Comma-separated list of snapshot horizons to predict (e.g. '1,4')",
    )
    parser.add_argument(
        "--min-new-edges",
        type=int,
        default=25,
        help="Minimum number of new edges required to keep a temporal training batch",
    )

    # Device
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    args.forecast_horizons = [
        int(h.strip()) for h in str(args.forecast_horizons).split(",") if h.strip()
    ] or [1]

    if args.head_loss_weights:
        head_loss_weights = [
            float(w.strip()) for w in str(args.head_loss_weights).split(",") if w.strip()
        ]
    else:
        head_loss_weights = None

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    LOGGER.info("=" * 60)
    LOGGER.info("TGNN Training and Evaluation")
    LOGGER.info("=" * 60)
    LOGGER.info(f"Device: {device}")
    LOGGER.info(f"Granularity: {args.granularity}")
    LOGGER.info(f"Max snapshots: {args.max_snapshots or 'all'}")
    LOGGER.info(f"Forecast horizons: {args.forecast_horizons}")

    # Load best params if available
    if args.best_params_path and Path(args.best_params_path).exists():
        LOGGER.info(f"Loading best params from {args.best_params_path}")
        best_params = json.loads(Path(args.best_params_path).read_text())
        if "best_params" in best_params:
            best_params = best_params["best_params"]

        args.hidden = int(best_params.get("hidden", args.hidden))
        args.out_channels = int(best_params.get("out_channels", args.out_channels))
        args.num_layers = int(best_params.get("num_layers", args.num_layers))
        args.dropout = float(best_params.get("dropout", args.dropout))
        args.lr = float(best_params.get("lr", args.lr))
        args.temporal_mode = str(best_params.get("temporal_mode", args.temporal_mode))
        args.epochs = int(best_params.get("epochs", args.epochs))

        args.temporal_decay = float(best_params.get("temporal_decay", args.temporal_decay))
        fh_param = best_params.get("forecast_horizons") or best_params.get("forecast_horizon")
        if fh_param is not None:
            if isinstance(fh_param, (list, tuple)):
                args.forecast_horizons = [int(v) for v in fh_param]
            else:
                args.forecast_horizons = [
                    int(v.strip()) for v in str(fh_param).split(",") if v.strip()
                ]

        LOGGER.info(
            f"  Loaded: hidden={args.hidden}, out={args.out_channels}, layers={args.num_layers}"
        )
        LOGGER.info(
            f"          dropout={args.dropout:.3f}, lr={args.lr:.6f}, mode={args.temporal_mode}, epochs={args.epochs}"
        )
        LOGGER.info(f"          temporal_decay={args.temporal_decay:.3f}")
        LOGGER.info(f"          forecast_horizons={args.forecast_horizons}")

    # Setup output directories
    out_dir = Path(args.out_dir)
    artifacts_dir = Path(args.artifacts_dir)
    _ensure_dir(out_dir)
    _ensure_dir(artifacts_dir)

    # Load data
    graph = load_graph(Path(args.adj), args.undirected)
    node2vec_path = Path(args.node2vec_emb) if args.node2vec_emb else None
    x_np, _feat_df = load_features(Path(args.features), node2vec_path=node2vec_path)
    x = torch.from_numpy(x_np).float()

    # Degree buckets for negative sampling
    if graph.in_deg.size:
        quantiles = np.quantile(graph.in_deg, [0.25, 0.5, 0.75])
        bucket_ids = np.digitize(graph.in_deg, quantiles, right=True)
    else:
        bucket_ids = np.zeros(graph.num_nodes, dtype=np.int64)
    bucket_to_nodes: dict[int, np.ndarray] = {}
    for bucket in np.unique(bucket_ids):
        nodes = np.where(bucket_ids == bucket)[0]
        if nodes.size:
            bucket_to_nodes[int(bucket)] = nodes.astype(np.int64)
    twohop_cache = TwoHopCache(graph.csr, graph.csc)

    # Build temporal snapshots
    _windows, edge_lists, _num_nodes_check = build_snapshots(
        Path(args.splits_root),
        args.granularity,
        args.max_snapshots,
    )

    # Handle seeds
    if args.seed is not None:
        seeds = [args.seed]
    elif args.multi_seed:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = [42]

    LOGGER.info(f"Running {len(seeds)} seed(s): {seeds}")

    # Run for each seed
    for seed_idx, seed in enumerate(seeds):
        LOGGER.info("=" * 60)
        LOGGER.info(f"SEED {seed} ({seed_idx + 1}/{len(seeds)})")
        LOGGER.info("=" * 60)

        # Setup seed-specific directories
        if len(seeds) > 1:
            seed_out_dir = out_dir / f"seed_{seed}"
            seed_artifacts_dir = artifacts_dir / f"seed_{seed}"
        else:
            seed_out_dir = out_dir
            seed_artifacts_dir = artifacts_dir

        _ensure_dir(seed_out_dir)
        _ensure_dir(seed_artifacts_dir)

        # Set random seeds (full four-call pattern for multi-GPU reproducibility)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create model
        model = TemporalGNN(
            in_channels=x.size(1),
            hidden_channels=args.hidden,
            out_channels=args.out_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            temporal_mode=args.temporal_mode,
            snapshot_batch_size=args.snapshot_batch_size,
            checkpoint_snapshots=args.gcn_checkpoint,
            checkpoint_threshold=args.checkpoint_threshold,
            temporal_decay=args.temporal_decay,
            num_score_heads=len(args.forecast_horizons),
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        LOGGER.info(f"Model parameters: {num_params:,}")

        # Train
        train_start = time.time()
        losses = train_temporal(
            model,
            x,
            graph,
            edge_lists,
            epochs=args.epochs,
            lr=args.lr,
            neg_ratio=args.neg_ratio,
            device=device,
            seed=seed,
            use_amp=args.use_amp,
            forecast_horizons=args.forecast_horizons,
            min_new_edges=args.min_new_edges,
            bucket_ids=bucket_ids,
            bucket_to_nodes=bucket_to_nodes,
            twohop_cache=twohop_cache,
            head_loss_weights=head_loss_weights,
            short_head_window=args.short_head_window,
        )
        train_time = time.time() - train_start

        LOGGER.info(f"Training complete ({train_time:.1f}s)")
        if losses:
            LOGGER.info(f"  Initial loss: {losses[0]:.4f}")
            LOGGER.info(f"  Final loss: {losses[-1]:.4f}")
            LOGGER.info(f"  Improvement: {losses[0] - losses[-1]:.4f}")
        else:
            LOGGER.info("  No losses recorded (check temporal task configuration)")

        # Save model
        model_path = seed_artifacts_dir / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "in_channels": x.size(1),
                    "hidden_channels": args.hidden,
                    "out_channels": args.out_channels,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "temporal_mode": args.temporal_mode,
                },
                "losses": losses,
                "seed": seed,
            },
            model_path,
        )
        LOGGER.info(f"  Model saved to {model_path}")

        # Evaluate with comprehensive slices (streaming evaluation like GraphSAGE)
        LOGGER.info("Getting final embeddings...")
        model.eval()
        x_dev = x.to(device)
        num_nodes_eval = x_dev.size(0)
        edge_adjs = [
            TemporalGNN.normalize_adjacency(
                torch.from_numpy(edges).long().to(device),
                num_nodes_eval,
            )
            for edges in edge_lists
        ]
        with torch.no_grad():
            skip_eval = model.skip_proj(x_dev)
            head_embs: list[Tensor] = []
            total_snapshots = len(edge_adjs)
            for head_idx in range(len(args.forecast_horizons)):
                if head_idx == 0:
                    start = max(0, total_snapshots - args.short_head_window)
                    subset = edge_adjs[start:]
                else:
                    subset = edge_adjs
                Z_head = model(x_dev, subset)
                proj = model.head_projections[head_idx](Z_head)
                gate = torch.sigmoid(model.head_skip_gates[head_idx])
                emb = proj + gate * skip_eval
                if args.normalize_emb:
                    emb = F.normalize(emb, p=2, dim=1)
                head_embs.append(emb)
            if args.normalize_emb:
                LOGGER.info("  Applied L2 normalization to embeddings")
            head_names = ["logit"]
            if len(head_embs) >= 2:
                head_names.append("logit_long")
            for idx in range(2, len(head_embs)):
                head_names.append(f"logit_head{idx + 1}")

        # Define K values for metrics
        Ks = [1, 10, 50, 100]

        primary_embedding = head_embs[0]
        long_embedding = head_embs[1] if len(head_embs) > 1 else None

        # Val
        LOGGER.info("Evaluating validation set...")
        val_writer = ScoreWriter(seed_out_dir / "scores_val.parquet")
        try:
            val_global, val_micro, val_slices, _val_counts = evaluate_split(
                name="val",
                Z=primary_embedding,
                device=device,
                g=graph,
                Ks=Ks,
                cand_path=Path(args.candidates_val),
                batch_size=2_000_000,
                splits_root=Path(args.splits_root),
                collect_slices=True,
                calibrate=args.calibrate,
                score_writer=val_writer,
                head_embeddings=head_embs,
                head_names=head_names,
            )
        finally:
            val_writer.close()
        val_global.to_csv(seed_out_dir / "global_val.csv", index=False)
        val_micro.to_csv(seed_out_dir / "micro_val.csv", index=False)
        val_slices.to_csv(seed_out_dir / "slices_val.csv", index=False)
        LOGGER.info(f"  Val: NDCG@100={val_global['ndcg@100'].iloc[0]:.6f}")
        LOGGER.info(f"  Val: saved metrics to {seed_out_dir}")

        # Test
        LOGGER.info("Evaluating test set...")
        test_writer = ScoreWriter(seed_out_dir / "scores_test.parquet")
        try:
            test_global, test_micro, test_slices, _test_counts = evaluate_split(
                name="test",
                Z=primary_embedding,
                device=device,
                g=graph,
                Ks=Ks,
                cand_path=Path(args.candidates_test),
                batch_size=2_000_000,
                splits_root=Path(args.splits_root),
                collect_slices=True,
                calibrate=args.calibrate,
                score_writer=test_writer,
                head_embeddings=head_embs,
                head_names=head_names,
            )
        finally:
            test_writer.close()
        test_global.to_csv(seed_out_dir / "global_test.csv", index=False)
        test_micro.to_csv(seed_out_dir / "micro_test.csv", index=False)
        test_slices.to_csv(seed_out_dir / "slices_test.csv", index=False)
        LOGGER.info(f"  Test: NDCG@100={test_global['ndcg@100'].iloc[0]:.6f}")
        LOGGER.info(f"  Test: saved metrics to {seed_out_dir}")

        if long_embedding is not None:
            LOGGER.info("Evaluating long-horizon head (head 2)...")
            val_writer_long = ScoreWriter(seed_out_dir / "scores_val_long.parquet")
            try:
                val_long_global, val_long_micro, val_long_slices, _ = evaluate_split(
                    name="val",
                    Z=long_embedding,
                    device=device,
                    g=graph,
                    Ks=Ks,
                    cand_path=Path(args.candidates_val),
                    batch_size=2_000_000,
                    splits_root=Path(args.splits_root),
                    collect_slices=True,
                    calibrate=args.calibrate,
                    score_writer=val_writer_long,
                    head_embeddings=[long_embedding],
                    head_names=["logit_long"],
                )
            finally:
                val_writer_long.close()
            val_long_global.to_csv(seed_out_dir / "global_val_long.csv", index=False)
            val_long_micro.to_csv(seed_out_dir / "micro_val_long.csv", index=False)
            val_long_slices.to_csv(seed_out_dir / "slices_val_long.csv", index=False)
            LOGGER.info(f"  Val (long head): NDCG@100={val_long_global['ndcg@100'].iloc[0]:.6f}")

            test_writer_long = ScoreWriter(seed_out_dir / "scores_test_long.parquet")
            try:
                test_long_global, test_long_micro, test_long_slices, _ = evaluate_split(
                    name="test",
                    Z=long_embedding,
                    device=device,
                    g=graph,
                    Ks=Ks,
                    cand_path=Path(args.candidates_test),
                    batch_size=2_000_000,
                    splits_root=Path(args.splits_root),
                    collect_slices=True,
                    calibrate=args.calibrate,
                    score_writer=test_writer_long,
                    head_embeddings=[long_embedding],
                    head_names=["logit_long"],
                )
            finally:
                test_writer_long.close()
            test_long_global.to_csv(seed_out_dir / "global_test_long.csv", index=False)
            test_long_micro.to_csv(seed_out_dir / "micro_test_long.csv", index=False)
            test_long_slices.to_csv(seed_out_dir / "slices_test_long.csv", index=False)
            LOGGER.info(f"  Test (long head): NDCG@100={test_long_global['ndcg@100'].iloc[0]:.6f}")

        # Summary
        summary = {
            "seed": seed,
            "config": vars(args),
            "train_time": train_time,
            "num_snapshots": len(edge_lists),
            "num_params": num_params,
        }
        if losses:
            summary["initial_loss"] = float(losses[0])
            summary["final_loss"] = float(losses[-1])
        else:
            summary["initial_loss"] = None
            summary["final_loss"] = None

        summary_path = seed_artifacts_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        LOGGER.info(f"  Summary saved to {summary_path}")

        # Aggressive GPU memory cleanup between seeds
        del model, losses
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        LOGGER.info("  GPU memory released for next seed")

    LOGGER.info("=" * 60)
    LOGGER.info("All seeds complete!")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
