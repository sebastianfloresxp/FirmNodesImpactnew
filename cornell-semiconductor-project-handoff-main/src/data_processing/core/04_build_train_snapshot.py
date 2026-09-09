#!/usr/bin/env python3
"""
Core Pipeline - Phase 4: Build Train Snapshot (Adjacency)

Builds directed CSR adjacency from the train split only, with degrees and optional
CSC transpose and neighbor cache sidecar. Strictly leakage-safe (train edges only).

Outputs (under --out-root):
  - adjacency/train_adj_T0.npz              (CSR, directed)
  - adjacency/train_adj_T0_csc.npz          (optional, CSC)
  - adjacency/out_degree.npy, in_degree.npy
  - adjacency/neighbor_cache_<K>.pt         (optional, torch int64 [N, K], -1 padded)
  - meta/adjacency_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Handle both module and script execution
try:
    from ._common.run_utils import ensure_run_root, write_json
except ImportError:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).parent))
    from _common.run_utils import ensure_run_root, write_json


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")
logger = logging.getLogger("core.snapshot")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build train-only adjacency snapshot")
    p.add_argument("--out-root", required=True, type=str)
    p.add_argument(
        "--train-file",
        type=str,
        help="Override train split parquet; default under --out-root/splits",
    )
    p.add_argument(
        "--mapping-file",
        type=str,
        help="Override mapping parquet; default under --out-root/mapping",
    )
    p.add_argument("--emit-csc", action="store_true", help="Also write CSC transpose")
    p.add_argument("--emit-undirected", action="store_true", help="Also write undirected adjacency")
    # Default to building a lightweight neighbor cache (K=50) unless explicitly disabled with 0
    p.add_argument(
        "--emit-neighbor-cache",
        type=int,
        default=50,
        help="Write neighbor cache with K neighbors per node (use 0 to disable)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _setup_logger(level: str) -> None:
    try:
        logging.getLogger().setLevel(getattr(logging, level.upper()))
    except Exception:
        logging.getLogger().setLevel(logging.INFO)


def _checksum_npz(path: Path) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_csr(num_nodes: int, edges: np.ndarray) -> sp.csr_matrix:
    # edges: shape [E, 2] with (src_id, dst_id)
    data = np.ones(edges.shape[0], dtype=np.uint8)
    rows = edges[:, 0].astype(np.int32)
    cols = edges[:, 1].astype(np.int32)
    csr = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.uint8)
    csr.sum_duplicates()  # ensure unique entries
    return csr


def _neighbor_cache_from_csr(csr: sp.csr_matrix, K: int) -> np.ndarray:
    N = csr.shape[0]  # type: ignore[index]
    # int32 is sufficient for node ids at this scale and halves storage
    cache = np.full((N, K), -1, dtype=np.int32)
    indptr = csr.indptr
    indices = csr.indices.astype(np.int32, copy=False)
    for i in range(N):
        start, end = indptr[i], indptr[i + 1]
        neigh = indices[start:end]
        if neigh.size:
            take = neigh[:K]
            cache[i, : take.size] = take
    return cache


def main() -> None:
    args = _parse_args()
    _setup_logger(args.log_level)

    out_root = Path(args.out_root)
    ensure_run_root(out_root, force=True, dry_run=args.dry_run)

    train_path = (
        Path(args.train_file) if args.train_file else (out_root / "splits" / "train_edges.parquet")
    )
    mapping_path = (
        Path(args.mapping_file)
        if args.mapping_file
        else (out_root / "mapping" / "entity_map.parquet")
    )

    adj_dir = out_root / "adjacency"
    meta_dir = out_root / "meta"
    if not args.dry_run:
        adj_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")

    logger.info("=== CORE TRAIN SNAPSHOT ===")
    logger.info(f"train_file:   {train_path}")
    logger.info(f"mapping_file: {mapping_path}")
    logger.info(
        f"emit_csc={args.emit_csc}, emit_undirected={args.emit_undirected}, neighbor_cache_K={args.emit_neighbor_cache}"
    )

    # Load mapping for node count
    num_nodes = int(pd.read_parquet(mapping_path).shape[0])

    # Load train edges
    df = pd.read_parquet(train_path, columns=["src_id", "dst_id", "ts"])
    # Validate ids
    bad = (~df["src_id"].between(0, num_nodes - 1)) | (~df["dst_id"].between(0, num_nodes - 1))
    if bad.any():
        raise ValueError(
            f"Found ids out of range [0,{num_nodes - 1}] in train edges: {int(bad.sum())} rows"
        )

    # Deduplicate (src_id, dst_id)
    ed = df[["src_id", "dst_id"]].drop_duplicates().to_numpy(copy=False)
    logger.info(f"Unique train edges: {ed.shape[0]:,}")

    if args.dry_run:
        logger.info("[dry-run] Would build CSR of shape (%d, %d)", num_nodes, num_nodes)
        return

    # Build CSR
    csr = _build_csr(num_nodes, ed)
    csr_path = adj_dir / "train_adj_T0.npz"
    sp.save_npz(csr_path, csr, compressed=True)

    # Degrees
    out_degree = np.diff(csr.indptr).astype(np.int32)
    in_degree = np.diff(csr.tocsc().indptr).astype(np.int32)
    np.save(adj_dir / "out_degree.npy", out_degree)
    np.save(adj_dir / "in_degree.npy", in_degree)

    # Optional CSC transpose
    csc_path: Path | None = None
    if args.emit_csc:
        csc = csr.tocsc(copy=True)
        csc_path = adj_dir / "train_adj_T0_csc.npz"
        sp.save_npz(csc_path, csc, compressed=True)

    # Optional undirected (symmetrized)
    undirected_path: Path | None = None
    if args.emit_undirected:
        und = csr.maximum(csr.transpose())
        undirected_path = adj_dir / "train_adj_T0_undirected.npz"
        sp.save_npz(undirected_path, und, compressed=True)

    # Optional neighbor cache
    cache_path: Path | None = None
    if args.emit_neighbor_cache is not None and args.emit_neighbor_cache > 0:
        K = int(args.emit_neighbor_cache)
        cache_np = _neighbor_cache_from_csr(csr, K)
        try:
            import torch  # lazy import

            cache_t = torch.from_numpy(cache_np)  # dtype=int32
            cache_path = adj_dir / f"neighbor_cache_{K}.pt"
            torch.save(cache_t, cache_path)
        except Exception as e:
            logger.warning(
                f"Failed to write torch neighbor cache; writing .npy instead. Error: {e}"
            )
            cache_path = adj_dir / f"neighbor_cache_{K}.npy"
            np.save(cache_path, cache_np)

    # Meta
    meta: dict[str, Any] = {
        "node_count": num_nodes,
        "edge_count_unique": int(ed.shape[0]),
        "directed": True,
        "files": {
            "csr": str(csr_path),
            "csc": str(csc_path) if csc_path else None,
            "undirected": str(undirected_path) if undirected_path else None,
            "out_degree": str(adj_dir / "out_degree.npy"),
            "in_degree": str(adj_dir / "in_degree.npy"),
            "neighbor_cache": str(cache_path) if cache_path else None,
        },
        "checksums": {
            "csr_sha1": _checksum_npz(csr_path),
            "csc_sha1": _checksum_npz(csc_path) if csc_path else None,
            "undirected_sha1": _checksum_npz(undirected_path) if undirected_path else None,
        },
        "train_ts": {
            "min": int(df["ts"].min()) if len(df) else None,
            "max": int(df["ts"].max()) if len(df) else None,
        },
        "degree_sums": {
            "sum_out": int(out_degree.sum()),
            "sum_in": int(in_degree.sum()),
        },
    }
    write_json(meta_dir / "adjacency_stats.json", meta)

    logger.info("=== TRAIN SNAPSHOT COMPLETE ===")
    logger.info(f"CSR: {csr_path}")
    if csc_path:
        logger.info(f"CSC: {csc_path}")
    if undirected_path:
        logger.info(f"Undirected: {undirected_path}")
    if cache_path:
        logger.info(f"Neighbor cache: {cache_path}")


if __name__ == "__main__":
    main()
