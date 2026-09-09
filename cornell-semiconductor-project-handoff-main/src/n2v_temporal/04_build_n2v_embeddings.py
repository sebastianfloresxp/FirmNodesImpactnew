#!/usr/bin/env python3
"""Build per-snapshot Node2Vec embeddings for the temporal TGNN pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn import Node2Vec

# Reuse snapshot builder from the main temporal pipeline
from n2v_temporal.run_n2v_temporal_eval import build_snapshots

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Node2Vec on each temporal snapshot and export embeddings"
    )
    parser.add_argument(
        "--splits-root",
        required=True,
        type=str,
        help="Path to temporal splits directory (contains train_edges.parquet)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/n2v_temporal/node2vec_snapshots",
        help="Directory where snapshot_XXX.npy files will be written",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="quarter",
        choices=["quarter", "annual"],
        help="Snapshot granularity to match TGNN training",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="Optional cap on number of snapshots (more recent snapshots kept)",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=128, help="Node2Vec embedding dimensionality"
    )
    parser.add_argument("--walk-length", type=int, default=40, help="Length of random walks")
    parser.add_argument("--context-size", type=int, default=10, help="Skip-gram context size")
    parser.add_argument(
        "--walks-per-node", type=int, default=10, help="Number of walks to start at each node"
    )
    parser.add_argument(
        "--num-negative-samples",
        type=int,
        default=1,
        help="Number of negative samples in skip-gram loss",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per snapshot")
    parser.add_argument("--batch-size", type=int, default=128, help="Random-walk batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Node2Vec DataLoader workers")
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate for SparseAdam optimizer"
    )
    parser.add_argument("--p", type=float, default=1.0, help="Return hyper-parameter p")
    parser.add_argument("--q", type=float, default=1.0, help="In-out hyper-parameter q")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cpu', or explicit CUDA device)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--undirected",
        dest="undirected",
        action="store_true",
        help="Treat snapshots as undirected by symmetrising edges",
    )
    parser.add_argument(
        "--directed", dest="undirected", action="store_false", help="Keep snapshot edges directed"
    )
    parser.set_defaults(undirected=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute embeddings even if snapshot file already exists",
    )
    parser.add_argument(
        "--log-every", type=int, default=5, help="Epoch interval for logging training loss"
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _prepare_edge_index(edges: np.ndarray, num_nodes: int, undirected: bool) -> torch.Tensor:
    edge_index = torch.from_numpy(edges).long()
    if undirected:
        rev = edge_index.flip(0)
        edge_index = torch.cat([edge_index, rev], dim=1)
    # Remove self-loops if all nodes isolated (rare)
    mask = edge_index[0] != edge_index[1]
    if mask.sum().item() == 0:
        return edge_index[:, :0]
    edge_index = edge_index[:, mask]
    return edge_index


def _train_node2vec(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    cfg: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    if edge_index.size(1) == 0:
        LOGGER.warning("Snapshot has no usable edges; returning random Gaussian embeddings")
        emb = np.random.normal(loc=0.0, scale=0.01, size=(num_nodes, cfg.embedding_dim)).astype(
            np.float32
        )
        return emb, {"epochs": 0, "training_time_sec": 0.0, "final_loss": None}

    model = Node2Vec(
        edge_index=edge_index.to(device),
        embedding_dim=cfg.embedding_dim,
        walk_length=cfg.walk_length,
        context_size=cfg.context_size,
        walks_per_node=cfg.walks_per_node,
        p=cfg.p,
        q=cfg.q,
        num_negative_samples=cfg.num_negative_samples,
        num_nodes=num_nodes,
        sparse=True,
    ).to(device)

    optimizer = torch.optim.SparseAdam(model.parameters(), lr=cfg.lr)
    loader = model.loader(batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    model.train()
    start = time.time()
    losses: list[float] = []
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for pos_rw, neg_rw in loader:
            pos_rw = pos_rw.to(device)
            neg_rw = neg_rw.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
        losses.append(epoch_loss)
        if cfg.log_every and ((epoch + 1) % cfg.log_every == 0 or epoch == 0):
            LOGGER.info("  epoch %d/%d loss=%.4f", epoch + 1, cfg.epochs, epoch_loss)

    duration = time.time() - start
    model.eval()
    with torch.no_grad():
        embeddings = model.embedding.weight.detach().cpu().numpy()

    stats = {
        "epochs": cfg.epochs,
        "training_time_sec": duration,
        "final_loss": losses[-1] if losses else None,
        "avg_loss": float(np.mean(losses)) if losses else None,
    }
    return embeddings, stats


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _set_seed(args.seed)
    device = _init_device(args.device)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Output directory: %s", args.out_dir)

    splits_root = Path(args.splits_root)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    LOGGER.info(
        "Building temporal snapshots from %s (granularity=%s, max=%s)",
        splits_root,
        args.granularity,
        args.max_snapshots or "all",
    )
    _windows, edge_lists, num_nodes = build_snapshots(
        splits_root=splits_root,
        granularity=args.granularity,
        max_snapshots=args.max_snapshots,
    )
    LOGGER.info("Loaded %d snapshots covering %d nodes", len(edge_lists), num_nodes)

    snapshot_stats: list[dict[str, object]] = []
    for idx, edges in enumerate(edge_lists):
        snap_name = f"snapshot_{idx:03d}.npy"
        out_path = out_dir / snap_name
        if out_path.exists() and not args.overwrite:
            LOGGER.info("[%s] already exists, skipping (use --overwrite to recompute)", snap_name)
            snapshot_stats.append(
                {
                    "snapshot": snap_name,
                    "status": "skipped",
                    "num_edges": int(edges.shape[1]),
                }
            )
            continue

        LOGGER.info("[%s] Training Node2Vec (%d edges)...", snap_name, edges.shape[1])
        edge_index = _prepare_edge_index(edges, num_nodes=num_nodes, undirected=args.undirected)
        embeddings, stats = _train_node2vec(edge_index, num_nodes, device, args)
        np.save(out_path, embeddings.astype(np.float32, copy=False))
        LOGGER.info("[%s] Saved embeddings to %s", snap_name, out_path)

        stats_payload: dict[str, object] = {
            "snapshot": snap_name,
            "num_edges": int(edges.shape[1]),
            "status": "ok",
        }
        stats_payload.update(stats)
        snapshot_stats.append(stats_payload)

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits_root": str(splits_root),
        "granularity": args.granularity,
        "max_snapshots": args.max_snapshots,
        "num_snapshots": len(edge_lists),
        "num_nodes": int(num_nodes),
        "config": {
            "embedding_dim": args.embedding_dim,
            "walk_length": args.walk_length,
            "context_size": args.context_size,
            "walks_per_node": args.walks_per_node,
            "num_negative_samples": args.num_negative_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "p": args.p,
            "q": args.q,
            "undirected": args.undirected,
            "device": str(device),
            "seed": args.seed,
        },
        "snapshots": snapshot_stats,
    }

    meta_path = out_dir / "node2vec_snapshot_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    LOGGER.info("Metadata saved to %s", meta_path)

    LOGGER.info(
        "Completed Node2Vec snapshot generation (%d snapshots processed)", len(snapshot_stats)
    )


if __name__ == "__main__":
    main()
