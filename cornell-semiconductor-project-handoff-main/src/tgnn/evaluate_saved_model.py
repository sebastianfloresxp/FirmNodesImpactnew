#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgnn.model import TemporalGNN
from tgnn.run_tgnn_eval import (
    ScoreWriter,
    build_snapshots,
    evaluate_split,
    load_features,
    load_graph,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved TGNN model heads")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--adj", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--candidates-val", type=str, required=True)
    parser.add_argument("--candidates-test", type=str, required=True)
    parser.add_argument("--splits-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--node2vec-emb", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--short-head-window", type=int, default=2)
    return parser.parse_args()


def _select_indices(history_idx: int, head_idx: int, window: int, total: int) -> list[int]:
    if head_idx == 0:
        start = max(0, total - window)
        return list(range(start, total))
    return list(range(total))


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model_path)
    state = torch.load(model_path, map_location="cpu")  # nosec B614 -- local pipeline artifacts
    config = state.get("config", {})

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    graph = load_graph(Path(args.adj), undirected=True)
    x_np, _ = load_features(
        Path(args.features), node2vec_path=Path(args.node2vec_emb) if args.node2vec_emb else None
    )
    x = torch.from_numpy(x_np).float()

    model = TemporalGNN(
        in_channels=config.get("in_channels", x.size(1)),
        hidden_channels=config.get("hidden_channels", 256),
        out_channels=config.get("out_channels", 128),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.0),
        temporal_mode=config.get("temporal_mode", "attention"),
        snapshot_batch_size=config.get("snapshot_batch_size", 2),
        checkpoint_snapshots=config.get("checkpoint_snapshots", False),
        checkpoint_threshold=config.get("checkpoint_threshold", 16000),
        temporal_decay=config.get("temporal_decay", 0.135),
        num_score_heads=config.get("num_score_heads", 2),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    x_dev = x.to(device)
    _windows, edge_lists, _ = build_snapshots(Path(args.splits_root), "quarter")
    edge_adjs: list[torch.sparse.Tensor] = []
    for edges in edge_lists:
        edge_tensor = torch.from_numpy(edges).long().to(device)
        adj = TemporalGNN.normalize_adjacency(edge_tensor, x_dev.size(0))
        edge_adjs.append(adj)

    short_window = max(1, int(args.short_head_window))
    skip_eval = model.skip_proj(x_dev)

    head_embs: list[torch.Tensor] = []
    total_snapshots = len(edge_adjs)
    for head_idx in range(model.num_score_heads):
        idx_range = _select_indices(total_snapshots - 1, head_idx, short_window, total_snapshots)
        subset = [edge_adjs[i] for i in idx_range]
        with torch.no_grad():
            z_head = model(x_dev, subset)
            proj = model.head_projections[head_idx](z_head)
            gate = torch.sigmoid(model.head_skip_gates[head_idx])
            emb = proj + gate * skip_eval
        head_embs.append(emb)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_emb = head_embs[0]
    long_emb = head_embs[1] if len(head_embs) > 1 else None

    Ks = [1, 10, 50, 100]

    def _run_eval(prefix: str, emb: torch.Tensor) -> None:
        writer_val = ScoreWriter(out_dir / f"scores_val{prefix}.parquet")
        try:
            val_global, val_micro, val_slices, _ = evaluate_split(
                name="val",
                Z=emb,
                device=device,
                g=graph,
                Ks=Ks,
                cand_path=Path(args.candidates_val),
                batch_size=2_000_000,
                splits_root=Path(args.splits_root),
                collect_slices=True,
                calibrate=args.calibrate,
                score_writer=writer_val,
                head_embeddings=[emb],
                head_names=["logit" if not prefix else f"logit{prefix}"],
            )
        finally:
            writer_val.close()
        val_global.to_csv(out_dir / f"global_val{prefix}.csv", index=False)
        val_micro.to_csv(out_dir / f"micro_val{prefix}.csv", index=False)
        val_slices.to_csv(out_dir / f"slices_val{prefix}.csv", index=False)

        writer_test = ScoreWriter(out_dir / f"scores_test{prefix}.parquet")
        try:
            test_global, test_micro, test_slices, _ = evaluate_split(
                name="test",
                Z=emb,
                device=device,
                g=graph,
                Ks=Ks,
                cand_path=Path(args.candidates_test),
                batch_size=2_000_000,
                splits_root=Path(args.splits_root),
                collect_slices=True,
                calibrate=args.calibrate,
                score_writer=writer_test,
                head_embeddings=[emb],
                head_names=["logit" if not prefix else f"logit{prefix}"],
            )
        finally:
            writer_test.close()
        test_global.to_csv(out_dir / f"global_test{prefix}.csv", index=False)
        test_micro.to_csv(out_dir / f"micro_test{prefix}.csv", index=False)
        test_slices.to_csv(out_dir / f"slices_test{prefix}.csv", index=False)

    _run_eval("", primary_emb)
    if long_emb is not None:
        _run_eval("_long", long_emb)


if __name__ == "__main__":
    main()
