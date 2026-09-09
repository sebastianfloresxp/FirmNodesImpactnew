from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.ensemble.utils import (
    ParquetAppend,
    apply_platt,
    load_calibration,
    load_json,
    now_iso,
)


def parse_seeds(root: Path, explicit: str | None) -> list[Path]:
    if explicit:
        seeds: list[Path] = []
        for item in explicit.split(","):
            item = item.strip()
            if not item:
                continue
            if item.isdigit():
                pattern = f"seed_{int(item)}"
                candidate = root / pattern
                if not candidate.exists():
                    raise FileNotFoundError(f"Seed directory not found: {candidate}")
                seeds.append(candidate)
            else:
                candidate = root / item
                if not candidate.exists():
                    raise FileNotFoundError(f"Seed directory not found: {candidate}")
                seeds.append(candidate)
        if not seeds:
            raise RuntimeError("No seeds resolved from explicit list")
        return seeds
    seeds = sorted(p for p in root.glob("seed_*") if p.is_dir())
    if not seeds:
        raise RuntimeError(f"No seed directories found under {root}")
    return seeds


def resolve_summary(path: Path) -> dict[str, object]:
    summary = path / "summary.json"
    if not summary.exists():
        return {}
    try:
        return load_json(summary)
    except Exception:
        return {}


def ensure_tmpdir() -> None:
    if "TMPDIR" not in os.environ:
        os.environ["TMPDIR"] = "/tmp"  # nosec B108 -- ephemeral HPO trial scratch space


def load_candidate_paths(summary: dict[str, object], split: str) -> str | None:
    candidates = summary.get("candidates", {}) if isinstance(summary, dict) else {}
    if isinstance(candidates, dict):
        path = candidates.get(split)
        if isinstance(path, str):
            return path
    return None


def _resolve_artifact_root(root: Path, model: str, tag: str) -> Path:
    candidate = root / model / tag
    if candidate.exists():
        return candidate
    fallback = root / model
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Artifacts directory not found for {model} (searched {candidate} and {fallback})"
    )


def export_graphsage(args: argparse.Namespace) -> None:
    from src.graphsage.run_graphsage_eval import (
        CandidateStreamer,
        GraphSAGE,
        csr_to_edge_index,
        load_features,
        load_graph,
    )

    ensure_tmpdir()

    art_root = Path(args.artifacts_root) / "graphsage" / args.tag
    res_root = Path(args.output_root) / "graphsage" / args.tag

    summary = resolve_summary(art_root)
    defaults = cast(
        dict[str, object], summary.get("hparams", {}) if isinstance(summary, dict) else {}
    )

    adj_path = Path(args.adj or cast(str, summary.get("adjacency", "")))
    feat_path = Path(args.features or cast(str, summary.get("features", "")))
    struct_path = Path(args.struct_feats) if args.struct_feats else None
    if struct_path is None:
        struct_path_str = summary.get("struct_features") if isinstance(summary, dict) else None
        if isinstance(struct_path_str, str) and struct_path_str:
            struct_path = Path(struct_path_str)
    splits_root = Path(args.splits_root or cast(str, summary.get("splits_root", "")))
    cand_val = Path(args.candidates_val or load_candidate_paths(summary, "val") or "")
    cand_test = Path(args.candidates_test or load_candidate_paths(summary, "test") or "")

    for p, name in [
        (adj_path, "adjacency"),
        (feat_path, "features"),
        (cand_val, "candidates_val"),
        (cand_test, "candidates_test"),
        (splits_root, "splits_root"),
    ]:
        if not str(p):
            raise RuntimeError(f"Missing required path for {name}")
        if not p.exists():
            raise FileNotFoundError(f"File or directory not found: {p}")

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    graph = load_graph(adj_path, undirected=bool(summary.get("undirected", False)))
    if struct_path is not None and not struct_path.exists():
        struct_path = None
    features, _ = load_features(feat_path, graph.num_nodes, struct_feats_path=struct_path)
    features = features.to(device)
    edge_index = csr_to_edge_index(graph.csr).to(device)

    seed_dirs = parse_seeds(art_root, args.seeds)
    for seed_dir in seed_dirs:
        seed_id = seed_dir.name.split("_")[-1]
        model_path = seed_dir / (args.model_filename or "model.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        calib = load_calibration(seed_dir / "calibration.json")

        ckpt = torch.load(model_path, map_location=device)  # nosec B614 -- local pipeline artifacts
        hparams = cast(dict[str, object], ckpt.get("hparams", {}))
        _def_hidden = int(cast(int, defaults.get("hidden", 128)))
        _def_layers = int(cast(int, defaults.get("layers", 2)))
        _def_dropout = float(cast(float, defaults.get("dropout", 0.0)))
        _def_norm = bool(defaults.get("normalize_emb", True))
        _def_id_emb = bool(defaults.get("use_id_emb", False))
        hidden = int(cast(int, hparams.get("hidden", _def_hidden)))
        layers = int(cast(int, hparams.get("layers", _def_layers)))
        dropout = float(cast(float, hparams.get("dropout", _def_dropout)))
        normalize = bool(hparams.get("normalize_emb", _def_norm))
        use_id_emb = bool(hparams.get("use_id_emb", _def_id_emb))
        if use_id_emb:
            raise RuntimeError(
                "GraphSAGE exporter does not yet support checkpoints with id embeddings"
            )
        in_channels = int(features.size(1))
        model = GraphSAGE(in_channels, hidden, layers, dropout).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        with torch.no_grad():
            embeddings = model.encode(features, edge_index)
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.cpu()

        scored_at = now_iso()
        for split, cand_path in [
            ("val", cand_val),
            ("test", cand_test),
        ]:
            if args.splits and split not in args.splits:
                continue
            out_path = res_root / seed_dir.name / f"scores_{split}.parquet"
            if out_path.exists() and not args.overwrite:
                if not args.quiet:
                    print(f"[SKIP] {out_path} exists")
                continue
            if not args.quiet:
                print(f"[INFO] Writing scores for seed {seed_id} split {split} -> {out_path}")
            writer = ParquetAppend(out_path)
            total_rows = 0
            streamer = CandidateStreamer(cand_path, batch_size=args.batch_size)
            for chunk in streamer:
                if chunk.empty:
                    continue
                pdf = chunk.copy()
                pdf["src_id"] = pdf["src_id"].astype(np.int64)
                pdf["dst_id"] = pdf["dst_id"].astype(np.int64)
                pdf["label"] = pdf["label"].astype(np.int8)
                src_idx = torch.from_numpy(pdf["src_id"].to_numpy(dtype=np.int64))
                dst_idx = torch.from_numpy(pdf["dst_id"].to_numpy(dtype=np.int64))
                with torch.no_grad():
                    zu = embeddings.index_select(0, src_idx)
                    zv = embeddings.index_select(0, dst_idx)
                    scores = (zv * zu).sum(dim=1).numpy()
                calibrated = apply_platt(scores, calib)
                pdf["logit"] = scores.astype(np.float32)
                pdf["calibrated"] = calibrated.astype(np.float32)
                pdf["model"] = args.model
                pdf["tag"] = args.tag
                pdf["seed"] = int(seed_id)
                pdf["split"] = split
                pdf["scored_at"] = scored_at
                writer.write(pdf)
                total_rows += len(pdf)
            writer.close()
            if not args.quiet:
                print(f"[DONE] {out_path} rows={total_rows:,}")


def export_node2vec(args: argparse.Namespace) -> None:
    from src.graphsage.run_graphsage_eval import CandidateStreamer

    ensure_tmpdir()

    art_root = Path(args.artifacts_root) / "node2vec" / args.tag
    res_root = Path(args.output_root) / "node2vec" / args.tag

    if not args.candidates_val or not args.candidates_test:
        raise RuntimeError("Node2Vec exporter requires --candidates-val and --candidates-test")
    if not args.splits_root:
        raise RuntimeError("Node2Vec exporter requires --splits-root")

    cand_val = Path(args.candidates_val)
    cand_test = Path(args.candidates_test)
    for p, name in [(cand_val, "candidates_val"), (cand_test, "candidates_test")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file for {name}: {p}")

    seed_dirs = parse_seeds(art_root, args.seeds)

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    for seed_dir in seed_dirs:
        seed_id = seed_dir.name.split("_")[-1]
        emb_path = seed_dir / (args.embedding_filename or "embeddings_dim256.pt")
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings tensor not found: {emb_path}")
        payload = torch.load(emb_path, map_location=device)  # nosec B614 -- local pipeline artifacts
        if isinstance(payload, dict) and "embeddings" in payload:
            embeddings = payload["embeddings"].to(device)
        elif torch.is_tensor(payload):
            embeddings = payload.to(device)
        else:
            raise RuntimeError(f"Unexpected Node2Vec embedding payload at {emb_path}")

        calib = load_calibration(seed_dir / "calibration.json")
        scored_at = now_iso()

        for split, cand_path in [("val", cand_val), ("test", cand_test)]:
            if args.splits and split not in args.splits:
                continue
            out_path = res_root / seed_dir.name / f"scores_{split}.parquet"
            if out_path.exists() and not args.overwrite:
                if not args.quiet:
                    print(f"[SKIP] {out_path} exists")
                continue
            if not args.quiet:
                print(f"[INFO] Node2Vec seed {seed_id} split {split} -> {out_path}")
            writer = ParquetAppend(out_path)
            total_rows = 0
            streamer = CandidateStreamer(cand_path, batch_size=args.batch_size)
            for chunk in streamer:
                if chunk.empty:
                    continue
                pdf = chunk.copy()
                pdf["src_id"] = pdf["src_id"].astype(np.int64)
                pdf["dst_id"] = pdf["dst_id"].astype(np.int64)
                pdf["label"] = pdf["label"].astype(np.int8)
                src_idx = torch.from_numpy(pdf["src_id"].to_numpy(dtype=np.int64))
                dst_idx = torch.from_numpy(pdf["dst_id"].to_numpy(dtype=np.int64))
                with torch.no_grad():
                    zu = embeddings.index_select(0, src_idx)
                    zv = embeddings.index_select(0, dst_idx)
                    scores = (zv * zu).sum(dim=1).cpu().numpy()
                calibrated = apply_platt(scores, calib)
                pdf["logit"] = scores.astype(np.float32)
                pdf["calibrated"] = calibrated.astype(np.float32)
                pdf["model"] = args.model
                pdf["tag"] = args.tag
                pdf["seed"] = int(seed_id)
                pdf["split"] = split
                pdf["scored_at"] = scored_at
                writer.write(pdf)
                total_rows += len(pdf)
            writer.close()
            if not args.quiet:
                print(f"[DONE] {out_path} rows={total_rows:,}")


def export_heuristics(args: argparse.Namespace) -> None:
    from src.heuristics.run_heuristics_eval import (
        CandidateStreamer,
        compute_heuristics_for_source,
        load_graph,
    )

    ensure_tmpdir()

    art_root = _resolve_artifact_root(Path(args.artifacts_root), "heuristics", args.tag)
    res_root = Path(args.output_root) / "heuristics" / args.tag

    summary = resolve_summary(art_root)

    adj_path = Path(args.adj or cast(str, summary.get("adjacency", "")))
    cand_val = Path(args.candidates_val or load_candidate_paths(summary, "val") or "")
    cand_test = Path(args.candidates_test or load_candidate_paths(summary, "test") or "")

    for p, name in [
        (adj_path, "adjacency"),
        (cand_val, "candidates_val"),
        (cand_test, "candidates_test"),
    ]:
        if not str(p):
            raise RuntimeError(f"Missing required path for {name}")
        if not p.exists():
            raise FileNotFoundError(f"File not found for {name}: {p}")

    directed = bool(summary.get("directed", True))
    graph = load_graph(adj_path, undirected=not directed)
    calib = load_calibration(art_root / "calibration.json")

    seed_name = "seed_base"
    scored_at = now_iso()
    flush_limit = max(200_000, int(args.batch_size // 10))

    for split, cand_path in [("val", cand_val), ("test", cand_test)]:
        if args.splits and split not in args.splits:
            continue
        out_path = res_root / seed_name / f"scores_{split}.parquet"
        if out_path.exists() and not args.overwrite:
            if not args.quiet:
                print(f"[SKIP] {out_path} exists")
            continue
        if not args.quiet:
            print(f"[INFO] Heuristics split {split} -> {out_path}")

        writer = ParquetAppend(out_path)
        total_rows = 0
        frames: list[pd.DataFrame] = []
        pending = 0
        tmp_buf: dict[str, object] = {}

        try:
            import pyarrow.parquet as pq

            has_ts = "ts" in set(pq.ParquetFile(cand_path).schema.names)
        except Exception:
            has_ts = False
        columns = ["src_id", "dst_id", "label"] + (["ts"] if has_ts else [])

        streamer = CandidateStreamer(cand_path, columns=columns, batch_size=args.batch_size)
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
                lbl = sub["label"].to_numpy(dtype=np.int8, copy=False)
                if vs.size == 0:
                    continue
                hs = compute_heuristics_for_source(u, vs, graph, tmp_buf)
                src_col = np.full(vs.shape, u, dtype=np.int64)
                data = {
                    "src_id": src_col,
                    "dst_id": vs.astype(np.int64),
                    "label": lbl.astype(np.int8),
                    "CN": hs.cn.astype(np.float32),
                    "Jaccard": hs.jaccard.astype(np.float32),
                    "AA": hs.aa.astype(np.float32),
                    "RA": hs.ra.astype(np.float32),
                    "COS": hs.cos.astype(np.float32),
                    "PA": hs.pa.astype(np.float32),
                }
                if "ts" in sub.columns:
                    data["ts"] = sub["ts"].to_numpy(dtype=np.float64, copy=False)
                df = pd.DataFrame(data)
                pa_scores = df["PA"].to_numpy(dtype=np.float64, copy=False)
                df["logit"] = pa_scores.astype(np.float32)
                df["calibrated"] = apply_platt(pa_scores, calib).astype(np.float32)
                df["model"] = args.model
                df["tag"] = args.tag
                df["seed"] = 0
                df["split"] = split
                df["scored_at"] = scored_at
                frames.append(df)
                pending += len(df)
                total_rows += len(df)
                if pending >= flush_limit:
                    writer.write(pd.concat(frames, ignore_index=True))
                    frames.clear()
                    pending = 0

        if frames:
            writer.write(pd.concat(frames, ignore_index=True))
        writer.close()
        if not args.quiet:
            print(f"[DONE] {out_path} rows={total_rows:,}")


def export_twotower(args: argparse.Namespace) -> None:
    from src.twotower.run_twotower_eval import (
        CandidateStreamer,
        Features,
        TwoTowerModel,
        compute_all_embeddings,
    )
    from src.twotower.run_twotower_eval import (
        load_features as tt_load_features,
    )

    ensure_tmpdir()

    art_root = _resolve_artifact_root(Path(args.artifacts_root), "twotower", args.tag)
    res_root = Path(args.output_root) / "twotower" / args.tag

    seed_dirs = [p for p in art_root.iterdir() if p.is_dir() and p.name.startswith("seed_")]
    if args.seeds:
        wanted = {f"seed_{int(s.strip())}" for s in args.seeds.split(",") if s.strip()}
        seed_dirs = [p for p in seed_dirs if p.name in wanted]
    if not seed_dirs:
        raise RuntimeError("No TwoTower seed directories discovered")

    adj_path = Path(args.adj or "data/processed/core/releases/core_v1/adjacency/train_adj_T0.npz")
    feat_path = Path(
        args.features or "data/processed/core/releases/core_v1/features/node_features_T0.parquet"
    )
    struct_path = Path(
        args.struct_feats
        or "data/processed/core/releases/core_v1/features/node_structural_v1.parquet"
    )
    if not adj_path.exists() or not feat_path.exists() or not struct_path.exists():
        raise FileNotFoundError("Invalid --adj/--features/--struct-feats for TwoTower")

    attr_keys = [
        k.strip()
        for k in str(
            args.attr_keys or "country,region,continent,entity_type,primary_sic_code"
        ).split(",")
        if k.strip()
    ]
    feats: Features = tt_load_features(feat_path, struct_path, attr_keys)

    cand_val = Path(
        args.candidates_val
        or "data/processed/core/releases/core_v1/candidates/val_candidates.parquet"
    )
    cand_test = Path(
        args.candidates_test
        or "data/processed/core/releases/core_v1/candidates/test_candidates.parquet"
    )
    if not cand_val.exists() or not cand_test.exists():
        raise FileNotFoundError(
            "Candidate parquet paths missing; supply --candidates-val/--candidates-test"
        )

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    struct_hidden_default = [int(x.strip()) for x in ["128", "64"] if x.strip()]
    attr_hidden_default = [int(x.strip()) for x in ["64", "64"] if x.strip()]

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[-1])
        model_path = seed_dir / (args.model_filename or "model.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing TwoTower model checkpoint: {model_path}")
        payload = torch.load(model_path, map_location=device)  # nosec B614 -- local pipeline artifacts
        params = payload.get("hparams", {})
        embed_dim = int(params.get("embed_dim", 64))
        struct_hidden = [
            int(x.strip())
            for x in str(params.get("struct_hidden", "128,64")).split(",")
            if x.strip()
        ] or struct_hidden_default
        attr_hidden = [
            int(x.strip()) for x in str(params.get("attr_hidden", "64,64")).split(",") if x.strip()
        ] or attr_hidden_default
        attr_emb_dim = int(params.get("attr_emb_dim", 32))
        dropout = float(params.get("dropout", 0.0))
        normalize = bool(params.get("normalize_emb", True))

        model = TwoTowerModel(
            struct_in=int(feats.struct.shape[1]),
            struct_hidden=struct_hidden,
            attr_num_cats=feats.num_cats,
            attr_emb_dim=attr_emb_dim,
            attr_hidden=attr_hidden,
            tower_out_dim=int(embed_dim // 2),
            final_dim=embed_dim,
            dropout=dropout,
            normalize=normalize,
        ).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()

        with torch.no_grad():
            ZU, ZV = compute_all_embeddings(model, feats, device=device)
        ZU = ZU.detach().cpu()
        ZV = ZV.detach().cpu()

        calib = load_calibration(seed_dir / "calibration.json")
        scored_at = now_iso()

        for split, cand_path in [("val", cand_val), ("test", cand_test)]:
            if args.splits and split not in args.splits:
                continue
            out_path = res_root / seed_dir.name / f"scores_{split}.parquet"
            if out_path.exists() and not args.overwrite:
                if not args.quiet:
                    print(f"[SKIP] {out_path} exists")
                continue
            if not args.quiet:
                print(f"[INFO] TwoTower seed {seed} split {split} -> {out_path}")
            writer = ParquetAppend(out_path)
            total_rows = 0
            streamer = CandidateStreamer(cand_path, batch_size=args.batch_size)
            for chunk in streamer:
                if chunk.empty:
                    continue
                chunk["src_id"] = chunk["src_id"].astype(np.int64)
                chunk["dst_id"] = chunk["dst_id"].astype(np.int64)
                chunk["label"] = chunk["label"].astype(np.int8)
                src_idx = torch.from_numpy(chunk["src_id"].to_numpy(dtype=np.int64))
                dst_idx = torch.from_numpy(chunk["dst_id"].to_numpy(dtype=np.int64))
                with torch.no_grad():
                    zu = ZU.index_select(0, src_idx)
                    zv = ZV.index_select(0, dst_idx)
                    scores = (zv * zu).sum(dim=1).numpy()
                calibrated = apply_platt(scores, calib)
                chunk["logit"] = scores.astype(np.float32)
                chunk["calibrated"] = calibrated.astype(np.float32)
                chunk["model"] = args.model
                chunk["tag"] = args.tag
                chunk["seed"] = seed
                chunk["split"] = split
                chunk["scored_at"] = scored_at
                writer.write(chunk)
                total_rows += len(chunk)
            writer.close()
            if not args.quiet:
                print(f"[DONE] {out_path} rows={total_rows:,}")


def export_tgnn(args: argparse.Namespace) -> None:
    """
    Export TGNN scores for a candidate pool. This mirrors evaluate_saved_model but writes the
    same schema as other exporters: logit, calibrated, model, tag, seed, split, scored_at.
    """
    from src.tgnn.model import TemporalGNN
    from src.tgnn.run_tgnn_eval import (
        CandidateStreamer,
        build_snapshots,
    )
    from src.tgnn.run_tgnn_eval import (
        load_features as tgnn_load_features,
    )

    ensure_tmpdir()

    art_root = _resolve_artifact_root(Path(args.artifacts_root), "tgnn", args.tag)
    res_root = Path(args.output_root) / "tgnn" / args.tag

    seed_dirs = parse_seeds(art_root, args.seeds)
    if not seed_dirs:
        raise RuntimeError("No TGNN seed directories found")

    # Resolve paths from the first seed's summary (all seeds share config)
    cfg: dict[str, object] = {}
    summary_first = resolve_summary(seed_dirs[0])
    if isinstance(summary_first, dict):
        cfg = cast(dict[str, object], summary_first.get("config", {})) or {}

    def _path_from_args_or_cfg(arg_val: str, cfg_key: str) -> Path:
        if arg_val:
            return Path(arg_val)
        val = cfg.get(cfg_key, "")
        if not isinstance(val, str) or not val:
            return Path("")
        return Path(val)

    adj_path = _path_from_args_or_cfg(args.adj, "adj")
    feat_path = _path_from_args_or_cfg(args.features, "features")
    node2vec_path = _path_from_args_or_cfg("", "node2vec_emb")
    cand_val = Path(args.candidates_val or cast(str, cfg.get("candidates_val", "")))
    cand_test = Path(args.candidates_test or cast(str, cfg.get("candidates_test", "")))
    splits_root = Path(args.splits_root or cast(str, cfg.get("splits_root", "")))

    for p, name in [
        (adj_path, "adjacency"),
        (feat_path, "features"),
        (cand_val, "candidates_val"),
        (cand_test, "candidates_test"),
        (splits_root, "splits_root"),
    ]:
        if not str(p):
            raise RuntimeError(f"Missing required path for {name}")
        if not p.exists():
            raise FileNotFoundError(f"File or directory not found: {p}")

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    # Build temporal snapshots and features once; reused across seeds
    max_snaps_raw_obj = cfg.get("max_snapshots", None)
    max_snapshots_tgnn: int | None = None
    if isinstance(max_snaps_raw_obj, int):
        max_snapshots_tgnn = max_snaps_raw_obj
    elif isinstance(max_snaps_raw_obj, str) and max_snaps_raw_obj.strip():
        try:
            max_snapshots_tgnn = int(max_snaps_raw_obj)
        except Exception:
            max_snapshots_tgnn = None
    _windows, edge_lists, _ = build_snapshots(
        splits_root=splits_root,
        granularity=str(cfg.get("granularity", "quarter")),
        max_snapshots=max_snapshots_tgnn,
    )
    node2vec_arg = node2vec_path if node2vec_path and node2vec_path.exists() else None
    x_np, _ = tgnn_load_features(feat_path, node2vec_path=node2vec_arg)
    x = torch.from_numpy(x_np).float().to(device)

    edge_adjs: list[torch.Tensor] = []
    for edges in edge_lists:
        edge_tensor = torch.from_numpy(edges).long().to(device)
        adj = TemporalGNN.normalize_adjacency(edge_tensor, x.size(0))
        edge_adjs.append(adj)

    short_window = int(cast(int, cfg.get("short_head_window", 2)) or 2)
    scored_at = now_iso()

    for seed_dir in seed_dirs:
        seed_id = seed_dir.name.split("_")[-1]
        model_path = seed_dir / (args.model_filename or "model.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"TGNN checkpoint not found: {model_path}")
        calib = load_calibration(seed_dir / "calibration.json")

        state = torch.load(model_path, map_location=device)  # nosec B614 -- local pipeline artifacts
        config = state.get("config", {})
        model = TemporalGNN(
            in_channels=int(config.get("in_channels", x.size(1))),
            hidden_channels=int(config.get("hidden_channels", config.get("hidden", 256))),
            out_channels=int(config.get("out_channels", 128)),
            num_layers=int(config.get("num_layers", 3)),
            dropout=float(config.get("dropout", 0.0)),
            temporal_mode=str(config.get("temporal_mode", "attention")),
            snapshot_batch_size=int(config.get("snapshot_batch_size", 2)),
            checkpoint_snapshots=bool(config.get("checkpoint_snapshots", False)),
            checkpoint_threshold=int(config.get("checkpoint_threshold", 16000)),
            temporal_decay=float(config.get("temporal_decay", 0.08)),
            num_score_heads=int(config.get("num_score_heads", 2)),
        ).to(device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        with torch.no_grad():
            skip_eval = model.skip_proj(x)
            head_embeddings: list[torch.Tensor] = []
            total_snapshots = len(edge_adjs)
            for head_idx in range(model.num_score_heads):
                if head_idx == 0:
                    idx_range = list(range(max(0, total_snapshots - short_window), total_snapshots))
                else:
                    idx_range = list(range(total_snapshots))
                subset = [edge_adjs[i] for i in idx_range]
                z_head = model(x, subset)
                proj = model.head_projections[head_idx](z_head)
                gate = torch.sigmoid(model.head_skip_gates[head_idx])
                emb = proj + gate * skip_eval
                if bool(config.get("normalize_emb", False)):
                    emb = F.normalize(emb, p=2, dim=1)
                head_embeddings.append(emb)

        primary_emb = head_embeddings[0].to(device)

        for split, cand_path in [("val", cand_val), ("test", cand_test)]:
            if args.splits and split not in args.splits:
                continue
            out_path = res_root / seed_dir.name / f"scores_{split}.parquet"
            if out_path.exists() and not args.overwrite:
                if not args.quiet:
                    print(f"[SKIP] {out_path} exists")
                continue
            if not args.quiet:
                print(f"[INFO] TGNN seed {seed_id} split {split} -> {out_path}")

            writer = ParquetAppend(out_path)
            total_rows = 0
            streamer = CandidateStreamer(cand_path, batch_size=args.batch_size)
            for chunk in streamer:
                if chunk.empty:
                    continue
                pdf = chunk.copy()
                pdf["src_id"] = pdf["src_id"].astype(np.int64)
                pdf["dst_id"] = pdf["dst_id"].astype(np.int64)
                pdf["label"] = pdf["label"].astype(np.int8)

                src_idx = torch.from_numpy(pdf["src_id"].to_numpy(dtype=np.int64)).to(device)
                dst_idx = torch.from_numpy(pdf["dst_id"].to_numpy(dtype=np.int64)).to(device)
                with torch.no_grad():
                    zu = primary_emb.index_select(0, src_idx)
                    zv = primary_emb.index_select(0, dst_idx)
                    scores = (zv * zu).sum(dim=1).cpu().numpy()
                calibrated = apply_platt(scores, calib)
                pdf["logit"] = scores.astype(np.float32)
                pdf["calibrated"] = calibrated.astype(np.float32)
                pdf["model"] = args.model
                pdf["tag"] = args.tag
                pdf["seed"] = int(seed_id)
                pdf["split"] = split
                pdf["scored_at"] = scored_at
                writer.write(pdf)
                total_rows += len(pdf)
            writer.close()
            if not args.quiet:
                print(f"[DONE] {out_path} rows={total_rows:,}")


def export_n2v_temporal(args: argparse.Namespace) -> None:
    """
    Export Node2Vec-Temporal (TemporalGNN with n2v snapshots) scores for a candidate pool.
    Mirrors the inference path in run_n2v_temporal_eval but writes the standard score schema.
    """
    from src.n2v_temporal.model import TemporalGNN as N2VTemporalGNN
    from src.n2v_temporal.run_n2v_temporal_eval import (
        CandidateStreamer,
        align_node2vec_embeddings,
        build_snapshots,
        load_n2v_embeddings,
        project_node2vec_embeddings,
    )
    from src.n2v_temporal.run_n2v_temporal_eval import (
        load_features as n2vt_load_features,
    )
    from src.n2v_temporal.run_n2v_temporal_eval import (
        load_graph as n2vt_load_graph,
    )

    ensure_tmpdir()

    art_root = _resolve_artifact_root(Path(args.artifacts_root), "n2v_temporal", args.tag)
    res_root = Path(args.output_root) / "n2v_temporal" / args.tag

    seed_dirs = parse_seeds(art_root, args.seeds)
    if not seed_dirs:
        raise RuntimeError("No n2v_temporal seed directories found")

    summary_first = resolve_summary(seed_dirs[0])
    cfg: dict[str, object] = cast(
        dict[str, object],
        summary_first.get("config", {}) if isinstance(summary_first, dict) else {},
    )

    def _path_from_args_or_cfg(arg_val: str, cfg_key: str) -> Path:
        if arg_val:
            return Path(arg_val)
        val = cfg.get(cfg_key, "")
        if isinstance(val, str) and val:
            return Path(val)
        return Path("")

    adj_path = _path_from_args_or_cfg(args.adj, "adj")
    feat_path = _path_from_args_or_cfg(args.features, "features")
    n2v_dir = Path(cast(str, cfg.get("n2v_dir", "")))
    cand_val = Path(args.candidates_val or cast(str, cfg.get("candidates_val", "")))
    cand_test = Path(args.candidates_test or cast(str, cfg.get("candidates_test", "")))
    splits_root = Path(args.splits_root or cast(str, cfg.get("splits_root", "")))

    for p, name in [
        (adj_path, "adjacency"),
        (feat_path, "features"),
        (n2v_dir, "n2v_dir"),
        (cand_val, "candidates_val"),
        (cand_test, "candidates_test"),
        (splits_root, "splits_root"),
    ]:
        if not str(p):
            raise RuntimeError(f"Missing required path for {name}")
        if not p.exists():
            raise FileNotFoundError(f"File or directory not found: {p}")

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    max_snaps_raw_n2v = cfg.get("max_snapshots")
    max_snapshots_n2v: int | None = None
    if isinstance(max_snaps_raw_n2v, int):
        max_snapshots_n2v = max_snaps_raw_n2v
    elif isinstance(max_snaps_raw_n2v, str) and max_snaps_raw_n2v.strip():
        try:
            max_snapshots_n2v = int(max_snaps_raw_n2v)
        except Exception:
            max_snapshots_n2v = None
    _windows, edge_lists, _ = build_snapshots(
        Path(splits_root),
        granularity=str(cfg.get("granularity", "quarter")),
        max_snapshots=max_snapshots_n2v,
    )

    x_np, _ = n2vt_load_features(feat_path)
    x_dev = torch.from_numpy(x_np).float().to(device)

    n2v_arrays = load_n2v_embeddings(n2v_dir, len(edge_lists), x_np.shape[0])
    if not n2v_arrays:
        raise RuntimeError(f"No Node2Vec snapshots loaded from {n2v_dir}")
    if bool(cfg.get("align_n2v", False)):
        n2v_arrays = align_node2vec_embeddings(n2v_arrays)
    n2v_target_dim = cfg.get("n2v_target_dim")
    if n2v_target_dim:
        try:
            n2v_td = int(n2v_target_dim)  # type: ignore[arg-type]
            if n2v_td > 0:
                n2v_arrays = project_node2vec_embeddings(n2v_arrays, n2v_td)
        except Exception:  # nosec B110 -- best-effort projection, pass is intentional
            pass
    n2v_dim = int(n2v_arrays[0].shape[1])
    n2v_tensors = [torch.from_numpy(arr).float().to(device) for arr in n2v_arrays]

    undirected = bool(cfg.get("undirected", True))
    graph = n2vt_load_graph(adj_path, undirected)
    edge_adjs: list[torch.Tensor]
    if bool(cfg.get("use_n2v_features", True)):
        edge_adjs = []
    else:
        edge_adjs = [
            N2VTemporalGNN.normalize_adjacency(
                torch.from_numpy(edges).long().to(device),
                graph.num_nodes,
            )
            for edges in edge_lists
        ]

    def _cfg_int(key: str, default: int) -> int:
        try:
            return int(cfg.get(key, default))  # type: ignore[arg-type]
        except Exception:
            return default

    def _cfg_float(key: str, default: float) -> float:
        try:
            return float(cast(float, cfg.get(key, default)))
        except Exception:
            return default

    model = N2VTemporalGNN(
        in_channels=int(x_np.shape[1]),
        hidden_channels=_cfg_int("hidden", 256),
        out_channels=_cfg_int("out_channels", n2v_dim),
        num_layers=_cfg_int("num_layers", 3),
        dropout=_cfg_float("dropout", 0.0),
        temporal_mode=str(cfg.get("temporal_mode", "attention")),
        snapshot_batch_size=_cfg_int("snapshot_batch_size", 1),
        checkpoint_snapshots=bool(cfg.get("gcn_checkpoint", False)),
        checkpoint_threshold=_cfg_int("checkpoint_threshold", 16000),
        temporal_decay=_cfg_float("temporal_decay", 0.135),
        num_score_heads=_cfg_int("num_score_heads", 1),
        n2v_dim=n2v_dim,
        use_n2v_features=bool(cfg.get("use_n2v_features", True)),
        concat_base_features=bool(cfg.get("concat_base_features", False)),
    ).to(device)

    for seed_dir in seed_dirs:
        seed_id = seed_dir.name.split("_")[-1]
        model_path = seed_dir / (args.model_filename or "model.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"n2v_temporal checkpoint not found: {model_path}")
        calib = load_calibration(seed_dir / "calibration.json")

        state = torch.load(model_path, map_location=device)  # nosec B614 -- local pipeline artifacts
        # Allow extra heads/keys (e.g., long-head projections); we only use primary head embeddings.
        model.load_state_dict(state["model_state_dict"], strict=False)
        model.eval()

        with torch.no_grad():
            Z = model(x_dev, edge_adjs, n2v_tensors)
            if bool(cfg.get("normalize_emb", False)):
                Z = F.normalize(Z, p=2, dim=1)
            head_embs = model.project_heads(Z)
            if bool(cfg.get("normalize_emb", False)):
                head_embs = [F.normalize(emb, p=2, dim=1) for emb in head_embs]

        primary_emb = head_embs[0]
        scored_at = now_iso()

        for split, cand_path in [("val", cand_val), ("test", cand_test)]:
            if args.splits and split not in args.splits:
                continue
            out_path = res_root / seed_dir.name / f"scores_{split}.parquet"
            if out_path.exists() and not args.overwrite:
                if not args.quiet:
                    print(f"[SKIP] {out_path} exists")
                continue
            if not args.quiet:
                print(f"[INFO] n2v_temporal seed {seed_id} split {split} -> {out_path}")

            writer = ParquetAppend(out_path)
            total_rows = 0
            streamer = CandidateStreamer(cand_path, batch_size=args.batch_size)
            for chunk in streamer:
                if chunk.empty:
                    continue
                pdf = chunk.copy()
                pdf["src_id"] = pdf["src_id"].astype(np.int64)
                pdf["dst_id"] = pdf["dst_id"].astype(np.int64)
                pdf["label"] = pdf["label"].astype(np.int8)

                src_idx = torch.from_numpy(pdf["src_id"].to_numpy(dtype=np.int64)).to(device)
                dst_idx = torch.from_numpy(pdf["dst_id"].to_numpy(dtype=np.int64)).to(device)
                with torch.no_grad():
                    zu = primary_emb.index_select(0, src_idx)
                    zv = primary_emb.index_select(0, dst_idx)
                    scores = (zv * zu).sum(dim=1).cpu().numpy()
                calibrated = apply_platt(scores, calib)
                pdf["logit"] = scores.astype(np.float32)
                pdf["calibrated"] = calibrated.astype(np.float32)
                pdf["model"] = args.model
                pdf["tag"] = args.tag
                pdf["seed"] = int(seed_id)
                pdf["split"] = split
                pdf["scored_at"] = scored_at
                writer.write(pdf)
                total_rows += len(pdf)
            writer.close()
            if not args.quiet:
                print(f"[DONE] {out_path} rows={total_rows:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-candidate scores for production models"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["graphsage", "twotower", "node2vec", "heuristics", "tgnn", "n2v_temporal"],
    )
    parser.add_argument("--tag", required=True, help="Model tag under artifacts/<model>/")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--seeds", default="", help="Comma-separated list of seeds or seed directories"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=2_000_000)
    parser.add_argument("--adj", default="")
    parser.add_argument("--features", default="")
    parser.add_argument("--struct-feats", default="")
    parser.add_argument("--candidates-val", default="")
    parser.add_argument("--candidates-test", default="")
    parser.add_argument("--splits-root", default="")
    parser.add_argument("--model-filename", default="model.pt")
    parser.add_argument("--splits", nargs="*", default=["val", "test"], help="Splits to export")
    parser.add_argument(
        "--embedding-filename",
        default="",
        help="Override default embedding filename for Node2Vec seeds",
    )
    parser.add_argument(
        "--attr-keys", default="", help="Comma-separated attribute keys for TwoTower features"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.model == "graphsage":
        export_graphsage(args)
    elif args.model == "twotower":
        export_twotower(args)
    elif args.model == "node2vec":
        export_node2vec(args)
    elif args.model == "heuristics":
        export_heuristics(args)
    elif args.model == "tgnn":
        export_tgnn(args)
    elif args.model == "n2v_temporal":
        export_n2v_temporal(args)
    else:
        raise NotImplementedError(f"Model {args.model} exporter not yet implemented")


if __name__ == "__main__":
    main()
