from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.ensemble.utils import ParquetAppend, apply_platt, ensure_dir
from src.graphsage.run_graphsage_eval import fit_platt
from src.heuristics.run_heuristics_eval import (
    CandidateStreamer,
    GraphData,
    _load_temporal_t0,
    classify_twohop,
    classify_warm_cold,
    classify_warm_cold_strict,
    load_graph,
)
from src.utils.horizons import assign_horizon_buckets, horizon_slice_names


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def build_base_features(
    *,
    splits_root: Path,
    adj_path: Path,
    cand_val: Path,
    cand_test: Path,
    out_dir: Path,
    folds: int,
    batch_size: int,
    assume_ts: bool = False,
    horizon_all: bool = False,
) -> dict[str, Path]:
    ensure_dir(out_dir)
    graph: GraphData = load_graph(adj_path, undirected=False)
    T0 = _load_temporal_t0(splits_root) or 0
    deg_q = np.quantile(graph.out_deg, [0.25, 0.5, 0.75])
    horizon_names = horizon_slice_names()

    def process_split(split: str, cand_path: Path) -> Path:
        out_path = out_dir / f"base_{split}.parquet"
        writer = ParquetAppend(out_path)
        columns = ["src_id", "dst_id", "label"]
        if assume_ts:
            has_ts = True
            columns.append("ts")
        else:
            try:
                import pyarrow.parquet as pq

                pq.read_table(cand_path, columns=["ts"], use_threads=False)
                has_ts = True
                columns.append("ts")
            except Exception:
                has_ts = False
        streamer = CandidateStreamer(cand_path, columns=columns, batch_size=batch_size)
        ts_sidecar: dict[tuple[int, int], int] | None = None
        if not has_ts:
            edges_path = splits_root / f"{split}_edges.parquet"
            if edges_path.exists():
                df_edges = pd.read_parquet(edges_path, columns=["src_id", "dst_id", "ts"])
                ts_sidecar = {
                    (int(r[0]), int(r[1])): int(r[2]) for r in df_edges.itertuples(index=False)
                }

        total = 0
        [f"slice_{name}" for name in horizon_names]
        print(f"[BASE] {split}: streaming {cand_path}")

        for chunk in streamer:
            if chunk.empty:
                continue
            chunk["src_id"] = chunk["src_id"].astype(np.int64)
            chunk["dst_id"] = chunk["dst_id"].astype(np.int64)
            chunk["label"] = chunk["label"].astype(np.int8)
            if "ts" in chunk.columns:
                chunk["ts"] = chunk["ts"].astype("float64")
            src_vals = chunk["src_id"].to_numpy(dtype=np.int64)
            fold_vals = (src_vals % folds).astype(np.int8)
            chunk["dst_id"].to_numpy(dtype=np.int64, copy=False)
            chunk["label"].to_numpy(dtype=np.int8, copy=False)

            data_rows: list[pd.DataFrame] = []
            idx_bounds = np.where(np.diff(src_vals) != 0)[0] + 1
            bounds = np.concatenate(([0], idx_bounds, [len(chunk)]))

            for i in range(len(bounds) - 1):
                a, b = int(bounds[i]), int(bounds[i + 1])
                sub = chunk.iloc[a:b]
                u = int(sub["src_id"].iloc[0])
                vs = sub["dst_id"].to_numpy(dtype=np.int64, copy=False)
                lbl = sub["label"].to_numpy(dtype=np.int8, copy=False)
                folds_local = fold_vals[a:b]
                wc = classify_warm_cold(u, vs, graph)
                wc3 = classify_warm_cold_strict(u, vs, graph, threshold=3)
                th_mask = classify_twohop(u, vs, graph)
                # Degree bin
                deg_u = graph.out_deg[u]
                deg_onehot = {
                    "slice_deg_q1": np.full(len(vs), 1 if deg_u <= deg_q[0] else 0, dtype=np.int8),
                    "slice_deg_q2": np.full(
                        len(vs), 1 if deg_q[0] < deg_u <= deg_q[1] else 0, dtype=np.int8
                    ),
                    "slice_deg_q3": np.full(
                        len(vs), 1 if deg_q[1] < deg_u <= deg_q[2] else 0, dtype=np.int8
                    ),
                    "slice_deg_q4": np.full(len(vs), 1 if deg_u > deg_q[2] else 0, dtype=np.int8),
                }
                # Horizon
                horizon_vals = np.zeros(len(vs), dtype=np.int8)
                if T0:
                    if "ts" in sub.columns:
                        ts_vals = sub["ts"].to_numpy(dtype=np.float64, copy=False)
                    elif ts_sidecar is not None:
                        ts_vals = np.full(len(sub), np.nan, dtype=np.float64)
                        pos_mask = lbl > 0
                        if pos_mask.any():
                            sa = sub["src_id"].to_numpy()[pos_mask]
                            da = sub["dst_id"].to_numpy()[pos_mask]
                            looked = [
                                ts_sidecar.get((int(s), int(d)), np.nan)
                                for s, d in zip(sa, da, strict=False)
                            ]
                            ts_vals[pos_mask] = np.array(looked, dtype=np.float64)
                    else:
                        ts_vals = np.full(len(sub), np.nan, dtype=np.float64)
                    if horizon_all:
                        idx_for_horizon = np.arange(len(sub), dtype=int)
                    else:
                        idx_for_horizon = np.nonzero(lbl > 0)[0]
                    if idx_for_horizon.size:
                        deltas = ts_vals[idx_for_horizon] - float(T0)
                        horizon_vals[idx_for_horizon] = assign_horizon_buckets(deltas)
                horizon_onehot = {
                    f"slice_{name}": np.zeros(len(vs), dtype=np.int8) for name in horizon_names
                }
                for idx, name in enumerate(horizon_names, start=1):
                    mask = horizon_vals == idx
                    horizon_onehot[f"slice_{name}"][mask] = 1

                rows_dict = {
                    "src_id": np.full(len(vs), u, dtype=np.int64),
                    "dst_id": vs.astype(np.int64),
                    "label": lbl.astype(np.int8),
                    "fold": folds_local.astype(np.int8),
                    "slice_WW": (wc == 0).astype(np.int8),
                    "slice_WC": (wc == 1).astype(np.int8),
                    "slice_CW": (wc == 2).astype(np.int8),
                    "slice_CC": (wc == 3).astype(np.int8),
                    "slice_WW3": (wc3 == 0).astype(np.int8),
                    "slice_WC3": (wc3 == 1).astype(np.int8),
                    "slice_CW3": (wc3 == 2).astype(np.int8),
                    "slice_CC3": (wc3 == 3).astype(np.int8),
                    "slice_twohop": th_mask.astype(np.int8),
                    "slice_gt2hop": (~th_mask).astype(np.int8),
                    **deg_onehot,
                }
                for name in horizon_names:
                    rows_dict[f"slice_{name}"] = horizon_onehot[f"slice_{name}"]

                df_rows = pd.DataFrame(rows_dict)
                data_rows.append(df_rows)

            merged = pd.concat(data_rows, ignore_index=True)
            writer.write(merged)
            total += len(merged)
            if total % 5_000_000 < len(merged):
                print(f"[BASE] {split}: processed {total:,} rows")

        writer.close()
        print(f"[BASE] {split}: finished {total:,} rows -> {out_path}")
        return out_path

    paths = {
        "val": process_split("val", cand_val),
        "test": process_split("test", cand_test),
    }
    return paths


def collect_samples(
    *,
    parquet_path: Path,
    folds: int,
    sample_per_fold: int,
    rng: np.random.Generator,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    samples_scores: dict[int, list[float]] = {f: [] for f in range(folds)}
    samples_labels: dict[int, list[float]] = {f: [] for f in range(folds)}
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=["src_id", "logit", "label"], batch_size=1_000_000):
        df = batch.to_pandas()
        src = df["src_id"].to_numpy(dtype=np.int64, copy=False)
        fold_arr = src % folds
        scores = df["logit"].to_numpy(dtype=np.float64, copy=False)
        labels = df["label"].to_numpy(dtype=np.float64, copy=False)
        for f in range(folds):
            if len(samples_scores[f]) >= sample_per_fold:
                continue
            idx = np.where(fold_arr == f)[0]
            if idx.size == 0:
                continue
            remaining = sample_per_fold - len(samples_scores[f])
            take = min(remaining, idx.size, 20000)
            if take <= 0:
                continue
            chosen = idx if take == idx.size else rng.choice(idx, size=take, replace=False)
            samples_scores[f].extend(scores[chosen].tolist())
            samples_labels[f].extend(labels[chosen].tolist())
    return {
        f: (
            np.asarray(samples_scores[f], dtype=np.float64),
            np.asarray(samples_labels[f], dtype=np.float64),
        )
        for f in range(folds)
    }


def compute_oof_parameters(
    samples: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[int, tuple[float, float]], tuple[float, float]]:
    fold_params: dict[int, tuple[float, float]] = {}
    all_scores = np.concatenate([scores for scores, _ in samples.values()])
    all_labels = np.concatenate([labels for _, labels in samples.values()])
    if all_scores.size == 0:
        global_params = (0.0, -6.0)
    else:
        global_params = fit_platt(all_scores, all_labels)
    for f, (_scores, _labels) in samples.items():
        comp_scores = (
            np.concatenate([samples[k][0] for k in samples if k != f])
            if len(samples) > 1
            else all_scores
        )
        comp_labels = (
            np.concatenate([samples[k][1] for k in samples if k != f])
            if len(samples) > 1
            else all_labels
        )
        if comp_scores.size == 0:
            fold_params[f] = global_params
        else:
            fold_params[f] = fit_platt(comp_scores, comp_labels)
    return fold_params, global_params


def apply_calibration_per_seed(
    *,
    input_val: Path,
    input_test: Path,
    out_val: Path,
    out_test: Path,
    folds: int,
    fold_params: dict[int, tuple[float, float]],
    global_params: tuple[float, float],
    batch_size: int,
) -> None:
    ensure_dir(out_val.parent)
    ensure_dir(out_test.parent)

    def transform_scores(path_in: Path, path_out: Path, use_fold_params: bool) -> None:
        writer = ParquetAppend(path_out)
        pf = pq.ParquetFile(path_in)
        for batch in pf.iter_batches(
            columns=["src_id", "dst_id", "logit", "label"], batch_size=batch_size
        ):
            df = batch.to_pandas()
            src = df["src_id"].to_numpy(dtype=np.int64, copy=False)
            logits = df["logit"].to_numpy(dtype=np.float64, copy=False)
            labels = df["label"].to_numpy(dtype=np.int8, copy=False)
            if use_fold_params:
                fold_arr = src % folds
                probs = np.empty_like(logits, dtype=np.float64)
                for f, params in fold_params.items():
                    mask = fold_arr == f
                    if not mask.any():
                        continue
                    A, B = params
                    probs[mask] = apply_platt(logits[mask], {"A": A, "B": B})
            else:
                A, B = global_params
                probs = apply_platt(logits, {"A": A, "B": B})
            out_df = pd.DataFrame(
                {
                    "src_id": df["src_id"].to_numpy(dtype=np.int64, copy=False),
                    "dst_id": df["dst_id"].to_numpy(dtype=np.int64, copy=False),
                    "label": labels,
                    "prob": probs.astype(np.float32),
                }
            )
            writer.write(out_df)
        writer.close()

    transform_scores(input_val, out_val, use_fold_params=True)
    transform_scores(input_test, out_test, use_fold_params=False)


def generate_model_probabilities(
    *,
    model_name: str,
    root: Path,
    splits_root: Path,
    folds: int,
    sample_per_fold: int,
    batch_size: int,
    out_dir: Path,
) -> dict[str, Path]:
    ensure_dir(out_dir)
    rng = np.random.default_rng(42)
    val_outputs: list[Path] = []
    test_outputs: list[Path] = []

    seed_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_"))
    if not seed_dirs:
        # single-seed case (heuristics)
        seed_dirs = [root]

    for seed_dir in seed_dirs:
        seed_name = seed_dir.name if seed_dir.name.startswith("seed_") else "seed_base"
        in_val = seed_dir / "scores_val.parquet"
        in_test = seed_dir / "scores_test.parquet"
        if not in_val.exists() or not in_test.exists():
            raise FileNotFoundError(f"Missing score parquet for {model_name}: {seed_dir}")
        print(f"[{model_name.upper()}] Calibrating seed {seed_name}")
        samples = collect_samples(
            parquet_path=in_val, folds=folds, sample_per_fold=sample_per_fold, rng=rng
        )
        fold_params, global_params = compute_oof_parameters(samples)
        out_val = seed_dir / "scores_val_oof.parquet"
        out_test = seed_dir / "scores_test_calib.parquet"
        apply_calibration_per_seed(
            input_val=in_val,
            input_test=in_test,
            out_val=out_val,
            out_test=out_test,
            folds=folds,
            fold_params=fold_params,
            global_params=global_params,
            batch_size=batch_size,
        )
        print(
            f"[{model_name.upper()}] Seed {seed_name}: OOF written -> {out_val}, test calib -> {out_test}"
        )
        val_outputs.append(out_val)
        test_outputs.append(out_test)

    model_tag = model_name

    def aggregate(paths: list[Path], split: str) -> Path:
        parquet_list = ",".join(f"'{p.as_posix()}'" for p in paths)
        sql = f"SELECT src_id, dst_id, AVG(prob) AS prob_{model_tag} FROM read_parquet([{parquet_list}]) GROUP BY 1,2"
        conn = duckdb.connect(database=":memory:")
        out_path = out_dir / f"{model_tag}_{split}.parquet"
        ensure_dir(out_path.parent)
        conn.execute(f"COPY ({sql}) TO '{out_path.as_posix()}' (FORMAT 'parquet')")
        conn.close()
        print(f"[{model_name.upper()}] Aggregated {split} -> {out_path}")
        return out_path

    return {
        "val": aggregate(val_outputs, "val"),
        "test": aggregate(test_outputs, "test"),
    }


def join_meta_dataset(
    *,
    base_paths: dict[str, Path],
    model_feature_paths: dict[str, dict[str, Path]],
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    for split in ["val", "test"]:
        base_path = base_paths[split]
        select_cols = ["base.*"]
        join_clauses = []
        for model, paths in model_feature_paths.items():
            path = paths[split]
            alias = f"{model}_{split}"
            select_cols.append(f"{alias}.prob_{model} AS prob_{model}")
            join_clauses.append(
                f"LEFT JOIN read_parquet('{path.as_posix()}') {alias} USING (src_id, dst_id)"
            )
        query = (
            "SELECT "
            + ", ".join(select_cols)
            + f" FROM read_parquet('{base_path.as_posix()}') base "
            + " ".join(join_clauses)
        )
        out_path = out_dir / f"meta_inputs_{split}.parquet"
        conn = duckdb.connect(database=":memory:")
        conn.execute(f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT 'parquet')")
        conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build meta-model dataset with OOF calibration and contextual flags"
    )
    p.add_argument("--splits-root", type=str, default="data/processed/core/releases/core_v1/splits")
    p.add_argument(
        "--adj", type=str, default="data/processed/core/releases/core_v1/adjacency/train_adj_T0.npz"
    )
    p.add_argument(
        "--candidates-val",
        type=str,
        default="data/processed/core/releases/core_v1/candidates/val_candidates.parquet",
    )
    p.add_argument(
        "--candidates-test",
        type=str,
        default="data/processed/core/releases/core_v1/candidates/test_candidates.parquet",
    )
    p.add_argument(
        "--candidate-has-ts",
        action="store_true",
        help="Set if candidate parquet already includes ts column",
    )
    p.add_argument(
        "--graphsage-root", type=str, default="results/graphsage/graphsage_production_v2"
    )
    p.add_argument("--node2vec-root", type=str, default="results/node2vec")
    p.add_argument("--heuristics-root", type=str, default="results/heuristics")
    p.add_argument("--twotower-root", type=str, default="results/twotower/twotower_production_v2")
    p.add_argument("--tgnn-root", type=str, default="results/tgnn/tgnn_production_v2")
    p.add_argument(
        "--n2v-temporal-root", type=str, default="results/n2v_temporal/n2v_temporal_production_v2"
    )
    p.add_argument("--out-dir", type=str, default="results/ensemble/meta_dataset")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--sample-per-fold", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=1_000_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits_root = Path(args.splits_root)
    adj_path = Path(args.adj)
    cand_val = Path(args.candidates_val)
    cand_test = Path(args.candidates_test)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    base_paths = build_base_features(
        splits_root=splits_root,
        adj_path=adj_path,
        cand_val=cand_val,
        cand_test=cand_test,
        out_dir=out_dir,
        folds=int(args.folds),
        batch_size=int(args.batch_size),
        assume_ts=bool(args.candidate_has_ts),
    )

    model_roots = {
        "graphsage": Path(args.graphsage_root),
        "node2vec": Path(args.node2vec_root),
        "heuristics": Path(args.heuristics_root),
        "twotower": Path(args.twotower_root),
        "tgnn": Path(args.tgnn_root),
        "n2v_temporal": Path(args.n2v_temporal_root),
    }

    model_feature_paths: dict[str, dict[str, Path]] = {}
    for model, root in model_roots.items():
        paths = generate_model_probabilities(
            model_name=model,
            root=root,
            splits_root=splits_root,
            folds=int(args.folds),
            sample_per_fold=int(args.sample_per_fold),
            batch_size=int(args.batch_size),
            out_dir=out_dir,
        )
        model_feature_paths[model] = paths

    join_meta_dataset(
        base_paths=base_paths,
        model_feature_paths=model_feature_paths,
        out_dir=out_dir,
    )
    print(f"Meta datasets written to {out_dir}")


if __name__ == "__main__":
    main()
