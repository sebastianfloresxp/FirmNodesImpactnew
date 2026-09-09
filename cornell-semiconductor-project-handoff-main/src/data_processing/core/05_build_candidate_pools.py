#!/usr/bin/env python3
"""
Core Pipeline - Phase 5: Build Evaluation Candidate Pools

For each source node with positives in validation or test, build a fixed-size
candidate set the models will rank against. Pools include all split positives
and exclude earlier-known positives to avoid leakage.

Policy implemented (dissertation defaults):
- Budget: 5,000 total candidates per source (including positives).
- Reachability: undirected two-hop (neighbors-of-neighbors via CSR+CSC); uniform random fallback.
- Exclusions: val pools exclude train positives; test pools exclude train ∪ val positives.
- No self-loops; unique (src_id, dst_id) per source.
- Deterministic with --seed (default 42).

Outputs (under --out-root):
- candidates/val_candidates.parquet
- candidates/test_candidates.parquet
- meta/candidate_pools_meta.json

Columns:
- src_id, dst_id, label (1=positive, 0=negative)
- src_degree_train, dst_degree_train
- source (pos|two_hop|random)
- Optional additional columns can be added later if needed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")
logger = logging.getLogger("core.candidates")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build evaluation candidate pools (undirected two-hop)")
    p.add_argument("--out-root", required=True, type=str, help="Run-scoped output root")
    p.add_argument(
        "--budget", type=int, default=5_000, help="Total candidates per source, incl. positives"
    )
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    p.add_argument(
        "--max-sources", type=int, default=None, help="Optional cap for sources per split (debug)"
    )
    p.add_argument("--dry-run", action="store_true", help="Plan only; no writes")
    p.add_argument(
        "--force", action="store_true", help="Allow overwrite of existing candidate files"
    )
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _setup_logger(level: str) -> None:
    try:
        logging.getLogger().setLevel(getattr(logging, level.upper()))
    except Exception:
        logging.getLogger().setLevel(logging.INFO)


def _load_splits(root: Path) -> dict[str, pd.DataFrame]:
    sdir = root / "splits"

    def rd(name: str) -> pd.DataFrame:
        return pd.read_parquet(
            sdir / f"{name}_edges.parquet", columns=["src_id", "dst_id", "ts"]
        )  # minimal

    return {
        "train": rd("train"),
        "val": rd("val"),
        "test": rd("test"),
    }


def _load_adj_and_degrees(
    root: Path,
) -> tuple[sp.csr_matrix, sp.csc_matrix, np.ndarray, np.ndarray, int]:
    adj_path = root / "adjacency" / "train_adj_T0.npz"
    csr = sp.load_npz(adj_path).tocsr(copy=False)
    # Load CSC if available (Phase 4 produced it); otherwise derive
    csc_path = root / "adjacency" / "train_adj_T0_csc.npz"
    if csc_path.exists():
        csc = sp.load_npz(csc_path).tocsc(copy=False)
    else:
        csc = csr.tocsc(copy=True)
    out_deg = np.load(root / "adjacency" / "out_degree.npy")
    in_deg = np.load(root / "adjacency" / "in_degree.npy")
    num_nodes = int(csr.shape[0])
    return (
        csr,
        csc,
        out_deg.astype(np.int32, copy=False),
        in_deg.astype(np.int32, copy=False),
        num_nodes,
    )


def _group_pos_by_src(df: pd.DataFrame) -> dict[int, set]:
    out: dict[int, set] = {}
    for s, d in zip(df["src_id"].to_numpy(), df["dst_id"].to_numpy(), strict=False):
        out.setdefault(int(s), set()).add(int(d))
    return out


def _neighbors_undirected(csr: sp.csr_matrix, csc: sp.csc_matrix, u: int) -> np.ndarray:
    """Return unique undirected 1-hop neighbors for node u using CSR out and CSC in."""
    out_indptr, out_idx = csr.indptr, csr.indices
    in_indptr, in_idx = csc.indptr, csc.indices
    outs = out_idx[out_indptr[u] : out_indptr[u + 1]]
    ins = in_idx[in_indptr[u] : in_indptr[u + 1]]
    if outs.size == 0 and ins.size == 0:
        return np.empty(0, dtype=np.int64)
    if outs.size == 0:
        return np.unique(ins.astype(np.int64, copy=False))
    if ins.size == 0:
        return np.unique(outs.astype(np.int64, copy=False))
    return np.unique(
        np.concatenate([outs.astype(np.int64, copy=False), ins.astype(np.int64, copy=False)])
    )


def _two_hop_undirected(
    csr: sp.csr_matrix, csc: sp.csc_matrix, u: int, budget_needed: int, excluded: set
) -> list[int]:
    r"""Gather up to budget_needed candidates by undirected two-hop, excluding one-hop and exclusions.

    one_hop(u) = out(u) ∪ in(u)
    two_hop(u) = (⋃_{v∈one_hop(u)} (out(v) ∪ in(v))) \ {u ∪ one_hop(u) ∪ excluded}
    """
    if budget_needed <= 0:
        return []
    one = _neighbors_undirected(csr, csc, u)
    if one.size == 0:
        return []
    out_indptr, out_idx = csr.indptr, csr.indices
    in_indptr, in_idx = csc.indptr, csc.indices
    sl: list[np.ndarray] = []
    for v in one:
        v = int(v)
        outs = out_idx[out_indptr[v] : out_indptr[v + 1]]
        ins = in_idx[in_indptr[v] : in_indptr[v + 1]]
        if outs.size > 0 and ins.size > 0:
            sl.append(np.concatenate([outs, ins]))
        elif outs.size > 0:
            sl.append(outs)
        elif ins.size > 0:
            sl.append(ins)
    if not sl:
        return []
    two_all = np.concatenate(sl).astype(np.int64, copy=False)
    two_unique = np.unique(two_all)
    # Build combined exclusion of self, one-hop, and provided exclusions
    if excluded:
        ex_arr = np.fromiter((int(x) for x in excluded), dtype=np.int64)
        ex_full = np.unique(np.concatenate([ex_arr, one]))
    else:
        ex_full = one
    mask = (~np.isin(two_unique, ex_full)) & (two_unique != u)
    cand = two_unique[mask]
    if cand.size == 0:
        return []
    take = int(min(budget_needed, cand.size))
    return cand[:take].astype(int).tolist()


def _random_fill(rng: np.random.Generator, num_nodes: int, needed: int, excluded: set) -> list[int]:
    if needed <= 0:
        return []
    res: list[int] = []
    # Sample in batches to reduce rejections
    batch = min(needed * 4, max(needed, 10_000))
    while len(res) < needed:
        cand = rng.integers(0, num_nodes, size=batch, endpoint=False, dtype=np.int64)
        for w in cand:
            iw = int(w)
            if iw in excluded:
                continue
            res.append(iw)
            if len(res) >= needed:
                break
    return res


def _build_pools_for_split(
    split_name: str,
    splits: dict[str, pd.DataFrame],
    csr: sp.csr_matrix,
    csc: sp.csc_matrix,
    out_deg: np.ndarray,
    in_deg: np.ndarray,
    num_nodes: int,
    budget: int,
    seed: int,
    max_sources: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert split_name in ("val", "test")
    train_df = splits["train"]
    val_df = splits["val"]
    splits["test"]

    pos = splits[split_name]
    pos_by_src = _group_pos_by_src(pos)
    train_pos_by_src = _group_pos_by_src(train_df)
    val_pos_by_src = _group_pos_by_src(val_df)

    sources = list(pos_by_src.keys())
    if max_sources is not None:
        sources = sources[:max_sources]

    logger.info(f"Split={split_name}: sources with positives: {len(sources):,}")

    rows: list[dict[str, Any]] = []
    rng_base = np.random.default_rng(seed)

    for idx, u in enumerate(sources):
        if idx % 1000 == 0:
            logger.info(f"  processed {idx}/{len(sources)} sources")
        P = pos_by_src.get(u, set())
        # Exclusions
        exclude = set(P)
        exclude.add(u)  # no self
        # Earlier positives to exclude as negatives
        if split_name == "val":
            exclude.update(train_pos_by_src.get(u, set()))
        else:  # test
            exclude.update(train_pos_by_src.get(u, set()))
            exclude.update(val_pos_by_src.get(u, set()))

        # Always add positives (label=1)
        for v in P:
            rows.append(
                {
                    "src_id": u,
                    "dst_id": v,
                    "label": 1,
                    "src_degree_train": int(out_deg[u]),
                    "dst_degree_train": int(in_deg[v]),
                    "source": "pos",
                }
            )

        # How many more to reach budget
        need = max(0, budget - len(P))
        # Two-hop undirected
        twohop = _two_hop_undirected(csr, csc, u, need, exclude)
        for v in twohop:
            rows.append(
                {
                    "src_id": u,
                    "dst_id": v,
                    "label": 0,
                    "src_degree_train": int(out_deg[u]),
                    "dst_degree_train": int(in_deg[v]),
                    "source": "two_hop",
                }
            )
        exclude.update(twohop)
        need -= len(twohop)

        # Random fallback if still needed
        if need > 0:
            # Per-source deterministic RNG derived from base seed and src
            rng = np.random.default_rng(rng_base.integers(0, 2**63 - 1) ^ int(u))
            rnd = _random_fill(rng, num_nodes, need, exclude)
            for v in rnd:
                rows.append(
                    {
                        "src_id": u,
                        "dst_id": v,
                        "label": 0,
                        "src_degree_train": int(out_deg[u]),
                        "dst_degree_train": int(in_deg[v]),
                        "source": "random",
                    }
                )

    # Assemble DataFrame
    cand_df = pd.DataFrame.from_records(
        rows,
        columns=["src_id", "dst_id", "label", "src_degree_train", "dst_degree_train", "source"],
    )

    # Acceptance checks
    # 1) Pool recall: all positives present
    # Only check positives for the sources we actually processed
    check_pos = pos[pos["src_id"].isin(sources)][["src_id", "dst_id"]].copy()
    # Build set of required positive pairs and set of candidate pairs
    pos_pairs = set(map(tuple, check_pos.to_numpy()))  # type: ignore[union-attr]
    cand_pairs = set(map(tuple, cand_df[cand_df["label"] == 1][["src_id", "dst_id"]].to_numpy()))  # type: ignore[union-attr]
    missing = pos_pairs - cand_pairs
    if missing:
        raise AssertionError(
            f"Pool recall failure: {len(missing)} positives missing from pools ({split_name})"
        )

    # 2) No self loops
    if int((cand_df["src_id"] == cand_df["dst_id"]).sum()) > 0:
        raise AssertionError("Found self-loops in candidate pools")

    # 3) Cardinality bounds per source
    grp = cand_df.groupby("src_id").size().reindex(sources, fill_value=0)
    posc = pos.groupby("src_id")["dst_id"].nunique().reindex(sources, fill_value=0)
    too_small = int((grp < posc).sum())
    if too_small > 0:
        raise AssertionError(f"Found sources with pool smaller than positives ({too_small})")

    meta = {
        "split": split_name,
        "budget": budget,
        "seed": seed,
        "sources": len(sources),
        "rows": len(cand_df),
        "policy": {
            "reachability": "undirected_two_hop",
            "random_fallback": "uniform",
            "exclusions": "val: exclude train; test: exclude train ∪ val",
        },
        "acceptance": {
            "pool_recall_ok": True,
            "no_self_loops": True,
            "cardinality_ok": True,
        },
    }
    return cand_df, meta


def main() -> None:
    args = _parse_args()
    _setup_logger(args.log_level)

    root = Path(args.out_root)
    cand_dir = root / "candidates"
    meta_dir = root / "meta"
    if not args.dry_run:
        cand_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    splits = _load_splits(root)
    csr, csc, out_deg, in_deg, num_nodes = _load_adj_and_degrees(root)
    logger.info(f"Train CSR: shape={csr.shape}, nnz={csr.nnz:,}")

    # Plan & scale warning
    src_val = splits["val"]["src_id"].nunique()
    src_test = splits["test"]["src_id"].nunique()
    est_rows = (src_val + src_test) * args.budget
    logger.info(
        f"Val sources={src_val:,}, Test sources={src_test:,}, est total rows ≈ {est_rows:,}"
    )
    if args.max_sources is not None:
        logger.info(f"Limiting to first {args.max_sources} sources per split for this run")

    if args.dry_run:
        logger.info("[dry-run] Planning only; no files will be written")
        return

    # VAL candidates
    val_df, val_meta = _build_pools_for_split(
        "val",
        splits,
        csr,
        csc,
        out_deg,
        in_deg,
        num_nodes,
        args.budget,
        args.seed,
        args.max_sources,
    )
    val_path = cand_dir / "val_candidates.parquet"
    if val_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing {val_path} (use --force)")
    val_df.to_parquet(val_path, index=False)

    # TEST candidates
    test_df, test_meta = _build_pools_for_split(
        "test",
        splits,
        csr,
        csc,
        out_deg,
        in_deg,
        num_nodes,
        args.budget,
        args.seed,
        args.max_sources,
    )
    test_path = cand_dir / "test_candidates.parquet"
    if test_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing {test_path} (use --force)")
    test_df.to_parquet(test_path, index=False)

    # Meta
    meta = {
        "run_root": str(root),
        "budget": args.budget,
        "seed": args.seed,
        "directed": True,
        "two_hop": "undirected",
        "sources": {
            "val": int(val_meta["sources"]),
            "test": int(test_meta["sources"]),
        },
        "rows": {
            "val": int(val_meta["rows"]),
            "test": int(test_meta["rows"]),
        },
        "acceptance": {
            "val": val_meta["acceptance"],
            "test": test_meta["acceptance"],
        },
        "policy": val_meta["policy"],
    }
    (meta_dir / "candidate_pools_meta.json").write_text(json.dumps(meta, indent=2))

    logger.info("=== CANDIDATE POOLS COMPLETE ===")
    logger.info(f"Val:  {val_path} ({len(val_df):,} rows)")
    logger.info(f"Test: {test_path} ({len(test_df):,} rows)")


if __name__ == "__main__":
    main()
