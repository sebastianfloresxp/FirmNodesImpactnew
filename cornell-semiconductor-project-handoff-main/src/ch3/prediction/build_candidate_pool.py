#!/usr/bin/env python3
"""Build DoD candidate pools for structural predictions.

Sources: in-SCR primes (node_ids from core_v1)
Destinations: large undirected k-hop neighborhoods, plus optional semis, hubs, and random background,
with per-source budget to keep size tractable.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from dotenv import load_dotenv

logger = logging.getLogger("ch3.candidates")


def load_seeds(prime_matches: Path, entity_map: Path) -> pd.DataFrame:
    matches = pd.read_parquet(prime_matches)
    entity = pd.read_parquet(entity_map)[["canonical_id", "node_id"]]
    seeds = matches[matches["factset_entity_id"].notna()].merge(
        entity, left_on="factset_entity_id", right_on="canonical_id", how="inner"
    )
    seeds = seeds[["node_id", "vendor_key"]].drop_duplicates()
    seeds["node_id"] = seeds["node_id"].astype(int)
    return seeds


def load_adj(adj_path: Path) -> sp.csr_matrix:
    mat = sp.load_npz(adj_path)
    if not sp.isspmatrix_csr(mat):
        mat = mat.tocsr()
    # make undirected
    undirected = mat + mat.T
    undirected.eliminate_zeros()
    return undirected


def load_semi_flags(semis_path: Path | None, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    is_core = np.zeros(n_nodes, dtype=bool)
    is_adj = np.zeros(n_nodes, dtype=bool)
    if semis_path and semis_path.exists():
        df = pd.read_parquet(semis_path)
        if "node_id" in df.columns:
            if "is_semi_core" in df.columns:
                is_core[df["node_id"].astype(int).values] = df["is_semi_core"].astype(bool).values
            if "is_semi_adjacent" in df.columns:
                is_adj[df["node_id"].astype(int).values] = (
                    df["is_semi_adjacent"].astype(bool).values
                )
    return is_core, is_adj


def compute_hubs(adj: sp.csr_matrix, top_frac: float) -> set[int]:
    # Cast to signed so downstream degree-based ordering is stable (avoid uint64 overflow on negation).
    deg = np.asarray(adj.sum(axis=1)).ravel().astype(np.int64)
    k = max(1, int(len(deg) * top_frac))
    idx = np.argpartition(deg, -k)[-k:]
    return {int(i) for i in idx}


def bfs_neighbors(adj: sp.csr_matrix, src: int, hop: int) -> dict[int, int]:
    seen: dict[int, int] = {src: 0}
    frontier = {src}
    for depth in range(1, hop + 1):
        next_frontier: set[int] = set()
        for node in frontier:
            neighbors = adj.indices[adj.indptr[node] : adj.indptr[node + 1]]
            for nb in neighbors:
                if nb not in seen:
                    seen[nb] = depth
                    next_frontier.add(nb)
        if not next_frontier:
            break
        frontier = next_frontier
    seen.pop(src, None)
    return seen


def build_candidates_for_seed(
    src: int,
    dist_map: dict[int, int],
    semi_core_idx: Iterable[int],
    semi_adj_idx: Iterable[int],
    hubs: set[int],
    rng: np.random.Generator,
    budget: int,
    random_k: int,
    degrees: np.ndarray,
) -> pd.DataFrame:
    # priority: lower is better
    candidates: dict[int, tuple[int, float]] = {}
    for dst, dist in dist_map.items():
        candidates[dst] = (dist, -degrees[dst])
    # ensure semis included with high priority
    semi_core_idx_set = {int(i) for i in semi_core_idx}
    semi_adj_idx_set = {int(i) for i in semi_adj_idx}
    for idx in semi_core_idx_set:
        if idx not in candidates:
            candidates[idx] = (0, -degrees[idx])
        else:
            candidates[idx] = (0, candidates[idx][1])
    for idx in semi_adj_idx_set:
        if idx not in candidates:
            candidates[idx] = (1, -degrees[idx])
    # hubs
    for h in hubs:
        if h not in candidates:
            candidates[h] = (2, -degrees[h])
    # random background
    if random_k > 0:
        all_nodes = len(degrees)
        choices = rng.choice(all_nodes, size=min(random_k, all_nodes), replace=False)
        for r in choices:
            if r not in candidates:
                candidates[r] = (5, -degrees[r])
    # sort and cap
    sorted_cands = sorted(candidates.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    if budget and len(sorted_cands) > budget:
        sorted_cands = sorted_cands[:budget]
    rows = []
    for dst, (priority, _degneg) in sorted_cands:
        rows.append(
            {
                "src_id": src,
                "dst_id": int(dst),
                "priority": priority,
                "distance_hops": dist_map.get(dst),
                "is_semi_core": dst in semi_core_idx_set,
                "is_semi_adjacent": dst in semi_adj_idx_set,
                "is_hub": dst in hubs,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")
    ap = argparse.ArgumentParser(description="Build candidate pool for in-SCR primes")
    ap.add_argument(
        "--prime-matches", type=Path, required=True, help="prime_matches.parquet (0.90 run)"
    )
    ap.add_argument(
        "--entity-map",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/mapping/entity_map.parquet"),
    )
    ap.add_argument(
        "--adjacency",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/adjacency/train_adj_T0_csc.npz"),
    )
    ap.add_argument(
        "--semis",
        type=Path,
        default=None,
        help="Optional semis flags parquet with node_id, is_semi_core, is_semi_adjacent",
    )
    ap.add_argument("--hop", type=int, default=3, help="Undirected hop radius")
    ap.add_argument(
        "--budget", type=int, default=10000, help="Max candidates per source (after union)"
    )
    ap.add_argument("--include-adjacent", action="store_true", help="Include semi-adjacent nodes")
    ap.add_argument(
        "--hub-top-frac", type=float, default=0.01, help="Top fraction by degree to treat as hubs"
    )
    ap.add_argument(
        "--random-per-source", type=int, default=500, help="Random background per source"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/ch3/prediction_upstream/candidates_prime_to_candidate.parquet"),
        help="Output candidate pairs (prime -> candidate). Use flip_candidates.py to orient as supplier->prime.",
    )
    args = ap.parse_args()

    seeds_df = load_seeds(args.prime_matches, args.entity_map)
    seeds = seeds_df["node_id"].tolist()
    logger.info("Seeds (in-SCR primes): %d", len(seeds))

    adj = load_adj(args.adjacency)
    n_nodes = adj.shape[0]
    # Cast to signed so we can safely negate degrees for descending-degree tie breaks.
    degrees = np.asarray(adj.sum(axis=1)).ravel().astype(np.int64)
    is_core, is_adj = load_semi_flags(args.semis, n_nodes)
    semi_core_idx = np.nonzero(is_core)[0].tolist()
    semi_adj_idx = np.nonzero(is_adj)[0].tolist() if args.include_adjacent else []
    hubs = compute_hubs(adj, args.hub_top_frac)
    rng = np.random.default_rng(args.seed)

    all_rows: list[pd.DataFrame] = []
    for i, src in enumerate(seeds, 1):
        dist_map = bfs_neighbors(adj, src, args.hop)
        df = build_candidates_for_seed(
            src=src,
            dist_map=dist_map,
            semi_core_idx=semi_core_idx,
            semi_adj_idx=semi_adj_idx,
            hubs=hubs,
            rng=rng,
            budget=args.budget,
            random_k=args.random_per_source,
            degrees=degrees,
        )
        all_rows.append(df)
        if i % 500 == 0:
            logger.info("Processed %d/%d seeds", i, len(seeds))

    out_df = pd.concat(all_rows, ignore_index=True)
    # Exporters expect a label column. For Chapter 3 inference, labels are placeholders
    # (ground truth is unknown in deployment), so we set label=0 for all pairs.
    out_df = out_df[["src_id", "dst_id"]].copy()
    out_df["label"] = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    logger.info("Wrote candidates: %s (rows=%d)", args.out, len(out_df))
    summary = {
        "seeds": len(seeds),
        "rows": len(out_df),
        "hop": args.hop,
        "budget": args.budget,
        "include_adjacent": args.include_adjacent,
        "hub_top_frac": args.hub_top_frac,
        "random_per_source": args.random_per_source,
    }
    with args.out.with_suffix(".json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
