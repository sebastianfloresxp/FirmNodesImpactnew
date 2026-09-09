#!/usr/bin/env python3
"""Build disclosed DoD baseline network (as-of filtered, depth-limited traversal).

Steps:
- Load matched primes (prime_matches.parquet) and obligations (dod_primes*.parquet).
- Map matched primes into core_v1 node_ids via entity_map.parquet.
- Filter SCR edges to an as-of date: start_date <= as_of AND (no end OR end >= as_of).
- Traverse upstream (supplier -> customer) from matched primes to depth D (default 3).
- Emit nodes/edges tables and a summary. Unmatched/matched-not-in-core primes are
  kept as Tier-1 isolates with in_scr=false.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


def load_primes(prime_matches: Path, primes_agg: Path, entity_map: Path) -> dict[str, pd.DataFrame]:
    """Load matched/unmatched primes and obligations; split by core presence."""
    matches = pd.read_parquet(prime_matches)
    obligations = pd.read_parquet(primes_agg)
    vendor_obs = (
        obligations.groupby("vendor_key")
        .agg(
            sum_total_dollars_obligated=("sum_total_dollars_obligated", "sum"),
            sum_federal_action_obligation=("sum_federal_action_obligation", "sum"),
            components=("dod_component", lambda x: sorted(set(x.dropna().tolist()))),
        )
        .reset_index()
    )
    primes = matches.merge(vendor_obs, on="vendor_key", how="left")
    entity = pd.read_parquet(entity_map)[["canonical_id", "node_id"]]
    primes_core = primes.merge(
        entity, left_on="factset_entity_id", right_on="canonical_id", how="inner"
    )
    primes_not_core = primes[
        (primes["factset_entity_id"].notna())
        & (~primes["factset_entity_id"].isin(primes_core["factset_entity_id"]))
    ].copy()
    primes_unmatched = primes[primes["factset_entity_id"].isna()].copy()
    return {
        "core": primes_core,
        "not_core": primes_not_core,
        "unmatched": primes_unmatched,
    }


def filter_edges(edges_path: Path, as_of: str) -> pd.DataFrame:
    """Filter edges active as-of date."""
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT src_id, dst_id, start_date, duration_days
        FROM read_parquet('{edges_path}')
        WHERE start_date <= DATE '{as_of}'
          AND (duration_days IS NULL OR start_date + (duration_days || ' days')::INTERVAL >= DATE '{as_of}')
        """
    ).fetchdf()
    con.close()
    return df


def traverse_upstream(
    edges: pd.DataFrame, seed_ids: list[int], max_depth: int
) -> dict[str, pd.DataFrame]:
    """Depth-limited BFS upstream (dst -> src)."""
    frontier: set[int] = set(seed_ids)
    visited: dict[int, int] = dict.fromkeys(seed_ids, 1)
    for depth in range(2, max_depth + 2):  # tiers start at 1 for primes
        mask = edges["dst_id"].isin(frontier)
        new_src = set(edges.loc[mask, "src_id"].tolist()) - set(visited.keys())
        if not new_src:
            break
        for nid in new_src:
            visited[nid] = depth
        frontier = new_src
    reachable = set(visited.keys())
    edges_sub = edges[(edges["src_id"].isin(reachable)) & (edges["dst_id"].isin(reachable))].copy()
    return {"tiers": visited, "edges": edges_sub}


def build_nodes(
    tiers: dict[int, int],
    primes_core: pd.DataFrame,
    primes_not_core: pd.DataFrame,
    primes_unmatched: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble node table with flags."""
    records = []
    tier_lookup = tiers
    # Core primes
    for _, row in primes_core.iterrows():
        nid = int(row["node_id"])
        records.append(
            {
                "node_id": nid,
                "factset_entity_id": row["factset_entity_id"],
                "vendor_key": row["vendor_key"],
                "node_type": "prime_core",
                "tier_from_dod": 1,
                "in_scr": True,
                "matched": True,
                "match_method": row.get("match_method"),
                "match_score": row.get("match_score"),
                "sum_total_dollars_obligated": row.get("sum_total_dollars_obligated"),
                "sum_federal_action_obligation": row.get("sum_federal_action_obligation"),
                "components": row.get("components"),
            }
        )
    # Not-core primes (no node_id)
    for _, row in primes_not_core.iterrows():
        records.append(
            {
                "node_id": None,
                "factset_entity_id": row["factset_entity_id"],
                "vendor_key": row["vendor_key"],
                "node_type": "prime_not_core",
                "tier_from_dod": 1,
                "in_scr": False,
                "matched": True,
                "match_method": row.get("match_method"),
                "match_score": row.get("match_score"),
                "sum_total_dollars_obligated": row.get("sum_total_dollars_obligated"),
                "sum_federal_action_obligation": row.get("sum_federal_action_obligation"),
                "components": row.get("components"),
            }
        )
    # Unmatched primes
    for _, row in primes_unmatched.iterrows():
        records.append(
            {
                "node_id": None,
                "factset_entity_id": None,
                "vendor_key": row["vendor_key"],
                "node_type": "prime_unmatched",
                "tier_from_dod": 1,
                "in_scr": False,
                "matched": False,
                "match_method": row.get("match_method"),
                "match_score": row.get("match_score"),
                "sum_total_dollars_obligated": row.get("sum_total_dollars_obligated"),
                "sum_federal_action_obligation": row.get("sum_federal_action_obligation"),
                "components": row.get("components"),
            }
        )
    # Suppliers reached
    for nid, tier in tier_lookup.items():
        if tier == 1:
            continue
        records.append(
            {
                "node_id": nid,
                "factset_entity_id": None,
                "vendor_key": None,
                "node_type": "supplier",
                "tier_from_dod": tier,
                "in_scr": True,
                "matched": False,
                "match_method": None,
                "match_score": None,
                "sum_total_dollars_obligated": None,
                "sum_federal_action_obligation": None,
                "components": None,
            }
        )
    return pd.DataFrame(records)


def write_summary(
    out_path: Path, nodes: pd.DataFrame, edges: pd.DataFrame, tiers: dict[int, int]
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "nodes_total": len(nodes),
        "edges_total": len(edges),
        "tiers": {str(t): int(list(tiers.values()).count(t)) for t in set(tiers.values())},
        "primes_core": int((nodes["node_type"] == "prime_core").sum()),
        "primes_not_core": int((nodes["node_type"] == "prime_not_core").sum()),
        "primes_unmatched": int((nodes["node_type"] == "prime_unmatched").sum()),
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build disclosed DoD baseline network")
    ap.add_argument("--prime-matches", type=Path, required=True, help="prime_matches.parquet")
    ap.add_argument(
        "--primes-agg", type=Path, required=True, help="dod_primes_fy*_asof.parquet (obligations)"
    )
    ap.add_argument(
        "--entity-map",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/mapping/entity_map.parquet"),
    )
    ap.add_argument(
        "--edges",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/splits/all_edges.parquet"),
    )
    ap.add_argument(
        "--as-of", type=str, default="2025-06-09", help="as-of date for active edges (YYYY-MM-DD)"
    )
    ap.add_argument("--depth", type=int, default=3, help="upstream traversal depth")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/ch3/network_upstream"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    primes = load_primes(args.prime_matches, args.primes_agg, args.entity_map)
    if primes["core"].empty:
        raise SystemExit("No primes mapped into core_v1; cannot traverse")
    seed_ids = primes["core"]["node_id"].astype(int).tolist()

    edges_df = filter_edges(args.edges, args.as_of)
    traversal = traverse_upstream(edges_df, seed_ids, args.depth)
    tiers = traversal["tiers"]
    edges_sub = traversal["edges"]

    nodes = build_nodes(tiers, primes["core"], primes["not_core"], primes["unmatched"])

    nodes_out = args.out_dir / f"dod_disclosed_nodes_asof_{args.as_of}_d{args.depth}.parquet"
    edges_out = args.out_dir / f"dod_disclosed_edges_asof_{args.as_of}_d{args.depth}.parquet"
    summary_out = args.out_dir / f"dod_disclosed_summary_asof_{args.as_of}_d{args.depth}.json"
    nodes.to_parquet(nodes_out, index=False)
    edges_sub.to_parquet(edges_out, index=False)
    write_summary(summary_out, nodes, edges_sub, tiers)
    print(f"Wrote nodes to {nodes_out}")
    print(f"Wrote edges to {edges_out}")
    print(f"Wrote summary to {summary_out}")


if __name__ == "__main__":
    main()
