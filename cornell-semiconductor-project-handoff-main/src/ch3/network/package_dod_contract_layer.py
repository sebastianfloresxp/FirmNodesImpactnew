#!/usr/bin/env python3
"""
Chapter 3: package Tier-0 DoD component nodes + Tier-1 prime vendor nodes into an
unweighted contract-layer graph, and optionally merge it with the current SCR
network snapshot (disclosed + predicted SCR Top-K).

This produces a single "network so far" file that includes:
  - DoD component nodes (Tier-0)
  - Prime vendor nodes (Tier-1, all 71,967 vendors)
  - Mapping edges from vendor -> SCR node_id where available
  - SCR disclosed/predicted edges (SCR node_id -> SCR node_id)

Outputs (default under artifacts/ch3/network):
  - dod_contract_nodes.parquet
  - dod_contract_edges.parquet
  - dod_network_nodes_top{K}.parquet
  - dod_network_edges_top{K}.parquet
  - summary_dod_network_top{K}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def main() -> None:
    p = argparse.ArgumentParser(
        description="Package DoD contract layer + merge with SCR network snapshot"
    )
    # Canonical build uses upstream-oriented prediction outputs.
    p.add_argument("--out-root", default="artifacts/ch3/network_upstream")
    p.add_argument(
        "--dod-primes", default="artifacts/ch3/usaspending/dod_primes_fy2022-2025_asof.parquet"
    )
    p.add_argument("--prime-matches", default="artifacts/ch3/matching/prime_matches_thr90.parquet")
    p.add_argument(
        "--entity-map", default="data/processed/core/releases/core_v1/mapping/entity_map.parquet"
    )
    p.add_argument(
        "--scr-nodes", default="artifacts/ch3/network_upstream/dod_scr_network_nodes_top5.parquet"
    )
    p.add_argument(
        "--scr-edges", default="artifacts/ch3/network_upstream/dod_scr_network_edges_top5.parquet"
    )
    p.add_argument(
        "--k", type=int, default=5, choices=[5, 10, 50], help="Which SCR Top-K snapshot to merge"
    )
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dod_primes = Path(args.dod_primes)
    prime_matches = Path(args.prime_matches)
    entity_map = Path(args.entity_map)
    scr_nodes = Path(args.scr_nodes)
    scr_edges = Path(args.scr_edges)
    for path in [dod_primes, prime_matches, entity_map, scr_nodes, scr_edges]:
        if not path.exists():
            raise FileNotFoundError(path)

    contract_nodes_path = out_root / "dod_contract_nodes.parquet"
    contract_edges_path = out_root / "dod_contract_edges.parquet"
    merged_nodes_path = out_root / f"dod_network_nodes_top{args.k}.parquet"
    merged_edges_path = out_root / f"dod_network_edges_top{args.k}.parquet"
    summary_path = out_root / f"summary_dod_network_top{args.k}.json"

    conn = duckdb.connect(database=":memory:")
    conn.execute("SET preserve_insertion_order=false;")

    # ----------------------- Contract layer nodes ------------------------ #
    # DoD components (Tier-0)
    # Use a slugged UID for stability (no spaces/punctuation in identifiers).
    conn.execute(
        f"""
        CREATE TABLE dod_nodes AS
        SELECT DISTINCT
          'dod:' || lower(regexp_replace(dod_component, '[^A-Za-z0-9]+', '_', 'g')) AS node_uid,
          'dod_component' AS node_type,
          dod_component AS name,
          NULL::VARCHAR AS vendor_key,
          NULL::VARCHAR AS factset_entity_id,
          NULL::VARCHAR AS match_method,
          NULL::DOUBLE AS match_score,
          NULL::BIGINT AS scr_node_id
        FROM read_parquet('{dod_primes.as_posix()}')
        WHERE dod_component IS NOT NULL;
        """
    )

    # Prime vendors (Tier-1): one node per vendor_key
    # Attach match status (FactSet id) and SCR node_id where available.
    conn.execute(
        f"""
        CREATE TABLE vendor_nodes AS
        SELECT
          'vendor:' || pm.vendor_key AS node_uid,
          'prime_vendor' AS node_type,
          pm.recipient_name AS name,
          pm.vendor_key,
          pm.factset_entity_id,
          pm.match_method,
          pm.match_score,
          em.node_id AS scr_node_id
        FROM read_parquet('{prime_matches.as_posix()}') pm
        LEFT JOIN read_parquet('{entity_map.as_posix()}') em
          ON pm.factset_entity_id = em.canonical_id;
        """
    )

    conn.execute(
        f"""
        COPY (
          SELECT * FROM dod_nodes
          UNION ALL
          SELECT * FROM vendor_nodes
        ) TO '{contract_nodes_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    # ----------------------- Contract layer edges ------------------------ #
    # Unweighted: a unique edge if a vendor has any prime award from a DoD component.
    conn.execute(
        f"""
        COPY (
          WITH base AS (
            SELECT
              dod_component,
              vendor_key,
              MIN(action_fy) AS fy_min,
              MAX(action_fy) AS fy_max
            FROM read_parquet('{dod_primes.as_posix()}')
            GROUP BY dod_component, vendor_key
          )
          SELECT
            'dod:' || lower(regexp_replace(dod_component, '[^A-Za-z0-9]+', '_', 'g')) AS src_uid,
            'vendor:' || vendor_key AS dst_uid,
            'dod_contract' AS edge_type,
            NULL::DOUBLE AS score,
            fy_min,
            fy_max
          FROM base
        ) TO '{contract_edges_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    # ----------------------- Merge with SCR snapshot --------------------- #
    # Create mapping edges (vendor -> scr node) where available.
    conn.execute(
        """
        CREATE TABLE vendor_to_scr_edges AS
        SELECT
          v.node_uid AS src_uid,
          'scr:' || CAST(v.scr_node_id AS VARCHAR) AS dst_uid,
          'maps_to_scr' AS edge_type,
          NULL::DOUBLE AS score,
          NULL::INTEGER AS fy_min,
          NULL::INTEGER AS fy_max
        FROM vendor_nodes v
        WHERE v.scr_node_id IS NOT NULL;
        """
    )

    # Convert SCR nodes/edges to UID space.
    conn.execute(
        f"""
        CREATE TABLE scr_nodes_uid AS
        SELECT
          'scr:' || CAST(node_id AS VARCHAR) AS node_uid,
          'scr_firm' AS node_type,
          NULL::VARCHAR AS name,
          NULL::VARCHAR AS vendor_key,
          NULL::VARCHAR AS factset_entity_id,
          NULL::VARCHAR AS match_method,
          NULL::DOUBLE AS match_score,
          node_id AS scr_node_id
        FROM read_parquet('{scr_nodes.as_posix()}');
        """
    )
    conn.execute(
        f"""
        CREATE TABLE scr_edges_uid AS
        SELECT
          'scr:' || CAST(src_id AS VARCHAR) AS src_uid,
          'scr:' || CAST(dst_id AS VARCHAR) AS dst_uid,
          edge_type,
          score,
          NULL::INTEGER AS fy_min,
          NULL::INTEGER AS fy_max
        FROM read_parquet('{scr_edges.as_posix()}');
        """
    )

    conn.execute(
        f"""
        COPY (
          SELECT * FROM read_parquet('{contract_edges_path.as_posix()}')
          UNION ALL
          SELECT * FROM vendor_to_scr_edges
          UNION ALL
          SELECT * FROM scr_edges_uid
        ) TO '{merged_edges_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    # Merge nodes: keep contract nodes + scr nodes. (Attributes for scr nodes live in the SCR node file.)
    conn.execute(
        f"""
        COPY (
          SELECT * FROM read_parquet('{contract_nodes_path.as_posix()}')
          UNION ALL
          SELECT * FROM scr_nodes_uid
        ) TO '{merged_nodes_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    # ------------------------------ Summary ------------------------------ #
    def count(path: Path) -> int:
        return int(
            conn.execute(f"SELECT COUNT(*) FROM read_parquet('{path.as_posix()}')").fetchone()[0]
        )

    summary = {
        "dod_components": int(conn.execute("SELECT COUNT(*) FROM dod_nodes").fetchone()[0]),
        "prime_vendors_total": int(conn.execute("SELECT COUNT(*) FROM vendor_nodes").fetchone()[0]),
        "prime_vendors_matched_factset": int(
            conn.execute(
                "SELECT COUNT(*) FROM vendor_nodes WHERE factset_entity_id IS NOT NULL"
            ).fetchone()[0]
        ),
        "prime_vendors_in_scr": int(
            conn.execute(
                "SELECT COUNT(*) FROM vendor_nodes WHERE scr_node_id IS NOT NULL"
            ).fetchone()[0]
        ),
        "dod_contract_edges": count(contract_edges_path),
        "vendor_to_scr_edges": int(
            conn.execute("SELECT COUNT(*) FROM vendor_to_scr_edges").fetchone()[0]
        ),
        "scr_edges_snapshot": int(conn.execute("SELECT COUNT(*) FROM scr_edges_uid").fetchone()[0]),
        "merged_nodes": count(merged_nodes_path),
        "merged_edges": count(merged_edges_path),
        "outputs": {
            "contract_nodes": str(contract_nodes_path),
            "contract_edges": str(contract_edges_path),
            "merged_nodes": str(merged_nodes_path),
            "merged_edges": str(merged_edges_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    conn.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
