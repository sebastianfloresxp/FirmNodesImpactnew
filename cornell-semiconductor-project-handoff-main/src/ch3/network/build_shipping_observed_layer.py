#!/usr/bin/env python3
"""
Chapter 3: build observed shipping layer (parent rollup) anchored to the DoD network.

Outputs (default under artifacts/ch3/network):
  - edges_observed_shipping.parquet
  - nodes_observed_shipping.parquet
  - summary_observed_shipping.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def main() -> None:
    p = argparse.ArgumentParser(description="Build observed shipping layer (anchored)")
    p.add_argument("--out-root", default="artifacts/ch3/network_upstream")
    p.add_argument(
        "--shipping-edges",
        default="data/processed/shipping/processed/shipping_edges_parent.parquet",
    )
    p.add_argument(
        "--canonical-nodes",
        default="artifacts/ch3/network_upstream/dod_network_nodes_top5_d99.parquet",
    )
    p.add_argument(
        "--entity-map",
        default="data/processed/core/releases/core_v1/mapping/entity_map.parquet",
    )
    p.add_argument(
        "--node-features",
        default="data/processed/core/releases/core_v1/features/node_features_T0.parquet",
    )
    p.add_argument(
        "--node-structural",
        default="data/processed/core/releases/core_v1/features/node_structural_v1.parquet",
    )
    p.add_argument("--window-start", default="2021-10-01")
    p.add_argument("--window-end", default="2025-06-09")
    p.add_argument("--rollup", default="parent", choices=["parent"])
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    shipping_edges = Path(args.shipping_edges)
    canonical_nodes = Path(args.canonical_nodes)
    entity_map = Path(args.entity_map)
    node_features = Path(args.node_features)
    node_structural = Path(args.node_structural)
    for path in [shipping_edges, canonical_nodes, entity_map, node_features, node_structural]:
        if not path.exists():
            raise FileNotFoundError(path)

    edges_out = out_root / "edges_observed_shipping.parquet"
    nodes_out = out_root / "nodes_observed_shipping.parquet"
    summary_out = out_root / "summary_observed_shipping.json"

    conn = duckdb.connect(database=":memory:")
    conn.execute("SET preserve_insertion_order=false;")

    # FactSet -> SCR node_id lookup (canonical_id + raw_id).
    conn.execute(
        f"""
        CREATE TEMP VIEW entity_lookup AS
        SELECT canonical_id AS factset_id, node_id FROM read_parquet('{entity_map.as_posix()}')
        UNION ALL
        SELECT raw_id AS factset_id, node_id FROM read_parquet('{entity_map.as_posix()}')
        """
    )

    # Anchor FactSet IDs: primes (factset_entity_id) + SCR nodes in canonical network.
    conn.execute(
        f"""
        CREATE TEMP VIEW anchor_factset AS
        WITH prime_ids AS (
          SELECT DISTINCT factset_entity_id AS factset_id
          FROM read_parquet('{canonical_nodes.as_posix()}')
          WHERE node_type = 'prime_vendor' AND factset_entity_id IS NOT NULL
        ),
        scr_ids AS (
          SELECT DISTINCT scr_node_id AS node_id
          FROM read_parquet('{canonical_nodes.as_posix()}')
          WHERE node_type = 'scr_firm' AND scr_node_id IS NOT NULL
        ),
        scr_factset AS (
          SELECT DISTINCT el.factset_id
          FROM scr_ids s
          JOIN entity_lookup el ON s.node_id = el.node_id
        )
        SELECT DISTINCT factset_id FROM prime_ids
        UNION
        SELECT DISTINCT factset_id FROM scr_factset
        """
    )

    # Filter window and anchor to DoD network.
    conn.execute(
        f"""
        CREATE TEMP VIEW shipping_window AS
        SELECT
          source_factset_entity_id AS src_fid,
          target_factset_entity_id AS dst_fid,
          record_date
        FROM read_parquet('{shipping_edges.as_posix()}')
        WHERE record_date BETWEEN DATE '{args.window_start}' AND DATE '{args.window_end}'
        """
    )
    conn.execute(
        """
        CREATE TEMP VIEW shipping_anchored AS
        SELECT w.*
        FROM shipping_window w
        WHERE w.src_fid IN (SELECT factset_id FROM anchor_factset)
           OR w.dst_fid IN (SELECT factset_id FROM anchor_factset)
        """
    )

    # Map to node_uids, preferring SCR node IDs when available.
    conn.execute(
        """
        CREATE TEMP VIEW shipping_mapped AS
        SELECT
          COALESCE('scr:' || CAST(src_map.node_id AS VARCHAR), 'factset:' || src_fid) AS src_uid,
          COALESCE('scr:' || CAST(dst_map.node_id AS VARCHAR), 'factset:' || dst_fid) AS dst_uid,
          record_date
        FROM shipping_anchored a
        LEFT JOIN entity_lookup src_map ON a.src_fid = src_map.factset_id
        LEFT JOIN entity_lookup dst_map ON a.dst_fid = dst_map.factset_id
        """
    )

    # Aggregate to observed edges.
    conn.execute(
        f"""
        COPY (
          SELECT
            src_uid,
            dst_uid,
            'observed_shipping' AS edge_type,
            COUNT(*)::BIGINT AS shipments_count,
            MIN(record_date) AS first_ship_date,
            MAX(record_date) AS last_ship_date,
            DATE '{args.window_start}' AS window_start,
            DATE '{args.window_end}' AS window_end,
            '{args.rollup}' AS rollup_level
          FROM shipping_mapped
          WHERE src_uid <> dst_uid
          GROUP BY src_uid, dst_uid
        ) TO '{edges_out.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    # Build nodes for shipping endpoints (scr + factset-only).
    conn.execute(
        f"""
        CREATE TEMP VIEW ship_nodes AS
        WITH all_nodes AS (
          SELECT DISTINCT src_uid AS node_uid FROM read_parquet('{edges_out.as_posix()}')
          UNION
          SELECT DISTINCT dst_uid AS node_uid FROM read_parquet('{edges_out.as_posix()}')
        ),
        scr_ids AS (
          SELECT
            node_uid,
            CAST(replace(node_uid, 'scr:', '') AS BIGINT) AS node_id
          FROM all_nodes
          WHERE node_uid LIKE 'scr:%'
        ),
        factset_ids AS (
          SELECT
            node_uid,
            replace(node_uid, 'factset:', '') AS factset_entity_id
          FROM all_nodes
          WHERE node_uid LIKE 'factset:%'
        )
        SELECT
          'scr:' || CAST(s.node_id AS VARCHAR) AS node_uid,
          'scr_firm' AS node_type,
          NULL::VARCHAR AS name,
          NULL::VARCHAR AS vendor_key,
          NULL::VARCHAR AS factset_entity_id,
          NULL::VARCHAR AS match_method,
          NULL::DOUBLE AS match_score,
          s.node_id AS scr_node_id,
          feat.entity_type,
          feat.gr_country,
          feat.gr_region,
          feat.gr_continent,
          feat.primary_sic_code,
          feat.industry_code,
          feat.sector_code,
          feat.l1_id,
          feat.l2_id,
          feat.l3_id,
          st.logdeg_in,
          st.logdeg_out,
          st.pagerank,
          st.hits_auth,
          st.hits_hub
        FROM scr_ids s
        LEFT JOIN read_parquet('{node_features.as_posix()}') feat ON s.node_id = feat.node_id
        LEFT JOIN read_parquet('{node_structural.as_posix()}') st ON s.node_id = st.node_id
        UNION ALL
        SELECT
          f.node_uid,
          'factset_firm' AS node_type,
          NULL::VARCHAR AS name,
          NULL::VARCHAR AS vendor_key,
          f.factset_entity_id,
          NULL::VARCHAR AS match_method,
          NULL::DOUBLE AS match_score,
          NULL::BIGINT AS scr_node_id,
          NULL::VARCHAR AS entity_type,
          NULL::VARCHAR AS gr_country,
          NULL::VARCHAR AS gr_region,
          NULL::VARCHAR AS gr_continent,
          NULL::INTEGER AS primary_sic_code,
          NULL::VARCHAR AS industry_code,
          NULL::VARCHAR AS sector_code,
          NULL::INTEGER AS l1_id,
          NULL::INTEGER AS l2_id,
          NULL::INTEGER AS l3_id,
          NULL::DOUBLE AS logdeg_in,
          NULL::DOUBLE AS logdeg_out,
          NULL::DOUBLE AS pagerank,
          NULL::DOUBLE AS hits_auth,
          NULL::DOUBLE AS hits_hub
        FROM factset_ids f
        """
    )

    conn.execute(
        f"""
        COPY (
          SELECT * FROM ship_nodes
        ) TO '{nodes_out.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    def count(path: Path) -> int:
        return int(
            conn.execute(f"SELECT COUNT(*) FROM read_parquet('{path.as_posix()}')").fetchone()[0]
        )

    summary = {
        "window_start": args.window_start,
        "window_end": args.window_end,
        "rollup": args.rollup,
        "edges_observed": count(edges_out),
        "nodes_observed": count(nodes_out),
        "nodes_scr": int(
            conn.execute("SELECT COUNT(*) FROM ship_nodes WHERE node_type='scr_firm'").fetchone()[0]
        ),
        "nodes_factset_only": int(
            conn.execute(
                "SELECT COUNT(*) FROM ship_nodes WHERE node_type='factset_firm'"
            ).fetchone()[0]
        ),
        "anchor_factset_ids": int(
            conn.execute("SELECT COUNT(*) FROM anchor_factset").fetchone()[0]
        ),
        "outputs": {
            "edges": str(edges_out),
            "nodes": str(nodes_out),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2))
    conn.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
