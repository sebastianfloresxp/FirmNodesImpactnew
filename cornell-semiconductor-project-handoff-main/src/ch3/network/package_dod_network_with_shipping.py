#!/usr/bin/env python3
"""
Chapter 3: merge canonical DoD network (d99 disclosed + Top-K predicted) with
observed shipping edges (anchored).

Outputs (default under artifacts/ch3/network):
  - dod_network_edges_top{K}_d99_shipping.parquet
  - dod_network_nodes_top{K}_d99_shipping.parquet
  - summary_dod_network_top{K}_d99_shipping.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def main() -> None:
    p = argparse.ArgumentParser(description="Package DoD network with observed shipping layer")
    p.add_argument("--out-root", default="artifacts/ch3/network_upstream")
    p.add_argument(
        "--canonical-edges",
        default="artifacts/ch3/network_upstream/dod_network_edges_top5_d99.parquet",
    )
    p.add_argument(
        "--canonical-nodes",
        default="artifacts/ch3/network_upstream/dod_network_nodes_top5_d99.parquet",
    )
    p.add_argument(
        "--shipping-edges", default="artifacts/ch3/network_upstream/edges_observed_shipping.parquet"
    )
    p.add_argument(
        "--shipping-nodes", default="artifacts/ch3/network_upstream/nodes_observed_shipping.parquet"
    )
    p.add_argument(
        "--contract-nodes", default="artifacts/ch3/network_upstream/dod_contract_nodes.parquet"
    )
    p.add_argument("--k", type=int, default=5, choices=[5, 10, 50])
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    canonical_edges = Path(args.canonical_edges)
    canonical_nodes = Path(args.canonical_nodes)
    shipping_edges = Path(args.shipping_edges)
    shipping_nodes = Path(args.shipping_nodes)
    contract_nodes = Path(args.contract_nodes)
    for path in [canonical_edges, canonical_nodes, shipping_edges, shipping_nodes, contract_nodes]:
        if not path.exists():
            raise FileNotFoundError(path)

    edges_out = out_root / f"dod_network_edges_top{args.k}_d99_shipping.parquet"
    nodes_out = out_root / f"dod_network_nodes_top{args.k}_d99_shipping.parquet"
    summary_out = out_root / f"summary_dod_network_top{args.k}_d99_shipping.json"

    conn = duckdb.connect(database=":memory:")
    conn.execute("SET preserve_insertion_order=false;")

    # Vendor -> FactSet mapping edges for primes not in SCR.
    conn.execute(
        f"""
        CREATE TEMP VIEW vendor_to_factset AS
        SELECT
          node_uid AS src_uid,
          'factset:' || factset_entity_id AS dst_uid,
          0::INTEGER AS is_contract,
          0::INTEGER AS is_maps_to_scr,
          1::INTEGER AS is_maps_to_factset,
          0::INTEGER AS is_disclosed,
          0::INTEGER AS is_predicted,
          0::INTEGER AS is_shipping,
          NULL::INTEGER AS fy_min,
          NULL::INTEGER AS fy_max,
          NULL::DATE AS disclosed_start_date,
          NULL::BIGINT AS disclosed_duration_days,
          NULL::DOUBLE AS pred_score,
          NULL::INTEGER AS pred_k,
          NULL::BIGINT AS shipments_count,
          NULL::DATE AS ship_first_date,
          NULL::DATE AS ship_last_date
        FROM read_parquet('{contract_nodes.as_posix()}')
        WHERE node_type = 'prime_vendor'
          AND factset_entity_id IS NOT NULL
          AND scr_node_id IS NULL
        """
    )

    # Canonical edges (add shipping columns + maps_to_factset)
    conn.execute(
        f"""
        CREATE TEMP VIEW canonical_edges AS
        SELECT
          src_uid,
          dst_uid,
          is_contract,
          is_maps_to_scr,
          0::INTEGER AS is_maps_to_factset,
          is_disclosed,
          is_predicted,
          0::INTEGER AS is_shipping,
          fy_min,
          fy_max,
          disclosed_start_date,
          disclosed_duration_days,
          pred_score,
          pred_k,
          NULL::BIGINT AS shipments_count,
          NULL::DATE AS ship_first_date,
          NULL::DATE AS ship_last_date
        FROM read_parquet('{canonical_edges.as_posix()}')
        """
    )

    # Shipping edges (observed)
    conn.execute(
        f"""
        CREATE TEMP VIEW shipping_edges AS
        SELECT
          src_uid,
          dst_uid,
          0::INTEGER AS is_contract,
          0::INTEGER AS is_maps_to_scr,
          0::INTEGER AS is_maps_to_factset,
          0::INTEGER AS is_disclosed,
          0::INTEGER AS is_predicted,
          1::INTEGER AS is_shipping,
          NULL::INTEGER AS fy_min,
          NULL::INTEGER AS fy_max,
          NULL::DATE AS disclosed_start_date,
          NULL::BIGINT AS disclosed_duration_days,
          NULL::DOUBLE AS pred_score,
          NULL::INTEGER AS pred_k,
          shipments_count,
          first_ship_date AS ship_first_date,
          last_ship_date AS ship_last_date
        FROM read_parquet('{shipping_edges.as_posix()}')
        """
    )

    # Merge edges with flags, dedup on src_uid,dst_uid.
    conn.execute(
        f"""
        COPY (
          WITH all_edges AS (
            SELECT * FROM canonical_edges
            UNION ALL
            SELECT * FROM vendor_to_factset
            UNION ALL
            SELECT * FROM shipping_edges
          )
          SELECT
            src_uid,
            dst_uid,
            MAX(is_contract) AS is_contract,
            MAX(is_maps_to_scr) AS is_maps_to_scr,
            MAX(is_maps_to_factset) AS is_maps_to_factset,
            MAX(is_disclosed) AS is_disclosed,
            MAX(is_predicted) AS is_predicted,
            MAX(is_shipping) AS is_shipping,
            MIN(fy_min) AS fy_min,
            MAX(fy_max) AS fy_max,
            MIN(disclosed_start_date) AS disclosed_start_date,
            MAX(disclosed_duration_days) AS disclosed_duration_days,
            MAX(pred_score) AS pred_score,
            MAX(pred_k) AS pred_k,
            MAX(shipments_count) AS shipments_count,
            MIN(ship_first_date) AS ship_first_date,
            MAX(ship_last_date) AS ship_last_date
          FROM all_edges
          GROUP BY src_uid, dst_uid
        ) TO '{edges_out.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    # Merge nodes (canonical + shipping nodes), dedup by node_uid.
    conn.execute(
        f"""
        COPY (
          WITH all_nodes AS (
            SELECT * FROM read_parquet('{canonical_nodes.as_posix()}')
            UNION ALL
            SELECT * FROM read_parquet('{shipping_nodes.as_posix()}')
          )
          SELECT
            node_uid,
            MAX(node_type) AS node_type,
            MAX(name) AS name,
            MAX(vendor_key) AS vendor_key,
            MAX(factset_entity_id) AS factset_entity_id,
            MAX(match_method) AS match_method,
            MAX(match_score) AS match_score,
            MAX(scr_node_id) AS scr_node_id,
            MAX(entity_type) AS entity_type,
            MAX(gr_country) AS gr_country,
            MAX(gr_region) AS gr_region,
            MAX(gr_continent) AS gr_continent,
            MAX(primary_sic_code) AS primary_sic_code,
            MAX(industry_code) AS industry_code,
            MAX(sector_code) AS sector_code,
            MAX(l1_id) AS l1_id,
            MAX(l2_id) AS l2_id,
            MAX(l3_id) AS l3_id,
            MAX(logdeg_in) AS logdeg_in,
            MAX(logdeg_out) AS logdeg_out,
            MAX(pagerank) AS pagerank,
            MAX(hits_auth) AS hits_auth,
            MAX(hits_hub) AS hits_hub
          FROM all_nodes
          GROUP BY node_uid
        ) TO '{nodes_out.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    def count(path: Path) -> int:
        return int(
            conn.execute(f"SELECT COUNT(*) FROM read_parquet('{path.as_posix()}')").fetchone()[0]
        )

    summary = {
        "k": int(args.k),
        "edges_total_dedup": count(edges_out),
        "nodes_total": count(nodes_out),
        "edges_contract": int(
            conn.execute("SELECT COUNT(*) FROM canonical_edges WHERE is_contract=1").fetchone()[0]
        ),
        "edges_maps_to_scr": int(
            conn.execute("SELECT COUNT(*) FROM canonical_edges WHERE is_maps_to_scr=1").fetchone()[
                0
            ]
        ),
        "edges_maps_to_factset": int(
            conn.execute("SELECT COUNT(*) FROM vendor_to_factset").fetchone()[0]
        ),
        "edges_disclosed_d99": int(
            conn.execute("SELECT COUNT(*) FROM canonical_edges WHERE is_disclosed=1").fetchone()[0]
        ),
        "edges_predicted_topk": int(
            conn.execute("SELECT COUNT(*) FROM canonical_edges WHERE is_predicted=1").fetchone()[0]
        ),
        "edges_observed_shipping": int(
            conn.execute("SELECT COUNT(*) FROM shipping_edges").fetchone()[0]
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
