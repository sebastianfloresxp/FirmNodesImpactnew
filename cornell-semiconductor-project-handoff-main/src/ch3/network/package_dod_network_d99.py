#!/usr/bin/env python3
"""
Chapter 3: package canonical DoD network using exhaustive disclosed edges (d99),
Top-K predicted edges, and the DoD contract layer.

Outputs (default under artifacts/ch3/network):
  - dod_network_edges_top{K}_d99.parquet
  - dod_network_nodes_top{K}_d99.parquet
  - summary_dod_network_top{K}_d99.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def main() -> None:
    p = argparse.ArgumentParser(
        description="Package canonical DoD network (d99 disclosed + Top-K predicted)"
    )
    p.add_argument("--out-root", default="artifacts/ch3/network_upstream")
    p.add_argument(
        "--contract-nodes", default="artifacts/ch3/network_upstream/dod_contract_nodes.parquet"
    )
    p.add_argument(
        "--contract-edges", default="artifacts/ch3/network_upstream/dod_contract_edges.parquet"
    )
    p.add_argument(
        "--disclosed-edges",
        default="artifacts/ch3/network_upstream/dod_disclosed_edges_asof_2025-06-09_d99.parquet",
    )
    p.add_argument(
        "--predicted-edges",
        default="artifacts/ch3/network_upstream/edges_predicted_scr_top5.parquet",
    )
    p.add_argument(
        "--node-features",
        default="data/processed/core/releases/core_v1/features/node_features_T0.parquet",
    )
    p.add_argument(
        "--node-structural",
        default="data/processed/core/releases/core_v1/features/node_structural_v1.parquet",
    )
    p.add_argument("--k", type=int, default=5, choices=[5, 10, 50])
    p.add_argument("--as-of", dest="as_of", default="2025-06-09")
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    contract_nodes = Path(args.contract_nodes)
    contract_edges = Path(args.contract_edges)
    disclosed_edges = Path(args.disclosed_edges)
    predicted_edges = Path(args.predicted_edges)
    node_features = Path(args.node_features)
    node_structural = Path(args.node_structural)
    for path in [
        contract_nodes,
        contract_edges,
        disclosed_edges,
        predicted_edges,
        node_features,
        node_structural,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    edges_out = out_root / f"dod_network_edges_top{args.k}_d99.parquet"
    nodes_out = out_root / f"dod_network_nodes_top{args.k}_d99.parquet"
    summary_out = out_root / f"summary_dod_network_top{args.k}_d99.json"

    conn = duckdb.connect(database=":memory:")
    conn.execute("SET preserve_insertion_order=false;")

    # Contract nodes/edges
    conn.execute(
        f"CREATE TABLE contract_nodes AS SELECT * FROM read_parquet('{contract_nodes.as_posix()}');"
    )
    conn.execute(
        f"CREATE TABLE contract_edges AS SELECT * FROM read_parquet('{contract_edges.as_posix()}');"
    )

    # Vendor -> SCR mapping edges (from contract nodes)
    conn.execute(
        """
        CREATE TABLE vendor_to_scr_edges AS
        SELECT
          node_uid AS src_uid,
          'scr:' || CAST(scr_node_id AS VARCHAR) AS dst_uid,
          NULL::DOUBLE AS pred_score,
          NULL::INTEGER AS pred_k,
          NULL::DATE AS disclosed_start_date,
          NULL::BIGINT AS disclosed_duration_days,
          NULL::INTEGER AS contract_fy_min,
          NULL::INTEGER AS contract_fy_max,
          0::INTEGER AS is_contract,
          1::INTEGER AS is_maps_to_scr,
          0::INTEGER AS is_disclosed,
          0::INTEGER AS is_predicted
        FROM contract_nodes
        WHERE node_type = 'prime_vendor' AND scr_node_id IS NOT NULL;
        """
    )

    # Disclosed SCR edges (d99, as-of filtered already)
    conn.execute(
        f"""
        CREATE TABLE disclosed_edges AS
        SELECT
          'scr:' || CAST(src_id AS VARCHAR) AS src_uid,
          'scr:' || CAST(dst_id AS VARCHAR) AS dst_uid,
          NULL::DOUBLE AS pred_score,
          NULL::INTEGER AS pred_k,
          CAST(start_date AS DATE) AS disclosed_start_date,
          duration_days::BIGINT AS disclosed_duration_days,
          NULL::INTEGER AS contract_fy_min,
          NULL::INTEGER AS contract_fy_max,
          0::INTEGER AS is_contract,
          0::INTEGER AS is_maps_to_scr,
          1::INTEGER AS is_disclosed,
          0::INTEGER AS is_predicted
        FROM read_parquet('{disclosed_edges.as_posix()}');
        """
    )

    # Predicted SCR edges (Top-K)
    conn.execute(
        f"""
        CREATE TABLE predicted_edges AS
        SELECT
          'scr:' || CAST(src_id AS VARCHAR) AS src_uid,
          'scr:' || CAST(dst_id AS VARCHAR) AS dst_uid,
          score AS pred_score,
          {int(args.k)}::INTEGER AS pred_k,
          NULL::DATE AS disclosed_start_date,
          NULL::BIGINT AS disclosed_duration_days,
          NULL::INTEGER AS contract_fy_min,
          NULL::INTEGER AS contract_fy_max,
          0::INTEGER AS is_contract,
          0::INTEGER AS is_maps_to_scr,
          0::INTEGER AS is_disclosed,
          1::INTEGER AS is_predicted
        FROM read_parquet('{predicted_edges.as_posix()}');
        """
    )

    # Contract edges (DoD -> vendor)
    conn.execute(
        """
        CREATE TABLE contract_edges_norm AS
        SELECT
          src_uid,
          dst_uid,
          NULL::DOUBLE AS pred_score,
          NULL::INTEGER AS pred_k,
          NULL::DATE AS disclosed_start_date,
          NULL::BIGINT AS disclosed_duration_days,
          fy_min::INTEGER AS contract_fy_min,
          fy_max::INTEGER AS contract_fy_max,
          1::INTEGER AS is_contract,
          0::INTEGER AS is_maps_to_scr,
          0::INTEGER AS is_disclosed,
          0::INTEGER AS is_predicted
        FROM contract_edges;
        """
    )

    # Deduped union with layer flags
    conn.execute(
        f"""
        COPY (
          WITH all_edges AS (
            SELECT * FROM contract_edges_norm
            UNION ALL
            SELECT * FROM vendor_to_scr_edges
            UNION ALL
            SELECT * FROM disclosed_edges
            UNION ALL
            SELECT * FROM predicted_edges
          )
          SELECT
            src_uid,
            dst_uid,
            MAX(is_contract) AS is_contract,
            MAX(is_maps_to_scr) AS is_maps_to_scr,
            MAX(is_disclosed) AS is_disclosed,
            MAX(is_predicted) AS is_predicted,
            MIN(contract_fy_min) AS fy_min,
            MAX(contract_fy_max) AS fy_max,
            MIN(disclosed_start_date) AS disclosed_start_date,
            MAX(disclosed_duration_days) AS disclosed_duration_days,
            MAX(pred_score) AS pred_score,
            MAX(pred_k) AS pred_k
          FROM all_edges
          GROUP BY src_uid, dst_uid
        ) TO '{edges_out.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    # SCR nodes referenced by disclosed or predicted edges
    conn.execute(
        f"""
        CREATE TABLE scr_nodes AS
        WITH scr_ids AS (
          SELECT DISTINCT CAST(replace(src_uid, 'scr:', '') AS BIGINT) AS node_id
          FROM disclosed_edges
          UNION
          SELECT DISTINCT CAST(replace(dst_uid, 'scr:', '') AS BIGINT) AS node_id
          FROM disclosed_edges
          UNION
          SELECT DISTINCT CAST(replace(src_uid, 'scr:', '') AS BIGINT) AS node_id
          FROM predicted_edges
          UNION
          SELECT DISTINCT CAST(replace(dst_uid, 'scr:', '') AS BIGINT) AS node_id
          FROM predicted_edges
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
        LEFT JOIN read_parquet('{node_structural.as_posix()}') st ON s.node_id = st.node_id;
        """
    )

    # Merge nodes (contract layer + SCR nodes)
    conn.execute(
        f"""
        COPY (
          SELECT
            node_uid,
            node_type,
            name,
            vendor_key,
            factset_entity_id,
            match_method,
            match_score,
            scr_node_id,
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
          FROM contract_nodes
          UNION ALL
          SELECT * FROM scr_nodes
        ) TO '{nodes_out.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    # Summary
    def count(path: Path) -> int:
        return int(
            conn.execute(f"SELECT COUNT(*) FROM read_parquet('{path.as_posix()}')").fetchone()[0]
        )

    summary = {
        "as_of": args.as_of,
        "k": int(args.k),
        "edges_total_dedup": count(edges_out),
        "nodes_total": count(nodes_out),
        "edges_contract": int(
            conn.execute("SELECT COUNT(*) FROM contract_edges_norm").fetchone()[0]
        ),
        "edges_maps_to_scr": int(
            conn.execute("SELECT COUNT(*) FROM vendor_to_scr_edges").fetchone()[0]
        ),
        "edges_disclosed_d99": int(
            conn.execute("SELECT COUNT(*) FROM disclosed_edges").fetchone()[0]
        ),
        "edges_predicted_topk": int(
            conn.execute("SELECT COUNT(*) FROM predicted_edges").fetchone()[0]
        ),
        "nodes_contract": int(conn.execute("SELECT COUNT(*) FROM contract_nodes").fetchone()[0]),
        "nodes_scr": int(conn.execute("SELECT COUNT(*) FROM scr_nodes").fetchone()[0]),
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
