#!/usr/bin/env python3
"""
Chapter 3: package SCR-based disclosed + predicted layers into clean Parquets with flags.

Inputs (current):
  - Prime matches: artifacts/ch3/matching/prime_matches_thr90.parquet
  - SCR node mapping: data/processed/core/releases/core_v1/mapping/entity_map.parquet
  - SCR edges: data/processed/core/releases/core_v1/splits/{train,val,test}_edges.parquet
  - Predicted edges (rank-based, canonical upstream): artifacts/ch3/prediction_upstream/top{K}_pred.parquet

Outputs (written under --out-root, default artifacts/ch3/network_upstream):
  - tier1_primes_scr.parquet
  - edges_disclosed_scr.parquet
  - edges_predicted_scr_top5.parquet / top10 / top50
  - edges_scr_layered_top5.parquet / top10 / top50
  - nodes_scr_layered_top5.parquet / top10 / top50
  - dod_scr_network_edges_top5.parquet
  - dod_scr_network_nodes_top5.parquet
  - summary_scr_layers.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def main() -> None:
    p = argparse.ArgumentParser(description="Package Chapter 3 SCR disclosed + predicted layers")
    p.add_argument("--out-root", default="artifacts/ch3/network_upstream")
    p.add_argument("--prime-matches", default="artifacts/ch3/matching/prime_matches_thr90.parquet")
    p.add_argument(
        "--entity-map", default="data/processed/core/releases/core_v1/mapping/entity_map.parquet"
    )
    p.add_argument(
        "--train-edges", default="data/processed/core/releases/core_v1/splits/train_edges.parquet"
    )
    p.add_argument(
        "--val-edges", default="data/processed/core/releases/core_v1/splits/val_edges.parquet"
    )
    p.add_argument(
        "--test-edges", default="data/processed/core/releases/core_v1/splits/test_edges.parquet"
    )
    p.add_argument(
        "--node-features",
        default="data/processed/core/releases/core_v1/features/node_features_T0.parquet",
    )
    p.add_argument(
        "--node-structural",
        default="data/processed/core/releases/core_v1/features/node_structural_v1.parquet",
    )
    p.add_argument("--pred-top5", default="artifacts/ch3/prediction_upstream/top5_pred.parquet")
    p.add_argument("--pred-top10", default="artifacts/ch3/prediction_upstream/top10_pred.parquet")
    p.add_argument("--pred-top50", default="artifacts/ch3/prediction_upstream/top50_pred.parquet")
    p.add_argument("--primary-k", type=int, default=5, choices=[5, 10, 50])
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = {k: Path(v) for k, v in vars(args).items() if k not in {"out_root", "primary_k"}}
    for name, path in paths.items():
        if name.startswith("pred_") and not path.exists():
            raise FileNotFoundError(f"Missing predicted edge file: {path}")
        if name.endswith("matches") and not path.exists():
            raise FileNotFoundError(f"Missing prime matches: {path}")

    conn = duckdb.connect(database=":memory:")
    conn.execute("SET preserve_insertion_order=false;")

    # 1) Deduped Tier-1 primes that map into SCR (FactSet canonical_id -> node_id)
    tier1_path = out_root / "tier1_primes_scr.parquet"
    conn.execute(
        f"""
        COPY (
          WITH mapped AS (
            SELECT DISTINCT
              pm.factset_entity_id,
              em.node_id
            FROM read_parquet('{Path(args.prime_matches).as_posix()}') pm
            JOIN read_parquet('{Path(args.entity_map).as_posix()}') em
              ON pm.factset_entity_id = em.canonical_id
            WHERE pm.factset_entity_id IS NOT NULL
          ),
          vendor_counts AS (
            SELECT
              em.node_id,
              COUNT(*) AS num_vendors_mapped,
              MIN(pm.recipient_name) AS example_recipient_name
            FROM read_parquet('{Path(args.prime_matches).as_posix()}') pm
            JOIN read_parquet('{Path(args.entity_map).as_posix()}') em
              ON pm.factset_entity_id = em.canonical_id
            GROUP BY em.node_id
          )
          SELECT
            m.node_id,
            vc.num_vendors_mapped,
            vc.example_recipient_name,
            m.factset_entity_id
          FROM mapped m
          JOIN vendor_counts vc USING (node_id)
        ) TO '{tier1_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    # 2) Disclosed edges from Tier-1 SCR primes (deduped on src_id,dst_id with ts min/max)
    disclosed_path = out_root / "edges_disclosed_scr.parquet"
    conn.execute(
        f"""
        COPY (
          WITH tier1 AS (
            SELECT DISTINCT node_id AS src_id FROM read_parquet('{tier1_path.as_posix()}')
          ),
          edges AS (
            SELECT src_id, dst_id, ts FROM read_parquet('{Path(args.train_edges).as_posix()}')
            UNION ALL
            SELECT src_id, dst_id, ts FROM read_parquet('{Path(args.val_edges).as_posix()}')
            UNION ALL
            SELECT src_id, dst_id, ts FROM read_parquet('{Path(args.test_edges).as_posix()}')
          )
          SELECT
            e.src_id,
            e.dst_id,
            'disclosed' AS edge_type,
            NULL::INTEGER AS policy_k,
            NULL::DOUBLE AS score,
            MIN(e.ts) AS ts_min,
            MAX(e.ts) AS ts_max
          FROM edges e
          JOIN tier1 t USING (src_id)
          GROUP BY e.src_id, e.dst_id
        ) TO '{disclosed_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    def write_pred(k: int, pred_path: Path) -> Path:
        out_path = out_root / f"edges_predicted_scr_top{k}.parquet"
        conn.execute(
            f"""
            COPY (
              SELECT
                src_id,
                dst_id,
                'predicted_scr' AS edge_type,
                {k}::INTEGER AS policy_k,
                meta_prob AS score,
                NULL::INTEGER AS ts_min,
                NULL::INTEGER AS ts_max
              FROM read_parquet('{pred_path.as_posix()}')
            ) TO '{out_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
            """
        )
        return out_path

    pred5 = write_pred(5, Path(args.pred_top5))
    pred10 = write_pred(10, Path(args.pred_top10))
    pred50 = write_pred(50, Path(args.pred_top50))

    # 3) Union: disclosed + predicted (layered)
    def write_layered(k: int, pred_edges: Path) -> Path:
        out_path = out_root / f"edges_scr_layered_top{k}.parquet"
        conn.execute(
            f"""
            COPY (
              SELECT * FROM read_parquet('{disclosed_path.as_posix()}')
              UNION ALL
              SELECT * FROM read_parquet('{pred_edges.as_posix()}')
            ) TO '{out_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
            """
        )
        return out_path

    layered5 = write_layered(5, pred5)
    layered10 = write_layered(10, pred10)
    layered50 = write_layered(50, pred50)

    # 4) Node tables per layered network (join to core_v1 node attributes)
    def write_nodes(k: int, layered_edges: Path) -> Path:
        out_path = out_root / f"nodes_scr_layered_top{k}.parquet"
        conn.execute(
            f"""
            COPY (
              WITH nodes AS (
                SELECT DISTINCT src_id AS node_id FROM read_parquet('{layered_edges.as_posix()}')
                UNION
                SELECT DISTINCT dst_id AS node_id FROM read_parquet('{layered_edges.as_posix()}')
              ),
              feat AS (
                SELECT * FROM read_parquet('{Path(args.node_features).as_posix()}')
              ),
              st AS (
                SELECT * FROM read_parquet('{Path(args.node_structural).as_posix()}')
              )
              SELECT
                n.node_id,
                (t.node_id IS NOT NULL) AS is_tier1_prime,
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
              FROM nodes n
              LEFT JOIN read_parquet('{tier1_path.as_posix()}') t ON n.node_id = t.node_id
              LEFT JOIN feat ON n.node_id = feat.node_id
              LEFT JOIN st ON n.node_id = st.node_id
            ) TO '{out_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
            """
        )
        return out_path

    nodes5 = write_nodes(5, layered5)
    nodes10 = write_nodes(10, layered10)
    nodes50 = write_nodes(50, layered50)

    # 4b) Canonical "network so far" files (Top-5 is primary operating point)
    # Include isolated Tier-1 primes even if they have zero disclosed/predicted edges.
    primary_k = int(args.primary_k)
    primary_edges = {5: layered5, 10: layered10, 50: layered50}[primary_k]
    primary_nodes = {5: nodes5, 10: nodes10, 50: nodes50}[primary_k]

    canonical_edges = out_root / f"dod_scr_network_edges_top{primary_k}.parquet"
    canonical_nodes = out_root / f"dod_scr_network_nodes_top{primary_k}.parquet"
    conn.execute(
        f"""
        COPY (
          SELECT * FROM read_parquet('{primary_edges.as_posix()}')
        ) TO '{canonical_edges.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )
    conn.execute(
        f"""
        COPY (
          WITH from_edges AS (
            SELECT * FROM read_parquet('{primary_nodes.as_posix()}')
          ),
          tier1 AS (
            SELECT node_id FROM read_parquet('{tier1_path.as_posix()}')
          ),
          missing AS (
            SELECT t.node_id
            FROM tier1 t
            LEFT JOIN from_edges n ON t.node_id = n.node_id
            WHERE n.node_id IS NULL
          ),
          feat AS (
            SELECT * FROM read_parquet('{Path(args.node_features).as_posix()}')
          ),
          st AS (
            SELECT * FROM read_parquet('{Path(args.node_structural).as_posix()}')
          )
          SELECT * FROM from_edges
          UNION ALL
          SELECT
            m.node_id,
            TRUE AS is_tier1_prime,
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
          FROM missing m
          LEFT JOIN feat ON m.node_id = feat.node_id
          LEFT JOIN st ON m.node_id = st.node_id
        ) TO '{canonical_nodes.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd');
        """
    )

    # 5) Summary JSON
    def count(path: Path) -> int:
        return int(
            conn.execute(f"SELECT COUNT(*) FROM read_parquet('{path.as_posix()}')").fetchone()[0]
        )

    def distinct_nodes(path: Path) -> int:
        return int(
            conn.execute(
                f"""
                WITH nodes AS (
                  SELECT DISTINCT src_id AS node_id FROM read_parquet('{path.as_posix()}')
                  UNION
                  SELECT DISTINCT dst_id AS node_id FROM read_parquet('{path.as_posix()}')
                )
                SELECT COUNT(*) FROM nodes
                """
            ).fetchone()[0]
        )

    summary = {
        "tier1_primes_in_scr": count(tier1_path),
        "edges_disclosed_dedup": count(disclosed_path),
        "edges_pred_top5": count(pred5),
        "edges_pred_top10": count(pred10),
        "edges_pred_top50": count(pred50),
        "edges_layered_top5": count(layered5),
        "edges_layered_top10": count(layered10),
        "edges_layered_top50": count(layered50),
        "nodes_layered_top5": count(nodes5),
        "nodes_layered_top10": count(nodes10),
        "nodes_layered_top50": count(nodes50),
        "unique_nodes_layered_top5": distinct_nodes(layered5),
        "unique_nodes_layered_top10": distinct_nodes(layered10),
        "unique_nodes_layered_top50": distinct_nodes(layered50),
        "canonical_primary_k": primary_k,
        "canonical_edges": str(canonical_edges),
        "canonical_nodes": str(canonical_nodes),
        "canonical_nodes_rows": count(canonical_nodes),
    }
    (out_root / "summary_scr_layers.json").write_text(json.dumps(summary, indent=2))

    conn.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
