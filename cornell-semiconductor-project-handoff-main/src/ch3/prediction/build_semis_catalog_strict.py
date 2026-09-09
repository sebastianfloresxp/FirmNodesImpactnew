#!/usr/bin/env python3
"""
Chapter 3: build strict semiconductor catalog using RBICS L4 (segments) + SIC 3674 fallback.

Outputs (default under artifacts/ch3/reference):
  - semis_catalog_strict_entities.parquet
  - semis_catalog_strict_l4.parquet
  - semis_flags_strict.parquet (node_uid-level for graph pruning)
  - semis_catalog_strict_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.db_client.connection import get_engine


def load_config(path: Path) -> dict[str, list[str]]:
    cfg = yaml.safe_load(path.read_text())
    return {k: [str(x) for x in v] for k, v in cfg.items()}


def fetch_rbics_l4_entities(engine, l4_codes: list[str]) -> pd.DataFrame:
    codes_str = ",".join(f"'{c}'" for c in l4_codes)
    sql = f"""
    SELECT DISTINCT r.factset_entity_id, s.l4_id, s.l4_name
    FROM rbics_v1.rbics_bus_seg_report r
    JOIN rbics_v1.rbics_bus_seg_item i ON r.report_id = i.report_id
    JOIN rbics_v1.rbics_structure s ON i.l6_id = s.l6_id
    WHERE s.l4_id IN ({codes_str})
    """
    return pd.read_sql(sql, engine)


def fetch_sic_entities(engine) -> pd.DataFrame:
    sql = """
    SELECT DISTINCT factset_entity_id, primary_sic_code
    FROM sym_v1.sym_entity_sector
    WHERE primary_sic_code IN (3674)
    """
    return pd.read_sql(sql, engine)


def build_node_factset_map(nodes: pd.DataFrame, entity_map: pd.DataFrame) -> pd.DataFrame:
    nodes = nodes.copy()
    emap = entity_map[["canonical_id", "node_id"]].drop_duplicates("node_id")
    emap["node_id"] = emap["node_id"].astype("Int64")
    nodes["scr_node_id"] = nodes["scr_node_id"].astype("Int64")
    nodes = nodes.merge(
        emap,
        left_on="scr_node_id",
        right_on="node_id",
        how="left",
        suffixes=("", "_emap"),
    )
    nodes["node_factset_id"] = nodes["factset_entity_id"].where(
        nodes["factset_entity_id"].notna(), nodes["canonical_id"]
    )
    return nodes.drop(columns=["node_id"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build strict semis catalog (RBICS L4 + SIC 3674 fallback)"
    )
    ap.add_argument("--config", type=Path, default=Path("src/ch3/config/semiconductor_codes.yml"))
    ap.add_argument(
        "--nodes",
        type=Path,
        default=Path("artifacts/ch3/network_upstream/dod_network_nodes_top5_d99_shipping.parquet"),
        help="Canonical DoD network nodes to map to node_uid",
    )
    ap.add_argument(
        "--entity-map",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/mapping/entity_map.parquet"),
    )
    ap.add_argument("--out-root", type=Path, default=Path("artifacts/ch3/reference"))
    ap.add_argument(
        "--catalog-entities",
        type=Path,
        default=None,
        help="Pre-exported entity catalog parquet (offline mode)",
    )
    ap.add_argument(
        "--catalog-l4",
        type=Path,
        default=None,
        help="Pre-exported RBICS L4 long table parquet (offline mode)",
    )
    args = ap.parse_args()

    if (
        args.catalog_entities
        and args.catalog_l4
        and args.catalog_entities.exists()
        and args.catalog_l4.exists()
    ):
        semis = pd.read_parquet(args.catalog_entities)
        rbics_df = pd.read_parquet(args.catalog_l4)
        print("Offline mode: loaded catalog from pre-exported parquets")
    else:
        cfg = load_config(args.config)
        l4_codes = [str(x) for x in cfg.get("rbics_core", [])]
        if not l4_codes:
            raise ValueError("rbics_core list is empty; cannot build strict semis catalog.")

        engine = get_engine()
        rbics_df = fetch_rbics_l4_entities(engine, l4_codes)
        sic_df = fetch_sic_entities(engine)

        # L4 long table
        rbics_df = rbics_df.drop_duplicates(["factset_entity_id", "l4_id"])
        rbics_df["factset_entity_id"] = rbics_df["factset_entity_id"].astype(str)
        rbics_df["l4_id"] = rbics_df["l4_id"].astype(str)

        # Entity-level catalog
        rbics_grouped = (
            rbics_df.groupby("factset_entity_id")
            .agg(
                rbics_l4_ids=("l4_id", lambda x: "|".join(sorted(set(x)))),
                rbics_l4_names=("l4_name", lambda x: "|".join(sorted(set(x)))),
            )
            .reset_index()
        )
        rbics_grouped["is_semi_rbics_l4"] = True

        sic_df = sic_df.drop_duplicates(["factset_entity_id"])
        sic_df["factset_entity_id"] = sic_df["factset_entity_id"].astype(str)
        sic_df["is_semi_sic3674"] = True

        semis = rbics_grouped.merge(
            sic_df[["factset_entity_id", "is_semi_sic3674"]],
            on="factset_entity_id",
            how="outer",
        )
        semis["is_semi_rbics_l4"] = semis["is_semi_rbics_l4"].fillna(False).astype(bool)
        semis["is_semi_sic3674"] = semis["is_semi_sic3674"].fillna(False).astype(bool)
        semis["is_semi_strict"] = semis["is_semi_rbics_l4"] | semis["is_semi_sic3674"]

    def _source(row) -> str:
        if row["is_semi_rbics_l4"] and row["is_semi_sic3674"]:
            return "rbics_l4+sic3674"
        if row["is_semi_rbics_l4"]:
            return "rbics_l4"
        if row["is_semi_sic3674"]:
            return "sic3674"
        return "none"

    semis["semi_source"] = semis.apply(_source, axis=1)

    # Node-level flags
    nodes = pd.read_parquet(args.nodes)
    entity_map = pd.read_parquet(args.entity_map)
    nodes = build_node_factset_map(nodes, entity_map)
    node_flags = nodes[["node_uid", "node_factset_id"]].rename(
        columns={"node_factset_id": "factset_entity_id"}
    )
    node_flags = node_flags.merge(semis, on="factset_entity_id", how="left")
    for col in ["is_semi_rbics_l4", "is_semi_sic3674", "is_semi_strict"]:
        node_flags[col] = node_flags[col].fillna(False).astype(bool)
    node_flags["semi_source"] = node_flags["semi_source"].fillna("none")

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    catalog_entities = out_root / "semis_catalog_strict_entities.parquet"
    catalog_l4 = out_root / "semis_catalog_strict_l4.parquet"
    semis_flags = out_root / "semis_flags_strict.parquet"
    summary_path = out_root / "semis_catalog_strict_summary.json"

    semis.to_parquet(catalog_entities, index=False)
    rbics_df.to_parquet(catalog_l4, index=False)
    node_flags.to_parquet(semis_flags, index=False)

    summary = {
        "entities_rbics_l4": int(semis["is_semi_rbics_l4"].sum()),
        "entities_sic3674": int(semis["is_semi_sic3674"].sum()),
        "entities_strict_total": int(semis["is_semi_strict"].sum()),
        "nodes_strict_total": int(node_flags["is_semi_strict"].sum()),
        "outputs": {
            "catalog_entities": str(catalog_entities),
            "catalog_l4": str(catalog_l4),
            "semis_flags": str(semis_flags),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
