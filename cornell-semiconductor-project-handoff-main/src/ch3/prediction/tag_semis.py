#!/usr/bin/env python3
"""Tag core_v1 nodes with semiconductor flags from RBICS/SIC codes (DB-based).

Pulls RBICS L4/L3 codes and SIC codes from FactSet SQL, filters to configured
code lists, maps to core_v1 node_ids via entity_map.parquet, and writes a
semis_flags.parquet with is_semi_core/is_semi_adjacent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from db_client.connection import get_engine


def load_codes(config_path: Path) -> dict[str, set[str]]:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    return {k: {str(x) for x in v} for k, v in cfg.items()}


def fetch_rbics_entities(codes: set[str], engine) -> set[str]:
    """Fetch factset_entity_id for RBICS l2 codes (sym_entity_sector_rbics only has l2)."""
    if not codes:
        return set()
    codes_str = ",".join(f"'{c}'" for c in codes)
    sql = f"SELECT DISTINCT factset_entity_id FROM sym_v1.sym_entity_sector_rbics WHERE l2_id IN ({codes_str})"
    df = pd.read_sql(sql, engine)
    return set(df["factset_entity_id"].astype(str).tolist())


def fetch_sic_entities(core: set[str], adj: set[str], engine) -> dict[str, str]:
    codes = core.union(adj)
    if not codes:
        return {}
    codes_str = ",".join(f"'{c}'" for c in codes)
    sql = f"SELECT factset_entity_id, primary_sic_code FROM sym_v1.sym_entity_sector WHERE primary_sic_code IN ({codes_str})"
    df = pd.read_sql(sql, engine)
    return dict(
        zip(df["factset_entity_id"].astype(str), df["primary_sic_code"].astype(str), strict=False)
    )


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Tag semiconductors (core/adjacent) from RBICS/SIC")
    ap.add_argument("--config", type=Path, default=Path("src/ch3/config/semiconductor_codes.yml"))
    ap.add_argument(
        "--entity-map",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/mapping/entity_map.parquet"),
    )
    ap.add_argument("--out", type=Path, default=Path("artifacts/ch3/reference/semis_flags.parquet"))
    args = ap.parse_args()

    codes = load_codes(args.config)
    engine = get_engine()

    rbics_core_ids = fetch_rbics_entities(codes.get("rbics_core", set()), engine)
    rbics_adj_ids = fetch_rbics_entities(codes.get("rbics_adjacent", set()), engine)
    sic_map = fetch_sic_entities(
        codes.get("sic_core", set()), codes.get("sic_adjacent", set()), engine
    )

    emap = pd.read_parquet(args.entity_map)[["canonical_id", "node_id"]]
    fid_to_node = dict(zip(emap["canonical_id"].astype(str), emap["node_id"], strict=False))

    records = []
    for fid, sic in sic_map.items():
        nid = fid_to_node.get(fid)
        if nid is None:
            continue
        is_core = sic in codes.get("sic_core", set())
        is_adj = sic in codes.get("sic_adjacent", set())
        if not (is_core or is_adj):
            continue
        records.append((nid, fid, is_core, is_adj))
    for fid in rbics_core_ids:
        nid = fid_to_node.get(fid)
        if nid is None:
            continue
        records.append((nid, fid, True, False))
    for fid in rbics_adj_ids:
        nid = fid_to_node.get(fid)
        if nid is None:
            continue
        records.append((nid, fid, False, True))

    df = pd.DataFrame(
        records, columns=["node_id", "factset_entity_id", "is_semi_core", "is_semi_adjacent"]
    )
    df = df.drop_duplicates(subset=["node_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    summary = {
        "rows": len(df),
        "core": int(df["is_semi_core"].sum()),
        "adjacent": int(df["is_semi_adjacent"].sum()),
    }
    with args.out.with_suffix(".json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {len(df)} rows to {args.out}")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
