#!/usr/bin/env python3
"""
Build an appendix-style LaTeX table for RBICS L4 semiconductor codes.

The table is intended to mirror `src/ch3/config/semiconductor_codes.yml` and add an
in-network coverage column: the number of unique firms (FactSet entity IDs) in the
locked DoD semiconductor network that are flagged by each RBICS L4 code.

Notes:
- Counts are computed at the firm level to avoid double-counting when a firm appears
  under multiple node representations (e.g., SCR node and/or prime vendor identity).
- Firms can report multiple RBICS L4 business segments, so a single firm may contribute
  to more than one L4 code count.

Output (default):
  - tables/chapter3/tab_app_semiconductor_codes_rbics_l4.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build appendix table: RBICS L4 semiconductor codes with coverage counts"
    )
    p.add_argument("--config", default="src/ch3/config/semiconductor_codes.yml")
    p.add_argument(
        "--nodes",
        default="artifacts/ch3/network_upstream/dod_semiconductor_nodes_top5_d99_shipping_strict.parquet",
        help="Locked strict DoD semiconductor network nodes parquet",
    )
    p.add_argument(
        "--entity-map",
        default="data/processed/core/releases/core_v1/mapping/entity_map.parquet",
        help="SCR node_id -> FactSet entity (canonical_id) map",
    )
    p.add_argument(
        "--catalog-l4",
        default="artifacts/ch3/reference/semis_catalog_strict_l4.parquet",
        help="Strict semis RBICS L4 long table (entity_id, l4_id, l4_name)",
    )
    p.add_argument(
        "--out",
        default="tables/chapter3/tab_app_semiconductor_codes_rbics_l4.tex",
        help="Output .tex file (tabular-only)",
    )
    return p.parse_args()


def load_codes(path: Path) -> tuple[list[str], list[str]]:
    cfg = yaml.safe_load(path.read_text())
    core = [str(x) for x in cfg.get("rbics_core", [])]
    adjacent = [str(x) for x in cfg.get("rbics_adjacent", [])]
    return core, adjacent


def get_entity_ids_in_network(nodes: pd.DataFrame, entity_map: pd.DataFrame) -> set[str]:
    """Return unique FactSet entity IDs represented in the locked network."""
    # factset_firm + prime_vendor nodes already carry factset_entity_id
    entity_ids = set(nodes["factset_entity_id"].dropna().astype(str))

    # scr_firm nodes need node_id -> canonical_id mapping
    scr = nodes[nodes["node_type"] == "scr_firm"].copy()
    if not scr.empty and "scr_node_id" in scr.columns:
        emap = entity_map.drop_duplicates("node_id").set_index("node_id")["canonical_id"]
        scr_ids = scr["scr_node_id"].dropna().astype(int)
        mapped = emap.loc[scr_ids].dropna().astype(str)
        entity_ids |= set(mapped.tolist())

    return entity_ids


def count_entities_by_l4(
    catalog_l4: pd.DataFrame, entity_ids_in_network: set[str]
) -> dict[str, int]:
    if catalog_l4.empty:
        return {}
    df = catalog_l4[catalog_l4["factset_entity_id"].astype(str).isin(entity_ids_in_network)].copy()
    if df.empty:
        return {}
    df["l4_id"] = df["l4_id"].astype(str)
    df["factset_entity_id"] = df["factset_entity_id"].astype(str)
    return (
        df.drop_duplicates(["factset_entity_id", "l4_id"])
        .groupby("l4_id")["factset_entity_id"]
        .nunique()
        .to_dict()
    )


def l4_name_lookup(catalog_l4: pd.DataFrame) -> dict[str, str]:
    if catalog_l4.empty:
        return {}
    df = catalog_l4.dropna(subset=["l4_id", "l4_name"]).copy()
    df["l4_id"] = df["l4_id"].astype(str)
    df["l4_name"] = df["l4_name"].astype(str)
    # Prefer most common name per L4.
    counts = (
        df.groupby(["l4_id", "l4_name"])
        .size()
        .reset_index(name="n")
        .sort_values(["l4_id", "n"], ascending=[True, False])
    )
    return counts.drop_duplicates("l4_id").set_index("l4_id")["l4_name"].to_dict()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    nodes_path = Path(args.nodes)
    entity_map_path = Path(args.entity_map)
    catalog_l4_path = Path(args.catalog_l4)
    out_path = Path(args.out)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not nodes_path.exists():
        raise FileNotFoundError(nodes_path)
    if not entity_map_path.exists():
        raise FileNotFoundError(entity_map_path)
    if not catalog_l4_path.exists():
        raise FileNotFoundError(catalog_l4_path)

    rbics_core, rbics_adj = load_codes(cfg_path)
    nodes = pd.read_parquet(
        nodes_path,
        columns=[
            "node_uid",
            "node_type",
            "scr_node_id",
            "factset_entity_id",
        ],
    )
    entity_map = pd.read_parquet(entity_map_path, columns=["node_id", "canonical_id"])
    catalog_l4 = pd.read_parquet(catalog_l4_path, columns=["factset_entity_id", "l4_id", "l4_name"])

    entity_ids_in_network = get_entity_ids_in_network(nodes, entity_map)
    counts = count_entities_by_l4(catalog_l4, entity_ids_in_network)
    names = l4_name_lookup(catalog_l4)

    # Fallback names for codes that may not appear in the strict RBICS catalog.
    default_names = {
        "55101010": "Electronic Components",
        "55102510": "Electronic Equipment Manufacturing",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{tabular}{@{}l l S[table-format=4.0, table-number-alignment=center] l@{}}",
        r"\toprule",
        r"\multicolumn{1}{c}{RBICS L4} & \multicolumn{1}{c}{Industry group} & \multicolumn{1}{c}{Firms} & \multicolumn{1}{c}{Role} \\",
        r"\midrule",
        r"\multicolumn{4}{@{}l}{\textit{Strict seed set (included in strict build)}} \\",
    ]

    for code in rbics_core:
        lines.append(
            f"{code} & {names.get(code, default_names.get(code, ''))} & {int(counts.get(code, 0))} & Included (strict) \\\\"
        )

    if rbics_adj:
        lines += [
            r"\midrule",
            r"\multicolumn{4}{@{}l}{\textit{Adjacent codes (defined but excluded from strict build)}} \\",
        ]
        for code in rbics_adj:
            lines.append(
                f"{code} & {names.get(code, default_names.get(code, ''))} &  & Excluded (adjacent) \\\\"
            )

    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out_path.write_text("\n".join(lines))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
