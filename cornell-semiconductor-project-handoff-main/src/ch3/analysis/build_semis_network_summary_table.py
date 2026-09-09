#!/usr/bin/env python3
"""
Build a compact LaTeX tabular for the strict DoD semiconductor network.

This is intended to be `\\input{...}` inside an outer LaTeX `table` environment
(i.e., tabular-only; no caption/label wrappers).

Default inputs target the canonical upstream (supplier->prime) build.

Output (default):
  - tables/chapter3/tab_3.4_dod_semiconductor_network_summary.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Chapter 3 strict semiconductor network summary tabular"
    )
    p.add_argument(
        "--nodes",
        default="artifacts/ch3/network_upstream/dod_semiconductor_nodes_top5_d99_shipping_strict.parquet",
        help="Nodes parquet for the strict semiconductor subgraph",
    )
    p.add_argument(
        "--summary",
        default="artifacts/ch3/network_upstream/summary_dod_semiconductor_top5_d99_shipping_strict.json",
        help="Summary JSON produced by build_semiconductor_subgraph.py",
    )
    p.add_argument(
        "--out",
        default="tables/chapter3/tab_3.4_dod_semiconductor_network_summary.tex",
        help="Output .tex file (tabular-only)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nodes_path = Path(args.nodes)
    summary_path = Path(args.summary)
    if not nodes_path.exists():
        raise FileNotFoundError(nodes_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    nodes = pd.read_parquet(
        nodes_path,
        columns=[
            "node_type",
            "is_semi_strict",
        ],
    )
    summary = json.loads(summary_path.read_text())

    node_counts = nodes["node_type"].value_counts().to_dict()
    rows = [
        ("Total Nodes", int(summary["nodes_total"])),
        ("Total Edges", int(summary["edges_total"])),
        ("DoD Agencies (Tier-0)", int(node_counts.get("dod_component", 0))),
        ("DoD Contractors (Tier-1)", int(node_counts.get("prime_vendor", 0))),
        ("Semiconductor Firms", int(nodes["is_semi_strict"].sum())),
        ("Edges: DoD contracts (DoD Agency to Contractor)", int(summary["edges_contract"])),
        ("Edges: Disclosed", int(summary["edges_disclosed"])),
        ("Edges: Predicted", int(summary["edges_predicted"])),
        ("Edges: Observed", int(summary["edges_shipping"])),
        ("Edges: Identity bridges to SCR", int(summary["edges_maps_to_scr"])),
        ("Edges: Identity bridges to FactSet", int(summary["edges_maps_to_factset"])),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    node_rows = rows[:5]   # Total Nodes … Semiconductor Firms
    edge_rows = rows[5:]   # Edges: DoD contracts … Identity bridges

    lines = [
        r"\begin{tabular}{@{}l S[table-format=6.0, table-number-alignment=center]@{}}",
        r"\toprule",
        r"\multicolumn{1}{c}{Metric} & \multicolumn{1}{c}{Count} \\",
        r"\midrule",
    ]
    for metric, count in node_rows:
        lines.append(f"{metric} & {count} \\\\")
    lines.append(r"\midrule")
    for metric, count in edge_rows[:4]:   # DoD contracts … Observed
        lines.append(f"{metric} & {count} \\\\")
    lines.append(r"\addlinespace")
    for metric, count in edge_rows[4:]:   # Identity bridges
        lines.append(f"{metric} & {count} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out_path.write_text("\n".join(lines))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
