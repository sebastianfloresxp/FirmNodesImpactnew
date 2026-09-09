#!/usr/bin/env python3
"""
Build a short baseline summary table for the canonical DoD network
with disclosed + predicted + observed layers.

This is an optional descriptive table for the *full* DoD network before
semiconductor pruning. It is not part of the locked Chapter 4 input.

Output (figs/chapter3/archive):
  - dod_network_baseline_summary.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    fig_root = Path("figs/chapter3/archive")
    fig_root.mkdir(parents=True, exist_ok=True)

    nodes_path = Path("artifacts/ch3/network_upstream/dod_network_nodes_top5_d99_shipping.parquet")
    summary_path = Path("artifacts/ch3/network_upstream/summary_dod_network_top5_d99_shipping.json")
    if not nodes_path.exists():
        raise FileNotFoundError(nodes_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    nodes = pd.read_parquet(nodes_path, columns=["node_type"])
    summary = json.loads(summary_path.read_text())

    node_counts = nodes["node_type"].value_counts().to_dict()
    rows = [
        ("Nodes total", summary["nodes_total"]),
        ("DoD components (Tier-0)", int(node_counts.get("dod_component", 0))),
        ("Prime vendors (Tier-1)", int(node_counts.get("prime_vendor", 0))),
        ("SCR firms", int(node_counts.get("scr_firm", 0))),
        ("FactSet-only shipping firms", int(node_counts.get("factset_firm", 0))),
        ("Edges total (deduped)", summary["edges_total_dedup"]),
        ("Edges: DoD contracts", summary["edges_contract"]),
        ("Edges: vendor → SCR map", summary["edges_maps_to_scr"]),
        ("Edges: vendor → FactSet map", summary["edges_maps_to_factset"]),
        ("Edges: disclosed SCR (d99)", summary["edges_disclosed_d99"]),
        ("Edges: predicted SCR (Top-5)", summary["edges_predicted_topk"]),
        ("Edges: observed shipping", summary["edges_observed_shipping"]),
    ]

    df = pd.DataFrame(rows, columns=["Metric", "Count"])
    df["Count"] = df["Count"].map(lambda x: f"{int(x):,}")

    latex = df.to_latex(
        index=False,
        caption="Baseline DoD network (disclosed + predicted + observed layers) summary counts.",
        label="tab:ch3_dod_network_baseline",
        escape=False,
    )
    out_path = fig_root / "dod_network_baseline_summary.tex"
    out_path.write_text(latex)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
