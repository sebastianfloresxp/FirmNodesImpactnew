#!/usr/bin/env python3
"""
Build a LaTeX tabular summarizing evidence overlap and incremental deltas.

This table is intended to support Chapter 3 (§3.7) by describing how the strict
DoD semiconductor network's supply-like edges decompose across evidence streams:
  - disclosed SCR (as-of snapshot)
  - predicted Top-K links
  - observed shipping (shipper->consignee)

The output is tabular-only (no caption/label wrapper) so it can be included via
`\\input{...}` inside a LaTeX `table` environment.

Default inputs target the locked strict semiconductor network:
  - artifacts/ch3/network_upstream/dod_semiconductor_edges_top5_d99_shipping_strict.parquet

Output (default):
  - tables/chapter3/tab_3.5_evidence_overlap_deltas.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Chapter 3 evidence overlap/delta tabular")
    p.add_argument(
        "--edges",
        default="artifacts/ch3/network_upstream/dod_semiconductor_edges_top5_d99_shipping_strict.parquet",
        help="Edges parquet for the strict DoD semiconductor network",
    )
    p.add_argument(
        "--out",
        default="tables/chapter3/tab_3.5_evidence_overlap_deltas.tex",
        help="Output .tex file (tabular-only)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    edges_path = Path(args.edges)
    if not edges_path.exists():
        raise FileNotFoundError(edges_path)

    edges = pd.read_parquet(edges_path, columns=["is_disclosed", "is_predicted", "is_shipping"])
    d = edges["is_disclosed"].astype(bool)
    p = edges["is_predicted"].astype(bool)
    s = edges["is_shipping"].astype(bool)

    disclosed = int(d.sum())
    disclosed_or_pred = int((d | p).sum())
    supply_union = int((d | p | s).sum())

    delta_pred = disclosed_or_pred - disclosed
    delta_ship = supply_union - disclosed_or_pred

    cats = {
        "Disclosed only": int((d & ~p & ~s).sum()),
        "Predicted only": int((p & ~d & ~s).sum()),
        "Observed only": int((s & ~d & ~p).sum()),
        "Disclosed ∩ Observed": int((d & s & ~p).sum()),
        "Predicted ∩ Observed": int((p & s & ~d).sum()),
        "Disclosed ∩ Predicted": int((d & p & ~s).sum()),
        "Disclosed ∩ Predicted ∩ Observed": int((d & p & s).sum()),
    }

    # Sanity check: mutually exclusive categories sum to union.
    if sum(cats.values()) != supply_union:
        raise RuntimeError(
            f"Overlap categories do not sum to union: {sum(cats.values())} != {supply_union}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{tabular}{@{}l S[table-format=6.0, table-number-alignment=center]@{}}",
        r"\toprule",
        r"\multicolumn{1}{c}{Metric} & \multicolumn{1}{c}{Count} \\",
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\textit{Incremental supply-edge dyads (directed)}} \\",
        f"Disclosed (as-of) & {disclosed} \\\\",
        f"+ Predicted (new vs disclosed) & {delta_pred} \\\\",
        f"Disclosed $\\cup$ predicted & {disclosed_or_pred} \\\\",
        f"+ Observed (new vs disclosed$\\cup$predicted) & {delta_ship} \\\\",
        f"Disclosed $\\cup$ predicted $\\cup$ observed & {supply_union} \\\\",
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\textit{Evidence overlap (mutually exclusive dyad categories)}} \\",
    ]
    for label, count in cats.items():
        lines.append(f"{label} & {count} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out_path.write_text("\n".join(lines))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
