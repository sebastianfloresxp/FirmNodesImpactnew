#!/usr/bin/env python3
"""Export Figure 4.3 action-matrix numbers for LaTeX rendering.

This script does not render any graphics. It writes a tidy cell table and a
figure-spec JSON so the matrix can be typeset directly in LaTeX.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG = "src/analysis/chapter4/config/ch4_v2_fix01.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Figure 4.3 matrix numbers")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional override for m9/action_matrix_summary.csv",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config must parse to mapping")
    return cfg


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    tables_root = Path(str(cfg.get("paths", {}).get("tables_root", "tables/chapter4/v2_fix01")))

    input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else (out_root / snapshot / "m9" / "action_matrix_summary.csv")
    )
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    out_dir = tables_root / "main_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(input_csv).copy()
    required_cols = {
        "impact_bin",
        "confidence_bin",
        "action_bucket",
        "n_nodes",
        "share_of_universe",
    }
    missing = required_cols - set(summary.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    summary["impact_bin"] = summary["impact_bin"].astype(str)
    summary["confidence_bin"] = summary["confidence_bin"].astype(str)
    summary["action_bucket"] = summary["action_bucket"].astype(str)
    summary["n_nodes"] = pd.to_numeric(summary["n_nodes"], errors="coerce").fillna(0).astype(int)
    summary["share_of_universe"] = pd.to_numeric(
        summary["share_of_universe"], errors="coerce"
    ).fillna(0.0)

    impact_order = ["high_impact", "lower_impact"]
    conf_order = ["high_confidence", "low_confidence"]
    labels = {
        ("high_impact", "high_confidence"): "ACT NOW",
        ("high_impact", "low_confidence"): "VALIDATE THEN ACT",
        ("lower_impact", "high_confidence"): "MONITOR / ROUTINE",
        ("lower_impact", "low_confidence"): "DEPRIORITIZE",
    }

    lookup = summary.set_index(["impact_bin", "confidence_bin"])
    n_total = int(summary["n_nodes"].sum())

    rows: list[dict[str, Any]] = []
    for impact in impact_order:
        for conf in conf_order:
            if (impact, conf) in lookup.index:
                row = lookup.loc[(impact, conf)]
                n = int(row["n_nodes"])
                share = float(row["share_of_universe"])
                action_bucket = str(row["action_bucket"])
            else:
                n = 0
                share = 0.0
                action_bucket = "missing"
            rows.append(
                {
                    "impact_bin": impact,
                    "confidence_bin": conf,
                    "action_bucket": action_bucket,
                    "display_label": labels[(impact, conf)],
                    "n_nodes": n,
                    "share_of_universe": share,
                    "percent_display": round(share * 100.0, 1),
                }
            )

    cells = pd.DataFrame(rows)
    cells_csv = out_dir / "fig_4_3_action_matrix_cells.csv"
    cells.to_csv(cells_csv, index=False)

    spec = {
        "figure_name": "fig_4_3_action_matrix",
        "view": "observed",
        "relevance_stream": "semiconductor_value_chain_strict",
        "n_universe": n_total,
        "impact_axis_order": impact_order,
        "confidence_axis_order": conf_order,
        "cells": cells.to_dict(orient="records"),
    }
    spec_json = out_dir / "fig_4_3_action_matrix_spec.json"
    spec_json.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    run_meta = {
        "module": "export_fig_4_3_numbers",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "inputs": {
            "action_matrix_summary_csv": str(input_csv),
            "action_matrix_summary_sha256": file_sha256(input_csv),
        },
        "outputs": {
            "cells_csv": str(cells_csv),
            "spec_json": str(spec_json),
        },
    }
    run_meta_json = out_dir / "fig_4_3_action_matrix_numbers_run_metadata.json"
    run_meta_json.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"[done] wrote {cells_csv}")
    print(f"[done] wrote {spec_json}")
    print(f"[done] wrote {run_meta_json}")


if __name__ == "__main__":
    main()
