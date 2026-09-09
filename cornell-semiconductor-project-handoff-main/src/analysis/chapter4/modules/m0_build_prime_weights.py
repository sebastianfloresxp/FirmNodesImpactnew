#!/usr/bin/env python3
"""Build Chapter 4 prime-weight inputs from Chapter 3 obligations."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chapter 4 prime weights")
    parser.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2_fix01.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--source-primes",
        default="artifacts/ch3/usaspending/dod_primes_fy2022-2025_asof.parquet",
        help="Chapter 3 as-of prime obligations parquet",
    )
    parser.add_argument(
        "--node-table",
        default="",
        help="Optional override for M0 node table parquet",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional override output parquet path",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config must parse to a mapping")
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


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(  # nosec B607 -- git is a well-known system executable
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            or None
        )
    except Exception:
        return None


def resolve_paths(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path, Path]:
    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))

    node_table_default = out_root / snapshot / "m0" / "node_table_contract.parquet"
    node_table = Path(args.node_table) if args.node_table else node_table_default

    output_default = Path(
        str(cfg.get("paths", {}).get("prime_weights", "artifacts/ch4/inputs/prime_weights.parquet"))
    )
    output = Path(args.output) if args.output else output_default

    source_primes = Path(args.source_primes)
    return source_primes, node_table, output


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    source_primes, node_table, output = resolve_paths(cfg, args)

    if not source_primes.exists():
        raise FileNotFoundError(f"missing source primes parquet: {source_primes}")
    if not node_table.exists():
        raise FileNotFoundError(f"missing node table parquet: {node_table}")

    nodes = pd.read_parquet(node_table)
    required_node_cols = {"analysis_uid", "has_prime_vendor", "rep_vendor_key"}
    if not required_node_cols.issubset(nodes.columns):
        missing = sorted(required_node_cols - set(nodes.columns))
        raise ValueError(f"node table missing required columns: {missing}")

    prime_nodes = nodes.loc[
        nodes["has_prime_vendor"].fillna(False).astype(bool), ["analysis_uid", "rep_vendor_key"]
    ].copy()
    prime_nodes["analysis_uid"] = prime_nodes["analysis_uid"].astype(str)
    prime_nodes["vendor_key"] = prime_nodes["rep_vendor_key"].astype(str)
    prime_nodes = prime_nodes[prime_nodes["vendor_key"].str.len() > 0]
    prime_nodes = prime_nodes.drop_duplicates(subset=["analysis_uid"], keep="first")

    primes = pd.read_parquet(source_primes)
    required_prime_cols = {
        "vendor_key",
        "sum_total_dollars_obligated",
        "sum_federal_action_obligation",
    }
    if not required_prime_cols.issubset(primes.columns):
        missing = sorted(required_prime_cols - set(primes.columns))
        raise ValueError(f"source primes missing required columns: {missing}")

    agg = (
        primes.groupby("vendor_key", as_index=False)[
            ["sum_total_dollars_obligated", "sum_federal_action_obligation"]
        ]
        .sum()
        .rename(
            columns={
                "sum_total_dollars_obligated": "obligation_raw",
                "sum_federal_action_obligation": "federal_action_obligation_raw",
            }
        )
    )
    agg["vendor_key"] = agg["vendor_key"].astype(str)

    weights = prime_nodes.merge(agg, how="left", on="vendor_key")
    weights["obligation_raw"] = pd.to_numeric(weights["obligation_raw"], errors="coerce").fillna(
        0.0
    )
    weights["federal_action_obligation_raw"] = pd.to_numeric(
        weights["federal_action_obligation_raw"], errors="coerce"
    ).fillna(0.0)

    nonneg_total = np.maximum(weights["obligation_raw"].to_numpy(np.float64), 0.0)
    nonneg_federal = np.maximum(weights["federal_action_obligation_raw"].to_numpy(np.float64), 0.0)

    weights["obligation_nonneg"] = nonneg_total
    weights["federal_action_obligation_nonneg"] = nonneg_federal
    weights["obligation_log1p"] = np.log1p(nonneg_total)
    weights["federal_action_obligation_log1p"] = np.log1p(nonneg_federal)

    # Backward-compatible fields consumed by Chapter 4 loaders.
    weights["obligation_weight"] = weights["obligation_log1p"]
    weights["total_obligations"] = weights["obligation_nonneg"]
    weights["weight"] = weights["obligation_weight"]

    out_cols = [
        "analysis_uid",
        "vendor_key",
        "obligation_raw",
        "obligation_nonneg",
        "obligation_log1p",
        "federal_action_obligation_raw",
        "federal_action_obligation_nonneg",
        "federal_action_obligation_log1p",
        "obligation_weight",
        "total_obligations",
        "weight",
    ]
    out_df = weights[out_cols].sort_values("analysis_uid", kind="mergesort").reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output, index=False)

    summary = {
        "module": "m0_build_prime_weights",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "platform": platform.platform(),
        "input_paths": {"source_primes": str(source_primes), "node_table": str(node_table)},
        "output_path": str(output),
        "coverage": {
            "prime_nodes_total": len(prime_nodes),
            "rows_output": len(out_df),
            "rows_with_positive_obligation": int((out_df["obligation_nonneg"] > 0).sum()),
            "rows_with_positive_weight": int((out_df["obligation_weight"] > 0).sum()),
            "obligation_nonneg_sum": float(out_df["obligation_nonneg"].sum()),
            "obligation_weight_sum": float(out_df["obligation_weight"].sum()),
        },
        "checksums": {
            "source_primes_sha256": file_sha256(source_primes),
            "node_table_sha256": file_sha256(node_table),
            "output_sha256": file_sha256(output),
        },
    }

    summary_path = output.with_name(output.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[done] wrote {output}")
    print(f"[done] wrote {summary_path}")
    print(
        "[summary] "
        f"rows={summary['coverage']['rows_output']} "
        f"positive_obligation={summary['coverage']['rows_with_positive_obligation']} "
        f"weight_sum={summary['coverage']['obligation_weight_sum']:.3f}"
    )


if __name__ == "__main__":
    main()
