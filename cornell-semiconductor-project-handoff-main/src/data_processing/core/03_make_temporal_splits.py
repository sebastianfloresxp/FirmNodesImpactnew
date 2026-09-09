#!/usr/bin/env python3
"""
Core Pipeline - Phase 3: Make Temporal Splits

Creates leakage-free train/val/test windows with explicit gap days and persists
split Parquet files plus split metadata. Joins node_id mapping for downstream use.

Outputs (under --out-root):
  - splits/train_edges.parquet
  - splits/val_edges.parquet
  - splits/test_edges.parquet
  - meta/temporal_splits.json

Each split file columns:
  event_id, src, dst, start_date, ts, timestamp, edge_feature, duration_days,
  src_id, dst_id
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

# Handle both module and script execution
try:
    from ._common.run_utils import ensure_run_root, write_json
except ImportError:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).parent))
    from _common.run_utils import ensure_run_root, write_json


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")
logger = logging.getLogger("core.splits")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create temporal train/val/test splits with gap days")
    p.add_argument(
        "--out-root", required=True, type=str, help="Run-scoped output root (same as prior phases)"
    )
    p.add_argument(
        "--events-file", type=str, help="Override events path; default resolves under --out-root"
    )
    p.add_argument(
        "--mapping-file", type=str, help="Override mapping path; default resolves under --out-root"
    )
    p.add_argument("--train-prop", type=float, default=0.70)
    p.add_argument("--val-prop", type=float, default=0.15)
    p.add_argument("--test-prop", type=float, default=0.15)
    # Dissertation default: use a more conservative 60-day gap
    p.add_argument("--gap-days", type=int, default=60)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _setup_logger(level: str) -> None:
    try:
        logging.getLogger().setLevel(getattr(logging, level.upper()))
    except Exception:
        logging.getLogger().setLevel(logging.INFO)


def _compute_boundaries(
    ts_min: int, ts_max: int, train_prop: float, val_prop: float, test_prop: float, gap_days: int
) -> dict[str, int]:
    total = train_prop + val_prop + test_prop
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Proportions must sum to 1.0; got {total}")
    if ts_max < ts_min:
        raise ValueError("Invalid ts range: max < min")

    D = int(ts_max - ts_min)
    # Compute cutpoints on integer day grid
    t0_end = round(ts_min + train_prop * D) - gap_days
    val_start = t0_end + gap_days
    val_end = round(val_start + val_prop * D) - gap_days
    test_start = val_end + gap_days

    # Clamp and sanity
    t0_end = max(ts_min, t0_end)
    val_start = max(t0_end + gap_days, val_start)
    val_end = max(val_start, val_end)
    test_start = max(val_end + gap_days, test_start)
    test_start = min(test_start, ts_max)  # ensure not beyond max

    return {
        "min_ts": int(ts_min),
        "T0_end": int(t0_end),
        "val_start": int(val_start),
        "val_end": int(val_end),
        "test_start": int(test_start),
        "max_ts": int(ts_max),
    }


def _join_mapping(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    m = mapping[["raw_id", "node_id"]].rename(columns={"raw_id": "src", "node_id": "src_id"})  # type: ignore[call-overload]
    out = df.merge(m, on="src", how="left")
    m2 = mapping[["raw_id", "node_id"]].rename(columns={"raw_id": "dst", "node_id": "dst_id"})  # type: ignore[call-overload]
    out = out.merge(m2, on="dst", how="left")
    # Verify coverage
    if out["src_id"].isna().any() or out["dst_id"].isna().any():  # type: ignore[misc]
        missing_src = int(out["src_id"].isna().sum())
        missing_dst = int(out["dst_id"].isna().sum())
        raise ValueError(
            f"Mapping coverage failure: missing src_id={missing_src}, dst_id={missing_dst}"
        )
    return out


def main() -> None:
    args = _parse_args()
    _setup_logger(args.log_level)

    out_root = Path(args.out_root)
    # Ensure standard dirs exist; allow existing run root
    ensure_run_root(out_root, force=True, dry_run=args.dry_run)

    events_path = (
        Path(args.events_file)
        if args.events_file
        else (out_root / "events" / "supply_chain_events.parquet")
    )
    mapping_path = (
        Path(args.mapping_file)
        if args.mapping_file
        else (out_root / "mapping" / "entity_map.parquet")
    )
    splits_dir = out_root / "splits"
    meta_dir = out_root / "meta"

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    logger.info("=== CORE TEMPORAL SPLITS ===")
    logger.info(f"events_file:  {events_path}")
    logger.info(f"mapping_file: {mapping_path}")
    logger.info(f"out_root:     {out_root}")
    logger.info(
        f"props: train={args.train_prop:.2f}, val={args.val_prop:.2f}, test={args.test_prop:.2f}, gap_days={args.gap_days}"
    )

    # Load minimal columns for performance
    cols = [
        "event_id",
        "src",
        "dst",
        "start_date",
        "ts",
        "timestamp",
        "edge_feature",
        "duration_days",
    ]
    ev = pd.read_parquet(events_path, columns=cols)
    ev["ts"] = ev["ts"].astype(int)

    ts_min = int(ev["ts"].min())
    ts_max = int(ev["ts"].max())
    bounds = _compute_boundaries(
        ts_min, ts_max, args.train_prop, args.val_prop, args.test_prop, args.gap_days
    )

    logger.info("Computed boundaries (epoch-days):")
    for k in ["min_ts", "T0_end", "val_start", "val_end", "test_start", "max_ts"]:
        logger.info(f"  {k}: {bounds[k]}")

    # Masks for splits and gap
    train_mask = ev["ts"] <= bounds["T0_end"]
    val_mask = (ev["ts"] >= bounds["val_start"]) & (ev["ts"] <= bounds["val_end"])
    test_mask = ev["ts"] >= bounds["test_start"]
    gap_mask = ~(train_mask | val_mask | test_mask)

    n_train, n_val, n_test, n_gap = (
        int(train_mask.sum()),
        int(val_mask.sum()),
        int(test_mask.sum()),
        int(gap_mask.sum()),
    )
    logger.info(
        f"Sizes: train={n_train:,}, val={n_val:,}, test={n_test:,}, gap_discarded={n_gap:,}"
    )

    # Prepare mapping and join src/dst ids
    mapping = pd.read_parquet(mapping_path, columns=["raw_id", "node_id"]).astype(
        {"raw_id": str, "node_id": int}
    )

    train_df = _join_mapping(ev.loc[train_mask].copy(), mapping)
    val_df = _join_mapping(ev.loc[val_mask].copy(), mapping)
    test_df = _join_mapping(ev.loc[test_mask].copy(), mapping)

    # Disjointness checks via event_id
    s_train = set(train_df["event_id"])
    s_val = set(val_df["event_id"])
    s_test = set(test_df["event_id"])
    assert s_train.isdisjoint(s_val) and s_train.isdisjoint(s_test) and s_val.isdisjoint(s_test), (
        "Split event_id sets are not disjoint"
    )

    # Dry-run summary only
    if args.dry_run:
        logger.info("[dry-run] Would write splits and meta under: %s", out_root)
        return

    splits_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Persist splits
    def keep_cols(df: pd.DataFrame) -> pd.DataFrame:
        return df[  # type: ignore[return-value]
            [
                "event_id",
                "src",
                "dst",
                "start_date",
                "ts",
                "timestamp",
                "edge_feature",
                "duration_days",
                "src_id",
                "dst_id",
            ]
        ]

    train_path = splits_dir / "train_edges.parquet"
    val_path = splits_dir / "val_edges.parquet"
    test_path = splits_dir / "test_edges.parquet"
    keep_cols(train_df).to_parquet(train_path, index=False)
    keep_cols(val_df).to_parquet(val_path, index=False)
    keep_cols(test_df).to_parquet(test_path, index=False)

    # Meta summary
    def split_meta(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "num_edges": len(df),
            "unique_src": int(df["src"].nunique()),
            "unique_dst": int(df["dst"].nunique()),
            "unique_src_id": int(df["src_id"].nunique()),
            "unique_dst_id": int(df["dst_id"].nunique()),
            "min_ts": int(df["ts"].min()) if len(df) else None,
            "max_ts": int(df["ts"].max()) if len(df) else None,
        }

    splits_meta: dict[str, Any] = {
        "boundaries": {
            **bounds,
            "gap_days": int(args.gap_days),
            "proportions": {"train": args.train_prop, "val": args.val_prop, "test": args.test_prop},
            "inclusion": {
                "train": "ts <= T0_end",
                "val": "val_start <= ts <= val_end",
                "test": "ts >= test_start",
            },
        },
        "sizes": {
            "train": split_meta(train_df),
            "val": split_meta(val_df),
            "test": split_meta(test_df),
            "gap_discarded": int(n_gap),
        },
        "disjoint": True,
    }

    write_json(meta_dir / "temporal_splits.json", splits_meta)

    logger.info("=== TEMPORAL SPLITS COMPLETE ===")
    logger.info(f"Wrote: {train_path}, {val_path}, {test_path}")


if __name__ == "__main__":
    main()
