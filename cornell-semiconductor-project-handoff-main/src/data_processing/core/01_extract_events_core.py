#!/usr/bin/env python3
"""
Core Pipeline - Phase 1: Extract Supply Chain Events

Produces a canonical, reproducible events table and run metadata without touching
any existing artifacts. All outputs are written under a caller-provided run root.

Outputs (under --out-root):
  - events/supply_chain_events.parquet
  - meta/duplicate_analysis.json
  - meta/event_summary.json
  - meta/run_info.json

Usage example:
  python src/data_processing/core/01_extract_events_core.py \
    --out-root data/processed/core/runs/2025-09-03_core_v1 \
    --source sql \
    --run-date 2025-09-03

Or from a local file with the same schema:
  python src/data_processing/core/01_extract_events_core.py \
    --out-root data/processed/core/runs/2025-09-03_core_v1 \
    --source parquet \
    --input-file path/to/supply_chain.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Handle both direct execution and module execution
try:
    from ._common.run_utils import (
        RunInfo,
        collect_env,
        detect_git_sha,
        ensure_run_root,
        to_dict,
        write_json,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from _common.run_utils import (
        RunInfo,
        collect_env,
        detect_git_sha,
        ensure_run_root,
        to_dict,
        write_json,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")
logger = logging.getLogger("core.events")


REQUIRED_COLS = [
    "supplier_factset_entity_id",
    "customer_factset_entity_id",
    "start_date",
    "end_date",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract and canonicalize supply chain events (core pipeline)"
    )
    p.add_argument(
        "--out-root", required=True, type=str, help="Run-scoped output root (must be new or empty)"
    )
    p.add_argument(
        "--source", choices=["sql", "parquet", "csv"], default="sql", help="Input source type"
    )
    p.add_argument(
        "--input-file", type=str, default=None, help="Local file path when source is parquet/csv"
    )
    p.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Run date (YYYY-MM-DD) used to close open intervals; default=today",
    )
    p.add_argument(
        "--min-start-date",
        type=str,
        default="2000-01-01",
        help="Drop events before this date (inclusive boundary)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Early slice after sorting by start_date (debug only)",
    )
    p.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.3,
        help="Fraction of shorter interval considered duplicate overlap",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Validate and print summary without writing outputs"
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing run root (avoid in production)",
    )
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return p.parse_args()


def _setup_logger(level: str) -> None:
    try:
        logging.getLogger().setLevel(getattr(logging, level.upper()))
    except Exception:
        logging.getLogger().setLevel(logging.INFO)


def _load_from_sql() -> pd.DataFrame:
    import os
    import urllib.parse

    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    load_dotenv()
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_DATABASE")
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    if not all([server, database, username, password]):
        raise ValueError(
            "Missing database environment variables: DB_SERVER, DB_DATABASE, DB_USERNAME, DB_PASSWORD"
        )

    params = urllib.parse.quote_plus(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};"
        f"Trusted_Connection=no;Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;"
    )
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    logger.info("Querying ent_v1.ent_scr_supply_chain …")
    query = (
        "SELECT id, supplier_factset_entity_id, customer_factset_entity_id, start_date, end_date,"
        "       revenue_pct, source_factset_entity_id "
        "FROM ent_v1.ent_scr_supply_chain ORDER BY start_date"
    )
    df = pd.read_sql(query, engine)
    return df


def _load_local(input_file: Path, source: str) -> pd.DataFrame:
    if source == "parquet":
        return pd.read_parquet(input_file)
    if source == "csv":
        return pd.read_csv(input_file)
    raise ValueError(f"Unsupported local source: {source}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns are present and date columns are datetime64[ns]."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    # Parse dates
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    # Ensure string ids
    out["supplier_factset_entity_id"] = out["supplier_factset_entity_id"].astype(str)
    out["customer_factset_entity_id"] = out["customer_factset_entity_id"].astype(str)
    return out


def _remove_self_loops(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["supplier_factset_entity_id"] != df["customer_factset_entity_id"]
    removed = int((~mask).sum())
    if removed:
        logger.info(f"Removed self-loops: {removed}")
    return df[mask].copy()  # type: ignore[return-value]


def _dedup_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    cols = ["supplier_factset_entity_id", "customer_factset_entity_id", "start_date", "end_date"]
    out = df.drop_duplicates(subset=cols).copy()
    return out, before - len(out)


def _dedup_overlap(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, int]:
    """Drop records that overlap > threshold of the shorter interval within each (src,dst)."""
    kept = []
    removed = 0
    for (_, _), group in df.groupby(  # type: ignore[misc]
        ["supplier_factset_entity_id", "customer_factset_entity_id"], sort=False
    ):
        if len(group) == 1:
            kept.append(group.iloc[0])
            continue
        g = group.sort_values("start_date").reset_index(drop=True)
        accepted: list[pd.Series] = []
        for _, row in g.iterrows():
            st = row["start_date"]
            en = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp.now(tz=None)  # type: ignore[misc]
            is_dup = False
            for prev in accepted:
                pst = prev["start_date"]
                pen = prev["end_date"] if pd.notna(prev["end_date"]) else pd.Timestamp.now(tz=None)  # type: ignore[misc]
                # overlap window
                o_st = max(st, pst)
                o_en = min(en, pen)
                if o_st < o_en:
                    overlap_days = (o_en - o_st).days  # type: ignore[union-attr]
                    cur_dur = max(1, (en - st).days)  # type: ignore[union-attr]
                    prev_dur = max(1, (pen - pst).days)  # type: ignore[union-attr]
                    if overlap_days > threshold * min(cur_dur, prev_dur):
                        is_dup = True
                        break
            if not is_dup:
                accepted.append(row)
            else:
                removed += 1
        if accepted:
            kept.extend(accepted)
    return pd.DataFrame(kept), removed


def _compute_fields(
    df: pd.DataFrame, run_date: pd.Timestamp, min_start_date: pd.Timestamp
) -> pd.DataFrame:
    out = df.copy()
    # Filter invalid/early starts
    out = out[out["start_date"].notna()].copy()
    out = out[out["start_date"] >= min_start_date].copy()
    # Filled end date and flags
    out["is_active"] = out["end_date"].isna()  # type: ignore[union-attr]
    out["end_date_filled"] = out["end_date"].fillna(run_date)  # type: ignore[union-attr]
    out["duration_days"] = (out["end_date_filled"] - out["start_date"]).dt.days  # type: ignore[union-attr]
    out = out[out["duration_days"] >= 0].copy()
    # Temporal features
    out["year"] = out["start_date"].dt.year  # type: ignore[union-attr]
    out["month"] = out["start_date"].dt.month  # type: ignore[union-attr]
    out["quarter"] = out["start_date"].dt.quarter  # type: ignore[union-attr]
    # Pair frequency after dedup
    freq = (
        out.groupby(["supplier_factset_entity_id", "customer_factset_entity_id"], as_index=False)  # type: ignore[union-attr]
        .size()
        .rename(columns={"size": "relationship_frequency"})  # type: ignore[call-overload]
    )
    out = out.merge(  # type: ignore[union-attr]
        freq,
        on=["supplier_factset_entity_id", "customer_factset_entity_id"],
        how="left",
    )
    # Canonical renames and features
    out["src"] = out["supplier_factset_entity_id"]
    out["dst"] = out["customer_factset_entity_id"]
    out["edge_feature"] = out["duration_days"]
    # epoch-days
    epoch = pd.Timestamp("1970-01-01")
    out["ts"] = ((out["start_date"] - epoch).dt.total_seconds() // (24 * 3600)).astype(int)
    # relative timestamp (days since min start)
    min_ts = out["start_date"].min()
    out["timestamp"] = (out["start_date"] - min_ts).dt.total_seconds() / (24 * 3600)

    # event_id (stable SHA1 short)
    def _eid(row: pd.Series) -> str:
        start_iso: str = row["start_date"].isoformat()  # type: ignore[union-attr]
        end_val = row["end_date"]
        end_iso = "" if pd.isna(end_val) else pd.Timestamp(end_val).isoformat()  # type: ignore[arg-type]
        s = f"{row['src']}|{row['dst']}|{start_iso}|{end_iso}"
        return hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]

    out["event_id"] = out.apply(_eid, axis=1)
    # sort
    out = out.sort_values("start_date").reset_index(drop=True)
    # final columns
    cols = [
        "event_id",
        "src",
        "dst",
        "start_date",
        "timestamp",
        "edge_feature",
        "duration_days",
        "relationship_frequency",
        "is_active",
        "year",
        "month",
        "quarter",
        "ts",
    ]
    return out[cols]  # type: ignore[return-value]


def _build_duplicate_meta(
    raw_rows: int, removed_self: int, removed_exact: int, removed_overlap: int
) -> dict[str, Any]:
    return {
        "raw_rows": raw_rows,
        "removed_self_loops": removed_self,
        "removed_exact_duplicates": removed_exact,
        "removed_overlap_duplicates": removed_overlap,
        "total_removed": removed_self + removed_exact + removed_overlap,
    }


def _build_event_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "total_events": len(df),
        "unique_entities": int(pd.concat([df["src"], df["dst"]]).nunique()),
        "temporal_span_days": float(df["timestamp"].max() - df["timestamp"].min())
        if len(df)
        else 0.0,
        "avg_duration_days": float(df["duration_days"].mean()) if len(df) else 0.0,
        "median_duration_days": float(df["duration_days"].median()) if len(df) else 0.0,
        "yearly_distribution": df["year"].value_counts().sort_index().astype(int).to_dict(),
    }


def main() -> None:
    args = _parse_args()
    _setup_logger(args.log_level)

    out_root = Path(args.out_root)
    _run_date_raw = (
        pd.to_datetime(args.run_date).normalize()
        if args.run_date
        else pd.Timestamp(datetime.now(timezone.utc).date())
    )
    if not isinstance(_run_date_raw, pd.Timestamp):
        raise ValueError(f"run_date resolved to non-Timestamp: {_run_date_raw!r}")
    run_date: pd.Timestamp = _run_date_raw
    min_start = pd.to_datetime(args.min_start_date).normalize()

    # Plan printout
    logger.info("=== CORE EVENTS EXTRACTION ===")
    logger.info(f"out_root: {out_root}")
    logger.info(f"source: {args.source}")
    if args.input_file:
        logger.info(f"input_file: {args.input_file}")
    logger.info(f"run_date: {run_date.date()}")
    logger.info(f"min_start_date: {min_start.date()}")
    if args.max_rows:
        logger.info(f"max_rows (debug slice): {args.max_rows}")
    logger.info(f"dry_run: {args.dry_run}")

    # Ensure a clean run root
    dirs = ensure_run_root(out_root, force=args.force, dry_run=args.dry_run)

    # Load input
    if args.source == "sql":
        df_raw = _load_from_sql()
    else:
        if not args.input_file:
            raise ValueError("--input-file is required when --source is parquet or csv")
        df_raw = _load_local(Path(args.input_file), args.source)

    row_counts = {"raw": len(df_raw)}
    logger.info(f"Loaded rows: {row_counts['raw']:,}")

    # Normalize & slice
    df = _normalize_columns(df_raw)
    if args.max_rows:
        df = df.sort_values("start_date").head(args.max_rows).copy()
        logger.info(f"Applied debug slice: {len(df):,} rows")

    # Self-loops
    before = len(df)
    df = _remove_self_loops(df)
    removed_self = before - len(df)

    # Exact dedup
    df, removed_exact = _dedup_exact(df)

    # Overlap dedup
    df, removed_overlap = _dedup_overlap(df, threshold=float(args.overlap_threshold))

    # Compute fields and final schema
    df_final = _compute_fields(df, run_date=run_date, min_start_date=min_start)

    # Acceptance checks
    assert df_final["duration_days"].ge(0).all(), "Negative durations detected"  # type: ignore[misc]
    assert abs(df_final["timestamp"].min() - 0.0) < 1e-9, "Timestamp min must be 0.0"
    assert df_final["event_id"].nunique() == len(df_final), "event_id must be unique"

    # Build meta
    duplicate_meta = _build_duplicate_meta(
        raw_rows=row_counts["raw"],
        removed_self=removed_self,
        removed_exact=removed_exact,
        removed_overlap=removed_overlap,
    )
    summary_meta = _build_event_summary(df_final)
    run_info = RunInfo(
        source=args.source,
        input_file=str(args.input_file) if args.input_file else None,
        run_date=str(run_date.date()),
        min_start_date=str(min_start.date()),
        args={
            "max_rows": args.max_rows,
            "overlap_threshold": args.overlap_threshold,
            "dry_run": bool(args.dry_run),
            "force": bool(args.force),
        },
        env_used=collect_env(["DB_SERVER", "DB_DATABASE", "DB_USERNAME", "DB_PASSWORD"]),
        git_sha=detect_git_sha(),
        row_counts={
            **row_counts,
            "after_self_loop": int(len(df) + removed_overlap + 0),
            "final": len(df_final),
        },
    )

    # Dry-run summary
    if args.dry_run:
        logger.info("[dry-run] Would write:")
        logger.info(f"  events: {dirs['events_dir'] / 'supply_chain_events.parquet'}")
        logger.info(f"  meta:   {dirs['meta_dir'] / 'duplicate_analysis.json'}")
        logger.info(f"          {dirs['meta_dir'] / 'event_summary.json'}")
        logger.info(f"          {dirs['meta_dir'] / 'run_info.json'}")
        return

    # Persist outputs
    events_path = dirs["events_dir"] / "supply_chain_events.parquet"
    df_final.to_parquet(events_path, index=False)
    write_json(dirs["meta_dir"] / "duplicate_analysis.json", duplicate_meta)
    write_json(dirs["meta_dir"] / "event_summary.json", summary_meta)
    write_json(dirs["meta_dir"] / "run_info.json", to_dict(run_info))

    # Console summary
    logger.info("=== EVENTS EXTRACTION COMPLETE ===")
    logger.info(f"Events: {events_path}")
    logger.info(f"Meta:   {dirs['meta_dir']}")
    logger.info(f"Total events: {len(df_final):,}")


if __name__ == "__main__":
    main()
