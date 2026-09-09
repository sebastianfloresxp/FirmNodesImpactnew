#!/usr/bin/env python3
"""
Core Pipeline - Phase 2: Build Entity Index (passthrough)

Creates a deterministic mapping from raw FactSet entity IDs to a contiguous node_id
space. Passthrough mode sets canonical_id = raw_id, then assigns node_id by sorting
canonical_id ascending for reproducibility.

Outputs (under --out-root):
  - mapping/entity_map.parquet  (raw_id, canonical_id, node_id)
  - mapping/entity_version.json (mode/rules/as_of)
  - meta/mapping_summary.json   (counts/checksums)

Usage:
  python src/data_processing/core/02_build_entity_index.py \
    --out-root data/processed/core/runs/2025-09-03_core_v1 \
    --events-file data/processed/core/runs/2025-09-03_core_v1/events/supply_chain_events.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

# Handle both direct execution and module execution
try:
    from ._common.run_utils import ensure_run_root, write_json
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from _common.run_utils import ensure_run_root, write_json


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")
logger = logging.getLogger("core.mapping")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build deterministic entity index (passthrough)")
    p.add_argument(
        "--out-root", required=True, type=str, help="Run-scoped output root (same as Phase 1)"
    )
    p.add_argument(
        "--events-file",
        type=str,
        help="Path to Phase 1 events parquet; default resolves under --out-root/events/…",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _setup_logger(level: str) -> None:
    import logging as _logging

    try:
        _logging.getLogger().setLevel(getattr(_logging, level.upper()))
    except Exception:
        _logging.getLogger().setLevel(_logging.INFO)


def _checksum_parquet(path: Path) -> str:
    """Lightweight checksum over file bytes for provenance."""
    h = hashlib.sha1(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = _parse_args()
    _setup_logger(args.log_level)

    out_root = Path(args.out_root)
    ensure_run_root(out_root, force=True, dry_run=args.dry_run)  # ensure subdirs exist
    events_path = (
        Path(args.events_file)
        if args.events_file
        else (out_root / "events" / "supply_chain_events.parquet")
    )

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    logger.info("=== CORE ENTITY INDEX (PASSTHROUGH) ===")
    logger.info(f"events_file: {events_path}")
    logger.info(f"out_root:    {out_root}")
    logger.info(f"dry_run:     {args.dry_run}")

    df = pd.read_parquet(events_path, columns=["src", "dst"])  # minimal read
    unique_ids = pd.unique(pd.concat([df["src"], df["dst"]], ignore_index=True))
    logger.info(f"Unique raw IDs: {len(unique_ids):,}")

    # Build mapping: canonical_id = raw_id; node_id assigned by sorted order
    canonical = (
        pd.Series(unique_ids, name="canonical_id", dtype="object")
        .sort_values(kind="mergesort")
        .reset_index(drop=True)
    )
    mapping_df = pd.DataFrame(
        {
            "raw_id": canonical,  # passthrough
            "canonical_id": canonical,
            "node_id": range(len(canonical)),
        }
    )

    # Acceptance checks
    assert mapping_df["node_id"].is_unique, "node_id must be unique"
    assert mapping_df["canonical_id"].is_unique, "canonical_id must be unique in passthrough"

    # Paths
    mapping_dir = out_root / "mapping"
    meta_dir = out_root / "meta"
    if not args.dry_run:
        mapping_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

    # Dry-run summary
    if args.dry_run:
        logger.info("[dry-run] Would write:")
        logger.info(f"  mapping: {mapping_dir / 'entity_map.parquet'}")
        logger.info(f"  version: {mapping_dir / 'entity_version.json'}")
        logger.info(f"  summary: {meta_dir / 'mapping_summary.json'}")
        return

    # Persist mapping
    map_path = mapping_dir / "entity_map.parquet"
    mapping_df.to_parquet(map_path, index=False)

    # Version and summary meta
    version_meta: dict[str, Any] = {
        "mode": "passthrough",
        "as_of": None,
        "rules": "canonical_id = raw_id; node_id assigned by ascending canonical_id",
        "counts": {
            "unique_raw": len(mapping_df),
            "unique_canonical": len(mapping_df),
        },
    }
    write_json(mapping_dir / "entity_version.json", version_meta)

    summary_meta: dict[str, Any] = {
        "mapping_file": str(map_path),
        "checksum_sha1": _checksum_parquet(map_path),
        "min_node_id": int(mapping_df["node_id"].min()),
        "max_node_id": int(mapping_df["node_id"].max()),
    }
    write_json(meta_dir / "mapping_summary.json", summary_meta)

    logger.info("=== ENTITY INDEX COMPLETE ===")
    logger.info(f"Mapping: {map_path}")
    logger.info(f"Entities: {len(mapping_df):,}")


if __name__ == "__main__":
    main()
