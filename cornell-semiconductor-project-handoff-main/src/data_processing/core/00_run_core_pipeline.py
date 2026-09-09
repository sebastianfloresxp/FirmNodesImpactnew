#!/usr/bin/env python3
"""
Core Pipeline Orchestrator (Phases 1–8)

Runs the full core pipeline end-to-end with sensible defaults:
  1) 01_extract_events_core.py          (SQL by default)
  2) 02_build_entity_index.py
  3) 03_make_temporal_splits.py         (gap=60 days default)
  4) 04_build_train_snapshot.py         (emit CSC by default; neighbor cache optional)
  5) 05_build_candidate_pools.py        (budget=5000; undirected two-hop)
  6) 06_build_features_T0.py            (DB-only, as-of T0)
  7) 07_validate_core_dataset.py
  8) 08_freeze_dataset.py               (require validation by default)

You can skip phases or run a dry-run plan. All steps write under a single
--out-root (run-scoped) directory.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")
logger = logging.getLogger("core.orchestrator")


def run_step(title: str, cmd: list[str], dry_run: bool = False) -> bool:
    logger.info("=== %s ===", title)
    logger.info("CMD: %s", " ".join(cmd))
    if dry_run:
        return True
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error(
                "Step failed (%s).\nSTDOUT:\n%s\nSTDERR:\n%s", title, res.stdout, res.stderr
            )
            return False
        if res.stdout:
            logger.info("%s", res.stdout.strip())
        return True
    except Exception as e:
        logger.exception("Exception while running step %s: %s", title, e)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Run core pipeline Phases 1–8")
    ap.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Run root (default: data/processed/core/runs/<YYYY-MM-DD>_core_v1)",
    )
    ap.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Run date for Phase 1 (YYYY-MM-DD). Default: today",
    )
    ap.add_argument(
        "--source",
        type=str,
        choices=["sql", "parquet", "csv"],
        default="sql",
        help="Phase 1 input source",
    )
    ap.add_argument(
        "--input-file", type=str, default=None, help="Phase 1 local input when source!=sql"
    )
    ap.add_argument(
        "--gap-days", type=int, default=60, help="Phase 3 temporal gap days (default 60)"
    )
    ap.add_argument(
        "--neighbor-cache-k",
        type=int,
        default=None,
        help="Phase 4: emit neighbor cache with K entries (optional)",
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=5000,
        help="Phase 5: candidate budget per source (default 5000)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Phase 5: RNG seed (default 42)")
    ap.add_argument(
        "--release-version",
        type=str,
        default="core_v1",
        help="Phase 8: release version tag (default core_v1)",
    )
    ap.add_argument(
        "--freeze-mode",
        type=str,
        choices=["copy", "link"],
        default="copy",
        help="Phase 8: copy or link (default copy)",
    )
    ap.add_argument("--skip", nargs="*", type=int, default=[], help="Skip phases by number (1..8)")
    ap.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    with contextlib.suppress(Exception):
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Resolve out-root
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        tag = datetime.now().strftime("%Y-%m-%d") + "_core_v1"
        out_root = Path("data/processed/core/runs") / tag

    logger.info("Run root: %s", out_root)
    logger.info("Skip phases: %s", args.skip if args.skip else "none")
    if args.dry_run:
        logger.info("DRY RUN: no commands will be executed")

    py = sys.executable
    core_dir = Path(__file__).parent

    # 1) Extract events
    if 1 not in args.skip:
        cmd = [
            py,
            str(core_dir / "01_extract_events_core.py"),
            "--out-root",
            str(out_root),
            "--source",
            args.source,
        ]
        if args.input_file:
            cmd += ["--input-file", args.input_file]
        if args.run_date:
            cmd += ["--run-date", args.run_date]
        if not run_step("Phase 1: Extract events", cmd, args.dry_run):
            sys.exit(1)

    # 2) Build entity index
    if 2 not in args.skip:
        cmd = [py, str(core_dir / "02_build_entity_index.py"), "--out-root", str(out_root)]
        if not run_step("Phase 2: Build entity index", cmd, args.dry_run):
            sys.exit(1)

    # 3) Temporal splits
    if 3 not in args.skip:
        cmd = [
            py,
            str(core_dir / "03_make_temporal_splits.py"),
            "--out-root",
            str(out_root),
            "--gap-days",
            str(args.gap_days),
        ]
        if not run_step("Phase 3: Temporal splits", cmd, args.dry_run):
            sys.exit(1)

    # 4) Train snapshot (adjacency)
    if 4 not in args.skip:
        cmd = [
            py,
            str(core_dir / "04_build_train_snapshot.py"),
            "--out-root",
            str(out_root),
            "--emit-csc",
        ]
        if args.neighbor_cache_k and args.neighbor_cache_k > 0:
            cmd += ["--emit-neighbor-cache", str(args.neighbor_cache_k)]
        if not run_step("Phase 4: Train snapshot", cmd, args.dry_run):
            sys.exit(1)

    # 5) Candidate pools (undirected two-hop; budget default 5000)
    if 5 not in args.skip:
        cmd = [
            py,
            str(core_dir / "05_build_candidate_pools.py"),
            "--out-root",
            str(out_root),
            "--budget",
            str(args.budget),
            "--seed",
            str(args.seed),
            "--force",
        ]
        if not run_step("Phase 5: Candidate pools", cmd, args.dry_run):
            sys.exit(1)

    # 6) Features as-of T0 (DB only)
    if 6 not in args.skip:
        cmd = [
            py,
            str(core_dir / "06_build_features_T0.py"),
            "--out-root",
            str(out_root),
            "--force",
        ]
        if not run_step("Phase 6: Features as-of T0", cmd, args.dry_run):
            sys.exit(1)

    # 7) Validation
    if 7 not in args.skip:
        cmd = [py, str(core_dir / "07_validate_core_dataset.py"), "--out-root", str(out_root)]
        if not run_step("Phase 7: Validate", cmd, args.dry_run):
            sys.exit(1)

    # 8) Freeze (require validation by default)
    if 8 not in args.skip:
        cmd = [
            py,
            str(core_dir / "08_freeze_dataset.py"),
            "--out-root",
            str(out_root),
            "--version",
            args.release_version,
            "--mode",
            args.freeze_mode,
            "--require_validated",
        ]
        if not run_step("Phase 8: Freeze dataset", cmd, args.dry_run):
            sys.exit(1)

    logger.info("=== CORE PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
