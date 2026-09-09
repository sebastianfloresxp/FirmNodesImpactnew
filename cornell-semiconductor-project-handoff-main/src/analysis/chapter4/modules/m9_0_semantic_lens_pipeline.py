#!/usr/bin/env python3
"""Module 9.0 orchestrator: run post-analysis semantic lens stage end-to-end.

This runner executes:
  M9.1 -> M9.2 -> M9.3

All three are thin aliases over existing `m0_5/m0_6/m0_7` logic, preserving
legacy artifact paths and manifests for reproducibility.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 9 semantic lens pipeline")
    parser.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2_fix01.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--encoder-pkl",
        default="data/processed/core/runs/2025-09-03_core_v1/features/encoders_T0.pkl",
        help="Encoder path passed to M9.1 (alias m0_5)",
    )
    parser.add_argument(
        "--topn",
        default="25,100",
        help="Comma-separated Top-N set for M9.1/M9.2 (e.g., 25,100,300)",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Optional as-of date (YYYY-MM-DD) passed to M9.2",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="FactSet query chunk size passed to M9.2",
    )
    parser.add_argument(
        "--codebook",
        default=None,
        help="Optional codebook override path passed to M9.3",
    )
    parser.add_argument("--skip-m9-1", action="store_true", help="Skip M9.1")
    parser.add_argument("--skip-m9-2", action="store_true", help="Skip M9.2")
    parser.add_argument("--skip-m9-3", action="store_true", help="Skip M9.3")
    return parser.parse_args()


def run_step(step: str, cmd: list[str]) -> None:
    print(f"[m9_0] running {step}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    this_dir = Path(__file__).resolve().parent

    py = sys.executable
    m91 = str(this_dir / "m9_1_code_universe.py")
    m92 = str(this_dir / "m9_2_factset_industry_enrichment.py")
    m93 = str(this_dir / "m9_3_semiconductor_lens.py")

    if not args.skip_m9_1:
        cmd = [
            py,
            m91,
            "--config",
            args.config,
            "--encoder-pkl",
            args.encoder_pkl,
            "--topn",
            args.topn,
        ]
        run_step("M9.1", cmd)

    if not args.skip_m9_2:
        cmd = [
            py,
            m92,
            "--config",
            args.config,
            "--topn",
            args.topn,
            "--chunk-size",
            str(args.chunk_size),
        ]
        if args.as_of_date:
            cmd.extend(["--as-of-date", args.as_of_date])
        run_step("M9.2", cmd)

    if not args.skip_m9_3:
        cmd = [py, m93, "--config", args.config]
        if args.codebook:
            cmd.extend(["--codebook", args.codebook])
        run_step("M9.3", cmd)

    print("[m9_0] complete")


if __name__ == "__main__":
    main()
