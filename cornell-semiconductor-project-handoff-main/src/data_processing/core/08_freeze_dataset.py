#!/usr/bin/env python3
"""
Core Pipeline - Phase 8: Freeze Dataset Release

Creates an immutable, versioned release of a validated run by copying (or
symlinking) core artifacts and emitting MANIFEST + DATASET_META with checksums
and provenance.

Outputs:
  releases/<version>/... (mirrors core artifacts)
  releases/<version>/MANIFEST.jsonl
  releases/<version>/DATASET_META.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")
logger = logging.getLogger("core.freeze")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ensure_empty_dir(d: Path, force: bool = False) -> None:
    if d.exists():
        if any(d.iterdir()) and not force:
            raise FileExistsError(f"Release directory not empty: {d}")
    else:
        d.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _expected_files(run_root: Path) -> list[Path]:
    # Core files normally present after phases 1-7
    ex: list[Path] = []
    ex += [run_root / "events" / "supply_chain_events.parquet"]
    ex += [run_root / "mapping" / "entity_map.parquet"]
    ex += [run_root / "splits" / f"{n}_edges.parquet" for n in ("train", "val", "test")]
    ex += [
        run_root / "adjacency" / "train_adj_T0.npz",
        run_root / "adjacency" / "out_degree.npy",
        run_root / "adjacency" / "in_degree.npy",
    ]
    # Optional adjacency sidecars
    opt = [run_root / "adjacency" / "train_adj_T0_csc.npz"]
    # Include any neighbor caches if present
    opt += (
        list((run_root / "adjacency").glob("neighbor_cache_*.*"))
        if (run_root / "adjacency").exists()
        else []
    )
    ex += [run_root / "candidates" / f"{n}_candidates.parquet" for n in ("val", "test")]
    ex += [
        run_root / "features" / "node_features_T0.parquet",
        run_root / "features" / "encoders_T0.pkl",
    ]
    # Meta
    meta_always = [
        "temporal_splits.json",
        "candidate_pools_meta.json",
        "adjacency_stats.json",
        "feature_schema.json",
        "validation_report.json",
    ]
    for m in meta_always:
        ex.append(run_root / "meta" / m)
    # Optional meta
    for m in ("run_info.json", "contamination_gap30_vs_gap60.json"):
        p = run_root / "meta" / m
        if p.exists():
            ex.append(p)
    # Add optional files if they exist
    ex += [p for p in opt if p.exists()]
    return ex


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        try:
            # Create relative symlink for portability
            rel = os.path.relpath(src, dst.parent)
            if dst.exists():
                dst.unlink()
            os.symlink(rel, dst)
            return
        except Exception as e:
            logger.warning(f"Symlink failed for {src} -> {dst}, falling back to copy. Error: {e}")
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze a validated run into a versioned release")
    ap.add_argument(
        "--out-root",
        required=True,
        type=str,
        help="Run root to freeze (e.g., .../runs/2025-09-03_core_v1)",
    )
    ap.add_argument(
        "--version", type=str, default="core_v1", help="Release version tag (e.g., core_v1)"
    )
    ap.add_argument(
        "--release-dir",
        type=str,
        default="data/processed/core/releases",
        help="Base directory for releases",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["copy", "link"],
        default="copy",
        help="Copy files (default) or create symlinks",
    )
    ap.add_argument(
        "--require_validated", action="store_true", help="Require validation overall_pass=true"
    )
    ap.add_argument(
        "--force", action="store_true", help="Allow overwriting a non-empty release directory"
    )
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    with contextlib.suppress(Exception):
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    run_root = Path(args.out_root)
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    # Require validation pass unless overridden
    val_path = run_root / "meta" / "validation_report.json"
    if args.require_validated:
        if not val_path.exists():
            raise FileNotFoundError(f"Validation report not found: {val_path}")
        report = _read_json(val_path)
        if not report.get("summary", {}).get("overall_pass", False):
            raise RuntimeError(
                "Validation did not pass; use --force or rerun validation after fixing issues."
            )

    # Prepare release path
    release_base = Path(args.release_dir)
    release_path = release_base / args.version
    _ensure_empty_dir(release_path, force=args.force)

    # Enumerate expected files
    files = _expected_files(run_root)
    missing = [str(p) for p in files if not p.exists()]
    if missing and not args.force:
        raise FileNotFoundError(
            f"Missing expected artifacts (use --force to continue): {missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )

    # Copy/link and hash
    manifest_lines: list[str] = []
    copied: list[tuple[str, int, str]] = []
    for src in files:
        if not src.exists():
            continue  # allow optional files to be absent
        rel_dst = release_path / src.relative_to(run_root)
        _copy_or_link(src, rel_dst, args.mode)
        # Compute checksum on destination (stable hash of release contents)
        try:
            size = rel_dst.stat().st_size
            sha = _sha256(rel_dst)
            copied.append((str(rel_dst.relative_to(release_path)), size, sha))
            manifest_lines.append(
                json.dumps(
                    {
                        "path": str(rel_dst.relative_to(release_path)),
                        "size_bytes": size,
                        "sha256": sha,
                        "source": str(src.relative_to(run_root)),
                    }
                )
            )
        except Exception as e:
            logger.warning(f"Checksum failed for {rel_dst}: {e}")

    # Write MANIFEST
    (release_path / "MANIFEST.jsonl").write_text(
        "\n".join(manifest_lines) + ("\n" if manifest_lines else "")
    )

    # Aggregate META
    meta: dict[str, Any] = {
        "run_root": str(run_root),
        "version": args.version,
        "created_at": __import__("datetime").datetime.utcnow().isoformat(),
        "mode": args.mode,
        "require_validated": bool(args.require_validated),
        "files": {
            "count": len(copied),
            "total_size_bytes": int(sum(sz for _, sz, _ in copied)),
        },
        "manifests": {
            "manifest_file": str((release_path / "MANIFEST.jsonl").relative_to(release_path))
        },
        "boundaries": {},
        "candidates_policy": {},
        "features_schema": {},
        "validation": {},
    }
    # Pull boundaries
    ts_meta = run_root / "meta" / "temporal_splits.json"
    if ts_meta.exists():
        meta["boundaries"] = _read_json(ts_meta).get("boundaries", {})
    # Candidates policy
    cp_meta = run_root / "meta" / "candidate_pools_meta.json"
    if cp_meta.exists():
        cpm = _read_json(cp_meta)
        meta["candidates_policy"] = {
            "budget": cpm.get("budget"),
            "two_hop": cpm.get("two_hop"),
            "directed": cpm.get("directed"),
            "seed": cpm.get("seed"),
        }
    # Features schema
    fs = run_root / "meta" / "feature_schema.json"
    if fs.exists():
        meta["features_schema"] = _read_json(fs)
    # Validation summary
    if val_path.exists():
        try:
            vr = _read_json(val_path)
            meta["validation"] = vr.get("summary", {})
        except Exception:
            pass

    (release_path / "DATASET_META.json").write_text(json.dumps(meta, indent=2))
    logger.info("Freeze complete. Release: %s", release_path)


if __name__ == "__main__":
    main()
