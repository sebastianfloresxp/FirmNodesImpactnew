#!/usr/bin/env python3
"""
Utilities for core data pipeline runs: safe run roots, JSON meta IO, and helpers.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunInfo:
    source: str
    input_file: str | None
    run_date: str
    min_start_date: str
    args: dict[str, Any]
    env_used: dict[str, str | None]
    git_sha: str | None
    row_counts: dict[str, int]


def ensure_run_root(out_root: Path, force: bool = False, dry_run: bool = False) -> dict[str, Path]:
    """Ensure a fresh run root with standard subdirectories.

    Returns a dict with keys: root, events_dir, meta_dir.
    Refuses to overwrite existing trees unless force=True (callers should avoid using force in production).
    """
    root = Path(out_root)
    events_dir = root / "events"
    meta_dir = root / "meta"

    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(
            f"Run root already exists and is not empty: {root}. Use --force to override (not recommended)."
        )

    if not dry_run:
        events_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

    return {"root": root, "events_dir": events_dir, "meta_dir": meta_dir}


def write_json(path: Path, data: dict[str, Any], dry_run: bool = False) -> None:
    """Write a JSON file with pretty indentation, creating parent dirs."""
    if dry_run:
        logger.info(f"[dry-run] Would write JSON: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def detect_git_sha(repo_root: Path | None = None) -> str | None:
    """Best-effort detection of current git SHA without invoking git.

    Reads .git/HEAD and referenced ref file if present. Returns None if not available.
    """
    try:
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[4]
        git_dir = root / ".git"
        head = git_dir / "HEAD"
        if not head.exists():
            return None
        content = head.read_text().strip()
        if content.startswith("ref:"):
            ref_path = content.split(" ", 1)[1].strip()
            ref_file = git_dir / ref_path
            if ref_file.exists():
                return ref_file.read_text().strip()[:40]
            return None
        # Detached HEAD contains SHA directly
        return content[:40]
    except Exception:
        return None


def collect_env(keys: list[str]) -> dict[str, str | None]:
    """Collect selected environment variables (values masked except presence)."""
    out: dict[str, str | None] = {}
    for k in keys:
        val = os.getenv(k)
        out[k] = "<set>" if val else None
    return out


def to_dict(dataclass_obj: Any) -> dict[str, Any]:
    return asdict(dataclass_obj)
