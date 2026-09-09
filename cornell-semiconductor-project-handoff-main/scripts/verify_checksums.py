#!/usr/bin/env python3
"""Verify frozen artifact checksums against checksums.json."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "checksums.json"

    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found")
        return 1

    with open(manifest_path) as f:
        entries = json.load(f)

    passed = 0
    failed = 0
    missing = 0

    for entry in entries:
        rel_path = entry["path"]
        expected_hash = entry["sha256"]
        full_path = repo_root / rel_path

        if not full_path.exists():
            print(f"  MISSING  {rel_path}")
            missing += 1
            continue

        actual_hash = sha256_file(full_path)
        if actual_hash == expected_hash:
            print(f"  PASS     {rel_path}")
            passed += 1
        else:
            print(f"  FAIL     {rel_path}")
            print(f"           expected: {expected_hash}")
            print(f"           actual:   {actual_hash}")
            failed += 1

    print("")
    print(f"Results: {passed} passed, {failed} failed, {missing} missing (of {len(entries)} total)")

    if failed > 0 or missing > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
