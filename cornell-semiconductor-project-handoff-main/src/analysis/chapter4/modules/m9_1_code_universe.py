#!/usr/bin/env python3
"""Module 9.1 wrapper: code-universe build for post-analysis semantic lensing.

This is a thin alias over `m0_5_code_universe.py` so Chapter 4 can describe
the semantic lens stage as a post-analysis module (M9) without changing logic
or outputs.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().with_name("m0_5_code_universe.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
