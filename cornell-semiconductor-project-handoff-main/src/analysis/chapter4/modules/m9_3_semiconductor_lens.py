#!/usr/bin/env python3
"""Module 9.3 wrapper: post-analysis semiconductor semantic lens.

Thin alias over `m0_7_semiconductor_lens.py` so the step can be run and cited
as M9.3 without changing implementation details or output formats.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().with_name("m0_7_semiconductor_lens.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
