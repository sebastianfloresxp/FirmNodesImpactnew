#!/usr/bin/env python3
"""Module 9.2 wrapper: FactSet industry enrichment for semantic lens inputs.

Thin alias over `m0_6_factset_industry_enrichment.py` to keep module labeling
aligned with the post-analysis M9 stage while preserving existing behavior.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().with_name("m0_6_factset_industry_enrichment.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
