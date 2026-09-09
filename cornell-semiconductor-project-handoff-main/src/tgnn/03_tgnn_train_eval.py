#!/usr/bin/env python3
"""
03_tgnn_train_eval.py
---------------------
Thin CLI wrapper that delegates to tgnn.run_tgnn_eval.main().
This mirrors the pattern used by other model families for consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ on path so we can import tgnn.* as a package
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tgnn.run_tgnn_eval import main

if __name__ == "__main__":
    main()
