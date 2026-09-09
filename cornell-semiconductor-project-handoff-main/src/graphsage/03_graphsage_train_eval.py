#!/usr/bin/env python3
"""
03_graphsage_train_eval.py
--------------------------
Thin CLI wrapper for GraphSAGE training + evaluation.
Delegates to run_graphsage_eval.main() to preserve import compatibility
while providing a numbered, descriptive entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the graphsage directory to Python path
GRAPHSAGE_DIR = Path(__file__).resolve().parent
if str(GRAPHSAGE_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPHSAGE_DIR))

# Import the main function directly
from run_graphsage_eval import main

if __name__ == "__main__":
    main()
