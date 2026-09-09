"""
Import helper utility for flexible script execution.

This module provides utilities to handle imports that work both when:
1. Running scripts directly: `python script.py`
2. Running as modules: `python -m package.script`

Usage:
    from ._common.import_helper import flexible_import

    # Instead of: from ._common.run_utils import some_function
    flexible_import('_common.run_utils', ['some_function'])
"""

import sys
from pathlib import Path
from typing import Any


def flexible_import(module_path: str, imports: list[str]) -> list[Any]:
    """
    Import from a module using flexible path resolution.

    Args:
        module_path: Relative module path (e.g., '_common.run_utils')
        imports: List of names to import from the module

    Returns:
        List of imported objects in the same order as imports

    Example:
        # Instead of: from ._common.run_utils import ensure_run_root, write_json
        ensure_run_root, write_json = flexible_import('_common.run_utils', ['ensure_run_root', 'write_json'])
    """
    try:
        # Try relative import first (works when run as module)
        module = __import__(module_path, fromlist=imports)
        return [getattr(module, name) for name in imports]
    except ImportError:
        # Fallback for direct execution
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        try:
            module = __import__(module_path, fromlist=imports)
            return [getattr(module, name) for name in imports]
        finally:
            # Clean up the path modification
            if str(script_dir) in sys.path:
                sys.path.remove(str(script_dir))
