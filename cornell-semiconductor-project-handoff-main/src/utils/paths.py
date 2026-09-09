"""
Path management utilities for the semiconductor supply chain analysis project.
"""

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_config_path() -> Path:
    """Get the path to the global configuration file."""
    return get_project_root() / "configs" / "global_config.yaml"


def load_config() -> dict[str, Any]:
    """Load the global configuration."""
    config_path = get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_paths(model_name: str) -> dict[str, Path]:
    """Get paths for a specific model."""
    root = get_project_root()
    config = load_config()

    return {
        "src": root / config["models"][model_name],
        "artifacts": root / config["outputs"]["artifacts_dir"] / model_name,
        "results": root / config["outputs"]["results_dir"] / model_name,
        "logs": root / config["outputs"]["logs_dir"] / model_name,
        "configs": root / config["outputs"]["configs_dir"] / model_name,
    }


def get_data_paths() -> dict[str, Path]:
    """Get data directory paths."""
    root = get_project_root()
    config = load_config()

    return {
        "raw": root / config["data"]["raw_dir"],
        "processed": root / config["data"]["processed_dir"],
    }


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    root = get_project_root()
    config = load_config()

    # Create data directories
    (root / config["data"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    (root / config["data"]["processed_dir"]).mkdir(parents=True, exist_ok=True)

    # Create output directories
    (root / config["outputs"]["artifacts_dir"]).mkdir(parents=True, exist_ok=True)
    (root / config["outputs"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    (root / config["outputs"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    (root / config["outputs"]["configs_dir"]).mkdir(parents=True, exist_ok=True)

    # Create model-specific directories
    for model in ["heuristics", "node2vec", "graphsage", "tgn", "evolvegcn", "dygformer"]:
        model_paths = get_model_paths(model)
        for path in model_paths.values():
            path.mkdir(parents=True, exist_ok=True)


def get_artifact_path(model_name: str, filename: str) -> Path:
    """Get path for a model artifact."""
    return get_model_paths(model_name)["artifacts"] / filename


def get_result_path(model_name: str, filename: str) -> Path:
    """Get path for a model result."""
    return get_model_paths(model_name)["results"] / filename


def get_log_path(model_name: str, filename: str) -> Path:
    """Get path for a model log."""
    return get_model_paths(model_name)["logs"] / filename


def get_config_path_for_model(model_name: str, filename: str) -> Path:
    """Get path for a model configuration file."""
    return get_model_paths(model_name)["configs"] / filename
