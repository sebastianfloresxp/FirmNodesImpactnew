#!/usr/bin/env python3
"""Run Chapter 4 gap-extension modules in sequence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 4 gap extensions")
    parser.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2_fix01.yaml",
        help="Config YAML path",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config must parse to mapping")
    return cfg


def ts() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_cmd(cmd: list[str]) -> None:
    print(f"[runner] {ts()} start: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[runner] {ts()} done: {' '.join(cmd)}", flush=True)


def write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2))


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    status_path = out_root / snapshot / "gap_extensions_runner_status.json"

    views = [str(v) for v in cfg.get("m6_1", {}).get("views", cfg.get("views", {}).keys())]
    if not views:
        views = ["disclosed", "observed", "full"]

    status: dict[str, Any] = {
        "status": "running",
        "started_utc": ts(),
        "config_path": str(cfg_path),
        "views": views,
        "steps": [],
    }
    write_status(status_path, status)

    try:
        for view in views:
            step = {"module": "m6_1", "view": view, "started_utc": ts(), "status": "running"}
            status["steps"].append(step)
            write_status(status_path, status)
            run_cmd(
                [
                    "python",
                    "src/analysis/chapter4/modules/m6_1_weighted_disruption.py",
                    "--config",
                    str(cfg_path),
                    "--view",
                    view,
                ]
            )
            step["status"] = "completed"
            step["finished_utc"] = ts()
            write_status(status_path, status)

        for module in [
            "m6_2_stratified_impact.py",
            "m6_3_candidate_guardrails.py",
            "m2_2_seam_dod_harm.py",
        ]:
            step = {"module": module, "started_utc": ts(), "status": "running"}
            status["steps"].append(step)
            write_status(status_path, status)
            run_cmd(
                ["python", f"src/analysis/chapter4/modules/{module}", "--config", str(cfg_path)]
            )
            step["status"] = "completed"
            step["finished_utc"] = ts()
            write_status(status_path, status)

        status["status"] = "completed"
        status["finished_utc"] = ts()
        write_status(status_path, status)
        print(f"[runner] {ts()} all extensions completed", flush=True)
    except Exception as exc:
        status["status"] = "failed"
        status["failed_utc"] = ts()
        status["error"] = repr(exc)
        write_status(status_path, status)
        print(f"[runner] {ts()} failed: {exc}", flush=True)
        raise


if __name__ == "__main__":
    main()
