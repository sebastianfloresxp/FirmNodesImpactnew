#!/usr/bin/env python3
"""Run fix03 effective-extension modules in a staged, persistent workflow."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 4 fix03 effective extensions")
    parser.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2_fix03_effective_extensions.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Polling interval while waiting for M6.1 jobs to finish",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Config must parse to mapping")
    return cfg


def write_status(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def m6_1_is_running() -> bool:
    try:
        output = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)  # nosec B607 -- git is a well-known system executable
    except Exception:
        return False
    self_pid = os.getpid()
    for line in output.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        args = parts[1]
        if pid == self_pid:
            continue
        if "m6_1_weighted_disruption.py" not in args:
            continue
        if "python" not in args:
            continue
        if "run_fix03_effective_extensions.py" in args:
            continue
        return True
    return False


def run_step(step: dict, status_path: Path) -> None:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    step["status"] = "running"
    step["started_utc"] = started
    write_status(status_path, status)
    print(f"[fix03] starting {step['module']}", flush=True)
    result = subprocess.run(step["cmd"], check=False)
    if result.returncode != 0:
        step["status"] = "failed"
        step["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        step["returncode"] = int(result.returncode)
        write_status(status_path, status)
        raise SystemExit(result.returncode)
    step["status"] = "completed"
    step["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_status(status_path, status)
    print(f"[fix03] completed {step['module']}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    run_dir = out_root / snapshot
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "fix03_effective_extensions_status.json"

    modules = [
        {
            "module": "m2_3",
            "cmd": [
                sys.executable,
                "src/analysis/chapter4/modules/m2_3_hard_chokepoints.py",
                "--config",
                str(cfg_path),
            ],
            "status": "pending",
        },
        {
            "module": "m3_2",
            "cmd": [
                sys.executable,
                "src/analysis/chapter4/modules/m3_2_effective_reach.py",
                "--config",
                str(cfg_path),
            ],
            "status": "pending",
        },
        {
            "module": "m6_4",
            "cmd": [
                sys.executable,
                "src/analysis/chapter4/modules/m6_4_effective_disruption.py",
                "--config",
                str(cfg_path),
            ],
            "status": "pending",
        },
    ]

    status = {
        "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "steps": modules,
    }
    write_status(status_path, status)

    # Step 1/2 run immediately.
    run_step(modules[0], status_path)
    run_step(modules[1], status_path)

    # Step 3 waits for any active m6_1 process to avoid heavy overlap.
    modules[2]["status"] = "waiting_for_m6_1"
    modules[2]["waiting_started_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_status(status_path, status)
    while m6_1_is_running():
        print("[fix03] waiting for active m6_1_weighted_disruption.py to finish...", flush=True)
        time.sleep(max(10, int(args.poll_seconds)))

    run_step(modules[2], status_path)

    status["status"] = "completed"
    status["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_status(status_path, status)
    print(f"[fix03] done; status written to {status_path}", flush=True)
