#!/usr/bin/env python3
"""Export Chapter 4 Table 4.1 (observed interdiction summary) for LaTeX."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG = "src/analysis/chapter4/config/ch4_v2_fix01.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Table 4.1 interdiction summary (observed)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional override for m6_1/interdiction_performance_observed.csv",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config must parse to mapping")
    return cfg


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def latex_escape(text: Any) -> str:
    s = "" if text is None else str(text)
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def build_latex_tabular(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("\\begin{tabular}{l r r r r}")
    lines.append("\\toprule")
    lines.append("Objective & $k$ & $H1_{\\log}$ & $H2$ growth & Disconnect share \\\\")
    lines.append("\\midrule")
    for row in df.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.Objective)} & "
            f"{int(row.k)} & "
            f"{row.H1_log:.4f} & "
            f"{row.H2_growth:.4f} & "
            f"{row.Disconnect_share:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    tables_root = Path(str(cfg.get("paths", {}).get("tables_root", "tables/chapter4/v2_fix01")))

    input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else out_root / snapshot / "m6_1" / "interdiction_performance_observed.csv"
    )
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    out_dir = tables_root / "main_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    src = pd.read_csv(input_csv).copy()
    required_cols = {
        "objective",
        "k_target",
        "step",
        "h1_log_obligation_any_support",
        "h2_path_growth",
        "h2_disconnect_share",
    }
    missing = required_cols - set(src.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    # Keep final interdiction outcome at each budget k.
    src = src[src["step"] == src["k_target"]].copy()

    # Keep chapter headline objectives only.
    keep_obj = {
        "deny_h1_log_obligation_any_support": "Deny (H1)",
        "delay_h2": "Delay (H2)",
    }
    src = src[src["objective"].isin(keep_obj.keys())].copy()

    src["Objective"] = src["objective"].map(keep_obj)
    src["k"] = pd.to_numeric(src["k_target"], errors="coerce").fillna(0).astype(int)
    src["H1_log"] = pd.to_numeric(src["h1_log_obligation_any_support"], errors="coerce").fillna(0.0)
    src["H2_growth"] = pd.to_numeric(src["h2_path_growth"], errors="coerce").fillna(0.0)
    src["Disconnect_share"] = pd.to_numeric(src["h2_disconnect_share"], errors="coerce").fillna(0.0)

    obj_order = {"Deny (H1)": 0, "Delay (H2)": 1}
    out = src[["Objective", "k", "H1_log", "H2_growth", "Disconnect_share"]].copy()
    out["obj_order"] = out["Objective"].map(obj_order).fillna(99).astype(int)
    out = (
        out.sort_values(["obj_order", "k"], ascending=[True, True])
        .drop(columns=["obj_order"])
        .reset_index(drop=True)
    )

    csv_out = out_dir / "tab_4_1_interdiction_observed.csv"
    tex_out = out_dir / "tab_4_1_interdiction_observed.tex"
    spec_out = out_dir / "tab_4_1_interdiction_observed_spec.json"
    meta_out = out_dir / "tab_4_1_interdiction_observed_run_metadata.json"

    out.to_csv(csv_out, index=False)
    tex_out.write_text(build_latex_tabular(out), encoding="utf-8")

    spec = {
        "table_name": "tab_4_1_interdiction_observed",
        "view": "observed",
        "source": str(input_csv),
        "objectives_included": list(keep_obj.keys()),
        "rows": len(out),
        "columns": out.columns.tolist(),
        "rounding": {"H1_log": 4, "H2_growth": 4, "Disconnect_share": 4},
    }
    spec_out.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    run_meta = {
        "module": "export_tab_4_1",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "inputs": {
            "interdiction_performance_observed_csv": str(input_csv),
            "interdiction_performance_observed_sha256": file_sha256(input_csv),
        },
        "outputs": {
            "table_csv": str(csv_out),
            "table_tex": str(tex_out),
            "table_spec": str(spec_out),
        },
    }
    meta_out.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"[done] wrote {csv_out}")
    print(f"[done] wrote {tex_out}")
    print(f"[done] wrote {spec_out}")
    print(f"[done] wrote {meta_out}")


if __name__ == "__main__":
    main()
