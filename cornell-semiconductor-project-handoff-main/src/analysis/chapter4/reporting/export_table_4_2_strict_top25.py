#!/usr/bin/env python3
"""Export Chapter 4 Table 4.2 (strict observed Top-25) for LaTeX.

Confidence for this table is computed directly from cross-view strict Top-25
presence (disclosed / observed / full), not from M7's broader candidate table.
"""

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
    parser = argparse.ArgumentParser(description="Export Table 4.2 strict observed Top-25")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional override for strict observed top25 CSV",
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


def role_label(rule_class: str, corridor_class: str) -> str:
    rc = str(rule_class or "").strip().lower()
    cc = str(corridor_class or "").strip().lower()
    if "core_semiconductor_include" in rc:
        if cc == "near_prime":
            return "near-prime semiconductor intermediary"
        return "semiconductor corridor intermediary"
    if "adjacent_manufacturing_include" in rc:
        if cc == "near_prime":
            return "near-prime manufacturing intermediary"
        return "manufacturing corridor intermediary"
    if cc == "near_prime":
        return "near-prime value-chain intermediary"
    return "value-chain corridor intermediary"


def confidence_label_from_presence(presence_count: int) -> str:
    if int(presence_count) >= 3:
        return "High"
    if int(presence_count) == 2:
        return "Medium"
    return "Low"


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
    lines.append("\\begin{tabularx}{\\linewidth}{r X l l r l}")
    lines.append("\\toprule")
    lines.append("Rank & Firm & Role & Corridor & $H1_{\\log}$ & Conf. \\\\")
    lines.append("\\midrule")
    for row in df.itertuples(index=False):
        lines.append(
            f"{int(row.Rank)} & "
            f"{latex_escape(row.Firm)} & "
            f"{latex_escape(row.Role)} & "
            f"{latex_escape(row.Corridor)} & "
            f"{row.H1_log_obligation:.4f} & "
            f"{latex_escape(row.Confidence)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
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
        else out_root
        / snapshot
        / "m0_7"
        / "top25_high_impact_semiconductor_value_chain_strict_observed.csv"
    )
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    out_dir = tables_root / "main_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    src = pd.read_csv(input_csv).copy()
    disclosed_csv = (
        out_root
        / snapshot
        / "m0_7"
        / "top25_high_impact_semiconductor_value_chain_strict_disclosed.csv"
    )
    full_csv = (
        out_root / snapshot / "m0_7" / "top25_high_impact_semiconductor_value_chain_strict_full.csv"
    )
    for p in [disclosed_csv, full_csv]:
        if not p.exists():
            raise FileNotFoundError(p)
    disclosed = pd.read_csv(disclosed_csv, usecols=["analysis_uid"]).copy()
    full = pd.read_csv(full_csv, usecols=["analysis_uid"]).copy()
    disclosed_set = set(disclosed["analysis_uid"].astype(str).tolist())
    full_set = set(full["analysis_uid"].astype(str).tolist())
    observed_set = set(src["analysis_uid"].astype(str).tolist())
    required_cols = {
        "rank_semiconductor_lens",
        "name",
        "selected_rule_classes",
        "tier_bin",
        "impact_stratum",
        "h1_log_obligation_any_support",
        "confidence_class",
    }
    missing = required_cols - set(src.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    src = (
        src.sort_values(["rank_semiconductor_lens", "name"], ascending=[True, True]).head(25).copy()
    )

    out = pd.DataFrame(
        {
            "Rank": pd.to_numeric(src["rank_semiconductor_lens"], errors="coerce")
            .fillna(0)
            .astype(int),
            "Firm": src["name"].fillna("Unknown").astype(str),
            "Role": [
                role_label(rc, cc)
                for rc, cc in zip(src["selected_rule_classes"], src["impact_stratum"], strict=False)
            ],
            "Corridor": src["impact_stratum"].fillna("unknown").astype(str),
            "H1_log_obligation": pd.to_numeric(
                src["h1_log_obligation_any_support"], errors="coerce"
            ).fillna(0.0),
            "Confidence": [
                confidence_label_from_presence(
                    int(uid in observed_set) + int(uid in disclosed_set) + int(uid in full_set)
                )
                for uid in src["analysis_uid"].astype(str).tolist()
            ],
        }
    )

    csv_out = out_dir / "tab_4_2_strict_top25_observed.csv"
    tex_out = out_dir / "tab_4_2_strict_top25_observed.tex"
    spec_out = out_dir / "tab_4_2_strict_top25_observed_spec.json"
    meta_out = out_dir / "tab_4_2_strict_top25_observed_run_metadata.json"

    out.to_csv(csv_out, index=False)
    tex_out.write_text(build_latex_tabular(out), encoding="utf-8")

    spec = {
        "table_name": "tab_4_2_strict_top25_observed",
        "view": "observed",
        "source": str(input_csv),
        "columns": out.columns.tolist(),
        "rows": len(out),
        "role_label_rule": "derived from selected_rule_classes + impact_stratum",
        "confidence_rule": "High if present in strict Top-25 across all 3 views; Medium if present in 2; Low if present in 1",
        "h1_rounding": 4,
    }
    spec_out.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    run_meta = {
        "module": "export_tab_4_2",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "inputs": {
            "strict_top25_csv": str(input_csv),
            "strict_top25_sha256": file_sha256(input_csv),
            "strict_top25_disclosed_csv": str(disclosed_csv),
            "strict_top25_disclosed_sha256": file_sha256(disclosed_csv),
            "strict_top25_full_csv": str(full_csv),
            "strict_top25_full_sha256": file_sha256(full_csv),
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
