#!/usr/bin/env python3
"""Export Figure 4.2 (observed mechanism decomposition by cohort, deny-only)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG = "src/analysis/chapter4/config/ch4_v2_fix01.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Figure 4.2 mechanism decomposition (observed)"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional override for m6_6/mechanism_interdiction_summary_observed.csv",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=25,
        help="Interdiction budget k_target to visualize (default: 25)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional override output directory (default: <figs_root>/main_text)",
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


def ensure_columns(df: pd.DataFrame, cols: set[str]) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")


def cohort_label(cohort: str) -> str:
    mapping = {
        "corridor_intermediary": "Corridor\nIntermediary",
        "upstream_of_semi": "Upstream\nof Semis",
        "prime_adjacent": "Prime\nAdjacent",
    }
    return mapping.get(cohort, cohort.replace("_", " ").title())


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    figs_root = Path(str(cfg.get("paths", {}).get("figs_root", "figs/chapter4/v2_fix01")))

    input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else out_root / snapshot / "m6_6" / "mechanism_interdiction_summary_observed.csv"
    )
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    out_dir = Path(args.out_dir) if args.out_dir else figs_root / "main_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv).copy()
    ensure_columns(
        df,
        {
            "view",
            "cohort",
            "objective",
            "k_target",
            "share_of_unrestricted_h1",
            "share_of_unrestricted_h2",
        },
    )
    df = df[df["view"].astype(str) == "observed"].copy()
    df["k_target"] = pd.to_numeric(df["k_target"], errors="coerce").astype("Int64")
    k_target = int(args.k)

    m6_1_cfg = cfg.get("m6_1", {})
    h1_key = str(m6_1_cfg.get("primary_h1_key", "log_obligation_any_support"))
    deny_objective = f"deny_h1_{h1_key}"

    cohort_order = ["corridor_intermediary", "prime_adjacent", "upstream_of_semi"]
    rows: list[dict[str, Any]] = []
    sub_k = df[df["k_target"] == int(k_target)].copy()
    if sub_k.empty:
        raise ValueError(f"no observed rows found for k_target={k_target}")
    for cohort in cohort_order:
        sub = sub_k[sub_k["cohort"].astype(str) == cohort].copy()
        if sub.empty:
            continue

        deny_row = sub[sub["objective"].astype(str) == deny_objective]
        deny_share = (
            float(deny_row["share_of_unrestricted_h1"].iloc[0]) if len(deny_row) else np.nan
        )

        rows.append(
            {
                "cohort": cohort,
                "cohort_label": cohort_label(cohort),
                "deny_share": deny_share,
                "deny_pct": deny_share * 100.0 if np.isfinite(deny_share) else np.nan,
                "k_target": int(k_target),
            }
        )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise ValueError(f"no cohort rows available after filtering for k_target={k_target}")

    # Global x-limit across panels for direct visual comparability.
    # Deny shares are normalized to unrestricted deny at same k; keep fixed scale for interpretability.
    xlim_max = 100.0

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.7), constrained_layout=False)
    sub = plot_df.copy()
    sub["cohort_order"] = (
        sub["cohort"]
        .astype(str)
        .apply(lambda c: cohort_order.index(c) if c in cohort_order else 999)
    )
    sub = sub.sort_values("cohort_order", ascending=True)
    y = np.arange(len(sub))
    deny_vals = sub["deny_pct"].to_numpy(dtype=float)

    # Print-safe style.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="x", color="#d9d9d9", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", which="major", labelsize=8.8, width=0.8, length=3.5)

    # Lollipop stems from zero to deny share per cohort.
    for idx, d_val in enumerate(deny_vals):
        ax.hlines(
            y=idx,
            xmin=0.0,
            xmax=float(d_val),
            color="#8c8c8c",
            linewidth=1.4,
            zorder=2,
        )

    ax.scatter(
        deny_vals,
        y,
        s=52,
        marker="o",
        facecolor="#1f1f1f",
        edgecolor="#1f1f1f",
        linewidth=0.7,
        zorder=3,
    )

    ax.axvline(100.0, color="#7a7a7a", linewidth=1.0, linestyle=":", zorder=1)
    ax.set_xlim(0.0, xlim_max)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["cohort_label"].tolist(), fontsize=8.8)
    ax.set_ylabel("Cohort", fontsize=10)
    ax.invert_yaxis()  # first cohort at top

    # Direct value labels for fast read in print.
    for idx, d_val in enumerate(deny_vals):
        label = f"{d_val:.1f}%"
        if d_val >= 97.5:
            # Keep 100.0% labels inside the plotting window.
            text_x = float(d_val) - 1.8
            ha = "right"
        else:
            text_x = float(d_val) + 1.8
            ha = "left"
        ax.text(
            text_x,
            float(idx),
            label,
            va="center",
            ha=ha,
            fontsize=8.0,
            color="#1f1f1f",
        )

    ax.set_xlabel("Cohort-constrained deny harm (% of unrestricted maximum)", fontsize=10)
    legend_font = 8.4
    # Slightly increase right padding so the 100 tick label does not clip.
    fig.subplots_adjust(left=0.23, right=0.965, bottom=0.18, top=0.90, wspace=0.12)

    out_pdf = out_dir / "fig_4_2_mechanism_observed.pdf"
    out_svg = out_dir / "fig_4_2_mechanism_observed.svg"
    out_png = out_dir / "fig_4_2_mechanism_observed.png"
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_png, format="png", dpi=600)
    plt.close(fig)

    plot_csv = out_dir / "fig_4_2_mechanism_observed_plot_data.csv"
    plot_df.to_csv(plot_csv, index=False)

    spec = {
        "figure_name": "fig_4_2_mechanism_observed",
        "view": "observed",
        "k_target": int(k_target),
        "source_csv": str(input_csv),
        "source_csv_sha256": file_sha256(input_csv),
        "output_files": {
            "pdf": str(out_pdf),
            "svg": str(out_svg),
            "png": str(out_png),
            "plot_data_csv": str(plot_csv),
        },
        "figure_size_inches": [6.5, 3.7],
        "font_sizes": {"axes_label": 10, "ticks": 9, "legend": legend_font},
        "x_axis": {
            "label": "Cohort-constrained deny harm (% of unrestricted maximum)",
            "range_pct": [0.0, 100.0],
            "reference_line_pct": 100.0,
        },
        "y_axis": {"label": "Cohort"},
        "lollipop": {
            "series": "deny",
            "label": "Deny share",
            "stem_color": "#8c8c8c",
            "stem_linewidth": 1.5,
            "marker": "o",
            "marker_facecolor": "#1f1f1f",
            "marker_edgecolor": "#1f1f1f",
            "label_precision_decimals": 1,
        },
        "cohorts": (
            plot_df[["cohort", "cohort_label"]]
            .drop_duplicates()
            .sort_values("cohort", ascending=True)
            .to_dict(orient="records")
        ),
    }
    spec_json = out_dir / "fig_4_2_mechanism_observed_spec.json"
    spec_json.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    run_meta = {
        "module": "export_fig_4_2_mechanism_observed",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "inputs": {
            "mechanism_summary_csv": str(input_csv),
            "mechanism_summary_sha256": file_sha256(input_csv),
            "k_target_requested": int(k_target),
        },
        "outputs": {
            "pdf": str(out_pdf),
            "svg": str(out_svg),
            "png": str(out_png),
            "plot_data_csv": str(plot_csv),
            "spec_json": str(spec_json),
        },
    }
    run_meta_json = out_dir / "fig_4_2_mechanism_observed_run_metadata.json"
    run_meta_json.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"[done] wrote {out_pdf}")
    print(f"[done] wrote {out_svg}")
    print(f"[done] wrote {out_png}")
    print(f"[done] wrote {plot_csv}")
    print(f"[done] wrote {spec_json}")
    print(f"[done] wrote {run_meta_json}")


if __name__ == "__main__":
    main()
