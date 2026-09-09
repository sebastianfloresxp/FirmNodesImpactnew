#!/usr/bin/env python3
"""Export Figure 4.1 (observed robustness: random vs targeted, H1)."""

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
    parser = argparse.ArgumentParser(description="Export Figure 4.1 robustness plot (observed)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional override for m6_1/robustness_summary_observed.csv",
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


def style_map() -> dict[str, dict[str, Any]]:
    return {
        "random": {
            "label": "Random baseline",
            "color": "#111111",
            "linestyle": "-",
            "linewidth": 1.9,
            "marker": "o",
            "markersize": 4.8,
            "zorder": 4,
        },
        "target_pagerank": {
            "label": "Targeted (PageRank)",
            "color": "#333333",
            "linestyle": "--",
            "linewidth": 1.8,
            "marker": "s",
            "markersize": 4.8,
            "zorder": 5,
        },
        "target_m4_bottleneck": {
            "label": "Targeted (Corridor bottleneck)",
            "color": "#555555",
            "linestyle": ":",
            "linewidth": 2.0,
            "marker": "^",
            "markersize": 5.0,
            "zorder": 5,
        },
        "target_m3_prime_reach": {
            "label": "Targeted (Prime-reach)",
            "color": "#8c8c8c",
            "linestyle": "-.",
            "linewidth": 1.4,
            "marker": "D",
            "markersize": 3.8,
            "zorder": 3,
        },
    }


def ensure_columns(df: pd.DataFrame, cols: set[str]) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")


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
        else out_root / snapshot / "m6_1" / "robustness_summary_observed.csv"
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
            "strategy",
            "frac_removed",
            "h1_reach_loss_mean",
            "h1_reach_loss_p05",
            "h1_reach_loss_p95",
        },
    )
    df = df[df["view"].astype(str) == "observed"].copy()
    if df.empty:
        raise ValueError("no observed rows found in robustness summary")

    for col in ("frac_removed", "h1_reach_loss_mean", "h1_reach_loss_p05", "h1_reach_loss_p95"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["frac_removed", "h1_reach_loss_mean"]).copy()

    # Main-text figure intentionally focuses on the clean random vs two-targeted comparison.
    strategy_order = ["random", "target_pagerank", "target_m4_bottleneck"]
    styles = style_map()
    available = [s for s in strategy_order if s in set(df["strategy"].astype(str))]
    if "random" not in available:
        raise ValueError("random strategy is required for Figure 4.1")

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=False)

    # Minimal, print-safe frame.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color="#d9d9d9", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", which="major", labelsize=9, width=0.8, length=3.5)

    plot_rows: list[dict[str, Any]] = []

    # Random baseline line + uncertainty band.
    random_df = df[df["strategy"] == "random"].sort_values("frac_removed").copy()
    x_pct = random_df["frac_removed"].to_numpy(dtype=float) * 100.0
    y_pct = random_df["h1_reach_loss_mean"].to_numpy(dtype=float) * 100.0
    y_lo_pct = random_df["h1_reach_loss_p05"].to_numpy(dtype=float) * 100.0
    y_hi_pct = random_df["h1_reach_loss_p95"].to_numpy(dtype=float) * 100.0
    st = styles["random"]
    ax.plot(
        x_pct,
        y_pct,
        label=st["label"],
        color=st["color"],
        linestyle=st["linestyle"],
        linewidth=st["linewidth"],
        marker=st["marker"],
        markersize=st["markersize"],
        zorder=st["zorder"],
    )
    for xv, yv, yl, yh in zip(x_pct, y_pct, y_lo_pct, y_hi_pct, strict=False):
        plot_rows.append(
            {
                "strategy": "random",
                "series_label": st["label"],
                "x_pct_removed": float(xv),
                "h1_loss_pct_mean": float(yv),
                "h1_loss_pct_p05": float(yl),
                "h1_loss_pct_p95": float(yh),
            }
        )

    # Targeted lines.
    for strategy in [s for s in available if s != "random"]:
        sub = df[df["strategy"] == strategy].sort_values("frac_removed").copy()
        if sub.empty:
            continue
        st = styles.get(strategy)
        if st is None:
            continue
        x = sub["frac_removed"].to_numpy(dtype=float) * 100.0
        y = sub["h1_reach_loss_mean"].to_numpy(dtype=float) * 100.0
        ax.plot(
            x,
            y,
            label=st["label"],
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=st["linewidth"],
            marker=st["marker"],
            markersize=st["markersize"],
            zorder=st["zorder"],
        )
        for xv, yv in zip(x, y, strict=False):
            plot_rows.append(
                {
                    "strategy": strategy,
                    "series_label": st["label"],
                    "x_pct_removed": float(xv),
                    "h1_loss_pct_mean": float(yv),
                    "h1_loss_pct_p05": np.nan,
                    "h1_loss_pct_p95": np.nan,
                }
            )

    xticks = sorted({float(v) for v in (df["frac_removed"].to_numpy(dtype=float) * 100.0)})
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:g}%" for v in xticks], fontsize=9)
    ax.set_xlabel("Nodes removed", fontsize=10)
    ax.set_ylabel(r"$H1_{\log}$ loss (%)", fontsize=10)

    ax.set_xlim(min(xticks) - 0.05, max(xticks) + 0.15)
    ymax = float(np.nanmax(df["h1_reach_loss_mean"].to_numpy(dtype=float) * 100.0))
    ax.set_ylim(0.0, ymax * 1.10 if ymax > 0 else 1.0)

    legend_font = 8.4
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
        fontsize=legend_font,
        handlelength=2.5,
        borderaxespad=0.2,
        ncols=3,
        columnspacing=1.2,
    )

    # Leave room for axis labels while keeping clean whitespace.
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.27, top=0.98)

    out_pdf = out_dir / "fig_4_1_robustness_observed.pdf"
    out_svg = out_dir / "fig_4_1_robustness_observed.svg"
    out_png = out_dir / "fig_4_1_robustness_observed.png"
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_png, format="png", dpi=600)
    plt.close(fig)

    plot_data = pd.DataFrame(plot_rows).sort_values(
        ["strategy", "x_pct_removed"], ascending=[True, True]
    )
    plot_data_csv = out_dir / "fig_4_1_robustness_observed_plot_data.csv"
    plot_data.to_csv(plot_data_csv, index=False)

    spec = {
        "figure_name": "fig_4_1_robustness_observed",
        "view": "observed",
        "source_csv": str(input_csv),
        "source_csv_sha256": file_sha256(input_csv),
        "output_files": {
            "pdf": str(out_pdf),
            "svg": str(out_svg),
            "png": str(out_png),
            "plot_data_csv": str(plot_data_csv),
        },
        "figure_size_inches": [6.5, 4.0],
        "font_sizes": {"axes_label": 10, "ticks": 9, "legend": legend_font},
        "x_axis": {"label": "Nodes removed", "ticks_percent": xticks},
        "y_axis": {"label": "H1_log loss (%)"},
        "series": [
            {
                "strategy": s,
                "label": styles[s]["label"],
                "color": styles[s]["color"],
                "linestyle": styles[s]["linestyle"],
                "marker": styles[s]["marker"],
            }
            for s in available
            if s in styles
        ],
        "random_uncertainty_band": {"enabled": False},
    }
    spec_json = out_dir / "fig_4_1_robustness_observed_spec.json"
    spec_json.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    run_meta = {
        "module": "export_fig_4_1_robustness_observed",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "inputs": {
            "robustness_summary_csv": str(input_csv),
            "robustness_summary_sha256": file_sha256(input_csv),
        },
        "outputs": {
            "pdf": str(out_pdf),
            "svg": str(out_svg),
            "png": str(out_png),
            "plot_data_csv": str(plot_data_csv),
            "spec_json": str(spec_json),
        },
    }
    run_meta_json = out_dir / "fig_4_1_robustness_observed_run_metadata.json"
    run_meta_json.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"[done] wrote {out_pdf}")
    print(f"[done] wrote {out_svg}")
    print(f"[done] wrote {out_png}")
    print(f"[done] wrote {plot_data_csv}")
    print(f"[done] wrote {spec_json}")
    print(f"[done] wrote {run_meta_json}")


if __name__ == "__main__":
    main()
