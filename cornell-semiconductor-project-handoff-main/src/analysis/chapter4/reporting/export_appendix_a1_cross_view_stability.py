#!/usr/bin/env python3
"""Export Appendix A1 cross-view stability figure + summary table."""

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
    parser = argparse.ArgumentParser(description="Export Appendix A1 cross-view stability outputs")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
    parser.add_argument(
        "--quadrants-csv",
        default=None,
        help="Optional override for m7/high_impact_confidence_quadrants.csv",
    )
    parser.add_argument(
        "--confidence-summary-csv",
        default=None,
        help="Optional override for m7/high_impact_confidence_summary.csv",
    )
    parser.add_argument(
        "--out-fig-dir",
        default=None,
        help="Optional override output figure directory (default: <figs_root>/appendix)",
    )
    parser.add_argument(
        "--out-table-dir",
        default=None,
        help="Optional override output table directory (default: <tables_root>/appendix)",
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


def topn_set(df: pd.DataFrame, rank_col: str, n: int) -> set[str]:
    if rank_col not in df.columns:
        return set()
    sub = df[pd.to_numeric(df[rank_col], errors="coerce").notna()].copy()
    if sub.empty:
        return set()
    sub["_rank"] = pd.to_numeric(sub[rank_col], errors="coerce")
    return set(sub.loc[sub["_rank"] <= float(n), "analysis_uid"].astype(str))


def jaccard(a: set[str], b: set[str]) -> float:
    u = len(a | b)
    if u == 0:
        return 0.0
    return float(len(a & b) / u)


def pair_rank_corr(
    df: pd.DataFrame, col_a: str, col_b: str
) -> tuple[int, float | None, float | None]:
    if col_a not in df.columns or col_b not in df.columns:
        return 0, None, None
    sub = df[["analysis_uid", col_a, col_b]].copy()
    sub[col_a] = pd.to_numeric(sub[col_a], errors="coerce")
    sub[col_b] = pd.to_numeric(sub[col_b], errors="coerce")
    sub = sub.dropna(subset=[col_a, col_b]).copy()
    n = len(sub)
    if n < 2:
        return n, None, None
    spearman = float(sub[col_a].corr(sub[col_b], method="spearman"))
    kendall = float(sub[col_a].corr(sub[col_b], method="kendall"))
    return n, spearman, kendall


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        v = float(value)
    except Exception:
        return "--"
    if not np.isfinite(v):
        return "--"
    return f"{v:.{digits}f}"


def fmt_count(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return str(round(float(value)))
    except Exception:
        return "--"


def build_latex_tabular(summary_wide: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("\\begin{tabular}{l r r r r}")
    lines.append("\\toprule")
    lines.append("Metric & Disc.-Obs. & Obs.-Full & Disc.-Full & Overall \\\\")
    lines.append("\\midrule")
    for row in summary_wide.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.metric)} & "
            f"{latex_escape(row.disclosed_observed)} & "
            f"{latex_escape(row.observed_full)} & "
            f"{latex_escape(row.disclosed_full)} & "
            f"{latex_escape(row.overall)} \\\\"
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
    figs_root = Path(str(cfg.get("paths", {}).get("figs_root", "figs/chapter4/v2_fix01")))
    tables_root = Path(str(cfg.get("paths", {}).get("tables_root", "tables/chapter4/v2_fix01")))

    quadrants_csv = (
        Path(args.quadrants_csv)
        if args.quadrants_csv
        else out_root / snapshot / "m7" / "high_impact_confidence_quadrants.csv"
    )
    conf_summary_csv = (
        Path(args.confidence_summary_csv)
        if args.confidence_summary_csv
        else out_root / snapshot / "m7" / "high_impact_confidence_summary.csv"
    )
    for p in (quadrants_csv, conf_summary_csv):
        if not p.exists():
            raise FileNotFoundError(p)

    out_fig_dir = Path(args.out_fig_dir) if args.out_fig_dir else figs_root / "appendix"
    out_tbl_dir = Path(args.out_table_dir) if args.out_table_dir else tables_root / "appendix"
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_tbl_dir.mkdir(parents=True, exist_ok=True)

    q = pd.read_csv(quadrants_csv).copy()
    conf = pd.read_csv(conf_summary_csv).copy()
    if "analysis_uid" not in q.columns:
        raise ValueError("quadrants file missing analysis_uid")

    # ---------- Figure data: Top-N overlaps ----------
    top_ns = [10, 25, 50, 100, 250]
    pairs = [
        ("disclosed_observed", "rank_disclosed", "rank_observed", "Disclosed vs Observed"),
        ("observed_full", "rank_observed", "rank_full", "Observed vs Full"),
        ("disclosed_full", "rank_disclosed", "rank_full", "Disclosed vs Full"),
    ]
    plot_rows: list[dict[str, Any]] = []
    for n in top_ns:
        for pair_id, col_a, col_b, pair_label in pairs:
            a = topn_set(q, col_a, n)
            b = topn_set(q, col_b, n)
            jac = jaccard(a, b)
            plot_rows.append(
                {
                    "pair_id": pair_id,
                    "pair_label": pair_label,
                    "top_n": int(n),
                    "set_a_size": len(a),
                    "set_b_size": len(b),
                    "overlap_count": len(a & b),
                    "union_count": len(a | b),
                    "jaccard_overlap": float(jac),
                    "jaccard_pct": float(jac * 100.0),
                }
            )
    plot_df = pd.DataFrame(plot_rows)
    plot_data_csv = out_fig_dir / "fig_a_1_cross_view_overlap_curve_plot_data.csv"
    plot_df.to_csv(plot_data_csv, index=False)

    # ---------- Figure rendering ----------
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color="#d9d9d9", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", which="major", labelsize=9, width=0.8, length=3.5)

    style = {
        "disclosed_observed": {
            "color": "#111111",
            "linestyle": "-",
            "marker": "o",
            "label": "Disclosed vs Observed",
        },
        "observed_full": {
            "color": "#555555",
            "linestyle": "--",
            "marker": "s",
            "label": "Observed vs Full",
        },
        "disclosed_full": {
            "color": "#888888",
            "linestyle": ":",
            "marker": "^",
            "label": "Disclosed vs Full",
        },
    }
    for pair_id, grp in plot_df.groupby("pair_id", sort=False):
        grp = grp.sort_values("top_n")
        st = style[pair_id]
        ax.plot(
            grp["top_n"].to_numpy(dtype=float),
            grp["jaccard_pct"].to_numpy(dtype=float),
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=1.8,
            marker=st["marker"],
            markersize=4.6,
            label=st["label"],
        )

    ax.set_xlabel("Top-N threshold (N)", fontsize=10)
    ax.set_ylabel("Top-N overlap (Jaccard, %)", fontsize=10)
    ax.set_xlim(8, 255)
    ax.set_xticks(top_ns)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper right", frameon=False, fontsize=8.8, ncol=1)

    fig_pdf = out_fig_dir / "fig_a_1_cross_view_overlap_curve.pdf"
    fig_svg = out_fig_dir / "fig_a_1_cross_view_overlap_curve.svg"
    fig_png = out_fig_dir / "fig_a_1_cross_view_overlap_curve.png"
    fig.savefig(fig_pdf, format="pdf")
    fig.savefig(fig_svg, format="svg")
    fig.savefig(fig_png, format="png", dpi=600)
    plt.close(fig)

    # ---------- Summary table ----------
    # Pairwise top-25 overlaps.
    top25 = {}
    for pair_id, col_a, col_b, _ in pairs:
        a = topn_set(q, col_a, 25)
        b = topn_set(q, col_b, 25)
        top25[pair_id] = {
            "count": len(a & b),
            "jaccard": jaccard(a, b),
        }

    # Pairwise rank correlations on common ranked nodes.
    corr = {}
    for pair_id, col_a, col_b, _ in pairs:
        n_common, spearman, kendall = pair_rank_corr(q, col_a, col_b)
        corr[pair_id] = {
            "n_common": n_common,
            "spearman": spearman,
            "kendall": kendall,
        }

    conf_map = {
        "high_impact_high_confidence": 0,
        "high_impact_medium_confidence": 0,
        "high_impact_low_confidence": 0,
    }
    for row in conf.itertuples(index=False):
        cls = str(row.confidence_class)
        if cls in conf_map:
            conf_map[cls] = int(row.n_nodes)
    conf_total = int(sum(conf_map.values()))

    row_order = [
        "Top-25 overlap count",
        "Top-25 Jaccard overlap",
        "Common ranked nodes (n)",
        "Spearman rank correlation",
        "Kendall rank correlation",
        "High-impact high confidence (n)",
        "High-impact medium confidence (n)",
        "High-impact low confidence (n)",
        "High-impact total (n)",
    ]
    rows: list[dict[str, Any]] = []
    for metric in row_order:
        row = {
            "metric": metric,
            "disclosed_observed": "",
            "observed_full": "",
            "disclosed_full": "",
            "overall": "",
        }
        if metric == "Top-25 overlap count":
            row["disclosed_observed"] = fmt_count(top25["disclosed_observed"]["count"])
            row["observed_full"] = fmt_count(top25["observed_full"]["count"])
            row["disclosed_full"] = fmt_count(top25["disclosed_full"]["count"])
        elif metric == "Top-25 Jaccard overlap":
            row["disclosed_observed"] = fmt_num(top25["disclosed_observed"]["jaccard"], 3)
            row["observed_full"] = fmt_num(top25["observed_full"]["jaccard"], 3)
            row["disclosed_full"] = fmt_num(top25["disclosed_full"]["jaccard"], 3)
        elif metric == "Common ranked nodes (n)":
            row["disclosed_observed"] = fmt_count(corr["disclosed_observed"]["n_common"])
            row["observed_full"] = fmt_count(corr["observed_full"]["n_common"])
            row["disclosed_full"] = fmt_count(corr["disclosed_full"]["n_common"])
        elif metric == "Spearman rank correlation":
            row["disclosed_observed"] = fmt_num(corr["disclosed_observed"]["spearman"], 3)
            row["observed_full"] = fmt_num(corr["observed_full"]["spearman"], 3)
            row["disclosed_full"] = fmt_num(corr["disclosed_full"]["spearman"], 3)
        elif metric == "Kendall rank correlation":
            row["disclosed_observed"] = fmt_num(corr["disclosed_observed"]["kendall"], 3)
            row["observed_full"] = fmt_num(corr["observed_full"]["kendall"], 3)
            row["disclosed_full"] = fmt_num(corr["disclosed_full"]["kendall"], 3)
        elif metric == "High-impact high confidence (n)":
            row["overall"] = fmt_count(conf_map["high_impact_high_confidence"])
        elif metric == "High-impact medium confidence (n)":
            row["overall"] = fmt_count(conf_map["high_impact_medium_confidence"])
        elif metric == "High-impact low confidence (n)":
            row["overall"] = fmt_count(conf_map["high_impact_low_confidence"])
        elif metric == "High-impact total (n)":
            row["overall"] = fmt_count(conf_total)
        rows.append(row)
    table_df = pd.DataFrame(rows)

    table_csv = out_tbl_dir / "tab_a_1_cross_view_stability_summary.csv"
    table_tex = out_tbl_dir / "tab_a_1_cross_view_stability_summary.tex"
    table_spec = out_tbl_dir / "tab_a_1_cross_view_stability_summary_spec.json"
    run_meta = out_tbl_dir / "tab_a_1_cross_view_stability_summary_run_metadata.json"

    table_df.to_csv(table_csv, index=False)
    table_tex.write_text(build_latex_tabular(table_df), encoding="utf-8")

    spec = {
        "figure_name": "fig_a_1_cross_view_overlap_curve",
        "table_name": "tab_a_1_cross_view_stability_summary",
        "view_pairs": [p[0] for p in pairs],
        "top_n_values_for_curve": top_ns,
        "top_n_for_table_overlap": 25,
        "confidence_classes": conf_map,
        "confidence_total": conf_total,
        "outputs": {
            "figure_pdf": str(fig_pdf),
            "figure_svg": str(fig_svg),
            "figure_png": str(fig_png),
            "figure_plot_data_csv": str(plot_data_csv),
            "table_csv": str(table_csv),
            "table_tex": str(table_tex),
        },
    }
    table_spec.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    meta = {
        "module": "export_appendix_a1_cross_view_stability",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "inputs": {
            "quadrants_csv": str(quadrants_csv),
            "quadrants_sha256": file_sha256(quadrants_csv),
            "confidence_summary_csv": str(conf_summary_csv),
            "confidence_summary_sha256": file_sha256(conf_summary_csv),
        },
        "outputs": {
            "figure_pdf": str(fig_pdf),
            "figure_svg": str(fig_svg),
            "figure_png": str(fig_png),
            "figure_plot_data_csv": str(plot_data_csv),
            "table_csv": str(table_csv),
            "table_tex": str(table_tex),
            "table_spec_json": str(table_spec),
        },
    }
    run_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[done] wrote {fig_pdf}")
    print(f"[done] wrote {fig_svg}")
    print(f"[done] wrote {fig_png}")
    print(f"[done] wrote {plot_data_csv}")
    print(f"[done] wrote {table_csv}")
    print(f"[done] wrote {table_tex}")
    print(f"[done] wrote {table_spec}")
    print(f"[done] wrote {run_meta}")


if __name__ == "__main__":
    main()
