#!/usr/bin/env python3
"""Precision–yield frontier for the ensemble (defaults to test split)."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, LogLocator, NullFormatter

ANALYSIS_DIR = Path("results/ensemble/meta_ranker/meta_ranker_v4/analysis_tables")
OUTPUT_DIR_DEFAULT = Path("figs/chapter2")
OUTPUT_BASENAME_TEMPLATE = "fig_03_precision_yield_frontier_{split}"
ANNOTATION_KEYS = [
    "tau=1",
    "tau_precision_0.90",
    "tau_precision_0.30",
    "tau_precision_0.20",
    "K=10_floor=0.995_cap=50",
]

TOPK_K_KEEP = {1, 3, 10}
TOPK_FLOOR_KEEP = {0.995, 0.99, 0.90, 0.85}

DCS_PARQUET_PATH = Path("predictions/meta_ranker_v4/dcs_top5_from_fhpe.parquet")

K_COLOR_MAP = {
    1: "#1f77b4",
    3: "#ff7f0e",
    10: "#2ca02c",
}

CALL_OUT_AX_POSITIONS = {
    "tau=1": (0.22, 0.94),
    "tau_precision_0.90": (0.42, 0.78),
    "tau_precision_0.30": (0.68, 0.46),
    "tau_precision_0.20": (0.76, 0.32),
    "K=10_floor=0.995_cap=50": (0.62, 0.24),
}

CALL_OUT_ALIGNMENT = {
    "tau=1": ("left", "top"),
    "tau_precision_0.90": ("left", "top"),
    "tau_precision_0.30": ("left", "top"),
    "tau_precision_0.20": ("left", "top"),
    "K=10_floor=0.995_cap=50": ("left", "top"),
}

CALL_OUT_CONNECTIONS = {
    "tau=1": "angle3,angleA=0,angleB=90",
    "tau_precision_0.90": "angle3,angleA=0,angleB=65",
    "tau_precision_0.30": "angle3,angleA=0,angleB=45",
    "tau_precision_0.20": "angle3,angleA=0,angleB=25",
    "K=10_floor=0.995_cap=50": "angle3,angleA=0,angleB=-45",
}


@dataclass
class OperatingPoint:
    label: str
    precision: float
    recall: float
    yield_fraction: float
    predicted_edges: int | None = None
    lift: float | None = None

    def callout_metrics(self, total_candidates: int | None) -> str:
        edges = self.predicted_edges
        if edges is None and total_candidates is not None:
            edges = round(self.yield_fraction * total_candidates)

        precision_pct = self.precision * 100
        yield_pct = self.yield_fraction * 100
        yield_str = format_small_percent(yield_pct)
        parts = [f"P={precision_pct:.1f}%", f"Y={yield_str}%"]
        if edges and edges > 0:
            parts[-1] += f" (≈{edges:,.0f} edges)"
        if self.lift and self.lift > 0:
            lift_val = self.lift
            if lift_val >= 100:
                lift_str = f"≈{lift_val:.0f}×"
            elif lift_val >= 10:
                lift_str = f"≈{lift_val:.1f}×"
            else:
                lift_str = f"≈{lift_val:.2f}×"
            parts.append(f"Lift {lift_str}")
        return " | ".join(parts)


@dataclass
class BaseRateInfo:
    rate: float
    total_candidates: int


def read_csv(path: Path, **read_kwargs) -> pd.DataFrame | None:
    if not path.exists():
        print(f"ERROR: required input missing: {path}")
        return None
    try:
        return pd.read_csv(path, **read_kwargs)
    except Exception as exc:
        print(f"ERROR: failed to read {path}: {exc}")
        return None


def load_threshold_curve(path: Path, max_points: int = 4000) -> pd.DataFrame | None:
    if not path.exists():
        print(f"ERROR: required PR curve data not found at {path}")
        return None

    usecols = ["threshold", "precision", "recall", "yield"]
    reservoir: list[tuple[float, float, float, float]] = []
    total = 0
    first_row: pd.Series | None = None
    last_row: pd.Series | None = None
    random.seed(0)

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200000):
        chunk["precision"] = pd.to_numeric(chunk["precision"], errors="coerce")
        chunk["yield"] = pd.to_numeric(chunk["yield"], errors="coerce")
        chunk["recall"] = pd.to_numeric(chunk.get("recall"), errors="coerce")
        chunk = chunk.dropna(subset=["precision", "yield"])
        if chunk.empty:
            continue
        if first_row is None:
            first_row = chunk.iloc[0]
        last_row = chunk.iloc[-1]
        for row in chunk.itertuples(index=False, name=None):
            total += 1
            if len(reservoir) < max_points:
                reservoir.append(row)
            else:
                j = random.randint(0, total - 1)  # nosec B311 -- seeded for reproducible sampling, not cryptographic
                if j < max_points:
                    reservoir[j] = row

    if not reservoir and first_row is None:
        print("ERROR: PR curve contained no usable rows")
        return None

    records = []
    if first_row is not None:
        records.append(tuple(first_row[col] for col in usecols))
    records.extend(reservoir)
    if last_row is not None:
        records.append(tuple(last_row[col] for col in usecols))

    df = pd.DataFrame(records, columns=usecols)
    df = df.dropna(subset=["precision", "yield"])
    df = df.sort_values("yield")
    df = df.drop_duplicates(subset=["yield"], keep="first")
    return df


def load_topk_grid(path: Path, split: str) -> pd.DataFrame | None:
    df = read_csv(path)
    if df is None:
        return None
    df["split"] = df["split"].astype(str)
    df = df[df["split"].str.lower() == split.lower()].copy()
    df["cap_numeric"] = pd.to_numeric(df["cap"], errors="coerce")
    df = df[df["cap_numeric"] == 50].copy()
    if df.empty:
        print("WARNING: no Top-K rows with cap=50 found; plotting full grid instead")
        df = read_csv(path)
        if df is None:
            return None
        df["split"] = df["split"].astype(str)
        df = df[df["split"].str.lower() == split.lower()].copy()
    df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
    df = df.dropna(subset=["precision", "yield", "recall"])
    return df


def load_operating_points(path: Path, split: str) -> dict[str, OperatingPoint]:
    df = read_csv(path)
    points: dict[str, OperatingPoint] = {}
    if df is None:
        return points
    df = df[df["split"].astype(str).str.lower() == split.lower()].copy()
    for key, label in [
        ("tau_precision_0.95", "Core (FHPE) τ (≥95% precision)"),
        ("tau_precision_0.90", "Validation lock τ (≥90% precision)"),
        ("tau_precision_0.85", "Discovery τ (≥85% precision)"),
        ("K=10_floor=0.995_cap=50", "Chosen Core (K=10, floor=0.995, cap=50)"),
    ]:
        row = df[df["key"] == key].head(1)
        if row.empty:
            continue
        r = row.iloc[0]
        points[key] = OperatingPoint(
            label=label,
            precision=float(r["precision"]),
            recall=float(r["recall"]),
            yield_fraction=float(r["yield"]),
            predicted_edges=int(r["predicted_edges"])
            if not pd.isna(r["predicted_edges"])
            else None,
            lift=float(r["lift_vs_base"])
            if "lift_vs_base" in r and not pd.isna(r["lift_vs_base"])
            else None,
        )
    return points


def load_base_rate(path: Path, split: str) -> BaseRateInfo | None:
    df = read_csv(path)
    if df is None:
        return None
    row = df[(df["split"].astype(str).str.lower() == split.lower()) & (df["slice"] == "overall")]
    if row.empty:
        return None
    info = row.iloc[0]
    return BaseRateInfo(rate=float(info["base_rate"]), total_candidates=int(info["total_rows"]))


def load_dcs_top5_point(base_info: BaseRateInfo | None) -> OperatingPoint | None:
    if base_info is None or not DCS_PARQUET_PATH.exists():
        return None
    try:
        df = pd.read_parquet(DCS_PARQUET_PATH)
    except Exception as exc:
        print(f"ERROR: failed to read {DCS_PARQUET_PATH}: {exc}")
        return None
    if df.empty or "label" not in df.columns:
        return None

    total_candidates = base_info.total_candidates
    if total_candidates <= 0:
        return None

    selected = len(df)
    positives = float(df["label"].sum())
    precision = positives / selected if selected else 0.0

    total_positives = base_info.rate * total_candidates
    recall = positives / total_positives if total_positives > 0 else 0.0
    yield_fraction = selected / total_candidates
    lift = precision / base_info.rate if base_info.rate > 0 else None

    return OperatingPoint(
        label="DCS Top-5 coverage",
        precision=precision,
        recall=recall,
        yield_fraction=yield_fraction,
        predicted_edges=selected,
        lift=lift,
    )


def format_small_percent(value: float) -> str:
    if value == 0:
        return "0"
    if value < 0.1:
        return f"{value:.3g}"
    if value < 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.1f}"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def lighten_color(color: str, factor: float) -> str:
    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(rgb + (1 - rgb) * factor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precision–yield frontier plot")
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Dataset split to visualise (default: test)",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=ANALYSIS_DIR,
        help="Directory containing analysis tables (default: v4 analysis tables)",
    )
    parser.add_argument(
        "--pr-curve-path",
        type=Path,
        default=None,
        help="Optional override for pr_curve_<split>.csv",
    )
    parser.add_argument(
        "--topk-path",
        type=Path,
        default=None,
        help="Optional override for topk_grid_<split>.csv",
    )
    parser.add_argument(
        "--operating-ladder-path",
        type=Path,
        default=None,
        help="Optional override for operating_ladder.csv",
    )
    parser.add_argument(
        "--base-rates-path",
        type=Path,
        default=None,
        help="Optional override for base_rates.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Directory for figure output",
    )
    parser.add_argument(
        "--output-basename",
        type=str,
        default=None,
        help="Optional custom figure stem (without extension)",
    )
    parser.add_argument(
        "--max-pr-points",
        type=int,
        default=4000,
        help="Max PR curve samples to retain (reservoir down-sampling)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "basic", "axes-only"],
        default="basic",
        help="Figure detail level: axes-only, global τ sweep only (basic), or full annotations",
    )
    parser.add_argument(
        "--yield-min-pct",
        type=float,
        default=0.01,
        help="Lower bound for yield axis in percent (default: 0.01)",
    )
    parser.add_argument(
        "--yield-max-pct",
        type=float,
        default=5.0,
        help="Upper bound for yield axis in percent (default: 5.0)",
    )
    parser.add_argument(
        "--precision-max",
        type=float,
        default=None,
        help="Optional upper bound for precision axis in percent (default: auto)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive placement of annotations (drag to adjust offsets)",
    )
    return parser.parse_args()


def determine_yield_bounds(
    curve: pd.DataFrame,
    grid: pd.DataFrame,
    yield_min_pct: float | None,
    yield_max_pct: float,
) -> tuple[float, float, float]:
    positive = []
    for series in (curve["yield"], grid.get("yield", pd.Series(dtype=float))):
        series = pd.to_numeric(series, errors="coerce")
        positive.append(series[series > 0])

    combined = pd.concat(positive, ignore_index=True) if positive else pd.Series(dtype=float)
    min_positive_pct = float(combined.min() * 100) if not combined.empty else 1e-4
    if not math.isfinite(min_positive_pct) or min_positive_pct <= 0:
        min_positive_pct = 1e-4

    if yield_min_pct is not None:
        lower_pct = max(yield_min_pct, 1e-4)
    else:
        lower_pct = max(min_positive_pct, 1e-4)

    upper_pct = max(yield_max_pct, lower_pct * 10)
    focus_max_fraction = upper_pct / 100
    return lower_pct, upper_pct, focus_max_fraction


def make_plot(
    curve: pd.DataFrame,
    grid: pd.DataFrame,
    points: dict[str, OperatingPoint],
    base_info: BaseRateInfo | None,
    split: str,
    mode: str,
    yield_min_pct: float | None,
    yield_max_pct: float,
    precision_max: float | None,
    interactive: bool,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7.0, 5.8))

    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#4d4d4d")
        spine.set_linewidth(0.8)

    lower_pct, _upper_pct, focus_max_fraction = determine_yield_bounds(
        curve, grid, yield_min_pct, yield_max_pct
    )

    tau_line_color = "#2f4b7c"

    curve_sorted = curve.sort_values("yield").drop_duplicates(subset=["yield"], keep="first")
    curve_sorted.iloc[0] if not curve_sorted.empty else None
    grid_sorted = grid.sort_values("yield") if "yield" in grid.columns else grid.copy()

    curve_plot = curve_sorted.copy()
    if mode != "axes-only":
        curve_plot = curve_plot[curve_plot["yield"] > 0]
        curve_plot = curve_plot[curve_plot["yield"] <= focus_max_fraction]
        if curve_plot.empty:
            curve_plot = curve_sorted[curve_sorted["yield"] > 0]
        if curve_plot.empty:
            curve_plot = curve_sorted.copy()
        monotone_precision = np.minimum.accumulate(curve_plot["precision"].to_numpy())
        curve_plot = curve_plot.assign(precision_monotone=monotone_precision)

    filtered_grid = grid_sorted[
        grid_sorted["K"].isin(TOPK_K_KEEP) & grid_sorted["floor"].isin(TOPK_FLOOR_KEEP)
    ].copy()
    if not filtered_grid.empty:
        grid_sorted = filtered_grid
    grid_plot = grid_sorted.copy()
    if mode == "full":
        grid_plot = grid_plot[grid_plot["yield"] > 0]
        grid_plot = grid_plot[grid_plot["yield"] <= focus_max_fraction]
        if grid_plot.empty:
            grid_plot = grid_sorted.copy()

    curve_yield_frac = curve_plot["yield"].to_numpy()
    (grid_plot["yield"].to_numpy() if (mode == "full" and not grid_plot.empty) else np.array([]))
    y_curve = np.array([])

    min_positive_pct = max(lower_pct, 1e-4)

    if mode != "axes-only":
        x_curve_pct = np.clip(curve_yield_frac * 100, min_positive_pct, None)
        y_curve = curve_plot.get("precision_monotone", curve_plot["precision"]).to_numpy() * 100
        ax.plot(
            x_curve_pct,
            y_curve,
            color=tau_line_color,
            linewidth=1.6,
            label="Global τ sweep" if mode != "axes-only" else None,
            zorder=2,
        )

    if mode == "full" and not grid_plot.empty:
        unique_k = sorted([k for k in grid_plot["K"].unique().tolist() if not pd.isna(k)])
        if not unique_k:
            unique_k = [1]
        size_min, size_max = 36, 115
        if len(unique_k) <= 1:
            size_map = {unique_k[0]: (size_min + size_max) / 2}
        else:
            size_map = {
                k: size_min + (size_max - size_min) * idx / (len(unique_k) - 1)
                for idx, k in enumerate(unique_k)
            }

        x_grid_pct = np.clip(grid_plot["yield"].to_numpy() * 100, min_positive_pct, None)
        y_grid = grid_plot["precision"].to_numpy() * 100
        sizes = grid_plot["K"].map(size_map).to_numpy()

        colors: list[str] = []
        for _, row in grid_plot.iterrows():
            k = int(row["K"]) if not pd.isna(row["K"]) else None
            base = K_COLOR_MAP.get(k, "#636363")
            floor = float(row["floor"])
            colors.append(
                lighten_color(base, 0.4)
                if math.isclose(floor, 0.99, rel_tol=0, abs_tol=1e-6)
                else base
            )
        color_array = np.array(colors)

        ax.scatter(
            x_grid_pct,
            y_grid,
            s=sizes,
            c=color_array,
            alpha=0.88,
            linewidth=0.4,
            edgecolors="#fdfdfd",
            zorder=4,
            label="Top-K+floor grid",
        )

    base_rate_pct = None
    if base_info is not None and mode != "axes-only":
        base_rate_pct = base_info.rate * 100
        ax.axhline(
            base_rate_pct,
            color="#9a9a9a",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

    candidate_total = base_info.total_candidates if base_info is not None else None

    display_min_pct = max(min_positive_pct, lower_pct)

    points = points.copy()

    def add_topk_point(key: str, label: str, k_value: int, floor_value: float) -> None:
        matches = grid[(grid["K"] == k_value) & (np.isclose(grid["floor"], floor_value))]
        if matches.empty:
            return
        row = matches.iloc[0]
        lift = None
        if base_info is not None and base_info.rate > 0:
            lift = float(row["precision"]) / base_info.rate
        points[key] = OperatingPoint(
            label=label,
            precision=float(row["precision"]),
            recall=float(row.get("recall", 0.0) or 0.0),
            yield_fraction=float(row["yield"]),
            predicted_edges=int(row["predicted_edges"])
            if not pd.isna(row["predicted_edges"])
            else None,
            lift=lift,
        )

    add_topk_point("core_topk", "Core Top-K (K=5, floor=0.90)", 5, 0.90)
    add_topk_point("discovery_topk", "Discovery Top-K (K=15, floor=0.85)", 15, 0.85)

    annotations: list[plt.Annotation] = []
    points.get("core_topk")
    points.get("discovery_topk")
    dcs_point = points.get("dcs_top5")

    def format_policy_annotation(point: OperatingPoint, include_lift: bool = False) -> str:
        precision_pct = point.precision * 100
        recall_pct = point.recall * 100
        parts = [f"{precision_pct:.0f}% precision"]
        if recall_pct <= 0.05:
            parts.append("0% recall")
        else:
            parts.append(f"{recall_pct:.1f}% recall")
        if point.predicted_edges:
            parts.append(f"≈{point.predicted_edges:,.0f} edges")
        if include_lift and point.lift:
            lift_val = point.lift
            if lift_val >= 100:
                lift_str = f"≈{lift_val:.0f}×"
            elif lift_val >= 10:
                lift_str = f"≈{lift_val:.1f}×"
            else:
                lift_str = f"≈{lift_val:.2f}×"
            parts.append(f"Lift {lift_str}")
        return f"{point.label}\n" + " · ".join(parts)

    ax.set_xlabel("Yield (% of candidates accepted)", fontsize=11)
    ax.set_ylabel("Measured precision (% lower bound)", fontsize=11)

    x_min_pct = max(0.01, yield_min_pct or 0.01, min_positive_pct)
    x_max_pct = min(0.5, max(yield_max_pct, 0.5))
    ax.set_xscale("log")
    ax.set_xlim(x_min_pct, x_max_pct)

    major_xticks = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    major_xticks = major_xticks[(major_xticks >= x_min_pct) & (major_xticks <= x_max_pct)]
    if major_xticks.size:
        ax.set_xticks(major_xticks)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))

    def yield_formatter(value: float, _pos: int) -> str:
        if value <= 0:
            return "0%"
        if value <= 0.0105:
            return "0%"
        if value < 0.02:
            return f"{value:.3f}%"
        if value < 0.1:
            return f"{value:.2f}%"
        if value < 1:
            return f"{value:.1f}%"
        return f"{value:.0f}%"

    ax.xaxis.set_major_formatter(FuncFormatter(yield_formatter))

    y_candidates = []
    if y_curve.size:
        y_candidates.append(float(np.nanmax(y_curve)))
    if mode == "full" and grid_plot is not None and not grid_plot.empty:
        y_candidates.append(float(np.nanmax(grid_plot["precision"].to_numpy() * 100)))
    y_upper = 105.0
    ax.set_ylim(0, y_upper)
    ax.set_yticks(np.arange(0, y_upper, 10))
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    def pct_to_edges_per_million(pct_value: float) -> float:
        return pct_value * 10_000

    def edges_per_million_to_pct(edges_value: float) -> float:
        return edges_value / 10_000

    secax = ax.secondary_xaxis(
        "top", functions=(pct_to_edges_per_million, edges_per_million_to_pct)
    )
    secax.set_xscale("log")
    secax.set_xlim(ax.get_xlim())
    secax.set_xlabel("Predicted edges per million candidates", fontsize=11)
    if major_xticks.size:
        secax.set_xticks(pct_to_edges_per_million(major_xticks))
    secax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    secax.xaxis.set_minor_formatter(NullFormatter())

    def edges_formatter(value: float, _pos: int) -> str:
        if value <= 0:
            return "0"
        if value >= 100_000:
            return f"{value / 1000:.0f}k"
        if value >= 10_000:
            return f"{value / 1000:.1f}k"
        if value >= 1_000:
            return f"{value / 1000:.1f}k"
        if value >= 1:
            return f"{value:.0f}"
        return f"{value:.1f}"

    secax.xaxis.set_major_formatter(FuncFormatter(edges_formatter))
    secax.tick_params(axis="x", labelsize=9, pad=4)

    ax.grid(which="major", color="#e3e3e3", linewidth=0.6, alpha=0.9)
    ax.grid(which="minor", color="#f2f2f2", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    "Validation" if split.lower() == "val" else "Test"

    legend_handles: list[Line2D] = []
    if mode != "axes-only":
        legend_handles.append(
            Line2D([], [], color=tau_line_color, linewidth=1.6, label="Global τ sweep")
        )
    if base_info is not None and base_rate_pct is not None and mode != "axes-only":
        legend_handles.append(
            Line2D(
                [],
                [],
                color="#9a9a9a",
                linestyle="--",
                linewidth=0.8,
                label=f"Base rate ({base_rate_pct:.2f}%)",
            )
        )

    if mode == "basic":
        basic_specs = [
            {
                "key": "tau_precision_0.95",
                "offset": (42, -13),
                "ha": "left",
                "va": "center",
                "connection": "angle3,angleA=0,angleB=-32",
                "facecolor": "#264b96",
                "edgecolor": "#ffffff",
                "size": 78,
                "zorder": 5.6,
                "alpha": 1.0,
                "bbox_fc": "#ffffff",
                "bbox_ec": "#4d4d4d",
                "arrow_color": "#264b96",
                "text_mode": "summary",
                "include_lift": True,
            },
            {
                "key": "tau_precision_0.85",
                "offset": (44, -3),
                "ha": "left",
                "va": "center",
                "connection": "angle3,angleA=0,angleB=-24",
                "facecolor": "#6f4c9b",
                "edgecolor": "#ffffff",
                "size": 74,
                "zorder": 5.4,
                "alpha": 1.0,
                "bbox_fc": "#ffffff",
                "bbox_ec": "#4d4d4d",
                "arrow_color": "#6f4c9b",
                "text_mode": "summary",
                "include_lift": True,
            },
        ]
        if dcs_point is not None:
            basic_specs.append(
                {
                    "key": "dcs_top5",
                    "offset": (115, -35),
                    "ha": "left",
                    "va": "center",
                    "connection": "angle3,angleA=0,angleB=-18",
                    "facecolor": "#4b6272",
                    "edgecolor": "#ffffff",
                    "size": 70,
                    "zorder": 5.3,
                    "alpha": 1.0,
                    "bbox_fc": "#ffffff",
                    "bbox_ec": "#4d4d4d",
                    "arrow_color": "#4b6272",
                    "text_mode": "summary",
                    "include_lift": True,
                }
            )
        {spec["key"]: spec for spec in basic_specs}
        for spec in basic_specs:
            key = spec["key"]
            point = points.get(key)
            if point is None:
                continue

            x_pt = max(point.yield_fraction * 100, display_min_pct)
            y_pt = point.precision * 100
            ax.scatter(
                [x_pt],
                [y_pt],
                s=spec.get("size", 68),
                facecolors=spec.get("facecolor", tau_line_color),
                edgecolors=spec.get("edgecolor", "white"),
                linewidths=0.9,
                alpha=spec.get("alpha", 1.0),
                zorder=spec.get("zorder", 5),
            )

            text_mode = spec.get("text_mode", "default")
            if text_mode == "summary":
                text = format_policy_annotation(point, include_lift=spec.get("include_lift", False))
            elif text_mode == "custom":
                text = spec["custom_text"](point)
            else:
                text = f"{point.label}\n{point.callout_metrics(candidate_total)}"
            ann = ax.annotate(
                text,
                xy=(x_pt, y_pt),
                xytext=spec.get("offset", (32, 12)),
                textcoords="offset points",
                fontsize=8.0,
                ha=spec.get("ha", "left"),
                va=spec.get("va", "center"),
                bbox={
                    "boxstyle": "round,pad=0.26",
                    "fc": spec.get("bbox_fc", "#ffffff"),
                    "ec": spec.get("bbox_ec", "#4d4d4d"),
                    "linewidth": 0.55,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": spec.get("arrow_color", "#4d4d4d"),
                    "linewidth": 0.55,
                    "connectionstyle": spec.get("connection", "angle3,angleA=0,angleB=-30"),
                },
                annotation_clip=False,
            )
            annotations.append(ann)

    if mode == "full":
        for key in ANNOTATION_KEYS:
            point = points.get(key)
            if not point:
                continue
            x = point.yield_fraction * 100
            if x <= 0:
                x = min_positive_pct
            y = point.precision * 100

            if key == "K=10_floor=0.995_cap=50":
                marker_face = K_COLOR_MAP.get(10, "#636363")
                marker_edge = "white"
                size = 90
            elif key == "tau=1":
                marker_face = tau_line_color
                marker_edge = "white"
                size = 80
            else:
                marker_face = tau_line_color
                marker_edge = "white"
                size = 68

            ax.scatter(
                [x],
                [y],
                s=size,
                facecolors=marker_face,
                edgecolors=marker_edge,
                linewidths=0.9,
                zorder=6,
            )

            text = f"{point.label}\n{point.callout_metrics(candidate_total)}"
            text_xy = CALL_OUT_AX_POSITIONS.get(key)
            ha, va = CALL_OUT_ALIGNMENT.get(key, ("left", "top"))
            if text_xy is not None:
                ann = ax.annotate(
                    text,
                    xy=(x, y),
                    xycoords="data",
                    xytext=text_xy,
                    textcoords="axes fraction",
                    fontsize=8,
                    ha=ha,
                    va=va,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#505050",
                        "linewidth": 0.5,
                        "connectionstyle": CALL_OUT_CONNECTIONS.get(key, "arc3,rad=0"),
                    },
                    annotation_clip=False,
                )
                annotations.append(ann)
            else:
                ann = ax.annotate(
                    text,
                    xy=(x, y),
                    xytext=(8, -26),
                    textcoords="offset points",
                    fontsize=8,
                    ha=ha,
                    va=va,
                    arrowprops={"arrowstyle": "-", "color": "#505050", "linewidth": 0.5},
                    annotation_clip=False,
                )
                annotations.append(ann)

    if mode == "basic":
        for spec in basic_specs:
            point = points.get(spec["key"])
            if not point:
                continue
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=spec.get("facecolor", tau_line_color),
                    markeredgecolor=spec.get("edgecolor", "white"),
                    markersize=7,
                    label=point.label,
                )
            )
    if legend_handles:
        legend = ax.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="upper right",
            frameon=False,
            fontsize=8.5,
            handlelength=1.4,
            handletextpad=0.6,
            borderpad=0.35,
            labelspacing=0.5,
        )
        if legend:
            legend.set_title(None)

    if interactive and annotations:
        for ann in annotations:
            ann.draggable(True)

        def _on_release(event):
            if event.inaxes is not ax:
                return
            for ann in annotations:
                contains, _ = ann.contains(event)
                if contains:
                    x_off, y_off = ann.xyann
                    label = ann.get_text().splitlines()[0]
                    print(
                        f"{label}: offset=({x_off:.1f}, {y_off:.1f}) ha={ann.get_ha()} va={ann.get_va()}"
                    )
                    break

        fig.canvas.mpl_connect("button_release_event", _on_release)

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.94)
    return fig, ax


def main() -> None:
    args = parse_args()
    split = args.split.lower()
    analysis_dir = args.analysis_dir

    pr_curve_path = args.pr_curve_path or analysis_dir / f"pr_curve_{split}.csv"
    topk_path = args.topk_path or analysis_dir / f"topk_grid_{split}.csv"
    operating_path = args.operating_ladder_path or analysis_dir / "operating_ladder.csv"
    base_rates_path = args.base_rates_path or analysis_dir / "base_rates.csv"

    curve = load_threshold_curve(pr_curve_path, max_points=args.max_pr_points)
    grid = load_topk_grid(topk_path, split)
    points = load_operating_points(operating_path, split)
    base_info = load_base_rate(base_rates_path, split)

    if args.mode in {"basic", "full"}:
        dcs_point_global = load_dcs_top5_point(base_info)
        if dcs_point_global is not None:
            dcs_point_global.label = "DCS (Top-K=5 within FHPE)"
            points["dcs_top5"] = dcs_point_global

    if curve is None or grid is None:
        print("Aborting: required data missing.")
        return

    if grid is not None and points is not None:

        def add_topk_to_points(key: str, label: str, k_value: int, floor_value: float) -> None:
            matches = grid[(grid["K"] == k_value) & (np.isclose(grid["floor"], floor_value))]
            if matches.empty:
                return
            row = matches.iloc[0]
            lift = None
            if base_info is not None and base_info.rate > 0:
                lift = float(row["precision"]) / base_info.rate
            points[key] = OperatingPoint(
                label=label,
                precision=float(row["precision"]),
                recall=float(row.get("recall", 0.0) or 0.0),
                yield_fraction=float(row["yield"]),
                predicted_edges=int(row["predicted_edges"])
                if not pd.isna(row["predicted_edges"])
                else None,
                lift=lift,
            )

        add_topk_to_points("core_topk", "Core Top-K (K=5, floor=0.90)", 5, 0.90)
        add_topk_to_points("discovery_topk", "Discovery Top-K (K=15, floor=0.85)", 15, 0.85)

    output_dir = args.output_dir
    ensure_output_dir(output_dir)
    fig, _ = make_plot(
        curve,
        grid,
        points,
        base_info,
        split=split,
        mode=args.mode,
        yield_min_pct=args.yield_min_pct,
        yield_max_pct=args.yield_max_pct,
        precision_max=args.precision_max,
        interactive=args.interactive,
    )

    if args.interactive:
        print(
            "\nDrag the annotation boxes to desired positions. Offsets will be printed in the console when you release the mouse."
        )
        plt.show()
        return

    basename = args.output_basename or OUTPUT_BASENAME_TEMPLATE.format(split=split)
    pdf_path = output_dir / f"{basename}.pdf"
    png_path = output_dir / f"{basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    split_label = "Validation" if split == "val" else "Test"
    if args.mode == "axes-only":
        annotated_labels: list[str] = []
    elif args.mode == "basic":
        annotated_labels = []
        for key in ("tau_precision_0.95", "tau_precision_0.85", "dcs_top5"):
            point = points.get(key)
            if point:
                annotated_labels.append(point.label)
    else:
        annotated_labels = [points[k].label for k in ANNOTATION_KEYS if k in points]

    summary_lines = [
        f"Precision–yield frontier — {split_label}",
        f"Mode: {args.mode}",
        f"Global τ points: {len(curve)}",
        f"Top-K grid points (cap=50): {len(grid)}",
        f"Annotated ops: {', '.join(annotated_labels) if annotated_labels else 'None'}",
        f"Outputs: {pdf_path}, {png_path}",
    ]
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
