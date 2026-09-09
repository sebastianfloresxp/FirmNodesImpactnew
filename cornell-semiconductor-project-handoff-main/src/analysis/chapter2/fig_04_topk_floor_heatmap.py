#!/usr/bin/env python3
"""Single heatmap showing Top-K + floor trade-offs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


DEFAULT_GRID_DIR = Path("results/ensemble/meta_ranker/meta_ranker_v4/analysis_tables")
DEFAULT_SPLIT = "test"
DEFAULT_K_VALUES: Iterable[int] = (20, 15, 10, 5, 3, 2, 1)
DEFAULT_FLOORS: Iterable[float] = (
    0.980,
    0.950,
    0.900,
    0.850,
    0.800,
)


@dataclass(frozen=True)
class HighlightSpec:
    k: int
    floor: float


def parse_highlight(value: str) -> HighlightSpec:
    try:
        k_str, floor_str = value.split(":")
        return HighlightSpec(k=int(k_str), floor=float(floor_str))
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise argparse.ArgumentTypeError("Highlight specification must look like '5:0.90'") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a single heatmap of Top-K + floor yield/recall trade-offs.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["val", "test"],
        help="Dataset split to visualise (default: test)",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=None,
        help="Optional override for topk_grid_<split>.csv",
    )
    parser.add_argument(
        "--floors",
        type=float,
        nargs="*",
        default=None,
        help="Floors to include (fractions). Defaults to entire sweep.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="*",
        default=None,
        help="Per-supplier caps to include. Defaults to all present in the grid.",
    )
    parser.add_argument(
        "--highlight",
        type=parse_highlight,
        action="append",
        default=[],
        help="Policy cells to emphasise, e.g. 5:0.90",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figs/chapter2"),
        help="Where to write the figure",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional filename stem override",
    )
    parser.add_argument(
        "--value-min",
        type=float,
        default=None,
        help="Optional manual lower bound for colour scale (after percent conversion)",
    )
    parser.add_argument(
        "--value-max",
        type=float,
        default=None,
        help="Optional manual upper bound for colour scale (after percent conversion)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="Optional gamma for colour scaling (PowerNorm). Values <1 expand the low-recall region",
    )
    return parser.parse_args()


def _select(
    values: Sequence,
    requested: Sequence | None,
    *,
    key=float,
    defaults: Sequence | None = None,
) -> list:
    avail = sorted({key(v) for v in values}, reverse=True)
    if not requested:
        if defaults is None:
            return avail
        selected = [val for val in defaults if any(np.isclose(val, a, atol=5e-4) for a in avail)]
        if selected:
            return selected
        return avail
    selected: list = []
    missing: list = []
    for item in requested:
        match = next((v for v in avail if np.isclose(v, key(item), atol=5e-4)), None)
        if match is None:
            missing.append(item)
        else:
            selected.append(match)
    if missing:
        missing_str = ", ".join(str(m) for m in missing)
        print(f"[warn] requested values not in grid: {missing_str}")
    return selected


def load_grid(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"].str.lower() == split].copy()
    if df.empty:
        raise RuntimeError(f"No rows for split '{split}' in {path}")
    df = df[df["cap"].isna()].copy()  # focus on unconstrained caps
    df.drop_duplicates(subset=["K", "floor"], inplace=True)
    return df


def format_floor(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def render_heatmap(
    matrix: np.ndarray,
    precision_matrix: np.ndarray,
    recall_matrix: np.ndarray,
    yield_matrix_pct: np.ndarray,
    yield_per_million: np.ndarray,
    predicted_edges: np.ndarray,
    floors: list[float],
    k_values: list[int],
    *,
    highlights: list[HighlightSpec],
    output_dir: Path,
    basename: str,
    value_min: float | None,
    value_max: float | None,
    gamma: float | None,
) -> None:
    n_rows, n_cols = matrix.shape

    cmap = plt.get_cmap("viridis")
    auto_max = min(10.0, np.nanmax(matrix)) if np.isfinite(matrix).any() else 10.0
    if auto_max < 2.0:
        auto_max = 2.0
    vmin = 0.0 if value_min is None else value_min
    vmax = auto_max if value_max is None else value_max
    colorbar_label = "Recall (% of positives captured)"

    def formatter(v, _pos):
        return f"{v:.0f}%"

    if np.isclose(vmin, vmax):
        vmax = vmin + 0.1

    if value_min is not None and value_max is not None and value_min >= value_max:
        value_max = value_min + 0.1

    if gamma and gamma > 0:
        norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig_width = max(6.0, 1.1 * n_cols)
    fig_height = max(3.8, 0.75 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([format_floor(f) for f in floors], rotation=45, ha="right")
    ax.set_xlabel("Probability floor τ")

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([str(k) for k in k_values])
    ax.set_ylabel("Per-supplier cap K")

    ax.set_title("")

    for r_idx, _k in enumerate(k_values):
        for c_idx, _floor in enumerate(floors):
            value = matrix[r_idx, c_idx]
            if not np.isfinite(value):
                continue

            precision_matrix[r_idx, c_idx]
            recall_val = recall_matrix[r_idx, c_idx]
            yield_matrix_pct[r_idx, c_idx]
            yield_per_million[r_idx, c_idx]
            total_edges = predicted_edges[r_idx, c_idx]

            recall_str = f"R={recall_val:.1f}%"
            if total_edges >= 1_000_000:
                edges_str = f"≈{total_edges / 1_000_000:.1f}M edges"
            elif total_edges >= 1_000:
                edges_str = f"≈{total_edges / 1_000:.0f}k edges"
            else:
                edges_str = f"≈{int(total_edges)} edges"
            text = f"{recall_str}\n{edges_str}"

            normalized = norm(value)
            text_color = "white" if normalized < 0.35 else "#111111"
            ax.text(
                c_idx,
                r_idx,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    for spec in highlights:
        if spec.k not in k_values:
            continue
        floor_match = next((f for f in floors if np.isclose(f, spec.floor, atol=5e-4)), None)
        if floor_match is None:
            continue
        r_idx = k_values.index(spec.k)
        c_idx = floors.index(floor_match)
        rect = plt.Rectangle(
            (c_idx - 0.5, r_idx - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#f57c00",
            linewidth=2.0,
        )
        ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.04)
    cbar.ax.set_ylabel(colorbar_label, fontsize=10)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    cbar.ax.tick_params(labelsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{basename}.pdf"
    png_path = output_dir / f"{basename}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"[saved] {pdf_path}")
    print(f"[saved] {png_path}")


def main() -> None:
    args = parse_args()
    split = args.split.lower()

    grid_path = args.grid or DEFAULT_GRID_DIR / f"topk_grid_{split}.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"Top-K grid not found: {grid_path}")

    df = load_grid(grid_path, split)

    floors = _select(df["floor"], args.floors, defaults=DEFAULT_FLOORS)
    k_values = _select(df["K"], args.k_values, key=int, defaults=DEFAULT_K_VALUES)

    df = df[df["floor"].isin(floors) & df["K"].isin(k_values)].copy()
    if df.empty:
        raise RuntimeError("Filtered grid is empty")

    floors = sorted({float(f) for f in df["floor"].unique()}, reverse=True)
    k_values = sorted({int(k) for k in df["K"].unique()}, reverse=True)

    pivot_metric = df.pivot(index="K", columns="floor", values="recall")
    pivot_precision = df.pivot(index="K", columns="floor", values="precision")
    pivot_recall = df.pivot(index="K", columns="floor", values="recall") * 100.0
    pivot_yield = df.pivot(index="K", columns="floor", values="yield") * 100.0
    pivot_yield_per_million = df.pivot(index="K", columns="floor", values="yield") * 1_000_000.0
    pivot_edges = df.pivot(index="K", columns="floor", values="predicted_edges")

    pivot_metric = pivot_metric.reindex(index=k_values, columns=floors)
    pivot_precision = pivot_precision.reindex(index=k_values, columns=floors) * 100.0
    pivot_recall = pivot_recall.reindex(index=k_values, columns=floors)
    pivot_yield = pivot_yield.reindex(index=k_values, columns=floors)
    pivot_yield_per_million = pivot_yield_per_million.reindex(index=k_values, columns=floors)
    pivot_edges = pivot_edges.reindex(index=k_values, columns=floors)

    matrix = pivot_metric.to_numpy(dtype=float) * 100.0
    precision_matrix = pivot_precision.to_numpy(dtype=float)
    recall_matrix = pivot_recall.to_numpy(dtype=float)
    yield_matrix_pct = pivot_yield.to_numpy(dtype=float)
    yield_per_million = pivot_yield_per_million.to_numpy(dtype=float)
    predicted_edges = pivot_edges.to_numpy(dtype=float)

    basename = args.output_name or f"fig_04_topk_floor_heatmap_{args.metric}_{split}"

    render_heatmap(
        matrix,
        precision_matrix,
        recall_matrix,
        yield_matrix_pct,
        yield_per_million,
        predicted_edges,
        floors,
        k_values,
        highlights=args.highlight,
        output_dir=args.output_dir,
        basename=basename,
        value_min=args.value_min,
        value_max=args.value_max,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
