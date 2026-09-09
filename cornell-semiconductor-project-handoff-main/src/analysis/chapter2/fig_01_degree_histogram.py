#!/usr/bin/env python3
"""Disclosed-layer degree histogram for Chapter 2 (50-degree bins up to cap)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def load_degrees(edges_path: Path) -> tuple[np.ndarray, int]:
    if not edges_path.exists():
        raise FileNotFoundError(f"Edge list not found: {edges_path}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    query = (
        f"SELECT src_id, dst_id FROM read_parquet('{edges_path.as_posix()}') "
        "WHERE src_id IS NOT NULL AND dst_id IS NOT NULL AND src_id <> dst_id"
    )
    table = con.execute(query).fetch_arrow_table()
    con.close()

    src = np.array(table.column("src_id"))
    dst = np.array(table.column("dst_id"))

    stacked = np.stack((src, dst), axis=1)
    ordered = np.sort(stacked, axis=1)
    dtype = np.dtype([("u", ordered.dtype), ("v", ordered.dtype)])
    structured = ordered.view(dtype)
    unique = np.unique(structured)
    unique_pairs = unique.view(ordered.dtype).reshape(-1, 2)

    counter: Counter[int] = Counter()
    for u, v in unique_pairs:
        counter[int(u)] += 1
        counter[int(v)] += 1

    degrees = np.fromiter(counter.values(), dtype=np.int64)
    return degrees, unique_pairs.shape[0]


def render_histogram(
    degrees: np.ndarray, edge_count: int, output_dir: Path, width: int = 50, cap: int = 1000
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    capped = np.clip(degrees, 0, cap)
    bin_edges = np.arange(0, cap + width, width)
    counts, _ = np.histogram(capped, bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    tail_count = int((degrees > cap).sum())
    if tail_count:
        centers = np.append(centers, cap + width / 2)
        counts = np.append(counts, tail_count)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bar_widths = np.full(len(centers), width * 0.9, dtype=float)
    ax.bar(
        centers,
        counts,
        width=bar_widths,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.5,
        align="center",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Number of Edges")
    ax.set_ylabel("Number of Companies")

    xticks = np.array([0, 200, 400, 600, 800], dtype=float)
    xtick_labels = ["0", "200", "400", "600", "800"]
    if tail_count:
        xticks = np.append(xticks, cap + width / 2)
        xtick_labels.append(f">{cap}")
    else:
        xticks = np.append(xticks, 1000.0)
        xtick_labels.append("1000")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(0, cap + width)
    ax.set_ylim(bottom=1)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    base_ticks = np.array([0, 200, 400, 600, 800, min(cap, 1000)])
    ax.vlines(
        base_ticks,
        ymin=1,
        ymax=ax.get_ylim()[1],
        colors="grey",
        linestyles=":",
        linewidth=0.5,
        alpha=0.4,
    )

    def format_log_tick(value: float, _: int) -> str:
        if value <= 0:
            return ""
        thresholds = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
        labels = ["1", "10", "100", "1k", "10k", "100k", "1M"]
        for threshold, label in zip(thresholds[::-1], labels[::-1], strict=False):
            if value >= threshold:
                return label
        return f"{value:g}"

    ax.yaxis.set_major_formatter(FuncFormatter(format_log_tick))

    base = output_dir / "fig_01_degree_histogram"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)

    print("FIG_STATS", degrees.size, edge_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Degree histogram for Chapter 2")
    parser.add_argument(
        "--edges",
        default="data/processed/core/releases/core_v1/splits/all_edges.parquet",
        help="Parquet file containing disclosed FactSet supply chain edges",
    )
    parser.add_argument(
        "--output-dir",
        default="figs/chapter2",
        help="Directory to write figure outputs",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=50,
        help="Degree bin width (default 50)",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=1000,
        help="Upper bound for explicit bins before using open tail (default 1000)",
    )
    args = parser.parse_args()

    degrees, edge_count = load_degrees(Path(args.edges))
    if degrees.size == 0:
        raise RuntimeError("No degrees computed from the provided edge list")

    render_histogram(degrees, edge_count, Path(args.output_dir), width=args.bin_width, cap=args.cap)


if __name__ == "__main__":
    main()
