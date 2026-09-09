#!/usr/bin/env python3
"""Export Figure 4.5: abstract corridor vs upstream schematic (non-empirical)."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

DEFAULT_CONFIG = "src/analysis/chapter4/config/ch4_v2_fix01.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Figure 4.5 schematic corridor/upstream diagram"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config YAML path")
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


def sample_positions(
    rng: np.random.Generator,
    n: int,
    x_lo: float,
    x_hi: float,
    y_lo: float = 0.10,
    y_hi: float = 0.90,
) -> np.ndarray:
    x = rng.uniform(x_lo, x_hi, size=n)
    y = rng.uniform(y_lo, y_hi, size=n)
    return np.stack([x, y], axis=1)


def build_edges(
    rng: np.random.Generator,
    upstream: list[str],
    semis: list[str],
    corridor: list[str],
    primes: list[str],
    dod: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    edges: list[tuple[str, str]] = []

    # Upstream -> Semis
    for u in upstream:
        k = int(rng.integers(1, 3))
        for s in rng.choice(semis, size=k, replace=False).tolist():
            edges.append((u, s))

    # Semis -> Corridor
    for s in semis:
        k = int(rng.integers(3, 6))
        for c in rng.choice(corridor, size=min(k, len(corridor)), replace=False).tolist():
            edges.append((s, c))

    # Some direct semi -> prime bypasses (rare)
    for s in semis:
        if float(rng.random()) < 0.40:
            p = str(rng.choice(primes))
            edges.append((s, p))

    # Corridor -> Prime
    for c in corridor:
        k = int(rng.integers(2, 4))
        for p in rng.choice(primes, size=min(k, len(primes)), replace=False).tolist():
            edges.append((c, p))

    # Prime -> DoD endpoint
    for p in primes:
        edges.append((p, dod))

    # De-duplicate preserving first order.
    dedup: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for e in edges:
        if e in dedup:
            continue
        dedup.add(e)
        ordered.append(e)

    # Highlight two canonical dependency routes for readability.
    highlight = [
        (upstream[0], semis[0]),
        (semis[0], corridor[0]),
        (corridor[0], primes[0]),
        (primes[0], dod),
        (upstream[1], semis[1]),
        (semis[1], corridor[2]),
        (corridor[2], primes[2]),
        (primes[2], dod),
    ]
    # Ensure highlight edges exist in context set.
    for e in highlight:
        if e not in dedup:
            ordered.append(e)
            dedup.add(e)
    return ordered, highlight


def draw_arrow(
    ax: plt.Axes,
    xy1: tuple[float, float],
    xy2: tuple[float, float],
    color: str,
    lw: float,
    alpha: float,
    z: float,
) -> None:
    patch = FancyArrowPatch(
        posA=xy1,
        posB=xy2,
        arrowstyle="-|>",
        mutation_scale=8,
        color=color,
        lw=lw,
        alpha=alpha,
        shrinkA=4,
        shrinkB=4,
        connectionstyle="arc3,rad=0.03",
        zorder=z,
    )
    ax.add_patch(patch)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    figs_root = Path(str(cfg.get("paths", {}).get("figs_root", "figs/chapter4/v2_fix01")))
    out_dir = Path(args.out_dir) if args.out_dir else (figs_root / "main_text")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)

    n_upstream = 16
    n_semis = 6
    n_corridor = 14
    n_primes = 7
    dod_uid = "dod_anchor"

    upstream_ids = [f"u{i + 1}" for i in range(n_upstream)]
    semi_ids = [f"s{i + 1}" for i in range(n_semis)]
    corridor_ids = [f"c{i + 1}" for i in range(n_corridor)]
    prime_ids = [f"p{i + 1}" for i in range(n_primes)]

    pos: dict[str, tuple[float, float]] = {}
    for uid, (x, y) in zip(
        upstream_ids, sample_positions(rng, n_upstream, 0.07, 0.29), strict=False
    ):
        pos[uid] = (float(x), float(y))
    for uid, (x, y) in zip(semi_ids, sample_positions(rng, n_semis, 0.33, 0.43), strict=False):
        pos[uid] = (float(x), float(y))
    for uid, (x, y) in zip(
        corridor_ids, sample_positions(rng, n_corridor, 0.47, 0.68), strict=False
    ):
        pos[uid] = (float(x), float(y))
    for uid, (x, y) in zip(prime_ids, sample_positions(rng, n_primes, 0.73, 0.87), strict=False):
        pos[uid] = (float(x), float(y))
    pos[dod_uid] = (0.94, 0.50)

    role: dict[str, str] = {}
    for uid in upstream_ids:
        role[uid] = "Upstream-of-Semi"
    for uid in semi_ids:
        role[uid] = "Strict Semi"
    for uid in corridor_ids:
        role[uid] = "Corridor Intermediary"
    for uid in prime_ids:
        role[uid] = "Prime Endpoint"
    role[dod_uid] = "DoD Endpoint"

    edges, highlight = build_edges(
        rng=rng,
        upstream=upstream_ids,
        semis=semi_ids,
        corridor=corridor_ids,
        primes=prime_ids,
        dod=dod_uid,
    )
    prime_set = set(prime_ids)
    prime_link_count = dict.fromkeys(corridor_ids, 0)
    for u, v in edges:
        if u in prime_link_count and v in prime_set:
            prime_link_count[u] += 1
    if prime_link_count:
        counts = np.array(list(prime_link_count.values()), dtype=float)
        cutoff = float(np.quantile(counts, 0.75))
        prime_adjacent_corridor_ids = [
            uid for uid, count in prime_link_count.items() if float(count) >= cutoff and count > 0
        ]
    else:
        prime_adjacent_corridor_ids = []
    if len(prime_adjacent_corridor_ids) == 0:
        prime_adjacent_corridor_ids = sorted(
            corridor_ids, key=lambda uid: pos[uid][0], reverse=True
        )[: max(1, n_corridor // 3)]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=False)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Region frame + dashed separators (no fill color).
    frame = Rectangle(
        (0.03, 0.04), 0.95, 0.92, facecolor="none", edgecolor="#8c8c8c", linewidth=0.9
    )
    ax.add_patch(frame)
    for x_sep in [0.33, 0.71]:
        ax.plot([x_sep, x_sep], [0.04, 0.96], color="#7a7a7a", lw=1.0, ls=(0, (4, 3)), zorder=2)

    ax.text(
        0.18, 0.978, "Upstream-of-Semis", ha="center", va="bottom", fontsize=10, color="#2f2f2f"
    )
    ax.text(
        0.52,
        0.978,
        "Semiconductor→Prime Corridor",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#2f2f2f",
    )
    ax.text(
        0.845,
        0.978,
        "Prime & DoD Endpoints",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#2f2f2f",
    )
    direction = FancyArrowPatch(
        posA=(0.08, 0.02),
        posB=(0.95, 0.02),
        arrowstyle="<|-|>",
        mutation_scale=9,
        color="#606060",
        lw=0.9,
        alpha=0.95,
        transform=ax.transAxes,
        clip_on=False,
        zorder=8,
    )
    ax.add_patch(direction)
    ax.text(
        0.08,
        0.005,
        "Upstream",
        ha="left",
        va="top",
        fontsize=8.2,
        color="#505050",
        transform=ax.transAxes,
    )
    ax.text(
        0.95,
        0.005,
        "Downstream",
        ha="right",
        va="top",
        fontsize=8.2,
        color="#505050",
        transform=ax.transAxes,
    )

    # Draw context edges first.
    hi_set = set(highlight)
    for u, v in edges:
        if (u, v) in hi_set:
            continue
        draw_arrow(ax, pos[u], pos[v], color="#8d8d8d", lw=0.55, alpha=0.22, z=1.0)

    # Draw highlighted routes.
    for u, v in highlight:
        draw_arrow(ax, pos[u], pos[v], color="#333333", lw=1.4, alpha=0.90, z=3.0)

    # Node styles.
    def draw_nodes(
        ids: list[str], marker: str, color: str, size: float, edge: str, lw: float, z: float
    ) -> None:
        x = [pos[i][0] for i in ids]
        y = [pos[i][1] for i in ids]
        ax.scatter(x, y, s=size, marker=marker, c=color, edgecolors=edge, linewidths=lw, zorder=z)

    draw_nodes(upstream_ids, "o", "#d7d7d7", 30, "#666666", 0.45, 4.0)
    draw_nodes(semi_ids, "^", "#59A14F", 64, "#3f3f3f", 0.55, 5.0)
    draw_nodes(corridor_ids, "o", "#9C755F", 44, "#4c4c4c", 0.55, 5.0)
    draw_nodes(prime_adjacent_corridor_ids, "o", "none", 86, "#6B3A2A", 1.0, 5.4)
    draw_nodes(prime_ids, "o", "#4E79A7", 62, "#2b3f56", 0.65, 6.0)
    draw_nodes([dod_uid], "D", "#8B1E3F", 82, "#2a0f1a", 0.90, 7.0)

    legend_node_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d7d7d7",
            markeredgecolor="#666666",
            markersize=6,
            label="Upstream enabler",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#59A14F",
            markeredgecolor="#3f3f3f",
            markersize=7,
            label="Strict semiconductor source",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9C755F",
            markeredgecolor="#4c4c4c",
            markersize=6,
            label="Corridor intermediary",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#6B3A2A",
            markeredgewidth=1.0,
            markersize=8,
            label="Prime-adjacent corridor",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#4E79A7",
            markeredgecolor="#2b3f56",
            markersize=7,
            label="Prime endpoint",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#8B1E3F",
            markeredgecolor="#2a0f1a",
            markersize=7,
            label="DoD endpoint",
        ),
    ]
    legend_nodes = ax.legend(
        handles=legend_node_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.135),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        columnspacing=1.15,
        handletextpad=0.45,
    )
    ax.add_artist(legend_nodes)
    legend_route = Line2D(
        [0, 1], [0, 0], color="#333333", lw=1.4, label="Illustrative dependency route"
    )
    ax.legend(
        handles=[legend_route],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.215),
        ncol=1,
        frameon=False,
        fontsize=7.6,
        handletextpad=0.55,
    )

    fig.subplots_adjust(left=0.02, right=0.995, top=0.975, bottom=0.245)

    out_pdf = out_dir / "fig_4_5_schematic_corridor_vs_upstream.pdf"
    out_svg = out_dir / "fig_4_5_schematic_corridor_vs_upstream.svg"
    out_png = out_dir / "fig_4_5_schematic_corridor_vs_upstream.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, format="png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    nodes_rows = [
        {"node_id": uid, "role": role[uid], "x": pos[uid][0], "y": pos[uid][1]}
        for uid in sorted(pos.keys())
    ]
    edges_rows = [
        {"src_id": u, "dst_id": v, "is_highlight": int((u, v) in hi_set)} for (u, v) in edges
    ]
    nodes_csv = out_dir / "fig_4_5_schematic_corridor_vs_upstream_nodes.csv"
    edges_csv = out_dir / "fig_4_5_schematic_corridor_vs_upstream_edges.csv"
    pd.DataFrame(nodes_rows).to_csv(nodes_csv, index=False)
    pd.DataFrame(edges_rows).to_csv(edges_csv, index=False)

    spec = {
        "figure_name": "fig_4_5_schematic_corridor_vs_upstream",
        "type": "abstract_schematic",
        "counts": {
            "n_upstream": n_upstream,
            "n_semis": n_semis,
            "n_corridor": n_corridor,
            "n_prime_adjacent_corridor": len(prime_adjacent_corridor_ids),
            "n_primes": n_primes,
            "n_dod": 1,
            "n_edges": len(edges),
            "n_highlight_edges": len(highlight),
        },
        "output_files": {
            "pdf": str(out_pdf),
            "svg": str(out_svg),
            "png": str(out_png),
            "nodes_csv": str(nodes_csv),
            "edges_csv": str(edges_csv),
        },
    }
    spec_json = out_dir / "fig_4_5_schematic_corridor_vs_upstream_spec.json"
    spec_json.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    run_meta = {
        "module": "export_fig_4_5_schematic_corridor_vs_upstream",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "inputs": {"config_sha256": file_sha256(cfg_path)},
        "outputs": {
            "pdf": str(out_pdf),
            "svg": str(out_svg),
            "png": str(out_png),
            "nodes_csv": str(nodes_csv),
            "edges_csv": str(edges_csv),
            "spec_json": str(spec_json),
        },
    }
    run_meta_json = out_dir / "fig_4_5_schematic_corridor_vs_upstream_run_metadata.json"
    run_meta_json.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"[done] wrote {out_pdf}")
    print(f"[done] wrote {out_svg}")
    print(f"[done] wrote {out_png}")
    print(f"[done] wrote {nodes_csv}")
    print(f"[done] wrote {edges_csv}")
    print(f"[done] wrote {spec_json}")
    print(f"[done] wrote {run_meta_json}")


if __name__ == "__main__":
    main()
