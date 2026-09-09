from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from src.ensemble.utils import ensure_dir, now_iso


def iter_calibration_stats(
    path: Path,
    prob_col: str,
    label_col: str,
    bins: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    pf = pq.ParquetFile(str(path))
    total = np.zeros(bins, dtype=np.float64)
    pos = np.zeros(bins, dtype=np.float64)
    prob_sum = np.zeros(bins, dtype=np.float64)
    global_count = 0
    brier_sum = 0.0
    logloss_sum = 0.0
    eps = 1e-12
    edges = np.linspace(0.0, 1.0, bins + 1)

    for batch in pf.iter_batches(columns=[prob_col, label_col], batch_size=batch_size):
        probs = batch.column(prob_col).to_numpy(zero_copy_only=False).astype(np.float64)
        labels = batch.column(label_col).to_numpy(zero_copy_only=False).astype(np.float64)
        if probs.size == 0:
            continue
        mask = np.isfinite(probs) & np.isfinite(labels)
        probs = probs[mask]
        labels = labels[mask]
        if probs.size == 0:
            continue
        global_count += int(probs.size)
        brier_sum += float(np.sum((probs - labels) ** 2))
        clipped = np.clip(probs, eps, 1.0 - eps)
        logloss_sum += float(
            -np.sum(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
        )
        bin_idx = np.clip(np.digitize(probs, edges, right=False) - 1, 0, bins - 1)
        totals = np.bincount(bin_idx, minlength=bins)
        pos_counts = np.bincount(bin_idx, weights=labels, minlength=bins)
        prob_sums = np.bincount(bin_idx, weights=probs, minlength=bins)
        total += totals
        pos += pos_counts
        prob_sum += prob_sums

    return total, pos, prob_sum, global_count, brier_sum, logloss_sum


def prepare_calibration_dataframe(
    total: np.ndarray,
    pos: np.ndarray,
    prob_sum: np.ndarray,
    global_count: int,
) -> list[dict[str, float]]:
    data: list[dict[str, float]] = []
    cumulative = 0
    for idx in range(len(total)):
        cnt = float(total[idx])
        if cnt <= 0:
            continue
        avg_pred = float(prob_sum[idx] / cnt)
        avg_true = float(pos[idx] / cnt)
        cumulative += cnt
        data.append(
            {
                "bin": idx,
                "count": cnt,
                "fraction": float(cnt / max(global_count, 1)),
                "predicted": avg_pred,
                "observed": avg_true,
                "cumulative_fraction": float(cumulative / max(global_count, 1)),
            }
        )
    return data


def plot_reliability_curve(
    calibration: list[dict[str, float]],
    title: str,
    out_path: Path,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=figsize)
    if not calibration:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
        ax.set_axis_off()
    else:
        preds = [row["predicted"] for row in calibration]
        obs = [row["observed"] for row in calibration]
        sizes = [max(row["fraction"], 1e-4) for row in calibration]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Perfect")
        ax.scatter(
            preds,
            obs,
            c="#1f77b4",
            s=np.array(sizes) * 2000.0,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.plot(preds, obs, color="#1f77b4", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Predicted probability", fontsize=12)
        ax.set_ylabel("Empirical frequency", fontsize=12)
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reliability plots from score parquet files"
    )
    parser.add_argument(
        "--scores", required=True, help="Path to parquet file with probabilities and labels"
    )
    parser.add_argument("--prob-column", default="meta_prob", help="Probability column name")
    parser.add_argument("--label-column", default="label", help="Label column name")
    parser.add_argument("--bins", type=int, default=20, help="Number of equal-width bins")
    parser.add_argument(
        "--batch-size", type=int, default=2_000_000, help="Rows per parquet batch when streaming"
    )
    parser.add_argument("--out-dir", default="results/ensemble/calibration")
    parser.add_argument("--name", default="meta_model", help="Name used in outputs")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figsize", default="6,6", help="Figure size in inches, e.g. 6,6")
    parser.add_argument(
        "--save-json", action="store_true", help="Persist calibration table as JSON"
    )
    parser.add_argument("--save-pdf", action="store_true", help="Also save PDF alongside PNG")

    args = parser.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores parquet not found: {scores_path}")

    figsize: tuple[float, float] = tuple(float(x.strip()) for x in args.figsize.split(","))  # type: ignore[assignment]

    totals, pos, prob_sum, count, brier_sum, logloss_sum = iter_calibration_stats(
        scores_path,
        args.prob_column,
        args.label_column,
        max(1, args.bins),
        max(100_000, args.batch_size),
    )

    calibration_rows = prepare_calibration_dataframe(totals, pos, prob_sum, count)
    metrics = {
        "created_at": now_iso(),
        "scores_path": str(scores_path),
        "prob_column": args.prob_column,
        "label_column": args.label_column,
        "bins": int(args.bins),
        "total_rows": int(count),
        "positive_rate": float(np.sum(pos) / max(np.sum(totals), 1.0)),
        "brier": float(brier_sum / max(count, 1)),
        "log_loss": float(logloss_sum / max(count, 1)),
    }

    out_root = Path(args.out_dir) / args.name
    ensure_dir(out_root)

    png_path = out_root / f"{args.name}_reliability.png"
    plot_title = f"Reliability Curve — {args.name.replace('_', ' ').title()}"
    plot_reliability_curve(calibration_rows, plot_title, png_path, args.dpi, figsize)

    if args.save_pdf:
        pdf_path = out_root / f"{args.name}_reliability.pdf"
        plot_reliability_curve(calibration_rows, plot_title, pdf_path, args.dpi, figsize)

    payload = {
        "metadata": metrics,
        "calibration": calibration_rows,
    }

    json_path = out_root / f"{args.name}_reliability.json"
    if args.save_json:
        ensure_dir(json_path.parent)
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
