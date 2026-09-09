from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb

from src.ensemble.utils import ensure_dir, now_iso

SLICE_DEFS: dict[str, str] = {
    "overall": "TRUE",
    "slice_CC": "slice_CC = 1",
    "slice_WW3": "slice_WW3 = 1",
    "slice_gt2hop": "slice_gt2hop = 1",
    "slice_deg_q1": "slice_deg_q1 = 1",
}

DEFAULT_THRESHOLDS: Sequence[float] = (1.0, 0.9235)
DEFAULT_PRECISION_TARGETS: Sequence[float] = (0.30, 0.20)
DEFAULT_TOPK_GRID: Sequence[str] = (
    "K=1,floor=0.995,cap=None",
    "K=1,floor=0.99,cap=None",
    "K=1,floor=0.985,cap=None",
    "K=1,floor=0.98,cap=None",
    "K=1,floor=0.995,cap=50",
    "K=1,floor=0.99,cap=50",
    "K=1,floor=0.985,cap=50",
    "K=1,floor=0.98,cap=50",
    "K=2,floor=0.995,cap=None",
    "K=2,floor=0.99,cap=None",
    "K=2,floor=0.985,cap=None",
    "K=2,floor=0.98,cap=None",
    "K=2,floor=0.995,cap=50",
    "K=2,floor=0.99,cap=50",
    "K=2,floor=0.985,cap=50",
    "K=2,floor=0.98,cap=50",
    "K=3,floor=0.995,cap=None",
    "K=3,floor=0.99,cap=None",
    "K=3,floor=0.985,cap=None",
    "K=3,floor=0.98,cap=None",
    "K=3,floor=0.995,cap=50",
    "K=3,floor=0.99,cap=50",
    "K=3,floor=0.985,cap=50",
    "K=3,floor=0.98,cap=50",
    "K=5,floor=0.995,cap=None",
    "K=5,floor=0.99,cap=None",
    "K=5,floor=0.985,cap=None",
    "K=5,floor=0.98,cap=None",
    "K=5,floor=0.995,cap=50",
    "K=5,floor=0.99,cap=50",
    "K=5,floor=0.985,cap=50",
    "K=5,floor=0.98,cap=50",
    "K=10,floor=0.995,cap=None",
    "K=10,floor=0.99,cap=None",
    "K=10,floor=0.985,cap=None",
    "K=10,floor=0.98,cap=None",
    "K=10,floor=0.995,cap=50",
    "K=10,floor=0.99,cap=50",
    "K=10,floor=0.985,cap=50",
    "K=10,floor=0.98,cap=50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selection policies on meta-ranker outputs"
    )
    parser.add_argument(
        "--scores-dir",
        required=True,
        help="Directory containing scores_val.parquet and scores_test.parquet",
    )
    parser.add_argument(
        "--meta-dir", required=True, help="Directory containing meta_inputs_{val,test}.parquet"
    )
    parser.add_argument(
        "--out-dir",
        default="results/ensemble/meta_ranker",
        help="Root directory for metrics output",
    )
    parser.add_argument("--tag", help="Tag under out-dir (defaults to name of scores directory)")
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Fixed thresholds to evaluate",
    )
    parser.add_argument(
        "--precision-targets",
        nargs="*",
        type=float,
        default=list(DEFAULT_PRECISION_TARGETS),
        help="Precision targets (evaluated on validation to derive taus)",
    )
    parser.add_argument(
        "--topk-config",
        action="append",
        default=list(DEFAULT_TOPK_GRID),
        help="Top-K + floor configuration in K=...,floor=...,cap=... format",
    )
    parser.add_argument("--prediction-root", help="Directory to store exported Parquet selections")
    parser.add_argument(
        "--export-threshold", action="append", default=[], help="Export NAME=TAU selection"
    )
    parser.add_argument(
        "--export-topk",
        action="append",
        default=[],
        help="Export NAME=K=...,floor=...,cap=... selection",
    )
    parser.add_argument("--threads", type=int, default=4, help="DuckDB thread count")
    return parser.parse_args()


@dataclass(frozen=True)
class TopKConfig:
    name: str
    k: int
    floor: float
    cap: int | None

    def key(self) -> str:
        cap_part = "None" if self.cap is None else str(self.cap)
        return f"K={self.k}_floor={self.floor}_cap={cap_part}"


class SelectionEvaluator:
    def __init__(self, scores_dir: Path, meta_dir: Path, threads: int) -> None:
        self.scores_dir = scores_dir
        self.meta_dir = meta_dir
        self.conn = duckdb.connect(database=":memory:")
        self.conn.execute("PRAGMA threads = ?", [max(1, threads)])
        self.views: dict[str, str] = {}
        self._register_split("val")
        self._register_split("test")
        self.base_counts = self._compute_base_counts()

    def close(self) -> None:
        self.conn.close()

    def _register_split(self, split: str) -> None:
        scores_path = self.scores_dir / f"scores_{split}.parquet"
        meta_path = self.meta_dir / f"meta_inputs_{split}.parquet"
        if not scores_path.exists():
            raise FileNotFoundError(f"Scores parquet missing: {scores_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta inputs parquet missing: {meta_path}")
        view_name = f"data_{split}"
        self.conn.execute(
            f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT s.src_id,
                   s.dst_id,
                   CAST(s.meta_prob AS DOUBLE) AS meta_prob,
                   CAST(s.label AS INTEGER) AS label,
                   CAST(m.slice_CC AS INTEGER) AS slice_CC,
                   CAST(m.slice_WW3 AS INTEGER) AS slice_WW3,
                   CAST(m.slice_gt2hop AS INTEGER) AS slice_gt2hop,
                   CAST(m.slice_deg_q1 AS INTEGER) AS slice_deg_q1
            FROM read_parquet('{scores_path.as_posix()}') s
            JOIN read_parquet('{meta_path.as_posix()}') m USING (src_id, dst_id)
            """
        )
        self.views[split] = view_name

    def _compute_base_counts(self) -> dict[str, dict[str, dict[str, int]]]:
        counts: dict[str, dict[str, dict[str, int]]] = {}
        for split, view_name in self.views.items():
            counts[split] = {}
            for slice_name, condition in SLICE_DEFS.items():
                row = self.conn.execute(
                    f"SELECT COUNT(*) AS total_rows, SUM(label) AS positives FROM {view_name} WHERE {condition}"
                ).fetchone()
                total_rows, positives = row if row is not None else (0, 0)
                counts[split][slice_name] = {
                    "total_rows": int(total_rows or 0),
                    "positives": int(positives or 0),
                }
        return counts

    def _selection_metrics(self, split: str, selection_sql: str) -> dict[str, dict[str, float]]:
        view_metrics: dict[str, dict[str, float]] = {}
        rows = self.conn.execute(
            f"""
            WITH filtered AS (
                {selection_sql}
            )
            SELECT 'overall' AS slice, COUNT(*) AS predicted, SUM(label) AS tp FROM filtered
            UNION ALL
            SELECT 'slice_CC' AS slice, COUNT(*) AS predicted, SUM(label) AS tp FROM filtered WHERE slice_CC = 1
            UNION ALL
            SELECT 'slice_WW3' AS slice, COUNT(*) AS predicted, SUM(label) AS tp FROM filtered WHERE slice_WW3 = 1
            UNION ALL
            SELECT 'slice_gt2hop' AS slice, COUNT(*) AS predicted, SUM(label) AS tp FROM filtered WHERE slice_gt2hop = 1
            UNION ALL
            SELECT 'slice_deg_q1' AS slice, COUNT(*) AS predicted, SUM(label) AS tp FROM filtered WHERE slice_deg_q1 = 1
            """
        ).fetchall()
        for slice_name, predicted, tp in rows:
            totals = self.base_counts[split][slice_name]
            predicted = int(predicted or 0)
            tp = int(tp or 0)
            total_rows = totals["total_rows"]
            positives = totals["positives"]
            precision = (tp / predicted) if predicted else 0.0
            recall = (tp / positives) if positives else 0.0
            yield_rate = (predicted / total_rows) if total_rows else 0.0
            view_metrics[slice_name] = {
                "predicted": predicted,
                "true_positives": tp,
                "precision": precision,
                "recall": recall,
                "yield": yield_rate,
            }
        return view_metrics

    def evaluate_threshold(self, tau: float) -> dict[str, dict[str, dict[str, float]]]:
        metrics: dict[str, dict[str, dict[str, float]]] = {}
        for split, view_name in self.views.items():
            selection_sql = (
                f"SELECT src_id, dst_id, meta_prob, label, slice_CC, slice_WW3, "
                f"slice_gt2hop, slice_deg_q1 FROM {view_name} WHERE meta_prob >= {tau}"
            )
            metrics[split] = self._selection_metrics(split, selection_sql)
        return metrics

    def derive_precision_tau(self, target: float) -> float:
        if not (0.0 < target < 1.0):
            raise ValueError("Precision target must be between 0 and 1")
        view_val = self.views["val"]
        row = self.conn.execute(
            f"""
            WITH data AS (
                SELECT meta_prob, label FROM {view_val}
            ),
            aggregated AS (
                SELECT meta_prob,
                       SUM(label) AS pos_at_prob,
                       COUNT(*) AS count_at_prob
                FROM data
                GROUP BY meta_prob
            ),
            ordered AS (
                SELECT meta_prob AS prob,
                       SUM(pos_at_prob) OVER (ORDER BY meta_prob DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_pos,
                       SUM(count_at_prob) OVER (ORDER BY meta_prob DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_count
                FROM aggregated
            )
            SELECT MIN(prob) AS threshold
            FROM (
                SELECT prob,
                       CASE WHEN cum_count = 0 THEN NULL ELSE cum_pos::DOUBLE / cum_count END AS precision
                FROM ordered
            )
            WHERE precision >= {target}
            """
        ).fetchone()
        tau = float(row[0]) if row and row[0] is not None else 1.0
        return tau

    def _topk_selection_sql(self, view_name: str, config: TopKConfig, include_ranks: bool) -> str:
        base_sql = (
            f"SELECT src_id, dst_id, meta_prob, label, slice_CC, slice_WW3, "
            f"slice_gt2hop, slice_deg_q1 FROM {view_name} WHERE meta_prob >= {config.floor}"
        )
        inner_select = (
            "SELECT src_id, dst_id, meta_prob, label, slice_CC, slice_WW3, slice_gt2hop, slice_deg_q1, "
            "ROW_NUMBER() OVER (PARTITION BY src_id ORDER BY meta_prob DESC, dst_id) AS rk_src"
        )
        if config.cap is not None:
            inner_select += ", ROW_NUMBER() OVER (PARTITION BY dst_id ORDER BY meta_prob DESC, src_id) AS rk_dst"
        inner_select += f" FROM ({base_sql})"
        where_clauses = [f"rk_src <= {config.k}"]
        if config.cap is not None:
            where_clauses.append(f"rk_dst <= {config.cap}")
        projected_cols = (
            "src_id, dst_id, meta_prob, label, slice_CC, slice_WW3, slice_gt2hop, slice_deg_q1"
        )
        if include_ranks:
            projected_cols += ", rk_src AS rank_src"
            if config.cap is not None:
                projected_cols += ", rk_dst AS rank_dst"
        selection_sql = f"SELECT {projected_cols} FROM ({inner_select}) WHERE " + " AND ".join(
            where_clauses
        )
        return selection_sql

    def evaluate_topk(self, config: TopKConfig) -> dict[str, dict[str, dict[str, float]]]:
        metrics: dict[str, dict[str, dict[str, float]]] = {}
        for split, view_name in self.views.items():
            selection_sql = self._topk_selection_sql(view_name, config, include_ranks=False)
            metrics[split] = self._selection_metrics(split, selection_sql)
        return metrics

    def export_threshold(self, name: str, tau: float, prediction_root: Path, tag: str) -> Path:
        export_path = prediction_root / tag / f"{name}.parquet"
        ensure_dir(export_path.parent)
        view_name = self.views["test"]
        selection_sql = (
            f"SELECT src_id, dst_id, meta_prob, label, slice_CC, slice_WW3, "
            f"slice_gt2hop, slice_deg_q1 FROM {view_name} WHERE meta_prob >= {tau}"
        )
        self.conn.execute(
            f"COPY ({selection_sql}) TO '{export_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd')"
        )
        return export_path

    def export_topk(self, name: str, config: TopKConfig, prediction_root: Path, tag: str) -> Path:
        export_path = prediction_root / tag / f"{name}.parquet"
        ensure_dir(export_path.parent)
        view_name = self.views["test"]
        selection_sql = self._topk_selection_sql(view_name, config, include_ranks=True)
        self.conn.execute(
            f"COPY ({selection_sql}) TO '{export_path.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd')"
        )
        return export_path


def parse_topk_string(raw: str, default_name: str | None = None) -> TopKConfig:
    parts = raw.split(",")
    items: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid top-k config: {raw}")
        key, value = part.split("=", 1)
        items[key.strip()] = value.strip()
    k = int(items.get("K") or items.get("k"))  # type: ignore[arg-type]
    floor = float(items.get("floor"))  # type: ignore[arg-type]
    cap_raw = items.get("cap")
    cap: int | None
    if cap_raw is None or cap_raw.lower() == "none":
        cap = None
    else:
        cap = int(cap_raw)
    name = default_name or f"K{k}_floor{floor}_cap{cap_raw}"
    return TopKConfig(name=name, k=k, floor=floor, cap=cap)


def parse_export_thresholds(raw_values: Iterable[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"Export threshold must be NAME=VALUE, got: {raw}")
        name, value = raw.split("=", 1)
        out[name.strip()] = float(value.strip())
    return out


def parse_export_topk(raw_values: Iterable[str]) -> dict[str, TopKConfig]:
    out: dict[str, TopKConfig] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"Export topk must be NAME=config, got: {raw}")
        name, config = raw.split("=", 1)
        out[name.strip()] = parse_topk_string(config.strip(), default_name=name.strip())
    return out


def main() -> None:
    args = parse_args()
    scores_dir = Path(args.scores_dir)
    meta_dir = Path(args.meta_dir)
    tag = args.tag or scores_dir.name
    out_dir = Path(args.out_dir) / tag
    ensure_dir(out_dir)

    evaluator = SelectionEvaluator(
        scores_dir=scores_dir, meta_dir=meta_dir, threads=int(args.threads)
    )

    fixed_thresholds = list(dict.fromkeys(float(t) for t in args.thresholds))
    precision_targets = list(dict.fromkeys(float(t) for t in args.precision_targets))
    topk_strings = list(dict.fromkeys(args.topk_config))

    threshold_metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    fixed_map: dict[str, float] = {}
    for tau in fixed_thresholds:
        name = f"tau={tau:g}"
        threshold_metrics[name] = evaluator.evaluate_threshold(tau)
        fixed_map[name] = tau

    precision_map: dict[float, float] = {}
    for target in precision_targets:
        tau = evaluator.derive_precision_tau(target)
        name = f"tau_precision_{target:.2f}"
        precision_map[target] = tau
        threshold_metrics[name] = evaluator.evaluate_threshold(tau)

    topk_metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    topk_configs: dict[str, TopKConfig] = {}
    for raw in topk_strings:
        cfg = parse_topk_string(raw)
        topk_configs[cfg.key()] = cfg
        topk_metrics[cfg.key()] = evaluator.evaluate_topk(cfg)

    exports: dict[str, str] = {}
    prediction_root = Path(args.prediction_root) if args.prediction_root else None
    if prediction_root:
        export_thresholds = parse_export_thresholds(args.export_threshold)
        export_topks = parse_export_topk(args.export_topk)
        for name, tau in export_thresholds.items():
            path = evaluator.export_threshold(name, tau, prediction_root, tag)
            exports[name] = path.as_posix()
        for name, cfg in export_topks.items():
            path = evaluator.export_topk(name, cfg, prediction_root, tag)
            exports[name] = path.as_posix()

    payload = {
        "created_at": now_iso(),
        "tag": tag,
        "inputs": {
            "scores_dir": scores_dir.as_posix(),
            "meta_dir": meta_dir.as_posix(),
        },
        "base_counts": evaluator.base_counts,
        "thresholds": {
            "fixed": fixed_map,
            "precision_targets": {f"{k:.2f}": v for k, v in precision_map.items()},
            "metrics": threshold_metrics,
        },
        "topk": dict(topk_metrics.items()),
        "exports": exports,
    }

    output_path = out_dir / "selection_summary.json"
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Selection summary written to {output_path}")

    evaluator.close()


if __name__ == "__main__":
    main()
