"""Evaluate precision lift from shipping-confirmed edges for ensemble selections."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import duckdb

DEFAULT_ENTITY_MAP = Path("data/processed/core/releases/core_v1/mapping/entity_map.parquet")
DEFAULT_SHIPPING_DIR = Path("data/processed/shipping/splits")
DEFAULT_SCORES_ROOT = Path("results/ensemble/meta_ranker/meta_ranker_v4")
DEFAULT_OUTPUT_DIR = Path("results/analysis/shipping_precision_v4")

SHIPPING_SCOPES = ("aligned", "union")


@dataclass(frozen=True)
class ThresholdScenario:
    name: str
    display: str
    threshold: float


@dataclass(frozen=True)
class TopKScenario:
    name: str
    display: str
    k: int
    floor: float
    cap: int | None


THRESHOLD_SCENARIOS: list[ThresholdScenario] = [
    ThresholdScenario(name="tau_gold", display="τ = 1.0 (Gold Anchor)", threshold=1.0),
    ThresholdScenario(name="tau_f1", display="τ ≈ 0.9235 (F1-opt)", threshold=0.9235),
]

TOPK_SCENARIOS: list[TopKScenario] = [
    TopKScenario(
        name="tau_core_topk",
        display="Top-K+floor (K=10,floor=0.995,cap=50)",
        k=10,
        floor=0.995,
        cap=50,
    ),
]


def register_entity_lookup(conn: duckdb.DuckDBPyConnection, entity_map: Path) -> None:
    path = entity_map.as_posix().replace("'", "''")
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW entity_lookup AS
        SELECT canonical_id AS factset_id, node_id FROM read_parquet('{path}')
        UNION ALL
        SELECT raw_id AS factset_id, node_id FROM read_parquet('{path}')
        """
    )


def _shipping_select(path: Path) -> str:
    path_str = path.as_posix().replace("'", "''")
    return (
        f"SELECT DISTINCT src.node_id AS src_id, dst.node_id AS dst_id "
        f"FROM read_parquet('{path_str}') AS s "
        f"JOIN entity_lookup AS src ON s.source_factset_entity_id = src.factset_id "
        f"JOIN entity_lookup AS dst ON s.target_factset_entity_id = dst.factset_id "
        f"WHERE src.node_id IS NOT NULL AND dst.node_id IS NOT NULL"
    )


def register_shipping_split(
    conn: duckdb.DuckDBPyConnection,
    split: str,
    shipping_dir: Path,
) -> None:
    path = shipping_dir / f"{split}_edges.parquet"
    if not path.exists():
        return
    view = f"shipping_{split}"
    conn.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS {_shipping_select(path)}")


def register_shipping_union(conn: duckdb.DuckDBPyConnection, shipping_dir: Path) -> None:
    selects: list[str] = []
    for name in ("train", "val", "test"):
        path = shipping_dir / f"{name}_edges.parquet"
        if path.exists():
            selects.append(_shipping_select(path))
    if not selects:
        conn.execute("CREATE OR REPLACE TEMP VIEW shipping_union AS SELECT 1 WHERE 0")
        return
    union_sql = " UNION ALL ".join(selects)
    conn.execute(
        f"CREATE OR REPLACE TEMP VIEW shipping_union AS SELECT DISTINCT * FROM ({union_sql})"
    )


def register_scores_split(conn: duckdb.DuckDBPyConnection, split: str, scores_root: Path) -> None:
    path = (scores_root / f"scores_{split}.parquet").as_posix().replace("'", "''")
    view = f"scores_{split}"
    conn.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT * FROM read_parquet('{path}')")


def selection_sql_threshold(view: str, threshold: float) -> str:
    return f"SELECT src_id, dst_id, label, meta_prob FROM {view} WHERE meta_prob >= {threshold}"


def selection_sql_topk(view: str, scenario: TopKScenario) -> str:
    cap_clause = ""
    select_cols = "src_id, dst_id, label, meta_prob, rk_src"
    over_dst = ""
    if scenario.cap is not None:
        select_cols += ", rk_dst"
        over_dst = (
            ", ROW_NUMBER() OVER (PARTITION BY dst_id ORDER BY meta_prob DESC, src_id) AS rk_dst"
        )
        cap_clause = f" AND rk_dst <= {scenario.cap}"
    return (
        "WITH filtered AS ("
        f"    SELECT *, ROW_NUMBER() OVER (PARTITION BY src_id ORDER BY meta_prob DESC, dst_id) AS rk_src"
        f"{over_dst} FROM {view} WHERE meta_prob >= {scenario.floor}"
        ") "
        f"SELECT src_id, dst_id, label, meta_prob FROM filtered "
        f"WHERE rk_src <= {scenario.k}{cap_clause}"
    )


def evaluate_selection(
    conn: duckdb.DuckDBPyConnection,
    selection_sql: str,
    shipping_view: str,
) -> dict[str, float]:
    query = f"""
    WITH selection AS ( {selection_sql} ),
    joined AS (
        SELECT
            sel.src_id,
            sel.dst_id,
            sel.label,
            sel.meta_prob,
            CASE WHEN ship.src_id IS NOT NULL THEN 1 ELSE 0 END AS shipping_flag
        FROM selection sel
        LEFT JOIN {shipping_view} ship
          ON sel.src_id = ship.src_id AND sel.dst_id = ship.dst_id
    )
    SELECT
        COUNT(*) AS predicted,
        SUM(label) AS true_positives,
        SUM(shipping_flag) AS shipping_hits,
        SUM(CASE WHEN label = 0 AND shipping_flag = 1 THEN 1 ELSE 0 END) AS shipping_new_tp,
        SUM(CASE WHEN label = 1 AND shipping_flag = 1 THEN 1 ELSE 0 END) AS shipping_existing_tp
    FROM joined
    """
    predicted, tp, shipping_hits, shipping_new_tp, shipping_existing_tp = conn.execute(
        query
    ).fetchone()
    predicted = int(predicted)
    tp = int(tp or 0)
    shipping_hits = int(shipping_hits or 0)
    shipping_new_tp = int(shipping_new_tp or 0)
    shipping_existing_tp = int(shipping_existing_tp or 0)
    precision = tp / predicted if predicted else 0.0
    precision_shipping = (tp + shipping_new_tp) / predicted if predicted else 0.0
    shipping_hit_rate = shipping_hits / predicted if predicted else 0.0
    return {
        "predicted": predicted,
        "true_positives": tp,
        "precision": precision,
        "shipping_hits": shipping_hits,
        "shipping_new_tp": shipping_new_tp,
        "shipping_existing_tp": shipping_existing_tp,
        "shipping_hit_rate": shipping_hit_rate,
        "precision_with_shipping": precision_shipping,
    }


def run_evaluation(
    scores_root: Path,
    entity_map: Path,
    shipping_dir: Path,
    output_dir: Path,
    splits: Iterable[str],
    shipping_scope: str,
) -> dict[str, dict[str, dict[str, float]]]:
    conn = duckdb.connect(database=":memory:")
    register_entity_lookup(conn, entity_map)
    if shipping_scope == "union":
        register_shipping_union(conn, shipping_dir)
    for split in splits:
        register_shipping_split(conn, split, shipping_dir)
        register_scores_split(conn, split, scores_root)

    results: dict[str, dict[str, dict[str, float]]] = {}

    for split in splits:
        view_scores = f"scores_{split}"
        shipping_view = "shipping_union" if shipping_scope == "union" else f"shipping_{split}"
        split_results: dict[str, dict[str, float]] = {}
        # Threshold scenarios
        for scenario in THRESHOLD_SCENARIOS:
            selection_sql = selection_sql_threshold(view_scores, scenario.threshold)
            metrics = evaluate_selection(conn, selection_sql, shipping_view)
            metrics["threshold"] = scenario.threshold
            metrics["display_name"] = scenario.display
            split_results[scenario.name] = metrics
        # Top-K scenarios
        for scenario in TOPK_SCENARIOS:
            selection_sql = selection_sql_topk(view_scores, scenario)
            metrics = evaluate_selection(conn, selection_sql, shipping_view)
            metrics["k"] = scenario.k
            metrics["floor"] = scenario.floor
            metrics["cap"] = scenario.cap if scenario.cap is None else int(scenario.cap)
            metrics["display_name"] = scenario.display
            split_results[scenario.name] = metrics
        results[split] = split_results

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "shipping_precision_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    rows: list[str] = [
        "split,scenario,display_name,predicted,true_positives,precision,shipping_hits,shipping_new_tp,shipping_existing_tp,shipping_hit_rate,precision_with_shipping,threshold,k,floor,cap"
    ]
    for split, scenarios in results.items():
        for name, metrics in scenarios.items():
            row = [
                split,
                name,
                metrics.get("display_name", ""),
                str(metrics["predicted"]),
                str(metrics["true_positives"]),
                f"{metrics['precision']:.6f}",
                str(metrics["shipping_hits"]),
                str(metrics["shipping_new_tp"]),
                str(metrics["shipping_existing_tp"]),
                f"{metrics['shipping_hit_rate']:.6f}",
                f"{metrics['precision_with_shipping']:.6f}",
                f"{metrics.get('threshold', '')}",
                f"{metrics.get('k', '')}",
                f"{metrics.get('floor', '')}",
                f"{metrics.get('cap', '')}",
            ]
            rows.append(",".join(row))
    csv_path = output_dir / "shipping_precision_summary.csv"
    csv_path.write_text("\n".join(rows))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble precision lift using shipping data"
    )
    parser.add_argument("--scores-root", type=Path, default=DEFAULT_SCORES_ROOT)
    parser.add_argument("--entity-map", type=Path, default=DEFAULT_ENTITY_MAP)
    parser.add_argument("--shipping-dir", type=Path, default=DEFAULT_SHIPPING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="*", default=["val", "test"], help="Splits to evaluate")
    parser.add_argument(
        "--shipping-scope",
        choices=SHIPPING_SCOPES,
        default="aligned",
        help="Use per-split shipping overlap (aligned) or union of all shipping edges",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_evaluation(
        scores_root=args.scores_root,
        entity_map=args.entity_map,
        shipping_dir=args.shipping_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        shipping_scope=args.shipping_scope,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
