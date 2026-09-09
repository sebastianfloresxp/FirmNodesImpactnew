"""Export FactSet supply chain relationships to a Parquet edge list and histogram."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd

from .query import iter_query

_REL_TYPE_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def _validate_rel_type(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        uppercase = value.upper()
        if not uppercase:
            raise ValueError("Empty relationship type supplied")
        if any(ch not in _REL_TYPE_ALLOWED for ch in uppercase):
            raise ValueError(f"Invalid characters in relationship type '{value}'")
        cleaned.append(uppercase)
    return cleaned


def _validate_date(value: str) -> str:
    datetime.strptime(value, "%Y-%m-%d")
    return value


def _build_sql(
    rel_types: list[str] | None,
    start_date_min: str | None,
    start_date_max: str | None,
    as_of_date: str | None,
) -> str:
    where = [
        "r.source_factset_entity_id IS NOT NULL",
        "r.target_factset_entity_id IS NOT NULL",
    ]

    if rel_types:
        cleaned = _validate_rel_type(rel_types)
        placeholders = ", ".join(f"'{val}'" for val in cleaned)
        where.append(f"r.rel_type IN ({placeholders})")

    if start_date_min:
        where.append(f"r.start_date >= '{_validate_date(start_date_min)}'")

    if start_date_max:
        where.append(f"r.start_date < '{_validate_date(start_date_max)}'")

    if as_of_date:
        as_of = _validate_date(as_of_date)
        where.append(
            "( (r.start_date IS NULL OR r.start_date <= '"
            + as_of
            + "') AND (r.end_date IS NULL OR r.end_date > '"
            + as_of
            + "') )"
        )

    sql = f"""
    SELECT
        r.id AS relationship_id,
        r.rel_type,
        t.rel_type_desc,
        r.source_factset_entity_id AS source_factset_entity_id,
        r.target_factset_entity_id AS target_factset_entity_id,
        CAST(r.start_date AS DATE) AS start_date,
        CAST(r.end_date AS DATE) AS end_date,
        r.revenue_pct
    FROM ent_v1.ent_scr_relationships AS r
    LEFT JOIN ref_v2.relationship_type_map AS t
        ON r.rel_type = t.rel_type_code
    WHERE {" AND ".join(where)}
    """
    return sql


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported output format for {path}. Use .parquet, .csv, or .json")


def export_supply_chain_edges(
    *,
    edges_path: Path,
    histogram_path: Path | None,
    rel_types: list[str] | None,
    start_date_min: str | None,
    start_date_max: str | None,
    as_of_date: str | None,
    chunksize: int,
) -> None:
    sql = _build_sql(rel_types, start_date_min, start_date_max, as_of_date)

    frames: list[pd.DataFrame] = []
    histogram: dict[tuple[str, str | None], int] = defaultdict(int)

    for chunk in iter_query(sql, params=None, chunksize=chunksize):
        if chunk.empty:
            continue
        chunk.rename(
            columns={
                "rel_type": "relationship_type",
                "rel_type_desc": "relationship_description",
            },
            inplace=True,
        )
        chunk["start_date"] = pd.to_datetime(chunk["start_date"], errors="coerce")
        chunk["end_date"] = pd.to_datetime(chunk["end_date"], errors="coerce")
        chunk["revenue_pct"] = pd.to_numeric(chunk["revenue_pct"], errors="coerce")

        frames.append(chunk)

        grouped = (
            chunk.groupby(["relationship_type", "relationship_description"], dropna=False)
            .size()
            .to_dict()
        )
        for key, value in grouped.items():
            histogram[key] += int(value)

    edges_df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[  # type: ignore[arg-type]
                "relationship_id",
                "relationship_type",
                "relationship_description",
                "source_factset_entity_id",
                "target_factset_entity_id",
                "start_date",
                "end_date",
                "revenue_pct",
            ]
        )
    )

    # Enforce column order and types before writing
    expected_order = [
        "relationship_id",
        "relationship_type",
        "relationship_description",
        "source_factset_entity_id",
        "target_factset_entity_id",
        "start_date",
        "end_date",
        "revenue_pct",
    ]
    edges_df = edges_df.reindex(columns=expected_order)

    _write_df(edges_df, edges_path)
    print(f"Wrote {len(edges_df):,} edges to {edges_path}")

    if histogram_path:
        hist_rows = [
            {
                "relationship_type": rel_type,
                "relationship_description": desc,
                "edge_count": count,
            }
            for (rel_type, desc), count in sorted(
                histogram.items(), key=lambda item: item[1], reverse=True
            )
        ]
        hist_df = pd.DataFrame(hist_rows)
        _write_df(hist_df, histogram_path)
        print(f"Wrote histogram with {len(hist_df):,} rows to {histogram_path}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export FactSet supply chain relationships to a Parquet edge list and histogram",
    )
    parser.add_argument(
        "--edges-out",
        required=True,
        type=Path,
        help="Path to write the edge list (parquet/csv/json)",
    )
    parser.add_argument(
        "--histogram-out",
        type=Path,
        help="Optional path to write histogram counts by relationship type (parquet/csv/json)",
    )
    parser.add_argument(
        "--rel-type",
        action="append",
        default=None,
        help="Filter to specific relationship types (may be repeated). Default is all types",
    )
    parser.add_argument(
        "--start-date-min",
        type=str,
        help="Keep relationships with start_date >= this YYYY-MM-DD value",
    )
    parser.add_argument(
        "--start-date-max",
        type=str,
        help="Keep relationships with start_date < this YYYY-MM-DD value",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        help="Only include relationships active on this YYYY-MM-DD date",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Rows per batch when streaming from SQL",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    export_supply_chain_edges(
        edges_path=args.edges_out,
        histogram_path=args.histogram_out,
        rel_types=args.rel_type,
        start_date_min=args.start_date_min,
        start_date_max=args.start_date_max,
        as_of_date=args.as_of_date,
        chunksize=args.chunksize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
