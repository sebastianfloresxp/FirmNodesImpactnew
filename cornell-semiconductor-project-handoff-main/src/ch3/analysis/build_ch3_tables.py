#!/usr/bin/env python3
"""
Generate Chapter 3 LaTeX tabulars for Top-K performance, score stratification,
and model agreement for the Chapter 3 prediction layer.

These files are intended to be `\\input{...}` inside an outer LaTeX `table`
environment (i.e., the outputs are *tabular-only*, no captions/labels).

Defaults target the canonical Chapter 3 inference run (upstream orientation).

Outputs (written under --out-dir, default tables/chapter3/):
  - tab_3.1_topk_precision_lift.tex
  - tab_3.2_table_deciles.tex
  - tab_3.3_model_agreement_precision.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

TRAIN = Path("data/processed/core/releases/core_v1/splits/train_edges.parquet")
VAL = Path("data/processed/core/releases/core_v1/splits/val_edges.parquet")
TEST = Path("data/processed/core/releases/core_v1/splits/test_edges.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Chapter 3 LaTeX inference diagnostics tables")
    p.add_argument(
        "--artifact-root",
        default="artifacts/ch3/prediction_upstream",
        help="Directory with prediction artifacts (e.g., artifacts/ch3/prediction_upstream)",
    )
    p.add_argument(
        "--out-dir",
        default="tables/chapter3",
        help="Directory to write LaTeX tabular-only tables (no outer table environment)",
    )
    return p.parse_args()


def ensure_dirs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def compute_base_rate(con: duckdb.DuckDBPyConnection, meta_scores: Path) -> float:
    return con.execute(
        """
        WITH known AS (
          SELECT DISTINCT src_id, dst_id FROM read_parquet(?)
          UNION SELECT DISTINCT src_id, dst_id FROM read_parquet(?)
          UNION SELECT DISTINCT src_id, dst_id FROM read_parquet(?)
        ), hits AS (
          SELECT COUNT(*) AS c
          FROM read_parquet(?) s
          JOIN known k USING (src_id, dst_id)
        ), total AS (
          SELECT COUNT(*) AS c FROM read_parquet(?)
        )
        SELECT hits.c::DOUBLE / total.c FROM hits, total
        """,
        [
            TRAIN.as_posix(),
            VAL.as_posix(),
            TEST.as_posix(),
            meta_scores.as_posix(),
            meta_scores.as_posix(),
        ],
    ).fetchone()[0]


def build_topk_table(
    con: duckdb.DuckDBPyConnection, base_rate: float, topk_files: dict[int, Path]
) -> pd.DataFrame:
    rows = []
    for k, path in topk_files.items():
        total, known = con.execute(
            """
            SELECT COUNT(*) AS total, SUM(is_known) AS known
            FROM read_parquet(?)
            """,
            [path.as_posix()],
        ).fetchone()
        precision = known / total if total else 0.0
        lift = precision / base_rate if base_rate else 0.0
        rows.append(
            {
                "K": k,
                "Predictions": total,
                "Known SCR": known,
                "Precision": precision,
                "Lift": lift,
                "New (predicted only)": total - known,
            }
        )
    df = pd.DataFrame(rows).sort_values("K")
    return df


def build_decile_table(
    con: duckdb.DuckDBPyConnection, base_rate: float, meta_scores: Path
) -> pd.DataFrame:
    df = con.execute(
        """
        WITH known AS (
          SELECT DISTINCT src_id, dst_id FROM read_parquet(?)
          UNION SELECT DISTINCT src_id, dst_id FROM read_parquet(?)
          UNION SELECT DISTINCT src_id, dst_id FROM read_parquet(?)
        ),
        scored AS (
          SELECT s.src_id,
                 s.dst_id,
                 s.meta_prob,
                 (k.src_id IS NOT NULL) AS is_known
          FROM read_parquet(?) s
          LEFT JOIN known k USING (src_id, dst_id)
        ),
        ranked AS (
          SELECT *,
                 NTILE(10) OVER (ORDER BY meta_prob DESC, src_id, dst_id) AS decile
          FROM scored
        )
        SELECT decile,
               COUNT(*) AS n,
               SUM(is_known) AS hits
        FROM ranked
        GROUP BY decile
        ORDER BY decile
        """,
        [TRAIN.as_posix(), VAL.as_posix(), TEST.as_posix(), meta_scores.as_posix()],
    ).fetchdf()
    df["Precision"] = df["hits"] / df["n"]
    df["Lift"] = df["Precision"] / base_rate
    # Convert decile to rank (1=highest scores)
    df["Decile"] = df["decile"]
    df[["n", "hits"]] = df[["n", "hits"]].astype(int)
    return df[["Decile", "n", "hits", "Precision", "Lift"]]


def build_agreement_table(
    con: duckdb.DuckDBPyConnection, base_rate: float, meta_inputs: Path
) -> pd.DataFrame:
    # Count how many base models give prob >= 0.5
    df = con.execute(
        """
        WITH known AS (
          SELECT DISTINCT src_id, dst_id FROM read_parquet(?)
          UNION SELECT DISTINCT src_id, dst_id FROM read_parquet(?)
          UNION SELECT DISTINCT src_id, dst_id FROM read_parquet(?)
        ),
        scored AS (
          SELECT m.*,
                 (k.src_id IS NOT NULL) AS is_known
          FROM read_parquet(?) m
          LEFT JOIN known k USING (src_id, dst_id)
        ),
        agg AS (
          SELECT
            ( (prob_graphsage >= 0.5)::INT
            + (prob_node2vec >= 0.5)::INT
            + (prob_heuristics >= 0.5)::INT
            + (prob_twotower >= 0.5)::INT
            + (prob_tgnn >= 0.5)::INT
            + (prob_n2v_temporal >= 0.5)::INT ) AS agree,
            is_known
          FROM scored
        )
        SELECT agree, COUNT(*) AS n, SUM(is_known) AS hits
        FROM agg
        GROUP BY agree
        ORDER BY agree DESC
        """,
        [TRAIN.as_posix(), VAL.as_posix(), TEST.as_posix(), meta_inputs.as_posix()],
    ).fetchdf()
    df["Precision"] = df["hits"] / df["n"]
    df["Lift"] = df["Precision"] / base_rate
    df.rename(columns={"agree": "Agree (models>=0.5)"}, inplace=True)
    df[["n", "hits"]] = df[["n", "hits"]].astype(int)
    return df[["Agree (models>=0.5)", "n", "hits", "Precision", "Lift"]]


def fmt_float(x: float) -> str:
    return f"{x:.3f}"


def write_topk_tabular(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"K & Predictions & Known Overlap & Precision & Lift & New (predicted only) \\",
        r"\midrule",
    ]
    for k, predictions, known, precision, lift, new_pred in df.sort_values("K").itertuples(
        index=False, name=None
    ):
        lines.append(
            f"{int(k)} & {int(predictions)} & {int(known)} & {fmt_float(float(precision))} & {fmt_float(float(lift))} & {int(new_pred)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out_path.write_text("\n".join(lines))


def write_deciles_tabular(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"\multicolumn{1}{c}{Decile} &",
        r"\multicolumn{1}{c}{$n$} &",
        r"\multicolumn{1}{c}{hits} &",
        r"\multicolumn{1}{c}{Precision} &",
        r"\multicolumn{1}{c}{Lift} \\",
        r"\midrule",
    ]
    for decile, n, hits, precision, lift in df.sort_values("Decile").itertuples(
        index=False, name=None
    ):
        lines.append(
            f"{int(decile)} & {int(n)} & {int(hits)} & {float(precision):.6f} & {fmt_float(float(lift))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out_path.write_text("\n".join(lines))


def write_agreement_tabular(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{ccccr}",
        r"\toprule",
        r"Agree (models>=0.5) & n & hits & Precision & Lift \\",
        r"\midrule",
    ]
    for agree, n, hits, precision, lift in df.sort_values(
        "Agree (models>=0.5)", ascending=False
    ).itertuples(index=False, name=None):
        lines.append(
            f"{int(agree)} & {int(n)} & {int(hits)} & {fmt_float(float(precision))} & {fmt_float(float(lift))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    meta_scores = artifact_root / "meta_scores_zero_slices.parquet"
    meta_inputs = artifact_root / "meta_inputs_zero_slices.parquet"
    topk_files = {
        5: artifact_root / "top5_all.parquet",
        10: artifact_root / "top10_all.parquet",
        50: artifact_root / "top50_all.parquet",
    }
    for p in [meta_scores, meta_inputs, *topk_files.values()]:
        if not p.exists():
            raise FileNotFoundError(p)

    con = duckdb.connect(database=":memory:")
    base_rate = compute_base_rate(con, meta_scores)

    topk = build_topk_table(con, base_rate, topk_files)
    deciles = build_decile_table(con, base_rate, meta_scores)
    agreement = build_agreement_table(con, base_rate, meta_inputs)
    con.close()

    write_topk_tabular(topk, out_dir / "tab_3.1_topk_precision_lift.tex")
    write_deciles_tabular(deciles, out_dir / "tab_3.2_table_deciles.tex")
    write_agreement_tabular(agreement, out_dir / "tab_3.3_model_agreement_precision.tex")
    print(f"[done] wrote Chapter 3 tabulars to {out_dir}")


if __name__ == "__main__":
    main()
