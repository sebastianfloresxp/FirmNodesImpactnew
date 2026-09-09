#!/usr/bin/env python3
"""Module 0.6: enrich Chapter 4 nodes with FactSet RBICS/SIC metadata.

This module is designed to support semiconductor-relevance codebook drafting.
It pulls:
  - SIC from sym_v1.sym_entity_sector
  - RBICS L2 from sym_v1.sym_entity_sector_rbics + ref_v2.rbics_structure_l2_curr
  - RBICS L4 (latest as-of snapshot) from rbics_v1 bus-segment tables
  - RBICS coverage flags from rbics_v1.rbics_coverage

Outputs under artifacts/.../m0_6:
  - node_industry_enrichment_factset.csv/.parquet
  - code_universe_rbics_l2.csv/.parquet
  - code_universe_rbics_l4.csv/.parquet
  - code_universe_sic_raw.csv/.parquet
  - run_metadata.json
  - manifest_m0_6.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
SRC_ROOT = THIS_FILE.parents[3]
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.append(p)

from db_client import run_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 0.6 FactSet industry enrichment")
    parser.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2_fix01.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="As-of date (YYYY-MM-DD) for RBICS bus-segment period filter; defaults to snapshot date from config",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Chunk size for SQL IN-list pulls"
    )
    parser.add_argument(
        "--topn",
        default="25,100",
        help="Comma-separated Top-N cutoffs used for impact prevalence columns (default: 25,100)",
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


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(  # nosec B607 -- git is a well-known system executable
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            or None
        )
    except Exception:
        return None


def discover_views(m6_1_dir: Path, cfg: dict[str, Any]) -> list[str]:
    configured = [str(v) for v in cfg.get("m6_1", {}).get("views", [])]
    if configured:
        return configured
    views: list[str] = []
    for p in sorted(m6_1_dir.glob("node_single_removal_impacts_*.csv")):
        view = p.stem.replace("node_single_removal_impacts_", "")
        if view:
            views.append(view)
    return views


def load_top_sets(
    m6_1_dir: Path, views: list[str], top_ns: list[int]
) -> tuple[dict[tuple[str, int], set[str]], dict[str, str]]:
    top_sets: dict[tuple[str, int], set[str]] = {}
    input_hashes: dict[str, str] = {}
    for view in views:
        path = m6_1_dir / f"node_single_removal_impacts_{view}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        input_hashes[str(path)] = file_sha256(path)
        df = pd.read_csv(path)
        metric = (
            "h1_log_obligation_any_support"
            if "h1_log_obligation_any_support" in df.columns
            else "h1_reach_loss"
        )
        ranked = df.sort_values([metric, "analysis_uid"], ascending=[False, True]).copy()
        ranked["analysis_uid"] = ranked["analysis_uid"].astype(str)
        for n in top_ns:
            top_sets[(view, n)] = set(ranked.head(n)["analysis_uid"].tolist())
    return top_sets, input_hashes


def quote_ids(ids: list[str]) -> str:
    # FactSet IDs are controlled IDs (alnum + hyphen); still escape single quotes defensively.
    quoted: list[str] = []
    for x in ids:
        safe = x.replace("'", "''")
        quoted.append("'" + safe + "'")
    return ", ".join(quoted)


def chunks(values: list[str], size: int):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def join_pipe(values: pd.Series) -> str:
    uniq = sorted(
        {
            str(v).strip()
            for v in values
            if pd.notna(v) and str(v).strip() and str(v).lower() != "nan"
        }
    )
    return "|".join(uniq) if uniq else ""


def pick_primary_l2(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        cols = [
            "factset_entity_id",
            "rbics_l2_primary_id",
            "rbics_l2_primary_name",
            "rbics_l1_primary_id",
            "rbics_l1_primary_name",
            "rbics_l2_primary_focus_flag",
            "rbics_l2_count",
            "rbics_l2_ids",
            "rbics_l2_names",
        ]
        return pd.DataFrame(columns=cols)

    work = df.copy()
    work["factset_entity_id"] = work["factset_entity_id"].astype(str)
    work["l2_id"] = work["l2_id"].astype(str)
    work["focus_flag"] = pd.to_numeric(work["focus_flag"], errors="coerce").fillna(0).astype(int)
    # Primary L2: focus_flag desc, then deterministic l2_id asc.
    primary = (
        work.sort_values(
            ["factset_entity_id", "focus_flag", "l2_id"], ascending=[True, False, True]
        )
        .drop_duplicates(subset=["factset_entity_id"], keep="first")
        .rename(
            columns={
                "l2_id": "rbics_l2_primary_id",
                "l2_name": "rbics_l2_primary_name",
                "l1_id": "rbics_l1_primary_id",
                "l1_name": "rbics_l1_primary_name",
                "focus_flag": "rbics_l2_primary_focus_flag",
            }
        )
    )

    agg = (
        work.groupby("factset_entity_id", as_index=False)
        .agg(
            rbics_l2_count=("l2_id", "nunique"),
            rbics_l2_ids=("l2_id", join_pipe),
            rbics_l2_names=("l2_name", join_pipe),
        )
        .copy()
    )
    out = primary[
        [
            "factset_entity_id",
            "rbics_l2_primary_id",
            "rbics_l2_primary_name",
            "rbics_l1_primary_id",
            "rbics_l1_primary_name",
            "rbics_l2_primary_focus_flag",
        ]
    ].merge(agg, on="factset_entity_id", how="left")
    return out


def pick_latest_l4(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "factset_entity_id",
        "rbics_l4_latest_period_end",
        "rbics_l4_count_latest",
        "rbics_l4_ids_latest",
        "rbics_l4_names_latest",
        "rbics_l4_top_id",
        "rbics_l4_top_name",
        "rbics_l4_top_revenue_pct",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    work = df.copy()
    work["factset_entity_id"] = work["factset_entity_id"].astype(str)
    work["period_end_date"] = pd.to_datetime(work["period_end_date"], errors="coerce")
    work = work[work["period_end_date"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=cols)
    work["l4_id"] = work["l4_id"].astype(str)
    work["revenue_pct"] = pd.to_numeric(work["revenue_pct"], errors="coerce")

    # Keep only latest period_end_date per entity.
    latest = work.groupby("factset_entity_id")["period_end_date"].transform("max")
    latest_df = work[work["period_end_date"] == latest].copy()

    # Top L4 by revenue_pct in latest report; tie-break on l4_id.
    top = (
        latest_df.assign(revenue_rank=latest_df["revenue_pct"].fillna(-1e12))
        .sort_values(["factset_entity_id", "revenue_rank", "l4_id"], ascending=[True, False, True])
        .drop_duplicates(subset=["factset_entity_id"], keep="first")
        .rename(
            columns={
                "l4_id": "rbics_l4_top_id",
                "l4_name": "rbics_l4_top_name",
                "revenue_pct": "rbics_l4_top_revenue_pct",
                "period_end_date": "rbics_l4_latest_period_end",
            }
        )
    )

    agg = (
        latest_df.groupby("factset_entity_id", as_index=False)
        .agg(
            rbics_l4_count_latest=("l4_id", "nunique"),
            rbics_l4_ids_latest=("l4_id", join_pipe),
            rbics_l4_names_latest=("l4_name", join_pipe),
        )
        .copy()
    )
    out = top[
        [
            "factset_entity_id",
            "rbics_l4_latest_period_end",
            "rbics_l4_top_id",
            "rbics_l4_top_name",
            "rbics_l4_top_revenue_pct",
        ]
    ].merge(agg, on="factset_entity_id", how="left")
    return out[cols]


def build_membership(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for row in enriched[
        [
            "analysis_uid",
            "sic_primary_code_raw",
            "rbics_l2_ids",
            "rbics_l2_names",
            "rbics_l4_ids_latest",
            "rbics_l4_names_latest",
        ]
    ].itertuples(index=False):
        uid = str(row.analysis_uid)

        # SIC
        sic = row.sic_primary_code_raw
        if pd.notna(sic):
            code = str(int(float(sic))).zfill(4)
            rows.append(
                {"analysis_uid": uid, "code_system": "sic_raw", "code_value": code, "code_name": ""}
            )

        # RBICS L2
        l2_ids = [
            x.strip()
            for x in str(row.rbics_l2_ids).split("|")
            if x and str(row.rbics_l2_ids).lower() != "nan" and x.strip()
        ]
        l2_names = [
            x.strip()
            for x in str(row.rbics_l2_names).split("|")
            if x and str(row.rbics_l2_names).lower() != "nan" and x.strip()
        ]
        if l2_ids:
            if len(l2_ids) == len(l2_names):
                for cid, cname in zip(l2_ids, l2_names, strict=False):
                    rows.append(
                        {
                            "analysis_uid": uid,
                            "code_system": "rbics_l2",
                            "code_value": cid,
                            "code_name": cname,
                        }
                    )
            else:
                for i, cid in enumerate(l2_ids):
                    cname = l2_names[i] if i < len(l2_names) else ""
                    rows.append(
                        {
                            "analysis_uid": uid,
                            "code_system": "rbics_l2",
                            "code_value": cid,
                            "code_name": cname,
                        }
                    )

        # RBICS L4 latest
        l4_ids = [
            x.strip()
            for x in str(row.rbics_l4_ids_latest).split("|")
            if x and str(row.rbics_l4_ids_latest).lower() != "nan" and x.strip()
        ]
        l4_names = [
            x.strip()
            for x in str(row.rbics_l4_names_latest).split("|")
            if x and str(row.rbics_l4_names_latest).lower() != "nan" and x.strip()
        ]
        if l4_ids:
            if len(l4_ids) == len(l4_names):
                for cid, cname in zip(l4_ids, l4_names, strict=False):
                    rows.append(
                        {
                            "analysis_uid": uid,
                            "code_system": "rbics_l4_latest",
                            "code_value": cid,
                            "code_name": cname,
                        }
                    )
            else:
                for i, cid in enumerate(l4_ids):
                    cname = l4_names[i] if i < len(l4_names) else ""
                    rows.append(
                        {
                            "analysis_uid": uid,
                            "code_system": "rbics_l4_latest",
                            "code_value": cid,
                            "code_name": cname,
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=["analysis_uid", "code_system", "code_value", "code_name"])
    out = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["analysis_uid", "code_system", "code_value", "code_name"])
        .copy()
    )
    return out


def aggregate_code_universe(
    membership: pd.DataFrame,
    enriched: pd.DataFrame,
    views: list[str],
    top_ns: list[int],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if membership.empty:
        return {
            "rbics_l2": pd.DataFrame(),
            "rbics_l4_latest": pd.DataFrame(),
            "sic_raw": pd.DataFrame(),
        }

    flags = enriched[
        ["analysis_uid", "is_prime", "is_semi"]
        + [c for c in enriched.columns if c.startswith("is_top")]
    ].copy()
    work = membership.merge(flags, on="analysis_uid", how="left")
    work["is_prime"] = work["is_prime"].fillna(False).astype(bool)
    work["is_semi"] = work["is_semi"].fillna(False).astype(bool)

    for system in ["rbics_l2", "rbics_l4_latest", "sic_raw"]:
        df = work[work["code_system"] == system].copy()
        if df.empty:
            out[system] = pd.DataFrame(
                columns=[
                    "code_system",
                    "code_value",
                    "code_name",
                    "node_count",
                    "prime_node_count",
                    "semi_node_count",
                ]
            )
            continue

        grouped = df.groupby(
            ["code_system", "code_value", "code_name"], dropna=False, as_index=False
        ).agg(
            node_count=("analysis_uid", "nunique"),
            prime_node_count=("is_prime", "sum"),
            semi_node_count=("is_semi", "sum"),
        )
        for view in views:
            for n in top_ns:
                col = f"is_top{n}_{view}"
                if col not in df.columns:
                    continue
                tmp = (
                    df.groupby(["code_system", "code_value", "code_name"], dropna=False)[col]
                    .sum()
                    .rename(f"top{n}_{view}_count")
                    .reset_index()
                )
                grouped = grouped.merge(
                    tmp, on=["code_system", "code_value", "code_name"], how="left"
                )
                grouped[f"top{n}_{view}_count"] = (
                    grouped[f"top{n}_{view}_count"].fillna(0).astype(int)
                )
                grouped[f"top{n}_{view}_share_within_code"] = np.where(
                    grouped["node_count"] > 0,
                    grouped[f"top{n}_{view}_count"] / grouped["node_count"],
                    0.0,
                )

        top25_cols = [c for c in grouped.columns if c.startswith("top25_") and c.endswith("_count")]
        top100_cols = [
            c for c in grouped.columns if c.startswith("top100_") and c.endswith("_count")
        ]
        grouped["top25_any_view_count"] = (
            grouped[top25_cols].sum(axis=1).astype(int) if top25_cols else 0
        )
        grouped["top100_any_view_count"] = (
            grouped[top100_cols].sum(axis=1).astype(int) if top100_cols else 0
        )
        grouped = grouped.sort_values(
            ["top100_any_view_count", "top25_any_view_count", "node_count", "code_value"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        out[system] = grouped

    return out


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    as_of = str(args.as_of_date or snapshot)

    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    m0_nodes_path = out_root / snapshot / "m0" / "node_table_contract.parquet"
    m6_1_dir = out_root / snapshot / "m6_1"
    entity_map_path = Path("data/processed/core/releases/core_v1/mapping/entity_map.parquet")
    out_dir = out_root / snapshot / "m0_6"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not m0_nodes_path.exists():
        raise FileNotFoundError(m0_nodes_path)
    if not entity_map_path.exists():
        raise FileNotFoundError(entity_map_path)

    top_ns = sorted({int(x.strip()) for x in str(args.topn).split(",") if x.strip()})
    if not top_ns:
        raise ValueError("topn must contain at least one integer")

    nodes = pd.read_parquet(
        m0_nodes_path,
        columns=[
            "analysis_uid",
            "entity_role",
            "name",
            "rep_scr_node_id",
            "rep_factset_entity_id",
            "has_prime_vendor",
            "is_semi_strict",
        ],
    ).copy()
    nodes["analysis_uid"] = nodes["analysis_uid"].astype(str)
    nodes["rep_scr_node_id"] = pd.to_numeric(nodes["rep_scr_node_id"], errors="coerce").astype(
        "Int64"
    )
    nodes["is_prime"] = nodes["has_prime_vendor"].fillna(False).astype(bool)
    nodes["is_semi"] = nodes["is_semi_strict"].fillna(False).astype(bool)

    emap = pd.read_parquet(entity_map_path, columns=["node_id", "canonical_id"]).drop_duplicates(
        "node_id"
    )
    emap["node_id"] = pd.to_numeric(emap["node_id"], errors="coerce").astype("Int64")
    nodes = nodes.merge(emap, left_on="rep_scr_node_id", right_on="node_id", how="left")
    nodes["factset_entity_id"] = (
        nodes["rep_factset_entity_id"].fillna(nodes["canonical_id"]).astype("string")
    )
    factset_ids = sorted(nodes["factset_entity_id"].dropna().astype(str).unique().tolist())

    views = discover_views(m6_1_dir=m6_1_dir, cfg=cfg)
    top_sets, impact_hashes = load_top_sets(m6_1_dir=m6_1_dir, views=views, top_ns=top_ns)
    for view in views:
        for n in top_ns:
            nodes[f"is_top{n}_{view}"] = nodes["analysis_uid"].isin(top_sets[(view, n)])

    print(f"[m0_6] mapped_factset_ids={len(factset_ids):,} chunk_size={args.chunk_size}")

    # 1) SIC pull
    sic_frames: list[pd.DataFrame] = []
    for chunk in chunks(factset_ids, args.chunk_size):
        q = (
            "SELECT factset_entity_id, primary_sic_code, industry_code, sector_code "
            "FROM sym_v1.sym_entity_sector "
            f"WHERE factset_entity_id IN ({quote_ids(chunk)})"
        )
        sic_frames.append(run_query(q))
    sic = (
        pd.concat(sic_frames, ignore_index=True)
        if sic_frames
        else pd.DataFrame(columns=["factset_entity_id"])
    )
    if not sic.empty:
        sic = sic.drop_duplicates(subset=["factset_entity_id"], keep="first").copy()
        for col in ["primary_sic_code", "industry_code", "sector_code"]:
            sic[col] = pd.to_numeric(sic[col], errors="coerce")
        sic = sic.rename(
            columns={
                "primary_sic_code": "sic_primary_code_raw",
                "industry_code": "sic_industry_code_raw",
                "sector_code": "sic_sector_code_raw",
            }
        )

    # 2) RBICS L2 pull
    l2_frames: list[pd.DataFrame] = []
    for chunk in chunks(factset_ids, args.chunk_size):
        q = (
            "SELECT r.factset_entity_id, r.l2_id, r.focus_flag, s.l1_id, s.l1_name, s.l2_name "
            "FROM sym_v1.sym_entity_sector_rbics r "
            "LEFT JOIN ref_v2.rbics_structure_l2_curr s ON r.l2_id = s.l2_id "
            f"WHERE r.factset_entity_id IN ({quote_ids(chunk)})"
        )
        l2_frames.append(run_query(q))
    l2_raw = (
        pd.concat(l2_frames, ignore_index=True)
        if l2_frames
        else pd.DataFrame(columns=["factset_entity_id"])
    )
    l2 = pick_primary_l2(l2_raw)

    # 3) RBICS L4 latest-as-of pull
    l4_frames: list[pd.DataFrame] = []
    for chunk in chunks(factset_ids, args.chunk_size):
        q = (
            "SELECT br.factset_entity_id, br.period_end_date, i.revenue_pct, s.l4_id, s.l4_name "
            "FROM rbics_v1.rbics_bus_seg_report br "
            "JOIN rbics_v1.rbics_bus_seg_item i ON br.report_id = i.report_id "
            "JOIN rbics_v1.rbics_structure s ON i.l6_id = s.l6_id "
            f"WHERE br.factset_entity_id IN ({quote_ids(chunk)}) "
            f"AND br.period_end_date <= '{as_of}'"
        )
        l4_frames.append(run_query(q))
    l4_raw = (
        pd.concat(l4_frames, ignore_index=True)
        if l4_frames
        else pd.DataFrame(columns=["factset_entity_id"])
    )
    l4 = pick_latest_l4(l4_raw)

    # 4) RBICS coverage flags
    cov_frames: list[pd.DataFrame] = []
    for chunk in chunks(factset_ids, args.chunk_size):
        q = (
            "SELECT factset_entity_id, actively_covered, focus_flag, revenue_flag, tradename_flag "
            "FROM rbics_v1.rbics_coverage "
            f"WHERE factset_entity_id IN ({quote_ids(chunk)})"
        )
        cov_frames.append(run_query(q))
    cov = (
        pd.concat(cov_frames, ignore_index=True)
        if cov_frames
        else pd.DataFrame(columns=["factset_entity_id"])
    )
    if not cov.empty:
        cov = cov.drop_duplicates(subset=["factset_entity_id"], keep="first").copy()
        cov = cov.rename(
            columns={
                "actively_covered": "rbics_cov_actively_covered",
                "focus_flag": "rbics_cov_focus_flag",
                "revenue_flag": "rbics_cov_revenue_flag",
                "tradename_flag": "rbics_cov_tradename_flag",
            }
        )
        for c in [
            "rbics_cov_actively_covered",
            "rbics_cov_focus_flag",
            "rbics_cov_revenue_flag",
            "rbics_cov_tradename_flag",
        ]:
            cov[c] = pd.to_numeric(cov[c], errors="coerce").astype("Int64")

    enriched = nodes.copy()
    if not sic.empty:
        enriched = enriched.merge(sic, on="factset_entity_id", how="left")
    if not l2.empty:
        enriched = enriched.merge(l2, on="factset_entity_id", how="left")
    if not l4.empty:
        enriched = enriched.merge(l4, on="factset_entity_id", how="left")
    if not cov.empty:
        enriched = enriched.merge(cov, on="factset_entity_id", how="left")

    enriched["has_sic_raw"] = enriched["sic_primary_code_raw"].notna()
    enriched["has_rbics_l2"] = enriched["rbics_l2_primary_id"].notna()
    enriched["has_rbics_l4_latest"] = enriched["rbics_l4_top_id"].notna()
    enriched["has_any_industry_label_factset"] = (
        enriched["has_sic_raw"] | enriched["has_rbics_l2"] | enriched["has_rbics_l4_latest"]
    )

    membership = build_membership(enriched)
    universes = aggregate_code_universe(
        membership=membership, enriched=enriched, views=views, top_ns=top_ns
    )

    # Write outputs
    enriched_csv = out_dir / "node_industry_enrichment_factset.csv"
    enriched_parquet = out_dir / "node_industry_enrichment_factset.parquet"
    rbics_l2_csv = out_dir / "code_universe_rbics_l2.csv"
    rbics_l2_parquet = out_dir / "code_universe_rbics_l2.parquet"
    rbics_l4_csv = out_dir / "code_universe_rbics_l4.csv"
    rbics_l4_parquet = out_dir / "code_universe_rbics_l4.parquet"
    sic_csv = out_dir / "code_universe_sic_raw.csv"
    sic_parquet = out_dir / "code_universe_sic_raw.parquet"
    membership_csv = out_dir / "node_code_membership_factset.csv"
    membership_parquet = out_dir / "node_code_membership_factset.parquet"

    enriched.to_csv(enriched_csv, index=False)
    enriched.to_parquet(enriched_parquet, index=False)
    universes["rbics_l2"].to_csv(rbics_l2_csv, index=False)
    universes["rbics_l2"].to_parquet(rbics_l2_parquet, index=False)
    universes["rbics_l4_latest"].to_csv(rbics_l4_csv, index=False)
    universes["rbics_l4_latest"].to_parquet(rbics_l4_parquet, index=False)
    universes["sic_raw"].to_csv(sic_csv, index=False)
    universes["sic_raw"].to_parquet(sic_parquet, index=False)
    membership.to_csv(membership_csv, index=False)
    membership.to_parquet(membership_parquet, index=False)

    input_hashes = {
        str(m0_nodes_path): file_sha256(m0_nodes_path),
        str(entity_map_path): file_sha256(entity_map_path),
        **impact_hashes,
    }
    outputs = {
        "node_industry_enrichment_factset_csv": str(enriched_csv),
        "node_industry_enrichment_factset_parquet": str(enriched_parquet),
        "node_code_membership_factset_csv": str(membership_csv),
        "node_code_membership_factset_parquet": str(membership_parquet),
        "code_universe_rbics_l2_csv": str(rbics_l2_csv),
        "code_universe_rbics_l2_parquet": str(rbics_l2_parquet),
        "code_universe_rbics_l4_csv": str(rbics_l4_csv),
        "code_universe_rbics_l4_parquet": str(rbics_l4_parquet),
        "code_universe_sic_raw_csv": str(sic_csv),
        "code_universe_sic_raw_parquet": str(sic_parquet),
    }

    summary = {
        "nodes_total": len(enriched),
        "mapped_factset_ids": len(factset_ids),
        "coverage_has_sic_raw": float(enriched["has_sic_raw"].mean()),
        "coverage_has_rbics_l2": float(enriched["has_rbics_l2"].mean()),
        "coverage_has_rbics_l4_latest": float(enriched["has_rbics_l4_latest"].mean()),
        "coverage_has_any_industry_label_factset": float(
            enriched["has_any_industry_label_factset"].mean()
        ),
        "rbics_l2_code_count": int(universes["rbics_l2"]["code_value"].nunique())
        if not universes["rbics_l2"].empty
        else 0,
        "rbics_l4_code_count": int(universes["rbics_l4_latest"]["code_value"].nunique())
        if not universes["rbics_l4_latest"].empty
        else 0,
        "sic_code_count": int(universes["sic_raw"]["code_value"].nunique())
        if not universes["sic_raw"].empty
        else 0,
    }

    run_metadata = {
        "module": "m0_6",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "platform": platform.platform(),
        "settings": {
            "as_of_date": as_of,
            "chunk_size": args.chunk_size,
            "views": views,
            "top_ns": top_ns,
        },
        "summary": summary,
    }
    run_metadata_path = out_dir / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m0_6",
        "snapshot": snapshot,
        "run_id": run_id,
        "inputs": input_hashes,
        "outputs": outputs,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "manifest_m0_6.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        "[m0_6] complete "
        f"nodes={summary['nodes_total']:,} "
        f"sic_cov={summary['coverage_has_sic_raw']:.3f} "
        f"rbics_l2_cov={summary['coverage_has_rbics_l2']:.3f} "
        f"rbics_l4_cov={summary['coverage_has_rbics_l4_latest']:.3f}"
    )
    print(f"[m0_6] outputs: {out_dir}")


if __name__ == "__main__":
    main()
