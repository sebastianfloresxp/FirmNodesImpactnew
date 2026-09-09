#!/usr/bin/env python3
"""
Chapter 3: Diagnostics for the Two-Tower cold-start layer.

Produces a compact, shareable readout (A–D) to understand whether the cold-start
predicted layer is behaving sensibly.

A) Feature diversity / missingness for cold primes (FactSet attributes at T0)
B) Hub dominance / candidate-composition effects in Top-K predictions
C) Destination concentration (top destination share, HHI/entropy, top dst names)
D) Score separation (within-vendor gaps across ranks)

Outputs:
  - artifacts/ch3/twotower/diagnostics_twotower_cold.json
  - artifacts/ch3/twotower/top_dst_top5.csv
  - artifacts/ch3/twotower/feature_tuple_counts.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ch3.prediction.score_twotower_cold_primes import (
    fetch_factset_attrs,
    load_cold_primes,
    load_encoder_bundle,
)
from src.db_client.connection import get_engine


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fixed_map_cat(mapping: dict[str, int], series: pd.Series) -> np.ndarray:
    # The encoders store missing as string "nan" (see encoders_T0.pkl).
    # Use pandas string dtype so NaNs stay NaN until fillna.
    s = series.astype("string").fillna("nan")
    # If something is genuinely unseen, fall back to the "nan" code when available.
    nan_code = mapping.get("nan", 0)
    codes = s.map(mapping).fillna(nan_code).astype("int64").to_numpy()
    return codes


def _current_map_cat(mapping: dict[str, int], series: pd.Series) -> np.ndarray:
    # Matches current behavior in score_twotower_cold_primes.py (problematic).
    s = series.astype(str).fillna("Unknown")
    codes = s.map(mapping).fillna(0).astype("int64").to_numpy()
    return codes


def _build_node_flags(
    *,
    semis_flags_path: Path,
    deg_out_path: Path,
    deg_in_path: Path,
    hub_top_frac: float,
) -> pd.DataFrame:
    semis = pd.read_parquet(
        semis_flags_path, columns=["node_id", "is_semi_core", "is_semi_adjacent"]
    ).copy()
    semis["node_id"] = semis["node_id"].astype("int64")

    outd = np.load(deg_out_path).astype(np.int64)
    ind = np.load(deg_in_path).astype(np.int64)
    deg = outd + ind
    n_nodes = int(deg.shape[0])

    k = max(1, int(n_nodes * float(hub_top_frac)))
    idx = np.argpartition(deg, -k)[-k:]
    hub_mask = np.zeros(n_nodes, dtype=bool)
    hub_mask[idx] = True

    flags = semis
    flags["is_hub"] = flags["node_id"].map(lambda x: bool(hub_mask[int(x)]))
    return flags


def _hhi_and_entropy(counts: np.ndarray) -> tuple[float, float]:
    total = float(counts.sum())
    if total <= 0:
        return 0.0, 0.0
    p = counts.astype(np.float64) / total
    hhi = float(np.sum(p * p))
    ent = float(-np.sum(np.where(p > 0, p * np.log(p), 0.0)))
    return hhi, ent


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostics for Two-Tower cold-start layer (A–D)")
    ap.add_argument(
        "--prime-matches",
        type=Path,
        default=Path("artifacts/ch3/matching/prime_matches_thr90.parquet"),
    )
    ap.add_argument(
        "--entity-map",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/mapping/entity_map.parquet"),
    )
    ap.add_argument(
        "--encoders",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/features/encoders_T0.pkl"),
    )
    ap.add_argument(
        "--semis-flags", type=Path, default=Path("artifacts/ch3/prediction/semis_flags.parquet")
    )
    ap.add_argument(
        "--deg-out",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/adjacency/out_degree.npy"),
    )
    ap.add_argument(
        "--deg-in",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/adjacency/in_degree.npy"),
    )
    ap.add_argument("--hub-top-frac", type=float, default=0.001)
    ap.add_argument("--as-of-date", type=str, default="2025-06-09")
    ap.add_argument("--version", type=str, default="twotower_cold_v2")
    ap.add_argument("--pool", type=str, default="semis_adjacent")
    ap.add_argument(
        "--iso-map", type=Path, default=Path("artifacts/ch3/reference/iso_region_continent_m49.csv")
    )
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--top50", type=Path, default=None)
    ap.add_argument("--edges-top5", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=Path("artifacts/ch3/twotower_cold_v2"))
    args = ap.parse_args()

    load_dotenv()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    version = str(args.version)
    pool = str(args.pool)

    if args.top50 is None:
        args.top50 = out_root / f"{version}_top50_{pool}.parquet"
    if args.edges_top5 is None:
        args.edges_top5 = Path(
            f"artifacts/ch3/network/edges_predicted_{version}_top5_{pool}.parquet"
        )

    # -------------------------
    # A) Feature diversity
    # -------------------------
    cold = load_cold_primes(args.prime_matches, args.entity_map)
    cold_fids = cold["factset_entity_id"].astype(str).tolist()
    unique_fids = sorted(set(cold_fids))

    engine = get_engine()
    raw = (
        fetch_factset_attrs(
            engine,
            unique_fids,
            as_of_date=str(args.as_of_date),
            chunk_size=int(args.chunk_size),
        )
        .drop_duplicates(subset=["factset_entity_id"])
        .rename(columns={"primary_sic_code": "primary_sic_raw"})
    )
    cold = cold.merge(raw, on="factset_entity_id", how="left", validate="m:1")

    # Raw missingness before fallback (GR-only)
    raw_missing = {
        "missing_gr_country_raw_frac": float(cold["gr_country"].isna().mean())
        if "gr_country" in cold
        else 1.0,
        "missing_gr_region_raw_frac": float(cold["gr_region"].isna().mean())
        if "gr_region" in cold
        else 1.0,
        "missing_gr_continent_raw_frac": float(cold["gr_continent"].isna().mean())
        if "gr_continent" in cold
        else 1.0,
    }

    # Apply the same geo fallback as cold-start inference (GR -> sym_entity.iso_country -> USAspending)
    iso_map = pd.read_csv(args.iso_map)
    iso_map["iso_country"] = iso_map["iso_country"].astype("string").str.strip()

    def _norm_iso(series: pd.Series) -> pd.Series:
        return series.astype("string").str.strip().str.upper()

    cold["gr_country"] = _norm_iso(cold.get("gr_country", pd.Series(dtype="string")))
    cold["entity_iso_country"] = _norm_iso(
        cold.get("entity_iso_country", pd.Series(dtype="string"))
    )
    cold["country_alpha2"] = _norm_iso(cold.get("country_alpha2", pd.Series(dtype="string")))
    cold["recipient_country_code"] = _norm_iso(
        cold.get("recipient_country_code", pd.Series(dtype="string"))
    )

    cold["gr_country"] = (
        cold["gr_country"]
        .fillna(cold["entity_iso_country"])
        .fillna(cold["country_alpha2"])
        .fillna(cold["recipient_country_code"])
    )
    cold["gr_country"] = cold["gr_country"].replace({"": pd.NA})

    cold = cold.merge(
        iso_map.rename(columns={"gr_region": "gr_region_iso", "gr_continent": "gr_continent_iso"}),
        left_on="gr_country",
        right_on="iso_country",
        how="left",
    )

    def _fill_geo(col: str, fallback_col: str) -> None:
        series = cold.get(col, pd.Series(dtype="string"))
        missing = series.isna() | series.isin(["Unknown", "nan", ""])
        cold.loc[missing, col] = cold.loc[missing, fallback_col]

    _fill_geo("gr_region", "gr_region_iso")
    _fill_geo("gr_continent", "gr_continent_iso")

    cold = cold.drop(
        columns=[
            c for c in ["iso_country", "gr_region_iso", "gr_continent_iso"] if c in cold.columns
        ]
    )

    enc = load_encoder_bundle(args.encoders)
    cat_maps = enc.get("categorical_mappings", {})
    maps = {k: {str(kk): int(vv) for kk, vv in cat_maps.get(k, {}).items()} for k in cat_maps}

    # Missingness (raw)
    a_missing = {
        "vendors": len(cold),
        "unique_factset_entities": len(unique_fids),
        "missing_primary_sic_raw_frac": float(cold["primary_sic_raw"].isna().mean())
        if "primary_sic_raw" in cold
        else 1.0,
        "missing_entity_type_frac": float(cold["entity_type"].isna().mean())
        if "entity_type" in cold
        else 1.0,
        "missing_gr_country_frac": float(cold["gr_country"].isna().mean())
        if "gr_country" in cold
        else 1.0,
        "missing_gr_region_frac": float(cold["gr_region"].isna().mean())
        if "gr_region" in cold
        else 1.0,
        "missing_gr_continent_frac": float(cold["gr_continent"].isna().mean())
        if "gr_continent" in cold
        else 1.0,
    }

    # Encode categorical attrs using (a) current logic, (b) fixed logic
    cur_country = _current_map_cat(
        maps.get("gr_country", {}), cold.get("gr_country", pd.Series(dtype="string"))
    )
    cur_region = _current_map_cat(
        maps.get("gr_region", {}), cold.get("gr_region", pd.Series(dtype="string"))
    )
    cur_cont = _current_map_cat(
        maps.get("gr_continent", {}), cold.get("gr_continent", pd.Series(dtype="string"))
    )
    cur_type = _current_map_cat(
        maps.get("entity_type", {}), cold.get("entity_type", pd.Series(dtype="string"))
    )

    fix_country = _fixed_map_cat(
        maps.get("gr_country", {}), cold.get("gr_country", pd.Series(dtype="string"))
    )
    fix_region = _fixed_map_cat(
        maps.get("gr_region", {}), cold.get("gr_region", pd.Series(dtype="string"))
    )
    fix_cont = _fixed_map_cat(
        maps.get("gr_continent", {}), cold.get("gr_continent", pd.Series(dtype="string"))
    )
    fix_type = _fixed_map_cat(
        maps.get("entity_type", {}), cold.get("entity_type", pd.Series(dtype="string"))
    )

    # Count how often encoders map to 0 (only happens under the current logic).
    a_zero_codes = {
        "current_zero_country_frac": float(np.mean(cur_country == 0)),
        "current_zero_region_frac": float(np.mean(cur_region == 0)),
        "current_zero_continent_frac": float(np.mean(cur_cont == 0)),
        "current_zero_entity_type_frac": float(np.mean(cur_type == 0)),
    }

    # Unique tuples (excluding SIC because it is handled separately in the twotower script)
    cur_tuples = pd.DataFrame(
        {
            "country": cur_country,
            "region": cur_region,
            "continent": cur_cont,
            "entity_type": cur_type,
        }
    )
    fix_tuples = pd.DataFrame(
        {
            "country": fix_country,
            "region": fix_region,
            "continent": fix_cont,
            "entity_type": fix_type,
        }
    )
    a_tuple_stats = {
        "unique_attr_tuples_current": int(cur_tuples.drop_duplicates().shape[0]),
        "unique_attr_tuples_fixed": int(fix_tuples.drop_duplicates().shape[0]),
        "top_tuple_share_current": float(cur_tuples.value_counts(normalize=True).iloc[0]),
        "top_tuple_share_fixed": float(fix_tuples.value_counts(normalize=True).iloc[0]),
    }

    # Export top tuples for quick eyeballing
    tuple_counts = (
        pd.concat(
            [
                cur_tuples.assign(version="current"),
                fix_tuples.assign(version="fixed"),
            ],
            ignore_index=True,
        )
        .groupby(["version", "country", "region", "continent", "entity_type"])
        .size()
        .reset_index(name="n")
        .sort_values(["version", "n"], ascending=[True, False])
    )
    tuple_counts.to_csv(out_root / f"feature_tuple_counts_{version}_{pool}.csv", index=False)

    # -------------------------
    # B–D) Output behavior
    # -------------------------
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")

    # Node flags: semis + hubs
    flags = _build_node_flags(
        semis_flags_path=args.semis_flags,
        deg_out_path=args.deg_out,
        deg_in_path=args.deg_in,
        hub_top_frac=float(args.hub_top_frac),
    )
    con.register("node_flags", flags)

    # Read Top-50 scored candidates (per vendor_key, rank<=50)
    top50_path = str(args.top50)
    edges_top5_path = str(args.edges_top5)

    n_top50 = int(con.sql(f"SELECT COUNT(*) FROM read_parquet('{top50_path}')").fetchone()[0])
    n_vendors = int(
        con.sql(f"SELECT COUNT(DISTINCT vendor_key) FROM read_parquet('{top50_path}')").fetchone()[
            0
        ]
    )

    # Top-5 slice from top50
    top5_df = con.sql(
        f"""
        SELECT vendor_key, dst_node_id, rank, score_raw, score_calib
        FROM read_parquet('{top50_path}')
        WHERE rank <= 5
        """
    ).df()
    # join node flags
    top5_df = top5_df.merge(flags, left_on="dst_node_id", right_on="node_id", how="left")

    b_comp = {
        "top5_edges": len(top5_df),
        "top5_dst_distinct": int(top5_df["dst_node_id"].nunique()),
        "frac_dst_is_hub": float(top5_df["is_hub"].fillna(False).mean()),
        "frac_dst_is_semi_core": float(top5_df["is_semi_core"].fillna(False).mean()),
        "frac_dst_is_semi_adjacent": float(top5_df["is_semi_adjacent"].fillna(False).mean()),
    }
    b_comp["frac_dst_is_semi_any"] = float(
        (top5_df["is_semi_core"].fillna(False) | top5_df["is_semi_adjacent"].fillna(False)).mean()
    )

    # Destination concentration
    dst_counts = top5_df.groupby("dst_node_id").size().sort_values(ascending=False)
    counts_np = dst_counts.to_numpy(dtype=np.int64)
    hhi, ent = _hhi_and_entropy(counts_np)
    c_conc = {
        "vendors": int(n_vendors),
        "top5_edges": len(top5_df),
        "distinct_dst_top5": int(dst_counts.shape[0]),
        "top1_dst_share": float(dst_counts.iloc[0] / float(len(top5_df))) if len(top5_df) else 0.0,
        "top10_dst_share": float(dst_counts.iloc[:10].sum() / float(len(top5_df)))
        if len(top5_df)
        else 0.0,
        "hhi_top5": float(hhi),
        "entropy_top5": float(ent),
    }

    # Map top destinations to FactSet names
    # node_id -> canonical_id (FactSet entity id) -> sym_entity.entity_proper_name
    entity_map = pd.read_parquet(args.entity_map, columns=["canonical_id", "node_id"]).copy()
    entity_map["node_id"] = entity_map["node_id"].astype("int64")
    entity_map["canonical_id"] = entity_map["canonical_id"].astype(str)
    top_dst = dst_counts.head(50).reset_index().rename(columns={0: "n_edges"})
    top_dst = top_dst.merge(entity_map, left_on="dst_node_id", right_on="node_id", how="left")

    # Fetch names for those 50 IDs
    top_fids = list(top_dst["canonical_id"].dropna().astype(str).unique().tolist())
    if top_fids:
        # SQL Server has a 2100-parameter limit; but 50 is safe.
        fid_list = "','".join(top_fids)
        q = (
            "SELECT factset_entity_id, entity_proper_name, iso_country, entity_type "
            "FROM sym_v1.sym_entity WHERE factset_entity_id IN ('" + fid_list + "')"
        )
        name_df = pd.read_sql(q, engine)
    else:
        name_df = pd.DataFrame(
            columns=["factset_entity_id", "entity_proper_name", "iso_country", "entity_type"]
        )
    name_df["factset_entity_id"] = name_df["factset_entity_id"].astype(str)
    top_dst = top_dst.merge(
        name_df, left_on="canonical_id", right_on="factset_entity_id", how="left"
    )

    # Fraction of vendors whose top-5 contains each top destination
    vendor_hit = top5_df[["vendor_key", "dst_node_id"]].drop_duplicates()
    vendor_count_by_dst = vendor_hit.groupby("dst_node_id").size().reindex(dst_counts.index)
    top_dst["vendors_with_in_top5"] = (
        top_dst["dst_node_id"].map(vendor_count_by_dst).fillna(0).astype(int)
    )
    top_dst["share_vendors"] = (
        top_dst["vendors_with_in_top5"] / float(n_vendors) if n_vendors else 0.0
    )

    top_dst_out = top_dst[
        [
            "dst_node_id",
            "n_edges",
            "vendors_with_in_top5",
            "share_vendors",
            "canonical_id",
            "entity_proper_name",
            "iso_country",
            "entity_type",
        ]
    ].copy()
    top_dst_out.to_csv(out_root / f"top_dst_top5_{version}_{pool}.csv", index=False)

    # Score separation
    # Pull per-vendor top1/top2/top50 gaps using duckdb for speed
    gaps = con.sql(
        f"""
        SELECT
          vendor_key,
          MAX(CASE WHEN rank=1 THEN score_raw END) AS s1,
          MAX(CASE WHEN rank=2 THEN score_raw END) AS s2,
          MAX(CASE WHEN rank=50 THEN score_raw END) AS s50,
          MAX(CASE WHEN rank=1 THEN score_calib END) AS p1,
          MAX(CASE WHEN rank=2 THEN score_calib END) AS p2,
          MAX(CASE WHEN rank=50 THEN score_calib END) AS p50
        FROM read_parquet('{top50_path}')
        GROUP BY vendor_key
        """
    ).df()
    gaps["gap_raw_1_2"] = gaps["s1"] - gaps["s2"]
    gaps["gap_raw_1_50"] = gaps["s1"] - gaps["s50"]
    gaps["gap_calib_1_2"] = gaps["p1"] - gaps["p2"]
    gaps["gap_calib_1_50"] = gaps["p1"] - gaps["p50"]

    def pct(x: pd.Series) -> dict[str, float]:
        qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
        out = {}
        arr = x.dropna().to_numpy(dtype=np.float64)
        if arr.size == 0:
            return {f"p{int(q * 100)}": float("nan") for q in qs}
        for q in qs:
            out[f"p{int(q * 100)}"] = float(np.quantile(arr, q))
        out["mean"] = float(np.mean(arr))
        return out

    d_sep = {
        "n_vendors": int(gaps.shape[0]),
        "gap_raw_1_2": pct(gaps["gap_raw_1_2"]),
        "gap_raw_1_50": pct(gaps["gap_raw_1_50"]),
        "gap_calib_1_2": pct(gaps["gap_calib_1_2"]),
        "gap_calib_1_50": pct(gaps["gap_calib_1_50"]),
    }

    # Hub dominance in final Top-5 edges file (string uids) for consistency check
    # src_uid is vendor:<vendor_key>, dst_uid is scr:<node_id>
    # Join by parsing dst_uid -> node_id.
    b_edges_file = (
        con.sql(
            f"""
        SELECT
          COUNT(*) AS n,
          SUM(CASE WHEN nf.is_hub THEN 1 ELSE 0 END) AS hub_n,
          SUM(CASE WHEN nf.is_semi_core THEN 1 ELSE 0 END) AS semi_core_n,
          SUM(CASE WHEN nf.is_semi_adjacent THEN 1 ELSE 0 END) AS semi_adj_n
        FROM (
          SELECT CAST(SUBSTR(dst_uid, 5) AS BIGINT) AS node_id
          FROM read_parquet('{edges_top5_path}')
        ) e
        LEFT JOIN node_flags nf ON e.node_id = nf.node_id
        """
        )
        .df()
        .iloc[0]
        .to_dict()
    )
    b_edges_file = {k: int(v) for k, v in b_edges_file.items()}
    b_edges_file["hub_frac"] = (
        float(b_edges_file["hub_n"] / b_edges_file["n"]) if b_edges_file["n"] else 0.0
    )
    b_edges_file["semi_any_frac"] = (
        float((b_edges_file["semi_core_n"] + b_edges_file["semi_adj_n"]) / b_edges_file["n"])
        if b_edges_file["n"]
        else 0.0
    )

    out = {
        "created_at": _now_iso(),
        "version": version,
        "pool": pool,
        "inputs": {
            "prime_matches": str(args.prime_matches),
            "entity_map": str(args.entity_map),
            "top50": str(args.top50),
            "edges_top5": str(args.edges_top5),
            "as_of_date": str(args.as_of_date),
            "hub_top_frac": float(args.hub_top_frac),
        },
        "A_missingness": a_missing,
        "A_missingness_raw": raw_missing,
        "A_zero_codes_current": a_zero_codes,
        "A_tuple_stats": a_tuple_stats,
        "B_composition_top5": b_comp,
        "B_composition_edges_file_top5": b_edges_file,
        "C_concentration_top5": c_conc,
        "D_score_separation": d_sep,
        "sanity": {
            "top50_rows": int(n_top50),
            "top50_vendors": int(n_vendors),
        },
    }

    (out_root / f"diagnostics_{version}_{pool}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
