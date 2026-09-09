#!/usr/bin/env python3
"""
Core Pipeline - Phase 6: Build Node Features as-of T0

Extracts simple, factual node features from FactSet DB strictly as-of the train
cutoff (T0). Fits encoders on train nodes only, then transforms the full
node_id universe for consistent inference across splits.

Outputs (under --out-root):
  - features/node_features_T0.parquet  (N rows = mapping size)
  - features/encoders_T0.pkl           (categorical maps, numeric min/max)
  - meta/feature_schema.json           (schema, dtypes, as-of date)

Feature set:
  - Industry: primary_sic_code, industry_code, sector_code
  - RBICS: l1_id, l2_id, l3_id, l3_count (unique L3s)
  - Entity: entity_type
  - Geography: gr_country, gr_region, gr_continent (latest report <= T0)

Encoders:
  - Categorical: label-encode (Unknown=0), mappings fit on train nodes
  - Numeric: min-max on train nodes (scaled to [0,1])
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import urllib.parse
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")
logger = logging.getLogger("core.features")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build as-of T0 node features from FactSet DB")
    p.add_argument("--out-root", required=True, type=str, help="Run-scoped output root")
    p.add_argument(
        "--t0-date",
        type=str,
        default=None,
        help="Override T0 date (YYYY-MM-DD); default from temporal_splits.json",
    )
    p.add_argument("--chunk-size", type=int, default=5000, help="Batch size for SQL IN queries")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _setup_logger(level: str) -> None:
    try:
        logging.getLogger().setLevel(getattr(logging, level.upper()))
    except Exception:
        logging.getLogger().setLevel(logging.INFO)


def _engine_from_env():
    load_dotenv()
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_DATABASE")
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    if not all([server, database, username, password]):
        raise ValueError("Missing DB env vars: DB_SERVER, DB_DATABASE, DB_USERNAME, DB_PASSWORD")
    params = urllib.parse.quote_plus(
        f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};"
        f"Trusted_Connection=no;Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=30;"
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")


def _load_mapping(out_root: Path) -> pd.DataFrame:
    mpath = out_root / "mapping" / "entity_map.parquet"
    if not mpath.exists():
        raise FileNotFoundError(f"Mapping not found: {mpath}")
    df = pd.read_parquet(mpath)
    # Expect columns: raw_id, canonical_id, node_id
    df = df[["raw_id", "node_id"]].astype({"raw_id": str, "node_id": int})
    return df  # type: ignore[return-value]


def _load_t0(out_root: Path, override: str | None) -> pd.Timestamp:
    if override:
        result = pd.to_datetime(override).normalize()
        if not isinstance(result, pd.Timestamp):
            raise ValueError(f"t0 override resolved to non-Timestamp: {result!r}")
        return result
    meta_path = out_root / "meta" / "temporal_splits.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Temporal splits meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    t0_days = int(meta["boundaries"]["T0_end"])  # epoch-days
    epoch = pd.Timestamp("1970-01-01")
    return (epoch + pd.Timedelta(days=t0_days)).normalize()  # type: ignore[return-value]


def _train_nodes(out_root: Path) -> np.ndarray:
    tpath = out_root / "splits" / "train_edges.parquet"
    if not tpath.exists():
        raise FileNotFoundError(f"Train split not found: {tpath}")
    df = pd.read_parquet(tpath, columns=["src_id", "dst_id"])
    nodes = pd.unique(pd.concat([df["src_id"], df["dst_id"]], ignore_index=True))
    return nodes.astype(np.int64)  # type: ignore[union-attr]


def _chunked(lst: list[str], size: int) -> Generator[list[str], None, None]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _q_industry(entity_list_sql: str) -> str:
    return (
        "SELECT factset_entity_id, primary_sic_code, industry_code, sector_code "
        "FROM sym_v1.sym_entity_sector WHERE factset_entity_id IN ('" + entity_list_sql + "')"
    )


def _q_entity(entity_list_sql: str) -> str:
    return (
        "SELECT factset_entity_id, entity_type FROM sym_v1.sym_entity "
        "WHERE factset_entity_id IN ('" + entity_list_sql + "')"
    )


def _q_rbics(entity_list_sql: str) -> str:
    return (
        "SELECT esr.factset_entity_id, rs.l1_id, rs.l2_id, rs.l3_id "
        "FROM sym_v1.sym_entity_sector_rbics esr "
        "JOIN rbics_v1.rbics_structure rs ON esr.l2_id = rs.l2_id "
        "WHERE esr.factset_entity_id IN ('" + entity_list_sql + "')"
    )


def _q_gr_country(entity_list_sql: str, t0: str) -> str:
    return (
        "WITH latest AS ("
        "  SELECT r.factset_entity_id, MAX(r.period_end_date) AS dt"
        "  FROM gr_v2.gr_report r WHERE r.period_end_date <= '" + t0 + "' "
        "  GROUP BY r.factset_entity_id"
        "), c AS ("
        "  SELECT r.factset_entity_id, i.iso_country, i.est_pct"
        "  FROM gr_v2.gr_item i JOIN gr_v2.gr_report r ON i.report_id = r.report_id"
        "  JOIN latest l ON r.factset_entity_id=l.factset_entity_id AND r.period_end_date=l.dt"
        "  WHERE i.est_pct > 0 AND r.factset_entity_id IN ('" + entity_list_sql + "')"
        ") SELECT * FROM c"
    )


def _q_gr_region(entity_list_sql: str, t0: str) -> str:
    return (
        "WITH latest AS ("
        "  SELECT r.factset_entity_id, MAX(r.period_end_date) AS dt"
        "  FROM gr_v2.gr_report r WHERE r.period_end_date <= '" + t0 + "' "
        "  GROUP BY r.factset_entity_id"
        "), x AS ("
        "  SELECT r.factset_entity_id, i4.region_id, i4.est_pct, rs.layer_number, rs.path"
        "  FROM gr_v2.gr_item_4tier i4"
        "  JOIN gr_v2.gr_report r ON i4.report_id = r.report_id"
        "  JOIN gr_v2.gr_region_structure_4t rs ON i4.region_id = rs.region_id"
        "  JOIN latest l ON r.factset_entity_id=l.factset_entity_id AND r.period_end_date=l.dt"
        "  WHERE i4.est_pct > 0 AND r.factset_entity_id IN ('" + entity_list_sql + "')"
        ") SELECT * FROM x"
    )


def _extract_features_db(
    engine, entity_ids: list[str], t0_date: pd.Timestamp, chunk_size: int
) -> pd.DataFrame:
    t0s = t0_date.strftime("%Y-%m-%d")
    pd.DataFrame({"factset_entity_id": entity_ids})

    # Industry
    frames = []
    for ch in _chunked(entity_ids, chunk_size):
        q = _q_industry("','".join(ch))
        try:
            frames.append(pd.read_sql(q, engine))
        except Exception as e:
            logger.warning(f"Industry chunk failed: {e}")
    df_ind = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["factset_entity_id", "primary_sic_code", "industry_code", "sector_code"]  # type: ignore[call-overload]
        )
    )

    # Entity type
    frames = []
    for ch in _chunked(entity_ids, chunk_size):
        q = _q_entity("','".join(ch))
        try:
            frames.append(pd.read_sql(q, engine))
        except Exception as e:
            logger.warning(f"Entity chunk failed: {e}")
    df_ent = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["factset_entity_id", "entity_type"])  # type: ignore[call-overload]
    )

    # RBICS basic
    frames = []
    for ch in _chunked(entity_ids, chunk_size):
        q = _q_rbics("','".join(ch))
        try:
            frames.append(pd.read_sql(q, engine))
        except Exception as e:
            logger.warning(f"RBICS chunk failed: {e}")
    df_rbics = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["factset_entity_id", "l1_id", "l2_id", "l3_id"])  # type: ignore[call-overload]
    )
    if not df_rbics.empty:
        l3c = df_rbics.groupby("factset_entity_id")["l3_id"].nunique().reset_index(name="l3_count")  # type: ignore[call-overload]
        df_rbics = (
            df_rbics.groupby("factset_entity_id", as_index=False)
            .first()
            .merge(l3c, on="factset_entity_id", how="left")
        )

    # Geography (as-of latest report <= T0)
    # Country
    frames = []
    for ch in _chunked(entity_ids, chunk_size):
        q = _q_gr_country("','".join(ch), t0s)
        try:
            frames.append(pd.read_sql(q, engine))
        except Exception as e:
            logger.warning(f"GR country chunk failed: {e}")
    df_country = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["factset_entity_id", "iso_country", "est_pct"])  # type: ignore[call-overload]
    )
    # Region/Continent
    frames = []
    for ch in _chunked(entity_ids, chunk_size):
        q = _q_gr_region("','".join(ch), t0s)
        try:
            frames.append(pd.read_sql(q, engine))
        except Exception as e:
            logger.warning(f"GR region chunk failed: {e}")
    df_reg = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["factset_entity_id", "region_id", "est_pct", "layer_number", "path"]  # type: ignore[call-overload]
        )
    )

    # Reduce geography to primary selections
    if not df_country.empty:
        # pick max est_pct per entity if >= 10, else Unknown
        country_primary = (
            df_country.sort_values(["factset_entity_id", "est_pct", "iso_country"])
            .groupby("factset_entity_id")
            .tail(1)
        )
        geo = country_primary[["factset_entity_id", "iso_country"]].rename(
            columns={"iso_country": "gr_country"}  # type: ignore[call-overload]
        )
    else:
        geo = pd.DataFrame(columns=["factset_entity_id", "gr_country"])  # type: ignore[call-overload]

    if not df_reg.empty:
        r2 = df_reg[df_reg["layer_number"] == 2].copy()
        r2["gr_region"] = r2["path"].str.split(">").str[-1].str.strip()  # type: ignore[union-attr]
        region_primary = (
            r2.sort_values(["factset_entity_id", "est_pct", "gr_region"])  # type: ignore[call-overload]
            .groupby("factset_entity_id")
            .tail(1)
        )
        geo = geo.merge(
            region_primary[["factset_entity_id", "gr_region"]], on="factset_entity_id", how="left"
        )

        r1 = df_reg[df_reg["layer_number"] == 1].copy()
        r1["gr_continent"] = r1["path"].str.split(">").str[-1].str.strip()  # type: ignore[union-attr]
        cont_primary = (
            r1.sort_values(["factset_entity_id", "est_pct", "gr_continent"])  # type: ignore[call-overload]
            .groupby("factset_entity_id")
            .tail(1)
        )
        geo = geo.merge(
            cont_primary[["factset_entity_id", "gr_continent"]], on="factset_entity_id", how="left"
        )
    else:
        geo["gr_region"] = np.nan
        geo["gr_continent"] = np.nan

    # Standardize region/continent labels (as in prior pipeline)
    region_mapping = {
        "North America": "North America",
        "Asia": "Asia",
        "Latin America": "Latin America",
        "Europe": "Europe",
        "Africa": "Africa",
        "Middle East": "Middle East",
        "European Union": "Europe",
        "Oceania": "Asia",
    }
    continent_mapping = {
        "Americas": "Americas",
        "Asia": "Asia/Pacific",
        "Asia/Pacific": "Asia/Pacific",
        "Europe": "Europe",
        "Africa": "Africa/Middle East",
        "Middle East": "Africa/Middle East",
    }
    if not geo.empty:
        geo["gr_region"] = geo["gr_region"].map(region_mapping).fillna("Unknown")  # type: ignore[arg-type]
        geo["gr_continent"] = geo["gr_continent"].map(continent_mapping).fillna("Unknown")  # type: ignore[arg-type]

    # Merge all
    out = pd.DataFrame({"factset_entity_id": entity_ids})
    for d in (df_ind, df_ent, df_rbics, geo):
        if not d.empty:
            out = out.merge(d, on="factset_entity_id", how="left")
    return out


def _fit_encoders(
    train_feats: pd.DataFrame, cat_cols: list[str], num_cols: list[str]
) -> dict[str, Any]:
    enc: dict[str, Any] = {"categorical_mappings": {}, "numeric_minmax": {}, "unknown_token": 0}
    # Categoricals: map str values to ints starting at 1; Unknown=0
    for col in cat_cols:
        vals = train_feats[col].astype(str).fillna("Unknown")
        uniq = sorted(set(vals.tolist()))
        mapping = {v: i + 1 for i, v in enumerate(uniq) if v != "Unknown"}
        enc["categorical_mappings"][col] = mapping
    # Numerics: record min/max
    for col in num_cols:
        s = pd.to_numeric(train_feats[col], errors="coerce")
        s = s.dropna()  # type: ignore[union-attr]
        if len(s) == 0:
            enc["numeric_minmax"][col] = {"min": None, "max": None}
        else:
            enc["numeric_minmax"][col] = {"min": float(s.min()), "max": float(s.max())}  # type: ignore[call-overload, arg-type]
    return enc


def _transform_all(
    all_feats: pd.DataFrame, enc: dict[str, Any], cat_cols: list[str], num_cols: list[str]
) -> pd.DataFrame:
    df = all_feats.copy()
    # Categorical
    for col in cat_cols:
        mapping = enc["categorical_mappings"].get(col, {})
        s = df[col].astype(str).fillna("Unknown")
        df[col] = s.map(mapping).fillna(0).astype(np.int32)
    # Numeric scaling
    for col in num_cols:
        mm = enc["numeric_minmax"].get(col, {"min": None, "max": None})
        x = pd.to_numeric(df[col], errors="coerce")
        if mm["min"] is None or mm["max"] is None or mm["max"] <= mm["min"]:
            df[col] = 0.0
        else:
            denominator = mm["max"] - mm["min"]
            if np.isinf(denominator) or denominator == 0:
                df[col] = 0.0
            else:
                df[col] = ((x - mm["min"]) / denominator).fillna(0.0)
        df[col] = df[col].astype(np.float32)
    return df


def main() -> None:
    args = _parse_args()
    _setup_logger(args.log_level)

    out_root = Path(args.out_root)
    feat_dir = out_root / "features"
    meta_dir = out_root / "meta"
    if not args.dry_run:
        feat_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

    mapping = _load_mapping(out_root)
    node_count = int(mapping.shape[0])
    t0_date = _load_t0(out_root, args.t0_date)
    train_node_ids = _train_nodes(out_root)

    logger.info(f"As-of T0 date: {t0_date.date()}")
    logger.info(f"Mapping nodes: {node_count:,}; Train nodes: {train_node_ids.size:,}")

    if args.dry_run:
        logger.info("[dry-run] Would extract features from DB and fit encoders on train nodes only")
        return

    engine = _engine_from_env()

    # Extract raw features for all mapping entities
    entity_ids = mapping["raw_id"].astype(str).tolist()
    raw = _extract_features_db(engine, entity_ids, t0_date, args.chunk_size)
    logger.info(f"Raw features extracted: {len(raw):,} rows")

    # Join to mapping (assign node_id)
    all_feats = mapping.merge(raw, left_on="raw_id", right_on="factset_entity_id", how="left")
    all_feats = (
        all_feats.drop(columns=["factset_entity_id"])
        if "factset_entity_id" in all_feats.columns
        else all_feats
    )

    # Define feature columns
    cat_cols = [
        "entity_type",
        "gr_country",
        "gr_region",
        "gr_continent",
    ]
    num_cols = [
        "primary_sic_code",
        "industry_code",
        "sector_code",
        "l1_id",
        "l2_id",
        "l3_id",
        "l3_count",
    ]

    # Fit encoders on train nodes only
    train_mask = all_feats["node_id"].isin(train_node_ids.tolist())  # type: ignore[union-attr]
    enc = _fit_encoders(all_feats.loc[train_mask], cat_cols, num_cols)

    # Transform full universe
    transformed = _transform_all(all_feats, enc, cat_cols, num_cols)
    # Final DF with node_id first
    cols_order = ["node_id", *cat_cols, *num_cols]
    final_df = transformed[["node_id"] + [c for c in cols_order if c != "node_id"]].copy()
    final_df = final_df.sort_values("node_id").reset_index(drop=True)  # type: ignore[call-overload]

    # Acceptance checks
    assert final_df.shape[0] == node_count, "Feature row count must equal mapping size"
    assert final_df["node_id"].is_unique, "node_id must be unique"
    assert final_df["node_id"].iloc[0] == 0, "node_id should start at 0"

    # Save outputs
    feat_path = feat_dir / "node_features_T0.parquet"
    if feat_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing {feat_path}; use --force")
    final_df.to_parquet(feat_path, index=False)

    enc_path = feat_dir / "encoders_T0.pkl"
    with open(enc_path, "wb") as f:
        pickle.dump(enc, f)

    schema = {
        "as_of_date": str(t0_date.date()),
        "node_count": node_count,
        "categorical": cat_cols,
        "numeric": num_cols,
        "dtypes": {c: str(final_df[c].dtype) for c in final_df.columns if c != "node_id"},
        "fit": {
            "train_nodes": int(train_node_ids.size),
        },
        "paths": {
            "features": str(feat_path),
            "encoders": str(enc_path),
        },
    }
    (meta_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    logger.info("=== FEATURES AS-OF T0 COMPLETE ===")
    logger.info(f"Features: {feat_path}")
    logger.info(f"Encoders: {enc_path}")


if __name__ == "__main__":
    main()
