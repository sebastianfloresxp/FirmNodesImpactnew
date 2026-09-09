#!/usr/bin/env python3
"""
Chapter 3: Two-Tower cold-start inference for Tier-1 primes matched to FactSet but not in SCR.

This script reuses the trained Chapter 2 Two-Tower model checkpoint (no retraining) and
produces a new predicted edge layer:
  vendor:<vendor_key>  ->  scr:<node_id>

Candidate strategy (Option A):
  - semis (core + optional adjacent)
  - hubs (top fraction by degree)
  - random tail
with a per-source budget (default 10k), then rank-based Top-K selection (default K=5/10/50).

Outputs (default under artifacts/ch3/twotower and artifacts/ch3/network):
  - artifacts/ch3/twotower/twotower_cold_top50.parquet
  - artifacts/ch3/network/edges_predicted_twotower_top{K}.parquet
  - artifacts/ch3/network/dod_network_edges_top5_with_twotower.parquet
  - artifacts/ch3/twotower/summary_twotower_cold.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.db_client.connection import get_engine
from src.ensemble.utils import apply_platt
from src.twotower.run_twotower_eval import TwoTowerModel
from src.twotower.run_twotower_eval import load_features as tt_load_features


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stable_int_hash(s: str) -> int:
    # Deterministic across runs/processes.
    h = 2166136261
    for ch in s.encode("utf-8", errors="ignore"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def load_country_mapping(encoders_path: Path) -> dict[str, int]:
    import pickle  # nosec B403 -- internal ML artifacts only

    with open(encoders_path, "rb") as fh:
        obj = pickle.load(fh)  # nosec B301 -- internal ML artifacts only
    cm = obj.get("categorical_mappings", {})
    return {str(k): int(v) for k, v in cm.get("gr_country", {}).items()}


def load_encoder_bundle(encoders_path: Path) -> dict[str, object]:
    import pickle  # nosec B403 -- internal ML artifacts only

    with open(encoders_path, "rb") as fh:
        return pickle.load(fh)  # nosec B301 -- internal ML artifacts only


def _chunked(lst: list[str], size: int) -> list[list[str]]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _q_industry(entity_list_sql: str) -> str:
    return (
        "SELECT factset_entity_id, primary_sic_code "
        "FROM sym_v1.sym_entity_sector WHERE factset_entity_id IN ('" + entity_list_sql + "')"
    )


def _q_entity(entity_list_sql: str) -> str:
    return (
        "SELECT factset_entity_id, entity_type, iso_country FROM sym_v1.sym_entity "
        "WHERE factset_entity_id IN ('" + entity_list_sql + "')"
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


def fetch_factset_attrs(
    engine,
    entity_ids: list[str],
    *,
    as_of_date: str,
    chunk_size: int,
) -> pd.DataFrame:
    # Mirrors src/data_processing/core/06_build_features_T0.py for the columns we need.
    frames = []
    for ch in _chunked(entity_ids, chunk_size):
        q = _q_industry("','".join(ch))
        frames.append(pd.read_sql(q, engine))
    df_ind = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["factset_entity_id", "primary_sic_code"])
    )

    frames = []
    for ch in _chunked(entity_ids, chunk_size):
        q = _q_entity("','".join(ch))
        frames.append(pd.read_sql(q, engine))
    df_ent = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["factset_entity_id", "entity_type"])
    )

    frames = []
    for ch in _chunked(entity_ids, chunk_size):
        q = _q_gr_country("','".join(ch), as_of_date)
        frames.append(pd.read_sql(q, engine))
    df_country = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["factset_entity_id", "iso_country", "est_pct"])
    )

    frames = []
    for ch in _chunked(entity_ids, chunk_size):
        q = _q_gr_region("','".join(ch), as_of_date)
        frames.append(pd.read_sql(q, engine))
    df_reg = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["factset_entity_id", "region_id", "est_pct", "layer_number", "path"]
        )
    )

    # Primary country: highest est_pct
    if not df_country.empty:
        country_primary = (
            df_country.sort_values(["factset_entity_id", "est_pct", "iso_country"])
            .groupby("factset_entity_id")
            .tail(1)
        )
        geo = country_primary[["factset_entity_id", "iso_country"]].rename(
            columns={"iso_country": "gr_country"}
        )
    else:
        geo = pd.DataFrame(columns=["factset_entity_id", "gr_country"])

    # Region/continent from 4-tier structure (same mapping as core pipeline)
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

    if not df_reg.empty:
        r2 = df_reg[df_reg["layer_number"] == 2].copy()
        r2["gr_region"] = r2["path"].str.split(">").str[-1].str.strip()
        region_primary = (
            r2.sort_values(["factset_entity_id", "est_pct", "gr_region"])
            .groupby("factset_entity_id")
            .tail(1)
        )
        geo = geo.merge(
            region_primary[["factset_entity_id", "gr_region"]], on="factset_entity_id", how="left"
        )

        r1 = df_reg[df_reg["layer_number"] == 1].copy()
        r1["gr_continent"] = r1["path"].str.split(">").str[-1].str.strip()
        cont_primary = (
            r1.sort_values(["factset_entity_id", "est_pct", "gr_continent"])
            .groupby("factset_entity_id")
            .tail(1)
        )
        geo = geo.merge(
            cont_primary[["factset_entity_id", "gr_continent"]], on="factset_entity_id", how="left"
        )

    geo["gr_region"] = (
        geo.get("gr_region", pd.Series(dtype="string")).map(region_mapping).fillna("Unknown")
    )
    geo["gr_continent"] = (
        geo.get("gr_continent", pd.Series(dtype="string")).map(continent_mapping).fillna("Unknown")
    )

    if "iso_country" in df_ent.columns:
        df_ent = df_ent.rename(columns={"iso_country": "entity_iso_country"})
    out = pd.DataFrame({"factset_entity_id": entity_ids})
    for d in (df_ind, df_ent, geo):
        if not d.empty:
            out = out.merge(d, on="factset_entity_id", how="left")
    return out


def build_primary_sic_string_to_code(node_features_path: Path) -> dict[str, int]:
    # Reproduce the factorization used by src/twotower/run_twotower_eval.py::load_features
    # for float-like 'primary_sic_code' (stored as float32 in node_features_T0.parquet).
    df = pd.read_parquet(node_features_path, columns=["node_id", "primary_sic_code"])
    df = df.sort_values("node_id").reset_index(drop=True)
    vals = df["primary_sic_code"].astype("string").fillna("<UNK>")
    cat = pd.Categorical(vals)
    return {str(cat.categories[i]): int(i) for i in range(len(cat.categories))}


def load_cold_primes(prime_matches: Path, entity_map: Path) -> pd.DataFrame:
    pm = pd.read_parquet(prime_matches)
    em = pd.read_parquet(entity_map, columns=["canonical_id", "node_id"])
    em_ids = set(em["canonical_id"].astype(str).tolist())
    pm = pm[pm["factset_entity_id"].notna()].copy()
    pm["factset_entity_id"] = pm["factset_entity_id"].astype(str)
    cold = pm[~pm["factset_entity_id"].isin(em_ids)].copy()
    # one row per vendor_key (already unique in this file)
    keep_cols = [
        "vendor_key",
        "recipient_name",
        "recipient_country_code",
        "country_alpha2",
        "factset_entity_id",
        "match_method",
        "match_score",
    ]
    cold = cold[[c for c in keep_cols if c in cold.columns]].copy()
    return cold.reset_index(drop=True)


def load_semis(semis_flags: Path, include_adjacent: bool) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(semis_flags)
    core = df[df["is_semi_core"].astype(bool)]["node_id"].astype(int).to_numpy()
    if include_adjacent:
        adj = df[df["is_semi_adjacent"].astype(bool)]["node_id"].astype(int).to_numpy()
    else:
        adj = np.empty(0, dtype=np.int64)
    return core.astype(np.int64), adj.astype(np.int64)


def load_hubs(deg_out_path: Path, deg_in_path: Path, top_frac: float) -> np.ndarray:
    outd = np.load(deg_out_path)
    ind = np.load(deg_in_path)
    deg = outd.astype(np.int64) + ind.astype(np.int64)
    k = max(1, int(len(deg) * float(top_frac)))
    idx = np.argpartition(deg, -k)[-k:]
    # deterministic ordering: descending degree
    idx = idx[np.argsort(-deg[idx])]
    return idx.astype(np.int64)


def build_candidates_for_vendor(
    *,
    vendor_key: str,
    budget: int,
    semis_base: np.ndarray,
    hubs: np.ndarray,
    pool_mode: str,
    geo_gated: bool,
    geo_min_candidates: int,
    vendor_country_code: int,
    vendor_region_code: int,
    nan_country_code: int,
    nan_region_code: int,
    dest_by_country: dict[int, np.ndarray],
    dest_by_region: dict[int, np.ndarray],
    rng_seed: int,
    random_k: int,
    n_nodes: int,
) -> np.ndarray:
    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be > 0")

    # Priority: geo-gated semis -> global semis -> hubs -> random (when enabled)
    out: list[int] = []
    seen = set()

    def add_many(arr: np.ndarray, cap: int | None = None) -> None:
        nonlocal out, seen
        if cap is None:
            cap = len(arr)
        added = 0
        for x in arr[:cap]:
            xi = int(x)
            if xi in seen:
                continue
            seen.add(xi)
            out.append(xi)
            added += 1
            if len(out) >= budget:
                return
            if added >= cap:
                return

    # Geo-gated semis: choose country if enough, else region if enough, else global
    if geo_gated:
        cand = np.empty(0, dtype=np.int64)
        if vendor_country_code != nan_country_code:
            cand = dest_by_country.get(int(vendor_country_code), np.empty(0, dtype=np.int64))
        if cand.size < geo_min_candidates and vendor_region_code != nan_region_code:
            cand = dest_by_region.get(int(vendor_region_code), np.empty(0, dtype=np.int64))
        if cand.size < geo_min_candidates:
            cand = semis_base
        if cand.size > budget:
            rng = np.random.default_rng(rng_seed ^ _stable_int_hash(vendor_key) ^ 0xA5A5A5A5)
            idx = rng.choice(len(cand), size=budget, replace=False)
            cand = cand[idx]
        add_many(cand)
    else:
        # Global semis pool
        if len(semis_base) >= budget:
            rng = np.random.default_rng(rng_seed ^ _stable_int_hash(vendor_key) ^ 0xA5A5A5A5)
            idx = rng.choice(len(semis_base), size=budget, replace=False)
            add_many(semis_base[idx])
            return np.array(out, dtype=np.int64)
        add_many(semis_base)

    # Hubs (only for semis_adjacent_hubs; only when using global pool)
    if pool_mode == "semis_adjacent_hubs" and not geo_gated and len(out) < budget and len(hubs) > 0:
        add_many(hubs)

    # Optional random tail (disabled by default)
    if len(out) < budget and random_k > 0:
        rng = np.random.default_rng(rng_seed ^ _stable_int_hash(vendor_key))
        target = min(int(random_k) * 3, n_nodes)
        choices = rng.choice(n_nodes, size=target, replace=False)
        add_many(choices)

    # If still short and random disabled, leave as-is (semis-only pool)

    return np.array(out, dtype=np.int64)


def compute_embeddings_side(
    model: TwoTowerModel,
    side: str,
    struct: np.ndarray,
    attr_codes: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    N = struct.shape[0]
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            s = torch.from_numpy(struct[start:end]).to(device=device, dtype=torch.float32)
            a = {
                k: torch.from_numpy(v[start:end]).to(device=device, dtype=torch.long)
                for k, v in attr_codes.items()
            }
            z = model.encode_nodes(side, s, a)
            out.append(z.detach().to("cpu"))
    return torch.cat(out, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Two-Tower cold-start inference for matched-but-not-in-SCR primes"
    )
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
    ap.add_argument(
        "--encoders",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/features/encoders_T0.pkl"),
    )
    ap.add_argument(
        "--node-features",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/features/node_features_T0.parquet"),
    )
    ap.add_argument(
        "--node-structural",
        type=Path,
        default=Path("data/processed/core/releases/core_v1/features/node_structural_v1.parquet"),
    )
    ap.add_argument(
        "--twotower-seed-dir",
        type=Path,
        default=Path("artifacts/twotower/twotower_production_v2/seed_42"),
    )
    ap.add_argument(
        "--chunk-size", type=int, default=5000, help="SQL IN() chunk size for FactSet pulls"
    )
    ap.add_argument(
        "--as-of-date",
        dest="as_of_date",
        type=str,
        default="2025-06-09",
        help="As-of date for FactSet attributes (YYYY-MM-DD)",
    )
    ap.add_argument(
        "--t0-date", dest="as_of_date", type=str, default=None, help="DEPRECATED: use --as-of-date"
    )
    ap.add_argument(
        "--iso-map", type=Path, default=Path("artifacts/ch3/reference/iso_region_continent_m49.csv")
    )
    ap.add_argument(
        "--geo-gated",
        action="store_true",
        help="Use geo-gated semis pool (country -> region -> global)",
    )
    ap.add_argument(
        "--geo-min-candidates",
        type=int,
        default=50,
        help="Minimum candidates to accept country/region pool before falling back",
    )
    ap.add_argument("--budget", type=int, default=10_000)
    ap.add_argument(
        "--pool",
        type=str,
        default="semis_adjacent",
        choices=["semis_core", "semis_adjacent", "semis_adjacent_hubs"],
    )
    ap.add_argument("--hub-top-frac", type=float, default=0.001)
    ap.add_argument("--random-per-source", type=int, default=0)
    ap.add_argument("--k-list", type=str, default="5,10,50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument(
        "--batch-emb", type=int, default=131072, help="Batch size for embedding computation"
    )
    ap.add_argument("--out-root", type=Path, default=Path("artifacts/ch3/twotower_cold_v2"))
    ap.add_argument("--network-root", type=Path, default=Path("artifacts/ch3/network"))
    ap.add_argument(
        "--limit-vendors", type=int, default=0, help="Debug: limit number of cold vendors scored"
    )
    args = ap.parse_args()

    load_dotenv()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    network_root = args.network_root
    network_root.mkdir(parents=True, exist_ok=True)

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    if sorted(k_list) != k_list or any(k <= 0 for k in k_list):
        raise ValueError("--k-list must be comma-separated positive ints (sorted preferred)")
    k_max = max(k_list)

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    cold = load_cold_primes(args.prime_matches, args.entity_map)
    if args.limit_vendors and args.limit_vendors > 0:
        cold = cold.iloc[: int(args.limit_vendors)].copy()
    vendor_keys = cold["vendor_key"].astype(str).tolist()
    print(f"[cold] vendors={len(cold):,} (matched FactSet, not in SCR)")
    cold_fids = cold["factset_entity_id"].astype(str).tolist()

    # Candidate sets shared across vendors
    include_adjacent = args.pool in {"semis_adjacent", "semis_adjacent_hubs"}
    semis_core, semis_adj = load_semis(args.semis_flags, include_adjacent=include_adjacent)
    semis_base = (
        semis_core
        if args.pool == "semis_core"
        else np.unique(np.concatenate([semis_core, semis_adj]))
    )
    hubs = np.empty(0, dtype=np.int64)
    if args.pool == "semis_adjacent_hubs":
        hubs = load_hubs(args.deg_out, args.deg_in, top_frac=float(args.hub_top_frac))
    n_nodes = int(np.load(args.deg_out).shape[0])
    print(
        f"[pool] mode={args.pool} geo_gated={bool(args.geo_gated)} semis_core={len(semis_core):,} semis_adj={len(semis_adj):,} hubs={len(hubs):,} n_nodes={n_nodes:,}"
    )

    # Load model + calibration
    model_path = args.twotower_seed_dir / "model.pt"
    calib_path = args.twotower_seed_dir / "calibration.json"
    payload = torch.load(model_path, map_location=device)  # nosec B614 -- local pipeline artifacts
    params = payload.get("hparams", {})
    embed_dim = int(params.get("embed_dim", 64))
    struct_hidden = [
        int(x.strip()) for x in str(params.get("struct_hidden", "128,64")).split(",") if x.strip()
    ]
    attr_hidden = [
        int(x.strip()) for x in str(params.get("attr_hidden", "64,64")).split(",") if x.strip()
    ]
    attr_emb_dim = int(params.get("attr_emb_dim", 32))
    dropout = float(params.get("dropout", 0.0))
    normalize = bool(params.get("normalize_emb", True))

    # Build SCR features (same as Chapter 2)
    attr_keys = ["country", "region", "continent", "entity_type", "primary_sic_code"]
    feats = tt_load_features(args.node_features, args.node_structural, attr_keys)
    model = TwoTowerModel(
        struct_in=int(feats.struct.shape[1]),
        struct_hidden=struct_hidden,
        attr_num_cats=feats.num_cats,
        attr_emb_dim=attr_emb_dim,
        attr_hidden=attr_hidden,
        tower_out_dim=int(embed_dim // 2),
        final_dim=embed_dim,
        dropout=dropout,
        normalize=normalize,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    # Build primary_sic float->category code mapping used by twotower load_features()
    primary_sic_to_code = build_primary_sic_string_to_code(args.node_features)
    # Default primary_sic_code category: most frequent code in SCR nodes
    primary_codes = feats.attr_codes["primary_sic_code"].astype(np.int64, copy=False)
    mode_code = int(np.argmax(np.bincount(primary_codes)))

    # Pull raw FactSet attributes for cold primes and encode using the same encoders used in core_v1 features.
    enc = load_encoder_bundle(args.encoders)
    cat_maps = enc.get("categorical_mappings", {})
    num_mm = enc.get("numeric_minmax", {})
    # Important: multiple vendor_keys can map to the same FactSet entity_id; query unique IDs
    # and keep one attribute row per FactSet entity to avoid exploding the vendor table on merge.
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

    # Geo fallback: GR (as-of) -> sym_entity.iso_country -> USAspending recipient country
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

    # Drop helper columns to keep the vendor table clean
    cold = cold.drop(
        columns=[
            c for c in ["iso_country", "gr_region_iso", "gr_continent_iso"] if c in cold.columns
        ]
    )

    def map_cat(col: str, series: pd.Series) -> np.ndarray:
        mapping = {str(k): int(v) for k, v in cat_maps.get(col, {}).items()}
        nan_code = mapping.get("nan", 0)
        s = series.astype("string").str.strip()
        s = s.replace({"Unknown": "nan", "": "nan"})
        s = s.fillna("nan")
        return s.map(mapping).fillna(nan_code).astype(np.int64).to_numpy()

    vendor_country = map_cat("gr_country", cold.get("gr_country", pd.Series(dtype="string")))
    vendor_region = map_cat("gr_region", cold.get("gr_region", pd.Series(dtype="string")))
    vendor_continent = map_cat("gr_continent", cold.get("gr_continent", pd.Series(dtype="string")))
    vendor_entity_type = map_cat("entity_type", cold.get("entity_type", pd.Series(dtype="string")))

    mm = num_mm.get("primary_sic_code", {"min": None, "max": None})
    sic_raw = pd.to_numeric(
        cold.get("primary_sic_raw", pd.Series(dtype="float64")), errors="coerce"
    )
    if mm.get("min") is None or mm.get("max") is None or float(mm["max"]) <= float(mm["min"]):
        sic_scaled = pd.Series(np.zeros(len(cold), dtype=np.float32))
    else:
        denom = float(mm["max"]) - float(mm["min"])
        sic_scaled = ((sic_raw - float(mm["min"])) / denom).fillna(0.0).astype(np.float32)
    sic_str = sic_scaled.astype("string")
    vendor_sic_code = np.array(
        [primary_sic_to_code.get(str(v), mode_code) for v in sic_str], dtype=np.int64
    )

    # Build cold vendor feature tensors:
    # - struct scalars: z-scored zeros (neutral; cold primes have no SCR graph structure)
    # - attr codes: FactSet-as-of-T0 attributes encoded to match core_v1 feature conventions
    vendor_struct = np.zeros((len(cold), feats.struct.shape[1]), dtype=np.float32)
    vendor_attr: dict[str, np.ndarray] = {
        "gr_country": vendor_country,
        "gr_region": vendor_region,
        "gr_continent": vendor_continent,
        "entity_type": vendor_entity_type,
        "primary_sic_code": vendor_sic_code,
    }

    # Build geo indexes for the semis pool (to gate candidates by country/region)
    dest_country = feats.attr_codes["gr_country"].astype(np.int64, copy=False)
    dest_region = feats.attr_codes["gr_region"].astype(np.int64, copy=False)
    semis_base = semis_base.astype(np.int64, copy=False)

    def _geo_index(nodes: np.ndarray, codes: np.ndarray) -> dict[int, np.ndarray]:
        d: dict[int, np.ndarray] = {}
        node_codes = codes[nodes]
        for code in np.unique(node_codes):
            d[int(code)] = nodes[node_codes == code]
        return d

    dest_by_country = _geo_index(semis_base, dest_country)
    dest_by_region = _geo_index(semis_base, dest_region)

    nan_country_code = cat_maps.get("gr_country", {}).get("nan", 0)
    nan_region_code = cat_maps.get("gr_region", {}).get("nan", 0)

    # Compute SCR destination embeddings (v-side)
    print(f"[embed] computing SCR embeddings on {device.type}...", flush=True)
    t0 = time.time()
    ZV = compute_embeddings_side(
        model=model,
        side="v",
        struct=feats.struct.astype(np.float32, copy=False),
        attr_codes={k: v.astype(np.int64, copy=False) for k, v in feats.attr_codes.items()},
        device=device,
        batch_size=int(args.batch_emb),
    )
    print(f"[embed] ZV shape={tuple(ZV.shape)} time={time.time() - t0:.1f}s", flush=True)

    # Compute vendor embeddings (u-side)
    print("[embed] computing vendor embeddings...", flush=True)
    t1 = time.time()
    ZU_vendor = compute_embeddings_side(
        model=model,
        side="u",
        struct=vendor_struct,
        attr_codes=vendor_attr,
        device=device,
        batch_size=min(int(args.batch_emb), max(1, len(cold))),
    )
    print(
        f"[embed] ZU_vendor shape={tuple(ZU_vendor.shape)} time={time.time() - t1:.1f}s", flush=True
    )

    # Load calibration
    calib = json.loads(calib_path.read_text()).get("params", {})

    # Score + select top-k_max per vendor
    top_rows: list[tuple[str, int, int, float, float]] = []
    start = time.time()
    zv_cpu = ZV  # [N, d] on CPU
    for i, vendor_key in enumerate(vendor_keys, 1):
        cand = build_candidates_for_vendor(
            vendor_key=vendor_key,
            budget=int(args.budget),
            semis_base=semis_base,
            hubs=hubs,
            pool_mode=str(args.pool),
            geo_gated=bool(args.geo_gated),
            geo_min_candidates=int(args.geo_min_candidates),
            vendor_country_code=int(vendor_country[i - 1]),
            vendor_region_code=int(vendor_region[i - 1]),
            nan_country_code=int(nan_country_code),
            nan_region_code=int(nan_region_code),
            dest_by_country=dest_by_country,
            dest_by_region=dest_by_region,
            rng_seed=int(args.seed),
            random_k=int(args.random_per_source),
            n_nodes=n_nodes,
        )

        zu = ZU_vendor[i - 1]  # [d] on CPU
        zv = zv_cpu.index_select(0, torch.from_numpy(cand.astype(np.int64)))
        scores = (zv * zu.unsqueeze(0)).sum(dim=1).numpy()
        # Calibrate (optional; selection is rank-based but this is useful for reporting)
        calibrated = apply_platt(scores.astype(np.float64), calib).astype(np.float32)

        if k_max < len(scores):
            idx = np.argpartition(scores, -k_max)[-k_max:]
            idx = idx[np.argsort(-scores[idx])]
        else:
            idx = np.argsort(-scores)
        for rk, j in enumerate(idx[:k_max], start=1):
            dst = int(cand[int(j)])
            top_rows.append((vendor_key, dst, rk, float(scores[int(j)]), float(calibrated[int(j)])))

        if i % 250 == 0 or i == len(vendor_keys):
            dt = time.time() - start
            rate = i / max(dt, 1e-6)
            print(f"[score] {i:,}/{len(vendor_keys):,} vendors ({rate:.1f} vendors/s)", flush=True)

    version = "twotower_cold_v2"
    pool_tag = f"{args.pool}_geo" if args.geo_gated else str(args.pool)
    top50_path = out_root / f"{version}_top50_{pool_tag}.parquet"
    df_top = pd.DataFrame(
        top_rows, columns=["vendor_key", "dst_node_id", "rank", "score_raw", "score_calib"]
    )
    df_top["pool"] = pool_tag
    df_top["version"] = version
    df_top.to_parquet(top50_path, index=False)
    print(f"[out] wrote {len(df_top):,} rows -> {top50_path}")

    # Write Top-K edge files in network UID format
    for k in k_list:
        out_edges = network_root / f"edges_predicted_{version}_top{k}_{pool_tag}.parquet"
        df_k = df_top[df_top["rank"] <= k].copy()
        df_k["src_uid"] = "vendor:" + df_k["vendor_key"].astype(str)
        df_k["dst_uid"] = "scr:" + df_k["dst_node_id"].astype(str)
        df_k["edge_type"] = "predicted_twotower"
        df_k["policy_k"] = int(k)
        df_k["score"] = df_k["score_raw"]
        df_k["pool"] = pool_tag
        df_k["version"] = version
        df_k = df_k[
            ["src_uid", "dst_uid", "edge_type", "policy_k", "rank", "score", "pool", "version"]
        ]
        df_k.to_parquet(out_edges, index=False)
        print(f"[out] {k=}: edges={len(df_k):,} -> {out_edges}")

    summary = {
        "created_at": _now_iso(),
        "device": str(device),
        "vendors_scored": len(vendor_keys),
        "unique_factset_entities_scored": len(set(cold_fids)),
        "budget_per_vendor": int(args.budget),
        "factset_attrs_asof": str(args.as_of_date),
        "factset_chunk_size": int(args.chunk_size),
        "pool": pool_tag,
        "version": version,
        "iso_map": str(args.iso_map),
        "geo_gated": bool(args.geo_gated),
        "geo_min_candidates": int(args.geo_min_candidates),
        "candidate_components": {
            "semis_core": len(semis_core),
            "semis_adjacent": len(semis_adj),
            "hubs": len(hubs),
            "random_per_source": int(args.random_per_source),
        },
        "k_list": k_list,
        "outputs": {
            "top50": str(top50_path),
            "edges_top5": str(network_root / f"edges_predicted_{version}_top5_{pool_tag}.parquet"),
            "edges_top10": str(
                network_root / f"edges_predicted_{version}_top10_{pool_tag}.parquet"
            ),
            "edges_top50": str(
                network_root / f"edges_predicted_{version}_top50_{pool_tag}.parquet"
            ),
        },
    }
    (out_root / f"summary_{version}_{pool_tag}.json").write_text(json.dumps(summary, indent=2))
    print("[done] twotower cold v2 complete")


if __name__ == "__main__":
    main()
