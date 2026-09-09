#!/usr/bin/env python3
"""Match DoD prime vendors to FactSet entities (precision-first linkage).

Approach:
- Load aggregated primes (as-of slice).
!- Build a vendor table (one row per vendor_key) with name/DBA/parent variants and geo.
!- Load FactSet sym_entity filtered to relevant countries via DuckDB.
!- Normalize names (upper, strip punctuation/suffixes) on both sides.
!- Exact matches in order: name+country, DBA+country, parent+country, then name without country if missing.
- For any residual unmatched vendors, run a conservative fuzzy match with blocking and
  accept only matches above a configured similarity threshold.
- Write outputs: accepted matches and review-needed (ambiguous or no match).

This step is designed to be auditable: it prefers precision over recall and records
unmatched/ambiguous cases explicitly.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

try:
    import pycountry
except ImportError:
    pycountry = None
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

from db_client.connection import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")
logger = logging.getLogger("ch3.match_primes")

# Basic normalization helpers
CORP_SUFFIXES = [
    "INCORPORATED",
    "INC",
    "LLC",
    "L.L.C",
    "CORPORATION",
    "CORP",
    "CO",
    "COMPANY",
    "LTD",
    "LIMITED",
    "SA",
    "AG",
    "BV",
    "GMBH",
    "SPA",
    "S.P.A",
    "PLC",
]
SUFFIX_PATTERN = re.compile(r"\\b(" + "|".join(CORP_SUFFIXES) + r")\\b")
PUNCT_PATTERN = re.compile(r"[^A-Z0-9 ]+")
MULTISPACE = re.compile(r"\\s+")


def normalize_name(name: str | None) -> str | None:
    if not name or not isinstance(name, str):
        return None
    n = name.upper()
    n = PUNCT_PATTERN.sub(" ", n)
    n = SUFFIX_PATTERN.sub(" ", n)
    n = MULTISPACE.sub(" ", n).strip()
    return n or None


def load_vendors(primes_path: Path, tx_path: Path) -> pd.DataFrame:
    """Load unique vendors, enrich with DBA/parent from transactions."""
    primes = pd.read_parquet(primes_path)
    # Get DBA/parent from transactions via DuckDB distinct
    con = duckdb.connect()
    vendor_vars = con.execute(
        f"""
        SELECT DISTINCT
            COALESCE(NULLIF(recipient_uei, ''), NULLIF(recipient_duns, ''), recipient_name) AS vendor_key,
            recipient_doing_business_as_name,
            recipient_parent_name,
            recipient_country_code,
            recipient_state_code,
            recipient_city_name
        FROM '{tx_path}'
        """
    ).fetchdf()
    con.close()
    vendors = primes[
        [
            "vendor_key",
            "recipient_name",
            "recipient_country_code",
            "recipient_state_code",
            "recipient_city_name",
            "sum_total_dollars_obligated",
        ]
    ].drop_duplicates(subset=["vendor_key"])
    vendors = vendors.merge(vendor_vars, on="vendor_key", how="left", suffixes=("", "_tx"))
    # Prefer country/state/city from primes, fall back to tx if missing
    for col in ["recipient_country_code", "recipient_state_code", "recipient_city_name"]:
        vendors[col] = vendors[col].fillna(vendors[f"{col}_tx"])
        vendors.drop(columns=[f"{col}_tx"], inplace=True)
    vendors = vendors.drop_duplicates(subset=["vendor_key"])

    # Convert country codes (USA -> US) if pycountry available
    def to_alpha2(code: str | None) -> str | None:
        if not code or not isinstance(code, str):
            return None
        if len(code) == 2:
            return code.upper()
        if len(code) == 3:
            # Fallback conversion: use pycountry if available, else strip to first 2 chars
            if pycountry:
                try:
                    return pycountry.countries.get(alpha_3=code.upper()).alpha_2  # type: ignore
                except Exception:  # nosec B110 -- best-effort country lookup, pass is intentional
                    pass
            return code[:2].upper()
        return None

    vendors["country_alpha2"] = vendors["recipient_country_code"].apply(to_alpha2)
    vendors["country_alpha2"].fillna(vendors["recipient_country_code"], inplace=True)
    # Normalize name variants
    vendors["name_norm"] = vendors["recipient_name"].apply(normalize_name)
    vendors["dba_norm"] = vendors["recipient_doing_business_as_name"].apply(normalize_name)
    vendors["parent_norm"] = vendors["recipient_parent_name"].apply(normalize_name)
    return vendors


def load_factset_entities(sym_path: Path, countries: set[str]) -> pd.DataFrame:
    """Load FactSet sym_entity filtered to relevant countries via DuckDB."""
    con = duckdb.connect()
    country_list = ",".join(f"'{c}'" for c in countries if c)
    where = f"WHERE iso_country IN ({country_list})" if country_list else ""
    logger.info("Loading sym_entity filtered to %d countries from %s", len(countries), sym_path)
    if sym_path.suffix.lower() == ".csv":
        df = con.execute(
            f"""
            SELECT factset_entity_id, entity_proper_name, iso_country
            FROM read_csv_auto('{sym_path}', header=True, all_varchar=TRUE, sample_size=-1)
            {where}
            """
        ).fetchdf()
    else:
        df = con.execute(
            f"""
            SELECT factset_entity_id, entity_proper_name, iso_country
            FROM read_parquet('{sym_path}')
            {where}
            """
        ).fetchdf()
    con.close()
    df["name_norm"] = df["entity_proper_name"].apply(normalize_name)
    return df.dropna(subset=["name_norm"])


def build_lookup(factset_df: pd.DataFrame) -> dict[tuple[str, str | None], list[str]]:
    """Map (name_norm, country) -> list of factset_entity_id."""
    lookup: dict[tuple[str, str | None], list[str]] = {}
    for _, row in factset_df.iterrows():
        key = (row["name_norm"], row["iso_country"])
        lookup.setdefault(key, []).append(row["factset_entity_id"])
    return lookup


def build_prefix_index(
    factset_df: pd.DataFrame, prefix_len: int = 6
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Map (country, prefix) -> list of (factset_entity_id, name_norm)."""
    idx: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for _, row in factset_df.iterrows():
        name_norm = row["name_norm"]
        if not name_norm:
            continue
        prefix = name_norm[:prefix_len]
        key = (row["iso_country"], prefix)
        idx.setdefault(key, []).append((row["factset_entity_id"], name_norm))
    return idx


def simple_ratio(a: str, b: str) -> float:
    """Lightweight similarity score in [0,1] using SequenceMatcher."""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def pick_match(candidates: list[str], method: str) -> tuple[str | None, str]:
    if not candidates:
        return None, method
    if len(candidates) == 1:
        return candidates[0], method
    # Ambiguous; send to review
    return None, method + "_ambiguous"


def match_vendor(
    row: pd.Series, lookup: dict[tuple[str, str | None], list[str]]
) -> tuple[str | None, str]:
    country = row.get("country_alpha2")
    names = [
        ("name_norm", row.get("name_norm")),
        ("dba_norm", row.get("dba_norm")),
        ("parent_norm", row.get("parent_norm")),
    ]
    # Exact with country
    for label, n in names:
        if not n:
            continue
        candidates = lookup.get((n, country))
        fid, method = pick_match(candidates or [], f"exact_{label}_country")
        if fid:
            return fid, method
    # Exact without country (only if missing country)
    if not country:
        for label, n in names:
            if not n:
                continue
            candidates = lookup.get((n, None)) or lookup.get((n, "")) or []
            fid, method = pick_match(candidates, f"exact_{label}_no_country")
            if fid:
                return fid, method
    return None, "unmatched"


def fuzzy_match_vendor(
    row: pd.Series,
    prefix_idx: dict[tuple[str, str], list[tuple[str, str]]],
    country: str | None,
    threshold: float = 0.92,
    prefix_len: int = 6,
) -> tuple[str | None, str, float]:
    """Fuzzy match using prefix blocking and simple ratio."""
    names = [
        ("name_norm", row.get("name_norm")),
        ("dba_norm", row.get("dba_norm")),
        ("parent_norm", row.get("parent_norm")),
    ]
    best_fid = None
    best_score = 0.0
    best_method = "unmatched"
    for label, n in names:
        if not n:
            continue
        prefix = n[:prefix_len]
        candidates = []
        if country:
            candidates = prefix_idx.get((country, prefix), [])
        else:
            # If no country, union across all countries for this prefix (not ideal but fallback)
            for (_c, pfx), lst in prefix_idx.items():
                if pfx == prefix:
                    candidates.extend(lst)
        if not candidates:
            continue
        scores = []
        for fid, fname in candidates:
            s = simple_ratio(n, fname)
            scores.append((s, fid))
        if not scores:
            continue
        scores.sort(reverse=True)
        top_score, top_fid = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        if top_score >= threshold and top_score - second >= 0.02 and top_score > best_score:
            best_score = top_score
            best_fid = top_fid
            best_method = f"fuzzy_{label}_prefix"
    return best_fid, best_method, best_score


def summarize(matches_df: pd.DataFrame, vendors_df: pd.DataFrame, out_path: Path) -> None:
    total = len(vendors_df)
    matched = matches_df["factset_entity_id"].notna().sum()
    summary = {
        "vendors_total": total,
        "vendors_matched": int(matched),
        "vendors_unmatched": int(total - matched),
        "match_rate": float(matched / total) if total else 0.0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        import json

        json.dump(summary, f, indent=2)
    logger.info("Match summary: %s", summary)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description="Match DoD primes to FactSet (deterministic exact pass)"
    )
    ap.add_argument(
        "--primes", type=Path, required=True, help="Aggregated primes parquet (as-of slice)"
    )
    ap.add_argument(
        "--transactions",
        type=Path,
        required=True,
        help="Transactions parquet (as-of slice) for name/DBA/parent variants",
    )
    ap.add_argument(
        "--sym-entity",
        type=Path,
        default=Path("data/raw/sym_entity_table.csv"),
        help="FactSet sym_entity CSV/Parquet",
    )
    ap.add_argument(
        "--fetch-sym-from-db",
        action="store_true",
        help="Fetch sym_entity from FactSet DB filtered to vendor countries",
    )
    ap.add_argument(
        "--sym-cache", type=Path, default=Path("artifacts/ch3/matching/sym_entity_cached.parquet")
    )
    ap.add_argument(
        "--out-matches", type=Path, default=Path("artifacts/ch3/matching/prime_matches.parquet")
    )
    ap.add_argument(
        "--out-review",
        type=Path,
        default=Path("artifacts/ch3/matching/prime_matches_review.parquet"),
    )
    ap.add_argument(
        "--out-summary", type=Path, default=Path("artifacts/ch3/matching/match_summary.json")
    )
    ap.add_argument(
        "--fuzzy-threshold", type=float, default=0.90, help="Fuzzy match acceptance threshold (0-1)"
    )
    args = ap.parse_args()

    vendors = load_vendors(args.primes, args.transactions)
    vendor_countries: set[str] = set(vendors["country_alpha2"].dropna().unique().tolist())
    sym_path = args.sym_entity
    if args.fetch_sym_from_db:
        sym_path = args.sym_cache
        sym_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Fetching sym_entity from DB into %s", sym_path)
        sql = "SELECT factset_entity_id, entity_proper_name, iso_country FROM sym_v1.sym_entity"
        if vendor_countries:
            country_list = ",".join(f"'{c}'" for c in vendor_countries)
            sql += f" WHERE iso_country IN ({country_list})"
        engine = get_engine()
        if pa is None or pq is None:
            raise SystemExit("pyarrow is required to cache sym_entity from DB")
        writer = None
        for chunk in pd.read_sql(sql, engine, chunksize=500_000):
            table = pa.Table.from_pandas(chunk)
            if writer is None:
                writer = pq.ParquetWriter(sym_path, table.schema)
            writer.write_table(table)
        if writer:
            writer.close()
        logger.info("sym_entity cached to %s", sym_path)

    factset_df = load_factset_entities(sym_path, vendor_countries)
    lookup = build_lookup(factset_df)
    prefix_idx = build_prefix_index(factset_df)

    matches: list[dict[str, object]] = []
    for _, row in vendors.iterrows():
        fid, method = match_vendor(row, lookup)
        score = 1.0 if fid else 0.0
        if not fid:
            # try fuzzy
            fid, method, score = fuzzy_match_vendor(
                row, prefix_idx, row.get("country_alpha2"), threshold=args.fuzzy_threshold
            )
        matches.append(
            {
                "vendor_key": row["vendor_key"],
                "recipient_name": row["recipient_name"],
                "recipient_doing_business_as_name": row["recipient_doing_business_as_name"],
                "recipient_parent_name": row["recipient_parent_name"],
                "recipient_country_code": row["recipient_country_code"],
                "country_alpha2": row["country_alpha2"],
                "recipient_state_code": row["recipient_state_code"],
                "recipient_city_name": row["recipient_city_name"],
                "factset_entity_id": fid,
                "match_method": method,
                "match_score": score if fid else 0.0,
            }
        )

    matches_df = pd.DataFrame(matches)
    matches_df.to_parquet(args.out_matches, index=False)
    review_df = matches_df[
        matches_df["factset_entity_id"].isna()
        | matches_df["match_method"].str.contains("ambiguous")
    ]
    review_df.to_parquet(args.out_review, index=False)
    summarize(matches_df, vendors, args.out_summary)


if __name__ == "__main__":
    main()
