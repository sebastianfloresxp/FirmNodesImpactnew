#!/usr/bin/env python3
"""
Core Pipeline - Phase 7: Validate Core Dataset

Runs integrity and leakage checks across events, splits, mapping, adjacency,
candidate pools, and features. Produces a machine-readable JSON report.

Output:
  - meta/validation_report.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import pyarrow.parquet as pq  # for row-group sampling
except Exception:  # pragma: no cover
    pq = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)8s %(message)s")
logger = logging.getLogger("core.validate")


@dataclass
class Check:
    name: str
    status: str  # PASS | FAIL | WARN
    details: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_events(root: Path) -> list[Check]:
    checks: list[Check] = []
    p = root / "events" / "supply_chain_events.parquet"
    if not p.exists():
        checks.append(Check("events.exists", "FAIL", {"path": str(p)}))
        return checks
    cols = [
        "event_id",
        "src",
        "dst",
        "start_date",
        "timestamp",
        "edge_feature",
        "duration_days",
        "relationship_frequency",
        "is_active",
        "year",
        "month",
        "quarter",
        "ts",
    ]
    df = pd.read_parquet(p, columns=cols)
    checks.append(Check("events.schema", "PASS", {"columns": cols}))
    # Uniqueness
    uniq_ok = df["event_id"].nunique() == len(df)
    checks.append(Check("events.event_id_unique", "PASS" if uniq_ok else "FAIL", {"rows": len(df)}))
    # timestamp min
    ts0_ok = abs(float(df["timestamp"].min()) - 0.0) < 1e-9
    checks.append(
        Check(
            "events.timestamp_min_zero",
            "PASS" if ts0_ok else "FAIL",
            {"min": float(df["timestamp"].min())},
        )
    )
    # non-negative durations
    dur_ok = bool((df["duration_days"] >= 0).all())
    checks.append(Check("events.duration_non_negative", "PASS" if dur_ok else "FAIL", {}))
    return checks


def _validate_splits(root: Path) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    meta_path = root / "meta" / "temporal_splits.json"
    if not meta_path.exists():
        checks.append(Check("splits.meta_exists", "FAIL", {"path": str(meta_path)}))
        return checks, {}
    meta = _read_json(meta_path)
    b = meta.get("boundaries", {})
    # Files exist
    sdir = root / "splits"
    files_ok = all((sdir / f"{n}_edges.parquet").exists() for n in ("train", "val", "test"))
    checks.append(Check("splits.files_exist", "PASS" if files_ok else "FAIL", {"dir": str(sdir)}))
    # Load minimal cols
    tr = pd.read_parquet(
        sdir / "train_edges.parquet", columns=["event_id", "ts", "src_id", "dst_id"]
    )
    va = pd.read_parquet(sdir / "val_edges.parquet", columns=["event_id", "ts", "src_id", "dst_id"])
    te = pd.read_parquet(
        sdir / "test_edges.parquet", columns=["event_id", "ts", "src_id", "dst_id"]
    )
    # Boundaries checks
    t0, vstart, vend, tstart = (
        int(b["T0_end"]),
        int(b["val_start"]),
        int(b["val_end"]),
        int(b["test_start"]),
    )
    gb = int(b.get("gap_days", 0))
    train_ok = bool((tr["ts"] <= t0).all())
    val_ok = bool(((va["ts"] >= vstart) & (va["ts"] <= vend)).all())
    test_ok = bool((te["ts"] >= tstart).all())
    checks.append(Check("splits.train_ts_bounds", "PASS" if train_ok else "FAIL", {"T0_end": t0}))
    checks.append(
        Check(
            "splits.val_ts_bounds",
            "PASS" if val_ok else "FAIL",
            {"val_start": vstart, "val_end": vend},
        )
    )
    checks.append(
        Check("splits.test_ts_bounds", "PASS" if test_ok else "FAIL", {"test_start": tstart})
    )
    # Gaps
    gap1 = int(vstart - t0)
    gap2 = int(tstart - vend)
    checks.append(
        Check(
            "splits.gap_after_train",
            "PASS" if gap1 == gb else "FAIL",
            {"expected": gb, "actual": gap1},
        )
    )
    checks.append(
        Check(
            "splits.gap_after_val",
            "PASS" if gap2 == gb else "FAIL",
            {"expected": gb, "actual": gap2},
        )
    )
    # Disjointness by event_id
    inter_tv = set(tr["event_id"]) & set(va["event_id"])
    inter_tt = set(tr["event_id"]) & set(te["event_id"])
    inter_vt = set(va["event_id"]) & set(te["event_id"])
    disjoint_ok = (len(inter_tv) == 0) and (len(inter_tt) == 0) and (len(inter_vt) == 0)
    checks.append(
        Check(
            "splits.disjoint_event_ids",
            "PASS" if disjoint_ok else "FAIL",
            {"train∩val": len(inter_tv), "train∩test": len(inter_tt), "val∩test": len(inter_vt)},
        )
    )
    return checks, {"train": tr, "val": va, "test": te, "boundaries": b}


def _validate_mapping(root: Path, splits: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    mpath = root / "mapping" / "entity_map.parquet"
    if not mpath.exists():
        checks.append(Check("mapping.exists", "FAIL", {"path": str(mpath)}))
        return checks
    m = pd.read_parquet(mpath)
    N = int(m.shape[0])
    # node_id uniqueness and range
    uniq = m["node_id"].nunique() == N
    min0 = int(m["node_id"].min()) == 0
    maxN = int(m["node_id"].max()) == N - 1
    checks.append(Check("mapping.node_id_unique", "PASS" if uniq else "FAIL", {"N": N}))
    checks.append(
        Check(
            "mapping.node_id_contiguous",
            "PASS" if (min0 and maxN) else "FAIL",
            {"min": int(m["node_id"].min()), "max": int(m["node_id"].max())},
        )
    )
    # Joinability: sample checks on splits (ensure no NA and in-range)
    for name in ("train", "val", "test"):
        df = splits[name]
        in_range = bool(
            (df["src_id"].between(0, N - 1)).all() and (df["dst_id"].between(0, N - 1)).all()
        )
        if not in_range:
            checks.append(Check(f"mapping.joinability_{name}", "FAIL", {}))
        else:
            checks.append(Check(f"mapping.joinability_{name}", "PASS", {"rows": len(df)}))
    return checks


def _validate_adjacency(root: Path, train_df: pd.DataFrame) -> list[Check]:
    checks: list[Check] = []
    adj_dir = root / "adjacency"
    csr_path = adj_dir / "train_adj_T0.npz"
    out_deg_path = adj_dir / "out_degree.npy"
    in_deg_path = adj_dir / "in_degree.npy"
    if not csr_path.exists():
        checks.append(Check("adjacency.csr_exists", "FAIL", {"path": str(csr_path)}))
        return checks
    csr = sp.load_npz(csr_path).tocsr(copy=False)
    nnz = int(csr.nnz)
    checks.append(Check("adjacency.shape", "PASS", {"shape": list(csr.shape), "nnz": nnz}))
    # unique train pairs
    uniq_pairs = train_df[["src_id", "dst_id"]].drop_duplicates().shape[0]
    pairs_ok = uniq_pairs == nnz
    checks.append(
        Check(
            "adjacency.nnz_matches_unique_pairs",
            "PASS" if pairs_ok else "FAIL",
            {"unique_pairs": int(uniq_pairs)},
        )
    )
    # degree sums
    outd = np.load(out_deg_path)
    ind = np.load(in_deg_path)
    deg_ok = int(outd.sum()) == nnz and int(ind.sum()) == nnz
    checks.append(
        Check(
            "adjacency.degree_sums_match",
            "PASS" if deg_ok else "FAIL",
            {"sum_out": int(outd.sum()), "sum_in": int(ind.sum())},
        )
    )
    return checks


def _sample_candidates_rg(path: Path, columns: list[str], rg_index: int = 0) -> pd.DataFrame:
    if pq is None:
        # Fallback: load a small chunk via pandas (may be slower)
        return pd.read_parquet(path, columns=columns).head(1_000_000)
    pf = pq.ParquetFile(path)
    return pf.read_row_group(rg_index, columns=columns).to_pandas()


def _validate_candidates(
    root: Path, splits: dict[str, Any], budget: int, sample_rg: int = 0
) -> list[Check]:
    checks: list[Check] = []
    cdir = root / "candidates"
    meta_path = root / "meta" / "candidate_pools_meta.json"
    if not meta_path.exists():
        checks.append(Check("candidates.meta_exists", "FAIL", {"path": str(meta_path)}))
        return checks
    meta = _read_json(meta_path)
    # Sanity: source counts align
    for split in ("val", "test"):
        split_src = int(splits[split]["src_id"].nunique())
        meta_src = int(meta["sources"][split])
        checks.append(
            Check(
                f"candidates.sources_{split}",
                "PASS" if split_src == meta_src else "FAIL",
                {"split": split, "meta": meta_src, "actual": split_src},
            )
        )
    # Sample-based validation per file
    for split in ("val", "test"):
        f = cdir / f"{split}_candidates.parquet"
        if not f.exists():
            checks.append(Check(f"candidates.file_exists_{split}", "FAIL", {"path": str(f)}))
            continue
        df = _sample_candidates_rg(f, ["src_id", "dst_id", "label", "source"], rg_index=sample_rg)
        # Per-source counts in sample should be <= budget and near budget for first few
        grp = df.groupby("src_id").size()
        per_ok = bool((grp <= budget).all())
        checks.append(
            Check(
                f"candidates.sample_per_source_le_budget_{split}",
                "PASS" if per_ok else "FAIL",
                {"sample_sources": int(grp.index.nunique())},
            )
        )
        # Pool recall on sample sources
        set(map(tuple, splits[split][["src_id", "dst_id"]].to_records(index=False)))
        cand_pos = set(
            map(tuple, df[df["label"] == 1][["src_id", "dst_id"]].to_records(index=False))  # type: ignore[union-attr]
        )
        # Only check for sources present in sample
        sample_sources = set(df["src_id"].unique().tolist())
        pos_subset = set(
            map(
                tuple,
                splits[split][splits[split]["src_id"].isin(sample_sources)][
                    ["src_id", "dst_id"]
                ].to_records(index=False),
            )
        )
        missing = len(pos_subset - cand_pos)
        checks.append(
            Check(
                f"candidates.sample_pool_recall_{split}",
                "PASS" if missing == 0 else "FAIL",
                {"missing": int(missing), "sample_sources": len(sample_sources)},
            )
        )
        # Leakage check on zeros
        zeros = set(map(tuple, df[df["label"] == 0][["src_id", "dst_id"]].to_records(index=False)))  # type: ignore[union-attr]
        train_pairs = set(map(tuple, splits["train"][["src_id", "dst_id"]].to_records(index=False)))
        if split == "test":
            val_pairs = set(map(tuple, splits["val"][["src_id", "dst_id"]].to_records(index=False)))
        else:
            val_pairs = set()
        leakage = [p for p in zeros if (p in train_pairs) or (p in val_pairs)]
        checks.append(
            Check(
                f"candidates.sample_leakage_{split}",
                "PASS" if len(leakage) == 0 else "FAIL",
                {"hits": len(leakage)},
            )
        )
    return checks


def _validate_features(root: Path, mapping_size: int) -> list[Check]:
    checks: list[Check] = []
    feat_dir = root / "features"
    feat_path = feat_dir / "node_features_T0.parquet"
    schema_path = root / "meta" / "feature_schema.json"
    if not feat_path.exists() or not schema_path.exists():
        checks.append(
            Check(
                "features.exists", "FAIL", {"features": str(feat_path), "schema": str(schema_path)}
            )
        )
        return checks
    _read_json(schema_path)
    df = pd.read_parquet(feat_path)
    # Row count and node_id uniqueness
    ok_rows = df.shape[0] == mapping_size
    ok_unique = df["node_id"].nunique() == mapping_size
    checks.append(
        Check(
            "features.row_count_equals_mapping",
            "PASS" if ok_rows else "FAIL",
            {"rows": int(df.shape[0]), "mapping": mapping_size},
        )
    )
    checks.append(Check("features.node_id_unique", "PASS" if ok_unique else "FAIL", {}))
    # Dtypes and NaNs
    dtypes = {c: str(df[c].dtype) for c in df.columns if c != "node_id"}
    has_nan = bool(df.drop(columns=["node_id"]).isna().any().any())  # type: ignore[union-attr]
    checks.append(Check("features.no_nans", "PASS" if not has_nan else "FAIL", {}))
    checks.append(Check("features.dtypes", "PASS", dtypes))
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate core dataset and emit report")
    ap.add_argument("--out-root", required=True, type=str)
    ap.add_argument(
        "--sample-rg", type=int, default=0, help="Parquet row group index to sample for candidates"
    )
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    with contextlib.suppress(Exception):
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    root = Path(args.out_root)
    report: dict[str, Any] = {
        "run_root": str(root),
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "summary": {},
        "checks": {},
    }

    # 1) Events
    ev_checks = _validate_events(root)
    report["checks"]["events"] = [asdict(c) for c in ev_checks]

    # 2) Splits
    sp_checks, sp_ctx = _validate_splits(root)
    report["checks"]["splits"] = [asdict(c) for c in sp_checks]

    # 3) Mapping
    mp_checks = _validate_mapping(root, sp_ctx)
    report["checks"]["mapping"] = [asdict(c) for c in mp_checks]

    # 4) Adjacency
    adj_checks = _validate_adjacency(root, sp_ctx.get("train", pd.DataFrame()))
    report["checks"]["adjacency"] = [asdict(c) for c in adj_checks]

    # 5) Candidates (sample)
    # get budget from meta if present
    meta_path = root / "meta" / "candidate_pools_meta.json"
    budget = 0
    if meta_path.exists():
        try:
            budget = int(_read_json(meta_path).get("budget", 0))
        except Exception:
            budget = 0
    cand_checks = _validate_candidates(root, sp_ctx, budget or 5_000, sample_rg=args.sample_rg)
    report["checks"]["candidates"] = [asdict(c) for c in cand_checks]

    # 6) Features
    # mapping size
    mpath = root / "mapping" / "entity_map.parquet"
    m = pd.read_parquet(mpath) if mpath.exists() else pd.DataFrame()
    ft_checks = _validate_features(root, int(m.shape[0]) if not m.empty else 0)
    report["checks"]["features"] = [asdict(c) for c in ft_checks]

    # Summary
    all_checks = [c for section in report["checks"].values() for c in section]
    failures = [c for c in all_checks if c["status"] == "FAIL"]
    warnings = [c for c in all_checks if c["status"] == "WARN"]
    report["summary"] = {
        "overall_pass": (len(failures) == 0),
        "failures": len(failures),
        "warnings": len(warnings),
    }

    out = root / "meta" / "validation_report.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info("Validation complete. Report: %s", out)


if __name__ == "__main__":
    main()
