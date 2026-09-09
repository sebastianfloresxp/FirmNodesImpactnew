#!/usr/bin/env python3
"""Module 0.5: build RBICS/SIC code universe tables for semiconductor relevance codebook drafting.

Outputs:
  - node_code_universe_inputs.csv/.parquet
  - code_universe_rbics_l4.csv/.parquet
  - code_universe_sic.csv/.parquet
  - run_metadata.json
  - manifest_m0_5.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle  # nosec B403 -- internal ML artifacts only
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

RBICS_L4_NAME_OVERRIDES = {
    "55102010": "Analog and Mixed Signal Semiconductors",
    "55102015": "Discrete Semiconductors",
    "55102020": "General Semiconductors",
    "55102025": "Memory Semiconductors",
    "55102030": "Processor Semiconductors",
    "55102035": "Programmable Logic and ASIC Semiconductors",
    "55102040": "Specialized Semiconductors",
    "55103010": "Semiconductor Manufacturing Capital Equipment",
    "55103015": "Semiconductor Manufacturing Services",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 0.5 code-universe extractor")
    parser.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2_fix01.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--encoder-pkl",
        default="data/processed/core/runs/2025-09-03_core_v1/features/encoders_T0.pkl",
        help="Path to encoders_T0.pkl used to decode normalized SIC feature back to SIC code",
    )
    parser.add_argument(
        "--topn",
        default="25,100",
        help="Comma-separated Top-N cutoffs for impact prevalence (e.g. 25,100)",
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


def parse_pipe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "none" or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split("|") if part and part.strip()]


def decode_sic_series(normalized: pd.Series, min_raw: float, max_raw: float) -> pd.Series:
    num = pd.to_numeric(normalized, errors="coerce")
    raw = num * (max_raw - min_raw) + min_raw
    sic = raw.round().astype("Int64")
    # Defensive bounds for SIC-like codes.
    sic = sic.where((sic >= 100) & (sic <= 9999), pd.NA)
    return sic


def load_sic_map(scr_nodes_path: Path, encoder_path: Path) -> pd.DataFrame:
    if not scr_nodes_path.exists():
        raise FileNotFoundError(scr_nodes_path)
    if not encoder_path.exists():
        raise FileNotFoundError(encoder_path)

    scr = pd.read_parquet(scr_nodes_path, columns=["scr_node_id", "primary_sic_code"]).copy()
    scr["scr_node_id"] = pd.to_numeric(scr["scr_node_id"], errors="coerce").astype("Int64")
    scr = scr[scr["scr_node_id"].notna()].copy()

    with encoder_path.open("rb") as handle:
        enc = pickle.load(handle)  # nosec B301 -- internal ML artifacts only
    mm = enc.get("numeric_minmax", {}).get("primary_sic_code")
    if not isinstance(mm, dict) or "min" not in mm or "max" not in mm:
        raise ValueError("primary_sic_code min/max not found in encoder pickle")
    min_raw = float(mm["min"])
    max_raw = float(mm["max"])

    scr["sic_code"] = decode_sic_series(scr["primary_sic_code"], min_raw=min_raw, max_raw=max_raw)
    scr = scr[scr["sic_code"].notna()].copy()
    scr["sic_code"] = scr["sic_code"].astype("Int64")
    scr["sic_code_str"] = scr["sic_code"].astype(int).astype(str).str.zfill(4)

    # Collapse potentially duplicated rows per SCR node deterministically.
    out = (
        scr.sort_values(["scr_node_id", "sic_code_str"])
        .drop_duplicates(subset=["scr_node_id"], keep="first")[["scr_node_id", "sic_code_str"]]
        .rename(columns={"scr_node_id": "rep_scr_node_id"})
        .reset_index(drop=True)
    )
    return out


def discover_views(m6_1_dir: Path, cfg: dict[str, Any]) -> list[str]:
    configured = [str(v) for v in cfg.get("m6_1", {}).get("views", [])]
    if configured:
        return configured
    views: list[str] = []
    for p in sorted(m6_1_dir.glob("node_single_removal_impacts_*.csv")):
        stem = p.stem
        view = stem.replace("node_single_removal_impacts_", "")
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


def build_rbics_membership(nodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for row in nodes[["analysis_uid", "rbics_l4_ids", "rbics_l4_names"]].itertuples(index=False):
        uid = str(row.analysis_uid)
        ids = parse_pipe_list(row.rbics_l4_ids)
        names = parse_pipe_list(row.rbics_l4_names)
        if not ids and not names:
            continue

        pairs: list[tuple[str | None, str | None]] = []
        if ids and names and len(ids) == len(names):
            pairs = list(zip(ids, names, strict=False))
        elif ids and names:
            # Keep all IDs and align available names by position if possible.
            for i, cid in enumerate(ids):
                cname = names[i] if i < len(names) else None
                pairs.append((cid, cname))
        elif ids:
            pairs = [(cid, None) for cid in ids]
        else:
            pairs = [(None, cname) for cname in names]

        seen: set[tuple[str, str]] = set()
        for cid, cname in pairs:
            code_value = str(cid).strip() if cid is not None else ""
            code_name = str(cname).strip() if cname is not None else ""
            if not code_value and not code_name:
                continue
            k = (code_value, code_name)
            if k in seen:
                continue
            seen.add(k)
            rows.append(
                {
                    "analysis_uid": uid,
                    "code_system": "rbics_l4",
                    "code_value": code_value,
                    "code_name": code_name,
                }
            )
    return pd.DataFrame(rows)


def build_sic_membership(nodes: pd.DataFrame) -> pd.DataFrame:
    sic_nodes = nodes[nodes["sic_code_str"].notna()].copy()
    if sic_nodes.empty:
        return pd.DataFrame(columns=["analysis_uid", "code_system", "code_value", "code_name"])
    out = sic_nodes[["analysis_uid", "sic_code_str"]].copy()
    out["code_system"] = "sic"
    out["code_value"] = out["sic_code_str"].astype(str)
    out["code_name"] = None
    out = out[["analysis_uid", "code_system", "code_value", "code_name"]]
    return out


def aggregate_universe(
    membership: pd.DataFrame,
    node_flags: pd.DataFrame,
    top_ns: list[int],
    views: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if membership.empty:
        cols = [
            "code_system",
            "code_value",
            "code_name",
            "node_count",
            "prime_node_count",
            "semi_node_count",
        ]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    mem = membership.drop_duplicates(
        subset=["analysis_uid", "code_system", "code_value", "code_name"]
    ).copy()
    mem = mem.merge(node_flags, on="analysis_uid", how="left")
    mem["is_prime"] = mem["is_prime"].fillna(False).astype(bool)
    mem["is_semi"] = mem["is_semi"].fillna(False).astype(bool)
    mem["code_value"] = mem["code_value"].fillna("").astype(str).str.strip()
    mem["code_name"] = mem["code_name"].fillna("").astype(str).str.strip()

    def _mode_name(series: pd.Series) -> str:
        non_empty = series[series.astype(str).str.len() > 0].astype(str)
        if non_empty.empty:
            return ""
        vc = non_empty.value_counts()
        return str(vc.index[0])

    def _summarize(system: str) -> pd.DataFrame:
        df = mem[mem["code_system"] == system].copy()
        if df.empty:
            return pd.DataFrame()
        grouped = df.groupby(["code_system", "code_value"], dropna=False, as_index=False).agg(
            code_name=("code_name", _mode_name),
            node_count=("analysis_uid", "nunique"),
            prime_node_count=("is_prime", "sum"),
            semi_node_count=("is_semi", "sum"),
        )
        if system == "rbics_l4":
            grouped["code_name"] = (
                grouped["code_value"].map(RBICS_L4_NAME_OVERRIDES).fillna(grouped["code_name"])
            )
        for view in views:
            for n in top_ns:
                col = f"is_top{n}_{view}"
                if col not in df.columns:
                    continue
                tmp = (
                    df.groupby(["code_system", "code_value"], dropna=False)[col]
                    .sum()
                    .rename(f"top{n}_{view}_count")
                    .reset_index()
                )
                grouped = grouped.merge(tmp, on=["code_system", "code_value"], how="left")
                grouped[f"top{n}_{view}_count"] = (
                    grouped[f"top{n}_{view}_count"].fillna(0).astype(int)
                )
                grouped[f"top{n}_{view}_share_within_code"] = np.where(
                    grouped["node_count"] > 0,
                    grouped[f"top{n}_{view}_count"] / grouped["node_count"],
                    0.0,
                )

        top100_cols = [
            c for c in grouped.columns if c.startswith("top100_") and c.endswith("_count")
        ]
        top25_cols = [c for c in grouped.columns if c.startswith("top25_") and c.endswith("_count")]
        if top25_cols:
            grouped["top25_any_view_count"] = grouped[top25_cols].sum(axis=1).astype(int)
        else:
            grouped["top25_any_view_count"] = 0
        if top100_cols:
            grouped["top100_any_view_count"] = grouped[top100_cols].sum(axis=1).astype(int)
        else:
            grouped["top100_any_view_count"] = 0

        grouped = grouped.sort_values(
            ["top100_any_view_count", "top25_any_view_count", "node_count", "code_value"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        return grouped

    rbics = _summarize("rbics_l4")
    sic = _summarize("sic")
    return rbics, sic


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    out_dir = out_root / snapshot / "m0_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    m0_nodes_path = out_root / snapshot / "m0" / "node_table_contract.parquet"
    m6_1_dir = out_root / snapshot / "m6_1"
    ch3_nodes_path = Path(
        str(
            cfg.get("paths", {}).get(
                "nodes",
                "artifacts/ch3/network_upstream/dod_semiconductor_nodes_top5_d99_shipping_strict.parquet",
            )
        )
    )
    encoder_path = Path(args.encoder_pkl)
    top_ns = sorted({int(x.strip()) for x in str(args.topn).split(",") if x.strip()})
    if not top_ns:
        raise ValueError("topn must contain at least one integer")

    for p in [m0_nodes_path, ch3_nodes_path, encoder_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    nodes = pd.read_parquet(m0_nodes_path).copy()
    nodes["analysis_uid"] = nodes["analysis_uid"].astype(str)
    nodes["rep_scr_node_id"] = pd.to_numeric(nodes.get("rep_scr_node_id"), errors="coerce").astype(
        "Int64"
    )
    nodes["is_prime"] = nodes.get("has_prime_vendor", False).fillna(False).astype(bool)
    nodes["is_semi"] = nodes.get("is_semi_strict", False).fillna(False).astype(bool)

    sic_map = load_sic_map(scr_nodes_path=ch3_nodes_path, encoder_path=encoder_path)
    nodes = nodes.merge(sic_map, on="rep_scr_node_id", how="left")

    views = discover_views(m6_1_dir=m6_1_dir, cfg=cfg)
    if not views:
        raise ValueError(f"Could not discover views from {m6_1_dir}")
    top_sets, impact_hashes = load_top_sets(m6_1_dir=m6_1_dir, views=views, top_ns=top_ns)

    node_flags = nodes[
        [
            "analysis_uid",
            "is_prime",
            "is_semi",
            "entity_role",
            "rbics_l4_ids",
            "rbics_l4_names",
            "sic_code_str",
        ]
    ].copy()
    for view in views:
        for n in top_ns:
            flag = f"is_top{n}_{view}"
            node_flags[flag] = node_flags["analysis_uid"].isin(top_sets[(view, n)])

    rbics_membership = build_rbics_membership(nodes=node_flags)
    sic_membership = build_sic_membership(nodes=node_flags)
    membership = pd.concat([rbics_membership, sic_membership], ignore_index=True)
    membership = membership.drop_duplicates(
        subset=["analysis_uid", "code_system", "code_value", "code_name"]
    ).copy()

    rbics_universe, sic_universe = aggregate_universe(
        membership=membership, node_flags=node_flags, top_ns=top_ns, views=views
    )

    node_inputs = node_flags.copy()
    node_inputs["has_rbics_l4"] = node_inputs["rbics_l4_ids"].fillna("").astype(str).str.len() > 0
    node_inputs["has_sic"] = node_inputs["sic_code_str"].notna()
    node_inputs["has_any_industry_label"] = node_inputs["has_rbics_l4"] | node_inputs["has_sic"]

    # Outputs
    node_inputs_csv = out_dir / "node_code_universe_inputs.csv"
    node_inputs_parquet = out_dir / "node_code_universe_inputs.parquet"
    rbics_csv = out_dir / "code_universe_rbics_l4.csv"
    rbics_parquet = out_dir / "code_universe_rbics_l4.parquet"
    sic_csv = out_dir / "code_universe_sic.csv"
    sic_parquet = out_dir / "code_universe_sic.parquet"
    membership_csv = out_dir / "node_code_membership.csv"
    membership_parquet = out_dir / "node_code_membership.parquet"

    node_inputs.to_csv(node_inputs_csv, index=False)
    node_inputs.to_parquet(node_inputs_parquet, index=False)
    rbics_universe.to_csv(rbics_csv, index=False)
    rbics_universe.to_parquet(rbics_parquet, index=False)
    sic_universe.to_csv(sic_csv, index=False)
    sic_universe.to_parquet(sic_parquet, index=False)
    membership.to_csv(membership_csv, index=False)
    membership.to_parquet(membership_parquet, index=False)

    input_hashes = {
        str(m0_nodes_path): file_sha256(m0_nodes_path),
        str(ch3_nodes_path): file_sha256(ch3_nodes_path),
        str(encoder_path): file_sha256(encoder_path),
        **impact_hashes,
    }
    outputs = {
        "node_code_universe_inputs_csv": str(node_inputs_csv),
        "node_code_universe_inputs_parquet": str(node_inputs_parquet),
        "node_code_membership_csv": str(membership_csv),
        "node_code_membership_parquet": str(membership_parquet),
        "code_universe_rbics_l4_csv": str(rbics_csv),
        "code_universe_rbics_l4_parquet": str(rbics_parquet),
        "code_universe_sic_csv": str(sic_csv),
        "code_universe_sic_parquet": str(sic_parquet),
    }

    run_metadata = {
        "module": "m0_5",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "platform": platform.platform(),
        "settings": {
            "views": views,
            "top_ns": top_ns,
            "encoder_pkl": str(encoder_path),
        },
        "summary": {
            "nodes_total": len(node_inputs),
            "nodes_with_rbics_l4": int(node_inputs["has_rbics_l4"].sum()),
            "nodes_with_sic": int(node_inputs["has_sic"].sum()),
            "nodes_with_any_label": int(node_inputs["has_any_industry_label"].sum()),
            "rbics_code_count": int(rbics_universe["code_value"].nunique())
            if not rbics_universe.empty
            else 0,
            "sic_code_count": int(sic_universe["code_value"].nunique())
            if not sic_universe.empty
            else 0,
        },
    }
    run_metadata_path = out_dir / "run_metadata.json"
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m0_5",
        "snapshot": snapshot,
        "run_id": run_id,
        "inputs": input_hashes,
        "outputs": outputs,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "manifest_m0_5.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        "[m0_5] complete "
        f"nodes={len(node_inputs):,} rbics_codes={run_metadata['summary']['rbics_code_count']:,} "
        f"sic_codes={run_metadata['summary']['sic_code_count']:,}"
    )
    print(f"[m0_5] outputs: {out_dir}")


if __name__ == "__main__":
    main()
