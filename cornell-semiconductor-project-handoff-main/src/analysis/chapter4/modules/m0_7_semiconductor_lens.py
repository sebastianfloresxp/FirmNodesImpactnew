#!/usr/bin/env python3
"""Module 0.7: apply semiconductor relevance codebook as a reporting lens.

This module does NOT modify graph topology or rerun structural metrics.
It applies a reproducible node-level relevance lens to existing Chapter 4
outputs and writes auditable include/exclude/uncertain decisions.

Outputs under artifacts/.../m0_7:
  - node_semiconductor_lens_decisions.csv/.parquet
  - node_semiconductor_lens_rule_hits.csv/.parquet
  - codebook_rule_coverage.csv
  - lens_decision_summary.csv
  - lens_topn_coverage_by_view.csv
  - node_impacts_semiconductor_lens_<view>.csv
  - node_impacts_semiconductor_lens_plus_uncertain_<view>.csv
  - top{N}_high_impact_semiconductor_lens_<view>.csv
  - top{N}_high_impact_semiconductor_lens_plus_uncertain_<view>.csv
  - manifest_m0_7.json
  - run_metadata.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from collections import deque
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

try:
    from db_client import run_query
except Exception:
    run_query = None


REQUIRED_CODEBOOK_COLS = [
    "rule_id",
    "code_system",
    "code_value",
    "code_name",
    "rule_class",
    "include_semiconductor_relevant",
    "include_semiconductor_relevant_plus_uncertain",
    "requires_evidence_gate",
    "evidence_gate_id",
    "rationale_short",
    "source_note",
    "version",
    "active_flag",
]

SYSTEM_PRIORITY = {
    "rbics_l4_latest": 1,
    "rbics_l2": 2,
    "sic_raw": 3,
}

DEFAULT_TOP_NS = [25, 100, 300]
DEFAULT_VIEWS = ["disclosed", "observed", "full"]
DEFAULT_TIE_BREAKERS = ["h2_disconnect_share", "h2_path_growth", "analysis_uid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chapter 4 Module 0.7 semiconductor relevance lens"
    )
    parser.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2_fix01.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--codebook",
        default=None,
        help="Optional override path to semiconductor codebook CSV",
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


def truthy_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_code_value(series: pd.Series) -> pd.Series:
    # Keep deterministic string keys for joins.
    out = clean_text(series)
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def to_bool(series: pd.Series) -> pd.Series:
    return truthy_int(series).astype(bool)


def pick_first_nonempty(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series([""] * len(frame), index=frame.index, dtype=object)
    for col in candidates:
        if col not in frame.columns:
            continue
        val = frame[col].fillna("").astype(str).str.strip()
        mask = out.astype(str).str.len().eq(0) & val.astype(str).str.len().gt(0)
        out.loc[mask] = val.loc[mask]
    return out


def load_codebook(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_CODEBOOK_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"codebook missing columns: {missing}")

    out = df.copy()
    out["rule_id"] = clean_text(out["rule_id"])
    out["code_system"] = clean_text(out["code_system"]).str.lower()
    out["code_value"] = normalize_code_value(out["code_value"])
    out["code_name"] = clean_text(out["code_name"])
    out["rule_class"] = clean_text(out["rule_class"])
    out["include_semiconductor_relevant"] = truthy_int(out["include_semiconductor_relevant"])
    out["include_semiconductor_relevant_plus_uncertain"] = truthy_int(
        out["include_semiconductor_relevant_plus_uncertain"]
    )
    out["requires_evidence_gate"] = truthy_int(out["requires_evidence_gate"])
    out["evidence_gate_id"] = clean_text(out["evidence_gate_id"])
    out["rationale_short"] = clean_text(out["rationale_short"])
    out["source_note"] = clean_text(out["source_note"])
    out["version"] = clean_text(out["version"])
    out["active_flag"] = truthy_int(out["active_flag"])
    out = out[out["active_flag"] == 1].copy()
    out["system_priority"] = out["code_system"].map(SYSTEM_PRIORITY).fillna(99).astype(int)
    if out["rule_id"].duplicated().any():
        dup = out[out["rule_id"].duplicated(keep=False)]["rule_id"].tolist()
        raise ValueError(f"duplicate rule_id values found in codebook: {sorted(set(dup))}")
    return out


def multi_source_reach(adjacency: list[list[int]], sources: np.ndarray) -> np.ndarray:
    n = len(adjacency)
    seen = np.zeros(n, dtype=bool)
    q: deque[int] = deque()
    for s in sources.tolist():
        si = int(s)
        if 0 <= si < n and not seen[si]:
            seen[si] = True
            q.append(si)
    while q:
        u = q.popleft()
        for v in adjacency[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return seen


def compute_empirical_gate_flags_for_view(
    node_uids: np.ndarray,
    edges: pd.DataFrame,
    include_any: list[str],
    semi_uids: np.ndarray,
    prime_uids: np.ndarray,
    require_empirical_support: bool = True,
) -> pd.DataFrame:
    if len(include_any) == 0:
        raise ValueError("include_any must not be empty for gate computation")

    required_cols = {"src_uid", "dst_uid", "is_disclosed", "is_observed_ship"}
    missing_required = sorted(required_cols - set(edges.columns))
    if missing_required:
        raise ValueError(
            f"edge table missing required columns for gate computation: {missing_required}"
        )
    missing_flags = sorted(set(include_any) - set(edges.columns))
    if missing_flags:
        raise ValueError(
            f"edge table missing include_any columns for gate computation: {missing_flags}"
        )

    uid_to_idx = {str(uid): i for i, uid in enumerate(node_uids.tolist())}
    n_nodes = len(node_uids)
    semi_idx = np.array(
        [uid_to_idx[u] for u in semi_uids.tolist() if u in uid_to_idx], dtype=np.int32
    )
    prime_idx = np.array(
        [uid_to_idx[u] for u in prime_uids.tolist() if u in uid_to_idx], dtype=np.int32
    )

    # View mask first, then optionally enforce empirical-only support (disclosed/shipping).
    view_mask = np.zeros(len(edges), dtype=bool)
    for col in include_any:
        view_mask |= edges[col].astype(bool).to_numpy()
    if require_empirical_support:
        empirical_support = (
            edges["is_disclosed"].astype(bool).to_numpy()
            | edges["is_observed_ship"].astype(bool).to_numpy()
        )
        mask = view_mask & empirical_support
    else:
        mask = view_mask

    edge_view = edges.loc[mask, ["src_uid", "dst_uid"]].copy()
    edge_view["src_uid"] = clean_text(edge_view["src_uid"])
    edge_view["dst_uid"] = clean_text(edge_view["dst_uid"])
    edge_view["src_idx"] = edge_view["src_uid"].map(uid_to_idx)
    edge_view["dst_idx"] = edge_view["dst_uid"].map(uid_to_idx)
    edge_view = edge_view[edge_view["src_idx"].notna() & edge_view["dst_idx"].notna()].copy()
    if not edge_view.empty:
        edge_view["src_idx"] = edge_view["src_idx"].astype(np.int32)
        edge_view["dst_idx"] = edge_view["dst_idx"].astype(np.int32)
        edge_view = edge_view[edge_view["src_idx"] != edge_view["dst_idx"]]
        edge_view = edge_view.drop_duplicates(subset=["src_idx", "dst_idx"], keep="first")

    forward: list[list[int]] = [[] for _ in range(n_nodes)]
    reverse: list[list[int]] = [[] for _ in range(n_nodes)]
    if not edge_view.empty:
        for u, v in edge_view[["src_idx", "dst_idx"]].itertuples(index=False, name=None):
            ui = int(u)
            vi = int(v)
            forward[ui].append(vi)
            reverse[vi].append(ui)

    from_semi = (
        multi_source_reach(forward, semi_idx)
        if len(semi_idx) > 0
        else np.zeros(n_nodes, dtype=bool)
    )
    to_prime = (
        multi_source_reach(reverse, prime_idx)
        if len(prime_idx) > 0
        else np.zeros(n_nodes, dtype=bool)
    )
    two_sided = from_semi & to_prime

    return pd.DataFrame(
        {
            "analysis_uid": node_uids.astype(str),
            "gate_empirical_two_sided_corridor": two_sided.astype(bool),
        }
    )


def load_confidence_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["analysis_uid", "confidence_class", "low_conf_reason", "presence_count"]
        )
    df = pd.read_csv(path)
    keep = [
        c
        for c in ["analysis_uid", "confidence_class", "low_conf_reason", "presence_count"]
        if c in df.columns
    ]
    out = df[keep].copy()
    out["analysis_uid"] = clean_text(out["analysis_uid"])
    if "confidence_class" in out.columns:
        out["confidence_class"] = clean_text(out["confidence_class"]).replace("", pd.NA)
    if "low_conf_reason" in out.columns:
        out["low_conf_reason"] = clean_text(out["low_conf_reason"]).replace("", pd.NA)
    if "presence_count" in out.columns:
        out["presence_count"] = pd.to_numeric(out["presence_count"], errors="coerce").astype(
            "Int64"
        )
    return out


def quote_ids(ids: list[str]) -> str:
    return ", ".join("'" + x.replace("'", "''") + "'" for x in ids)


def iter_chunks(values: list[str], size: int):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def fetch_factset_names(factset_ids: list[str], chunk_size: int = 1000) -> pd.DataFrame:
    cols = ["factset_entity_id", "entity_proper_name", "iso_country"]
    if run_query is None or len(factset_ids) == 0:
        return pd.DataFrame(columns=cols)

    frames: list[pd.DataFrame] = []
    for chunk in iter_chunks(factset_ids, max(1, int(chunk_size))):
        q = (
            "SELECT factset_entity_id, entity_proper_name, iso_country "
            "FROM sym_v1.sym_entity "
            f"WHERE factset_entity_id IN ({quote_ids(chunk)})"
        )
        batch = run_query(q)
        if batch is None or batch.empty:
            continue
        frames.append(batch.copy())

    if not frames:
        return pd.DataFrame(columns=cols)

    out = pd.concat(frames, ignore_index=True)
    out["factset_entity_id"] = clean_text(out["factset_entity_id"])
    out["entity_proper_name"] = clean_text(out["entity_proper_name"]).replace("", pd.NA)
    out["iso_country"] = clean_text(out["iso_country"]).replace("", pd.NA)
    out = out.sort_values(
        ["factset_entity_id", "entity_proper_name"], ascending=[True, True], kind="mergesort"
    )
    out = out.drop_duplicates(subset=["factset_entity_id"], keep="first").reset_index(drop=True)
    return out[cols]


def evaluate_node(uid: str, group: pd.DataFrame) -> dict[str, Any]:
    uid = str(uid)
    matched_count_total = len(group)
    gate_pass_rows = group[group["gate_pass"]].copy()
    matched_count_gate_pass = len(gate_pass_rows)

    if matched_count_total == 0:
        return {
            "analysis_uid": uid,
            "decision_strict": "uncertain_no_rule",
            "decision_plus_uncertain": "uncertain_no_rule",
            "include_semiconductor_lens": False,
            "include_semiconductor_lens_plus_uncertain": False,
            "selected_code_system": pd.NA,
            "selected_system_priority": pd.NA,
            "matched_rule_count_total": matched_count_total,
            "matched_rule_count_gate_pass": matched_count_gate_pass,
            "selected_rule_count": 0,
            "selected_rule_ids": "",
            "selected_code_values": "",
            "selected_rule_classes": "",
            "gate_required_any": False,
            "gate_failed_count": 0,
            "has_conflicting_selected_rules": False,
            "decision_reason_strict": "no_matching_codebook_rule",
            "decision_reason_plus_uncertain": "no_matching_codebook_rule",
        }

    gate_required_any = bool(group["requires_evidence_gate"].astype(bool).any())
    gate_failed_count = int((~group["gate_pass"]).sum())
    if gate_pass_rows.empty:
        return {
            "analysis_uid": uid,
            "decision_strict": "uncertain_gate_not_met",
            "decision_plus_uncertain": "uncertain_gate_not_met",
            "include_semiconductor_lens": False,
            "include_semiconductor_lens_plus_uncertain": False,
            "selected_code_system": pd.NA,
            "selected_system_priority": pd.NA,
            "matched_rule_count_total": matched_count_total,
            "matched_rule_count_gate_pass": matched_count_gate_pass,
            "selected_rule_count": 0,
            "selected_rule_ids": "",
            "selected_code_values": "",
            "selected_rule_classes": "",
            "gate_required_any": gate_required_any,
            "gate_failed_count": gate_failed_count,
            "has_conflicting_selected_rules": False,
            "decision_reason_strict": "all_matching_rules_failed_gate",
            "decision_reason_plus_uncertain": "all_matching_rules_failed_gate",
        }

    best_priority = int(gate_pass_rows["system_priority"].min())
    selected = gate_pass_rows[gate_pass_rows["system_priority"] == best_priority].copy()
    selected_code_system = str(selected["code_system"].iloc[0]) if len(selected) else ""

    strict_inc = bool((selected["include_semiconductor_relevant"] == 1).any())
    excl = bool(
        (
            (selected["include_semiconductor_relevant"] == 0)
            & (selected["include_semiconductor_relevant_plus_uncertain"] == 0)
        ).any()
    )
    plus_inc = bool((selected["include_semiconductor_relevant_plus_uncertain"] == 1).any())
    uncertain_only = bool(
        (
            (selected["include_semiconductor_relevant"] == 0)
            & (selected["include_semiconductor_relevant_plus_uncertain"] == 1)
        ).any()
    )

    if strict_inc and not excl:
        decision_strict = "include"
        reason_strict = "selected_rules_include"
    elif excl and not strict_inc:
        decision_strict = "exclude"
        reason_strict = "selected_rules_exclude"
    elif strict_inc and excl:
        decision_strict = "uncertain_conflict"
        reason_strict = "include_exclude_conflict_same_priority"
    elif uncertain_only:
        decision_strict = "uncertain_label"
        reason_strict = "selected_rules_uncertain_only"
    else:
        decision_strict = "uncertain_other"
        reason_strict = "selected_rules_no_decisive_signal"

    if plus_inc and not excl:
        decision_plus = "include"
        reason_plus = "selected_rules_include_or_uncertain"
    elif excl and not plus_inc:
        decision_plus = "exclude"
        reason_plus = "selected_rules_exclude"
    elif plus_inc and excl:
        decision_plus = "uncertain_conflict"
        reason_plus = "include_exclude_conflict_same_priority"
    else:
        decision_plus = "uncertain_other"
        reason_plus = "selected_rules_no_decisive_signal"

    selected_rule_ids = "|".join(sorted(set(clean_text(selected["rule_id"]).tolist())))
    selected_code_values = "|".join(sorted(set(clean_text(selected["code_value"]).tolist())))
    selected_rule_classes = "|".join(sorted(set(clean_text(selected["rule_class"]).tolist())))

    return {
        "analysis_uid": uid,
        "decision_strict": decision_strict,
        "decision_plus_uncertain": decision_plus,
        "include_semiconductor_lens": bool(decision_strict == "include"),
        "include_semiconductor_lens_plus_uncertain": bool(decision_plus == "include"),
        "selected_code_system": selected_code_system if selected_code_system else pd.NA,
        "selected_system_priority": int(best_priority),
        "matched_rule_count_total": matched_count_total,
        "matched_rule_count_gate_pass": matched_count_gate_pass,
        "selected_rule_count": len(selected),
        "selected_rule_ids": selected_rule_ids,
        "selected_code_values": selected_code_values,
        "selected_rule_classes": selected_rule_classes,
        "gate_required_any": gate_required_any,
        "gate_failed_count": gate_failed_count,
        "has_conflicting_selected_rules": bool(strict_inc and excl),
        "decision_reason_strict": reason_strict,
        "decision_reason_plus_uncertain": reason_plus,
    }


def build_decisions_for_view(
    view: str,
    hits_base: pd.DataFrame,
    node_base: pd.DataFrame,
    gate_flags_view: pd.DataFrame,
    apply_endpoint_overrides: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hits = hits_base.merge(gate_flags_view, on="analysis_uid", how="left")
    for c in ["gate_empirical_two_sided_corridor", "gate_empirical_intermediary_corridor"]:
        hits[c] = hits[c].fillna(False).astype(bool)

    def resolve_gate_pass(row: pd.Series) -> bool:
        if int(row.get("requires_evidence_gate", 0)) == 0:
            return True
        gate_id = str(row.get("evidence_gate_id", "")).strip().lower()
        if gate_id == "gate_empirical_two_sided_corridor":
            return bool(row.get("gate_empirical_two_sided_corridor", False))
        if gate_id == "gate_empirical_intermediary_corridor":
            return bool(row.get("gate_empirical_intermediary_corridor", False))
        # Unknown gate IDs are treated as failed to avoid silent over-inclusion.
        return False

    if not hits.empty:
        hits["gate_pass"] = hits.apply(resolve_gate_pass, axis=1).astype(bool)
    else:
        hits["gate_pass"] = pd.Series(dtype=bool)

    grouped = {uid: g for uid, g in hits.groupby("analysis_uid", sort=False)} if not hits.empty else {}  # noqa: C416 -- dict() on GroupBy raises TypeError in pandas ≥2.1
    decision_rows: list[dict[str, Any]] = []
    for uid in node_base["analysis_uid"].astype(str).tolist():
        g = grouped.get(uid)
        if g is None:
            g = pd.DataFrame(columns=hits.columns)
        decision_rows.append(evaluate_node(uid, g))

    decisions = pd.DataFrame(decision_rows)
    decisions["analysis_uid"] = clean_text(decisions["analysis_uid"])
    decisions["include_semiconductor_lens"] = (
        decisions["include_semiconductor_lens"].fillna(False).astype(bool)
    )
    decisions["include_semiconductor_lens_plus_uncertain"] = (
        decisions["include_semiconductor_lens_plus_uncertain"].fillna(False).astype(bool)
    )
    decisions = decisions.merge(gate_flags_view, on="analysis_uid", how="left")
    for c in ["gate_empirical_two_sided_corridor", "gate_empirical_intermediary_corridor"]:
        decisions[c] = decisions[c].fillna(False).astype(bool)
    decisions["view"] = view

    # Optional endpoint override: force include for prime/semi anchors.
    decisions["endpoint_override_applied"] = False
    if apply_endpoint_overrides:
        merged = decisions.merge(
            node_base[
                [
                    "analysis_uid",
                    "is_prime",
                    "has_prime_vendor",
                    "is_semi",
                    "is_semi_strict",
                ]
            ],
            on="analysis_uid",
            how="left",
        )
        for c in ["is_prime", "has_prime_vendor", "is_semi", "is_semi_strict"]:
            merged[c] = merged[c].fillna(False).astype(bool)
        is_prime_anchor = merged["is_prime"] | merged["has_prime_vendor"]
        is_semi_anchor = merged["is_semi"] | merged["is_semi_strict"]
        override_mask = is_prime_anchor | is_semi_anchor

        merged["decision_strict_pre_override"] = merged["decision_strict"]
        merged["decision_plus_uncertain_pre_override"] = merged["decision_plus_uncertain"]
        merged.loc[override_mask, "decision_strict"] = "include"
        merged.loc[override_mask, "decision_plus_uncertain"] = "include"
        merged.loc[override_mask, "include_semiconductor_lens"] = True
        merged.loc[override_mask, "include_semiconductor_lens_plus_uncertain"] = True
        merged.loc[override_mask, "decision_reason_strict"] = (
            "endpoint_override_prime_or_semi_anchor"
        )
        merged.loc[override_mask, "decision_reason_plus_uncertain"] = (
            "endpoint_override_prime_or_semi_anchor"
        )
        merged.loc[override_mask, "endpoint_override_applied"] = True
        merged.loc[
            override_mask
            & merged["selected_rule_classes"].fillna("").astype(str).str.strip().eq(""),
            "selected_rule_classes",
        ] = "endpoint_override"
        merged.loc[
            override_mask & merged["selected_rule_ids"].fillna("").astype(str).str.strip().eq(""),
            "selected_rule_ids",
        ] = "OVR_ENDPOINT"
        merged.loc[
            override_mask & merged["selected_code_system"].isna(),
            "selected_code_system",
        ] = "endpoint_override"

        decisions = merged.drop(
            columns=["is_prime", "has_prime_vendor", "is_semi", "is_semi_strict"]
        )

    return decisions, hits


def build_rule_coverage_for_view(
    view: str, hits: pd.DataFrame, decisions: pd.DataFrame
) -> pd.DataFrame:
    if hits.empty:
        return pd.DataFrame(
            columns=[
                "view",
                "rule_id",
                "code_system",
                "code_value",
                "code_name",
                "rule_class",
                "matched_nodes",
                "gate_pass_nodes",
                "gate_failed_nodes",
                "selected_nodes",
                "selected_share_within_matched",
            ]
        )

    selected_hits = hits.merge(
        decisions[["analysis_uid", "selected_rule_ids"]],
        on="analysis_uid",
        how="left",
    )
    if "code_name_rule" in selected_hits.columns:
        selected_hits["code_name_eff"] = clean_text(selected_hits["code_name_rule"])
    elif "code_name" in selected_hits.columns:
        selected_hits["code_name_eff"] = clean_text(selected_hits["code_name"])
    else:
        selected_hits["code_name_eff"] = ""

    selected_hits["is_selected_rule"] = selected_hits.apply(
        lambda r: (
            str(r["rule_id"]) in set(str(r.get("selected_rule_ids", "")).split("|"))
            if str(r.get("selected_rule_ids", "")).strip()
            else False
        ),
        axis=1,
    )

    coverage = (
        selected_hits.groupby(
            ["rule_id", "code_system", "code_value", "code_name_eff", "rule_class"],
            as_index=False,
        )
        .agg(
            matched_nodes=("analysis_uid", "nunique"),
            gate_pass_nodes=(
                "gate_pass",
                lambda s: int(selected_hits.loc[s.index[s.astype(bool)], "analysis_uid"].nunique()),
            ),
            gate_failed_nodes=(
                "gate_pass",
                lambda s: int(
                    selected_hits.loc[s.index[~s.astype(bool)], "analysis_uid"].nunique()
                ),
            ),
            selected_nodes=(
                "is_selected_rule",
                lambda s: int(selected_hits.loc[s.index[s.astype(bool)], "analysis_uid"].nunique()),
            ),
        )
        .copy()
    )
    coverage = coverage.rename(columns={"code_name_eff": "code_name"})
    coverage["selected_share_within_matched"] = np.where(
        coverage["matched_nodes"] > 0,
        coverage["selected_nodes"] / coverage["matched_nodes"],
        np.nan,
    )
    coverage["view"] = view
    coverage = coverage.sort_values(
        ["selected_nodes", "gate_pass_nodes", "matched_nodes", "rule_id"],
        ascending=[False, False, False, True],
    )
    return coverage


def rank_table(
    impacts: pd.DataFrame,
    rank_metric: str,
    tie_breakers: list[str],
) -> pd.DataFrame:
    sort_cols: list[str] = []
    ascending: list[bool] = []

    if rank_metric not in impacts.columns:
        fallback = "h1_reach_loss" if "h1_reach_loss" in impacts.columns else None
        if fallback is None:
            raise ValueError(
                f"ranking metric {rank_metric} missing and fallback h1_reach_loss unavailable"
            )
        rank_metric = fallback

    sort_cols.append(rank_metric)
    ascending.append(False)

    for c in tie_breakers:
        if c == rank_metric:
            continue
        if c in impacts.columns and c != "analysis_uid":
            sort_cols.append(c)
            ascending.append(False)
    if "analysis_uid" in impacts.columns:
        sort_cols.append("analysis_uid")
        ascending.append(True)

    out = impacts.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(
        drop=True
    )
    out["rank_semiconductor_lens"] = np.arange(1, len(out) + 1, dtype=np.int32)
    out["rank_metric_used"] = rank_metric
    return out


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "unspecified_run"))
    out_root = Path(str(cfg.get("paths", {}).get("out_root", "artifacts/ch4/v2_fix01")))
    out_dir = out_root / snapshot / "m0_7"
    out_dir.mkdir(parents=True, exist_ok=True)

    m0_7_cfg = cfg.get("m0_7", {})
    codebook_path = Path(
        args.codebook
        if args.codebook
        else str(m0_7_cfg.get("codebook_path", "configs/ch4_semiconductor_codebook_v1.csv"))
    )
    primary_decision_view = str(
        m0_7_cfg.get("primary_decision_view", m0_7_cfg.get("gate_view", "observed"))
    )
    views = [str(v) for v in m0_7_cfg.get("views", cfg.get("m7", {}).get("views", DEFAULT_VIEWS))]
    top_ns = [int(n) for n in m0_7_cfg.get("top_n", DEFAULT_TOP_NS)]
    ranking_metric = str(m0_7_cfg.get("ranking_metric", "h1_log_obligation_any_support"))
    tie_breakers = [str(x) for x in m0_7_cfg.get("tie_breakers", DEFAULT_TIE_BREAKERS)]
    fetch_factset_names_enabled = bool(m0_7_cfg.get("fetch_factset_names", True))
    factset_name_chunk_size = int(m0_7_cfg.get("factset_name_chunk_size", 1000))
    apply_endpoint_overrides = bool(m0_7_cfg.get("apply_endpoint_overrides", True))
    gate_empirical_only = bool(m0_7_cfg.get("gate_empirical_only", True))

    if primary_decision_view not in views:
        raise ValueError(
            f"primary_decision_view={primary_decision_view} not in configured views={views}"
        )

    m0_dir = out_root / snapshot / "m0"
    m0_6_dir = out_root / snapshot / "m0_6"
    m6_2_dir = out_root / snapshot / "m6_2"
    m6_1_dir = out_root / snapshot / "m6_1"
    m7_dir = out_root / snapshot / "m7"

    input_paths = {
        "codebook": codebook_path,
        "node_enrichment": m0_6_dir / "node_industry_enrichment_factset.parquet",
        "node_membership": m0_6_dir / "node_code_membership_factset.parquet",
        "m0_nodes": m0_dir / "node_table_contract.parquet",
        "edge_table": m0_dir / "edge_table_contract.parquet",
        "confidence_table": m7_dir / "high_impact_confidence_quadrants.csv",
    }
    for p in input_paths.values():
        if not p.exists():
            raise FileNotFoundError(p)

    codebook = load_codebook(input_paths["codebook"])
    enrichment = pd.read_parquet(input_paths["node_enrichment"]).copy()
    membership = pd.read_parquet(input_paths["node_membership"]).copy()
    m0_nodes = pd.read_parquet(
        input_paths["m0_nodes"], columns=["analysis_uid", "name", "gr_country", "gr_region"]
    ).copy()
    edges = pd.read_parquet(
        input_paths["edge_table"],
        columns=["src_uid", "dst_uid", "is_disclosed", "is_observed_ship", "is_predicted"],
    ).copy()
    confidence = load_confidence_map(input_paths["confidence_table"])

    enrichment["analysis_uid"] = clean_text(enrichment["analysis_uid"])
    membership["analysis_uid"] = clean_text(membership["analysis_uid"])
    membership["code_system"] = clean_text(membership["code_system"]).str.lower()
    membership["code_value"] = normalize_code_value(membership["code_value"])
    m0_nodes["analysis_uid"] = clean_text(m0_nodes["analysis_uid"])
    m0_nodes["name"] = clean_text(m0_nodes["name"]).replace("", pd.NA)
    m0_nodes["gr_country"] = clean_text(m0_nodes["gr_country"]).replace("", pd.NA)
    m0_nodes["gr_region"] = clean_text(m0_nodes["gr_region"]).replace("", pd.NA)

    # Node base table for decisions.
    node_base = enrichment.copy()
    node_base["name"] = pick_first_nonempty(node_base, ["name"])
    node_base = node_base.merge(m0_nodes, on="analysis_uid", how="left", suffixes=("", "_m0"))
    node_base["factset_entity_id"] = clean_text(
        node_base.get("factset_entity_id", pd.Series([""] * len(node_base)))
    )

    factset_lookup = pd.DataFrame(
        columns=["factset_entity_id", "entity_proper_name", "iso_country"]
    )
    if fetch_factset_names_enabled and run_query is not None:
        fs_ids = sorted({x for x in node_base["factset_entity_id"].tolist() if x})
        factset_lookup = fetch_factset_names(fs_ids, chunk_size=factset_name_chunk_size)
        if not factset_lookup.empty:
            node_base = node_base.merge(factset_lookup, on="factset_entity_id", how="left")

    node_base["name"] = pick_first_nonempty(
        node_base, ["name", "name_m0", "entity_proper_name"]
    ).replace("", pd.NA)
    node_base["iso_country"] = pick_first_nonempty(
        node_base, ["iso_country", "iso_country_x", "iso_country_y"]
    ).replace("", pd.NA)
    node_base["country_code_factset"] = pick_first_nonempty(node_base, ["gr_country"]).replace(
        "", pd.NA
    )
    node_base["region_code_factset"] = pick_first_nonempty(node_base, ["gr_region"]).replace(
        "", pd.NA
    )
    for c in ["is_prime", "has_prime_vendor", "is_semi", "is_semi_strict"]:
        if c not in node_base.columns:
            node_base[c] = False
        node_base[c] = node_base[c].fillna(False).astype(bool)
    node_base["is_firm_role"] = clean_text(node_base["entity_role"]).eq("firm")

    # Base rule hits (gate flags are applied per view below).
    hits_base = membership.merge(
        codebook[
            [
                "rule_id",
                "code_system",
                "code_value",
                "code_name",
                "rule_class",
                "include_semiconductor_relevant",
                "include_semiconductor_relevant_plus_uncertain",
                "requires_evidence_gate",
                "evidence_gate_id",
                "rationale_short",
                "source_note",
                "version",
                "system_priority",
            ]
        ],
        on=["code_system", "code_value"],
        how="inner",
        suffixes=("_node", "_rule"),
    )
    hits_base["analysis_uid"] = clean_text(hits_base["analysis_uid"])

    # Compute true two-sided empirical gates per view.
    node_uids = node_base["analysis_uid"].astype(str).to_numpy()
    semi_uids = (
        node_base.loc[node_base["is_semi"] | node_base["is_semi_strict"], "analysis_uid"]
        .astype(str)
        .to_numpy()
    )
    prime_uids = (
        node_base.loc[node_base["is_prime"] | node_base["has_prime_vendor"], "analysis_uid"]
        .astype(str)
        .to_numpy()
    )

    gate_flags_by_view: dict[str, pd.DataFrame] = {}
    gate_effective_include_any: dict[str, list[str]] = {}
    for view in views:
        include_any = [str(c) for c in cfg.get("views", {}).get(view, {}).get("include_any", [])]
        if not include_any:
            raise ValueError(f"no include_any configured for view={view}")
        effective_include_any = include_any
        if gate_empirical_only:
            # For empirical gate checks, predicted-only links are excluded.
            effective_include_any = [c for c in include_any if c != "is_predicted"]
            if len(effective_include_any) == 0:
                effective_include_any = ["is_disclosed", "is_observed_ship"]
        gate_effective_include_any[view] = effective_include_any

        gate_df = compute_empirical_gate_flags_for_view(
            node_uids=node_uids,
            edges=edges,
            include_any=effective_include_any,
            semi_uids=semi_uids,
            prime_uids=prime_uids,
            require_empirical_support=gate_empirical_only,
        )
        gate_df = gate_df.merge(
            node_base[
                [
                    "analysis_uid",
                    "is_firm_role",
                    "is_prime",
                    "has_prime_vendor",
                    "is_semi",
                    "is_semi_strict",
                ]
            ],
            on="analysis_uid",
            how="left",
        )
        for c in ["is_firm_role", "is_prime", "has_prime_vendor", "is_semi", "is_semi_strict"]:
            gate_df[c] = gate_df[c].fillna(False).astype(bool)
        gate_df["gate_empirical_intermediary_corridor"] = (
            gate_df["gate_empirical_two_sided_corridor"]
            & gate_df["is_firm_role"]
            & ~(gate_df["is_prime"] | gate_df["has_prime_vendor"])
            & ~(gate_df["is_semi"] | gate_df["is_semi_strict"])
        )
        gate_df["gate_definition"] = "two_sided_empirical_paths"
        gate_df["view"] = view
        gate_flags_by_view[view] = gate_df[
            [
                "analysis_uid",
                "gate_empirical_two_sided_corridor",
                "gate_empirical_intermediary_corridor",
                "gate_definition",
                "view",
            ]
        ].copy()

    decisions_by_view: dict[str, pd.DataFrame] = {}
    decision_tables_by_view: dict[str, pd.DataFrame] = {}
    hits_by_view: dict[str, pd.DataFrame] = {}
    coverage_parts: list[pd.DataFrame] = []
    for view in views:
        decisions_v, hits_v = build_decisions_for_view(
            view=view,
            hits_base=hits_base,
            node_base=node_base,
            gate_flags_view=gate_flags_by_view[view],
            apply_endpoint_overrides=apply_endpoint_overrides,
        )
        decisions_by_view[view] = decisions_v
        hits_by_view[view] = hits_v
        coverage_parts.append(
            build_rule_coverage_for_view(view=view, hits=hits_v, decisions=decisions_v)
        )

        table_v = node_base.merge(decisions_v, on="analysis_uid", how="left")
        table_v = table_v.merge(confidence, on="analysis_uid", how="left")
        table_v["confidence_class"] = clean_text(table_v["confidence_class"]).replace("", pd.NA)
        table_v["low_conf_reason"] = clean_text(table_v["low_conf_reason"]).replace("", pd.NA)
        table_v["presence_count"] = pd.to_numeric(
            table_v["presence_count"], errors="coerce"
        ).astype("Int64")
        decision_tables_by_view[view] = table_v

    decision_table = decision_tables_by_view[primary_decision_view].copy()
    decisions_by_view[primary_decision_view].copy()
    hits = hits_by_view[primary_decision_view].copy()
    coverage = pd.concat(coverage_parts, ignore_index=True) if coverage_parts else pd.DataFrame()

    decisions_long = pd.concat(
        [decision_tables_by_view[v].assign(view=v) for v in views], ignore_index=True
    )
    hits_long = pd.concat([hits_by_view[v].assign(view=v) for v in views], ignore_index=True)

    # Top-N filtered outputs by view.
    coverage_rows: list[dict[str, Any]] = []
    composition_rows: list[dict[str, Any]] = []
    generated_outputs: dict[str, str] = {}
    for view in views:
        impacts_path = m6_2_dir / f"node_impacts_stratified_{view}.csv"
        if impacts_path.exists():
            impacts = pd.read_csv(impacts_path)
        else:
            fallback_path = m6_1_dir / f"node_single_removal_impacts_{view}.csv"
            if not fallback_path.exists():
                raise FileNotFoundError(
                    f"missing impact input for view={view}: {impacts_path} and {fallback_path}"
                )
            impacts = pd.read_csv(fallback_path)
            impacts["tier_bin"] = pd.NA
            impacts["is_intermediary_corridor"] = pd.NA

        impacts["analysis_uid"] = clean_text(impacts["analysis_uid"])
        decision_table_view = decision_tables_by_view[view]
        impacts = impacts.merge(
            decision_table_view[
                [
                    "analysis_uid",
                    "name",
                    "canonical_id",
                    "iso_country",
                    "country_code_factset",
                    "region_code_factset",
                    "entity_role",
                    "is_semi",
                    "include_semiconductor_lens",
                    "include_semiconductor_lens_plus_uncertain",
                    "decision_strict",
                    "decision_plus_uncertain",
                    "decision_reason_strict",
                    "decision_reason_plus_uncertain",
                    "selected_code_system",
                    "selected_rule_ids",
                    "selected_code_values",
                    "selected_rule_classes",
                    "endpoint_override_applied",
                    "gate_empirical_two_sided_corridor",
                    "gate_empirical_intermediary_corridor",
                    "gate_definition",
                    "confidence_class",
                    "low_conf_reason",
                    "presence_count",
                ]
            ],
            on="analysis_uid",
            how="left",
            suffixes=("", "_lens"),
        )

        # Keep existing names when present; otherwise use enriched names.
        if "name_lens" in impacts.columns:
            base_name = clean_text(impacts.get("name", pd.Series([""] * len(impacts))))
            lens_name = clean_text(impacts["name_lens"])
            impacts["name"] = np.where(base_name.str.len() > 0, base_name, lens_name)
            impacts["name"] = clean_text(impacts["name"]).replace("", pd.NA)
            impacts.drop(columns=["name_lens"], inplace=True)

        for col in ["include_semiconductor_lens", "include_semiconductor_lens_plus_uncertain"]:
            impacts[col] = impacts[col].fillna(False).astype(bool)

        rule_class = clean_text(
            impacts.get("selected_rule_classes", pd.Series([""] * len(impacts)))
        )
        has_core_adj = rule_class.str.contains(
            "core_semiconductor_include"
        ) | rule_class.str.contains("adjacent_manufacturing_include")
        has_conditional = rule_class.str.contains("conditional_services_include")
        is_uncertain = impacts["decision_strict"].astype(str).str.startswith("uncertain")
        is_plus_only = (impacts["decision_strict"].astype(str) != "include") & (
            impacts["decision_plus_uncertain"].astype(str) == "include"
        )

        impacts["lens_product_stream"] = "other"
        impacts.loc[is_plus_only | is_uncertain, "lens_product_stream"] = (
            "semiconductor_adjacent_uncertain"
        )
        impacts.loc[
            has_conditional & impacts["include_semiconductor_lens"], "lens_product_stream"
        ] = "digital_infrastructure_dependencies"
        impacts.loc[has_core_adj & impacts["include_semiconductor_lens"], "lens_product_stream"] = (
            "semiconductor_value_chain_strict"
        )

        ranked_strict = rank_table(
            impacts[impacts["include_semiconductor_lens"]].copy(),
            rank_metric=ranking_metric,
            tie_breakers=tie_breakers,
        )
        ranked_plus = rank_table(
            impacts[impacts["include_semiconductor_lens_plus_uncertain"]].copy(),
            rank_metric=ranking_metric,
            tie_breakers=tie_breakers,
        )

        full_strict_out = out_dir / f"node_impacts_semiconductor_lens_{view}.csv"
        full_plus_out = out_dir / f"node_impacts_semiconductor_lens_plus_uncertain_{view}.csv"
        ranked_strict.to_csv(full_strict_out, index=False)
        ranked_plus.to_csv(full_plus_out, index=False)
        generated_outputs[f"node_impacts_semiconductor_lens_{view}"] = str(full_strict_out)
        generated_outputs[f"node_impacts_semiconductor_lens_plus_uncertain_{view}"] = str(
            full_plus_out
        )

        ranked_stream_strict = rank_table(
            impacts[impacts["lens_product_stream"] == "semiconductor_value_chain_strict"].copy(),
            rank_metric=ranking_metric,
            tie_breakers=tie_breakers,
        )
        ranked_stream_uncertain = rank_table(
            impacts[impacts["lens_product_stream"] == "semiconductor_adjacent_uncertain"].copy(),
            rank_metric=ranking_metric,
            tie_breakers=tie_breakers,
        )
        ranked_stream_digital = rank_table(
            impacts[impacts["lens_product_stream"] == "digital_infrastructure_dependencies"].copy(),
            rank_metric=ranking_metric,
            tie_breakers=tie_breakers,
        )
        stream_strict_out = out_dir / f"node_impacts_semiconductor_value_chain_strict_{view}.csv"
        stream_uncertain_out = out_dir / f"node_impacts_semiconductor_adjacent_uncertain_{view}.csv"
        stream_digital_out = (
            out_dir / f"node_impacts_digital_infrastructure_dependencies_{view}.csv"
        )
        ranked_stream_strict.to_csv(stream_strict_out, index=False)
        ranked_stream_uncertain.to_csv(stream_uncertain_out, index=False)
        ranked_stream_digital.to_csv(stream_digital_out, index=False)
        generated_outputs[f"node_impacts_semiconductor_value_chain_strict_{view}"] = str(
            stream_strict_out
        )
        generated_outputs[f"node_impacts_semiconductor_adjacent_uncertain_{view}"] = str(
            stream_uncertain_out
        )
        generated_outputs[f"node_impacts_digital_infrastructure_dependencies_{view}"] = str(
            stream_digital_out
        )

        n_total = len(impacts)
        n_strict = len(ranked_strict)
        n_plus = len(ranked_plus)

        for n in top_ns:
            top_base = impacts.copy()
            top_base = rank_table(
                top_base, rank_metric=ranking_metric, tie_breakers=tie_breakers
            ).head(n)

            top_strict = ranked_strict.head(n).copy()
            top_plus = ranked_plus.head(n).copy()
            top_stream_strict = ranked_stream_strict.head(n).copy()
            top_stream_uncertain = ranked_stream_uncertain.head(n).copy()
            top_stream_digital = ranked_stream_digital.head(n).copy()

            top_strict_out = out_dir / f"top{n}_high_impact_semiconductor_lens_{view}.csv"
            top_plus_out = (
                out_dir / f"top{n}_high_impact_semiconductor_lens_plus_uncertain_{view}.csv"
            )
            top_stream_strict_out = (
                out_dir / f"top{n}_high_impact_semiconductor_value_chain_strict_{view}.csv"
            )
            top_stream_uncertain_out = (
                out_dir / f"top{n}_high_impact_semiconductor_adjacent_uncertain_{view}.csv"
            )
            top_stream_digital_out = (
                out_dir / f"top{n}_high_impact_digital_infrastructure_dependencies_{view}.csv"
            )
            top_strict.to_csv(top_strict_out, index=False)
            top_plus.to_csv(top_plus_out, index=False)
            top_stream_strict.to_csv(top_stream_strict_out, index=False)
            top_stream_uncertain.to_csv(top_stream_uncertain_out, index=False)
            top_stream_digital.to_csv(top_stream_digital_out, index=False)
            generated_outputs[f"top{n}_high_impact_semiconductor_lens_{view}"] = str(top_strict_out)
            generated_outputs[f"top{n}_high_impact_semiconductor_lens_plus_uncertain_{view}"] = str(
                top_plus_out
            )
            generated_outputs[f"top{n}_high_impact_semiconductor_value_chain_strict_{view}"] = str(
                top_stream_strict_out
            )
            generated_outputs[f"top{n}_high_impact_semiconductor_adjacent_uncertain_{view}"] = str(
                top_stream_uncertain_out
            )
            generated_outputs[f"top{n}_high_impact_digital_infrastructure_dependencies_{view}"] = (
                str(top_stream_digital_out)
            )

            comp = (
                top_base["selected_rule_classes"]
                .fillna("missing_rule_class")
                .astype(str)
                .str.strip()
            )
            comp_counts = comp.value_counts(dropna=False)
            for rc, cnt in comp_counts.items():
                composition_rows.append(
                    {
                        "view": view,
                        "top_n": int(n),
                        "selected_rule_classes": rc,
                        "count": int(cnt),
                        "share": float(cnt / len(top_base)) if len(top_base) > 0 else np.nan,
                    }
                )

            coverage_rows.append(
                {
                    "view": view,
                    "top_n": int(n),
                    "ranking_metric": ranking_metric,
                    "n_total_nodes": n_total,
                    "n_semiconductor_lens": n_strict,
                    "n_semiconductor_lens_plus_uncertain": n_plus,
                    "share_semiconductor_lens": (n_strict / n_total) if n_total > 0 else np.nan,
                    "share_semiconductor_lens_plus_uncertain": (n_plus / n_total)
                    if n_total > 0
                    else np.nan,
                    "topn_baseline_count": len(top_base),
                    "topn_in_semiconductor_lens": int(top_base["include_semiconductor_lens"].sum()),
                    "topn_in_semiconductor_lens_plus_uncertain": int(
                        top_base["include_semiconductor_lens_plus_uncertain"].sum()
                    ),
                    "topn_share_in_semiconductor_lens": float(
                        top_base["include_semiconductor_lens"].mean()
                    )
                    if len(top_base) > 0
                    else np.nan,
                    "topn_share_in_semiconductor_lens_plus_uncertain": float(
                        top_base["include_semiconductor_lens_plus_uncertain"].mean()
                    )
                    if len(top_base) > 0
                    else np.nan,
                }
            )

    decision_summary = (
        decisions_long.groupby(
            ["view", "decision_strict", "decision_plus_uncertain", "entity_role"],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "node_count"})
        .sort_values(
            ["view", "decision_strict", "decision_plus_uncertain", "entity_role"],
            ascending=[True, True, True, True],
        )
    )

    coverage_by_view = pd.DataFrame(coverage_rows).sort_values(
        ["view", "top_n"], ascending=[True, True]
    )
    composition_by_rule = pd.DataFrame(composition_rows).sort_values(
        ["view", "top_n", "count", "selected_rule_classes"],
        ascending=[True, True, False, True],
    )

    # Write outputs.
    decision_csv = out_dir / "node_semiconductor_lens_decisions.csv"
    decision_parquet = out_dir / "node_semiconductor_lens_decisions.parquet"
    decision_by_view_csv = out_dir / "node_semiconductor_lens_decisions_by_view.csv"
    decision_by_view_parquet = out_dir / "node_semiconductor_lens_decisions_by_view.parquet"
    rule_hits_csv = out_dir / "node_semiconductor_lens_rule_hits.csv"
    rule_hits_parquet = out_dir / "node_semiconductor_lens_rule_hits.parquet"
    rule_hits_by_view_csv = out_dir / "node_semiconductor_lens_rule_hits_by_view.csv"
    rule_hits_by_view_parquet = out_dir / "node_semiconductor_lens_rule_hits_by_view.parquet"
    rule_cov_csv = out_dir / "codebook_rule_coverage.csv"
    summary_csv = out_dir / "lens_decision_summary.csv"
    topn_cov_csv = out_dir / "lens_topn_coverage_by_view.csv"
    topn_comp_csv = out_dir / "lens_topn_composition_by_rule_class.csv"

    decision_table.to_csv(decision_csv, index=False)
    decision_table.to_parquet(decision_parquet, index=False)
    decisions_long.to_csv(decision_by_view_csv, index=False)
    decisions_long.to_parquet(decision_by_view_parquet, index=False)
    hits.to_csv(rule_hits_csv, index=False)
    hits.to_parquet(rule_hits_parquet, index=False)
    hits_long.to_csv(rule_hits_by_view_csv, index=False)
    hits_long.to_parquet(rule_hits_by_view_parquet, index=False)
    coverage.to_csv(rule_cov_csv, index=False)
    decision_summary.to_csv(summary_csv, index=False)
    coverage_by_view.to_csv(topn_cov_csv, index=False)
    composition_by_rule.to_csv(topn_comp_csv, index=False)

    run_metadata = {
        "module": "m0_7",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "yaml": yaml.__version__,
        },
        "settings": {
            "primary_decision_view": primary_decision_view,
            "views": views,
            "top_n": top_ns,
            "ranking_metric": ranking_metric,
            "tie_breakers": tie_breakers,
            "codebook_path": str(codebook_path),
            "fetch_factset_names": fetch_factset_names_enabled,
            "factset_name_chunk_size": factset_name_chunk_size,
            "db_client_available": bool(run_query is not None),
            "gate_definition": "two_sided_empirical_paths",
            "gate_empirical_only": gate_empirical_only,
            "gate_effective_include_any": gate_effective_include_any,
            "apply_endpoint_overrides": apply_endpoint_overrides,
        },
        "counts": {
            "n_nodes": len(decision_table),
            "n_nodes_all_views_rows": len(decisions_long),
            "n_membership_rows": len(membership),
            "n_rule_hits": len(hits),
            "n_rule_hits_all_views_rows": len(hits_long),
            "n_rules_active": len(codebook),
            "n_include_strict": int(decision_table["include_semiconductor_lens"].sum()),
            "n_include_plus_uncertain": int(
                decision_table["include_semiconductor_lens_plus_uncertain"].sum()
            ),
            "n_decision_uncertain_strict": int(
                (decision_table["decision_strict"].astype(str).str.startswith("uncertain")).sum()
            ),
            "n_nodes_with_name": int(decision_table["name"].notna().sum())
            if "name" in decision_table.columns
            else 0,
            "n_nodes_with_iso_country": int(decision_table["iso_country"].notna().sum())
            if "iso_country" in decision_table.columns
            else 0,
            "n_factset_name_rows": len(factset_lookup),
            "n_endpoint_overrides_primary_view": int(
                decision_table["endpoint_override_applied"].fillna(False).sum()
            )
            if "endpoint_override_applied" in decision_table.columns
            else 0,
        },
    }
    run_metadata_out = out_dir / "run_metadata.json"
    run_metadata_out.write_text(json.dumps(run_metadata, indent=2))

    input_manifest = {
        "codebook": str(input_paths["codebook"]),
        "codebook_sha256": file_sha256(input_paths["codebook"]),
        "node_enrichment": str(input_paths["node_enrichment"]),
        "node_enrichment_sha256": file_sha256(input_paths["node_enrichment"]),
        "node_membership": str(input_paths["node_membership"]),
        "node_membership_sha256": file_sha256(input_paths["node_membership"]),
        "m0_nodes": str(input_paths["m0_nodes"]),
        "m0_nodes_sha256": file_sha256(input_paths["m0_nodes"]),
        "edge_table": str(input_paths["edge_table"]),
        "edge_table_sha256": file_sha256(input_paths["edge_table"]),
        "confidence_table": str(input_paths["confidence_table"]),
        "confidence_table_sha256": file_sha256(input_paths["confidence_table"]),
    }
    for view in views:
        p2 = m6_2_dir / f"node_impacts_stratified_{view}.csv"
        p1 = m6_1_dir / f"node_single_removal_impacts_{view}.csv"
        if p2.exists():
            input_manifest[f"impacts_{view}"] = str(p2)
            input_manifest[f"impacts_{view}_sha256"] = file_sha256(p2)
        else:
            input_manifest[f"impacts_{view}"] = str(p1)
            input_manifest[f"impacts_{view}_sha256"] = file_sha256(p1)

    output_manifest = {
        "node_semiconductor_lens_decisions_csv": str(decision_csv),
        "node_semiconductor_lens_decisions_parquet": str(decision_parquet),
        "node_semiconductor_lens_decisions_by_view_csv": str(decision_by_view_csv),
        "node_semiconductor_lens_decisions_by_view_parquet": str(decision_by_view_parquet),
        "node_semiconductor_lens_rule_hits_csv": str(rule_hits_csv),
        "node_semiconductor_lens_rule_hits_parquet": str(rule_hits_parquet),
        "node_semiconductor_lens_rule_hits_by_view_csv": str(rule_hits_by_view_csv),
        "node_semiconductor_lens_rule_hits_by_view_parquet": str(rule_hits_by_view_parquet),
        "codebook_rule_coverage_csv": str(rule_cov_csv),
        "lens_decision_summary_csv": str(summary_csv),
        "lens_topn_coverage_by_view_csv": str(topn_cov_csv),
        "lens_topn_composition_by_rule_class_csv": str(topn_comp_csv),
        "run_metadata": str(run_metadata_out),
    }
    output_manifest.update(generated_outputs)

    manifest = {
        "module": "m0_7",
        "snapshot": snapshot,
        "run_id": run_id,
        "config_path": str(cfg_path),
        "inputs": input_manifest,
        "outputs": output_manifest,
    }
    manifest_out = out_dir / "manifest_m0_7.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))

    print(f"[done] wrote {decision_csv}")
    print(f"[done] wrote {decision_parquet}")
    print(f"[done] wrote {decision_by_view_csv}")
    print(f"[done] wrote {decision_by_view_parquet}")
    print(f"[done] wrote {rule_hits_csv}")
    print(f"[done] wrote {rule_hits_parquet}")
    print(f"[done] wrote {rule_hits_by_view_csv}")
    print(f"[done] wrote {rule_hits_by_view_parquet}")
    print(f"[done] wrote {rule_cov_csv}")
    print(f"[done] wrote {summary_csv}")
    print(f"[done] wrote {topn_cov_csv}")
    print(f"[done] wrote {topn_comp_csv}")
    print(f"[done] wrote {manifest_out}")
    print(f"[done] wrote {run_metadata_out}")


if __name__ == "__main__":
    main()
