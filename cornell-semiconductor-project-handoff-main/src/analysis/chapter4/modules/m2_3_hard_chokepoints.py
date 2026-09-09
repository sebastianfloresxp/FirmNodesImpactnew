#!/usr/bin/env python3
"""Module 2.3: directed hard chokepoints via dominators and min-cut."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml

SUPER_SOURCE = "__SUPER_SEMI_SOURCE__"


def bool_series_with_fallback(
    df: pd.DataFrame, columns: list[str], *, default: bool = False
) -> pd.Series:
    for col in columns:
        if col in df.columns:
            return df[col].fillna(default).astype(bool)
    return pd.Series(np.full(len(df), default, dtype=bool), index=df.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 4 Module 2.3 hard chokepoints")
    parser.add_argument(
        "--config",
        default="src/analysis/chapter4/config/ch4_v2_fix01.yaml",
        help="Config YAML path",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config must parse to mapping")
    return cfg


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(  # nosec B607 -- git is a well-known system executable
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def get_view_mask(edges: pd.DataFrame, include_any: list[str]) -> np.ndarray:
    if not include_any:
        raise ValueError("include_any must not be empty")
    missing = [col for col in include_any if col not in edges.columns]
    if missing:
        raise ValueError(f"missing view flags: {missing}")
    mask = np.zeros(len(edges), dtype=bool)
    for col in include_any:
        mask |= edges[col].astype(bool).to_numpy()
    return mask


def build_view_edges(edges: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    cols = ["src_uid", "dst_uid", "is_disclosed", "is_observed_ship", "is_predicted"]
    df = edges.loc[mask, cols].copy()
    df = df[df["src_uid"] != df["dst_uid"]]
    df = (
        df.groupby(["src_uid", "dst_uid"], as_index=False)[
            ["is_disclosed", "is_observed_ship", "is_predicted"]
        ]
        .max()
        .sort_values(["src_uid", "dst_uid"], ascending=[True, True], kind="mergesort")
    )
    return df


def load_prime_weight_table(weights_path: Path, prime_uids: np.ndarray) -> pd.DataFrame:
    if not weights_path.exists():
        return pd.DataFrame(
            {
                "analysis_uid": prime_uids.astype(str),
                "weight_unit": np.ones(len(prime_uids), dtype=np.float64),
                "weight_log_obligation": np.ones(len(prime_uids), dtype=np.float64),
                "weight_raw_obligation": np.ones(len(prime_uids), dtype=np.float64),
            }
        )

    if weights_path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(weights_path)
    else:
        raw = pd.read_csv(weights_path)

    uid_col = next(
        (c for c in ["analysis_uid", "prime_uid", "uid", "node_uid"] if c in raw.columns), None
    )
    if uid_col is None:
        return pd.DataFrame(
            {
                "analysis_uid": prime_uids.astype(str),
                "weight_unit": np.ones(len(prime_uids), dtype=np.float64),
                "weight_log_obligation": np.ones(len(prime_uids), dtype=np.float64),
                "weight_raw_obligation": np.ones(len(prime_uids), dtype=np.float64),
            }
        )

    df = raw.copy()
    df["analysis_uid"] = df[uid_col].astype(str)
    for col in [
        "obligation_log1p",
        "obligation_weight",
        "weight",
        "obligation_nonneg",
        "total_obligations",
        "obligation_raw",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    out = pd.DataFrame({"analysis_uid": prime_uids.astype(str)})
    merged = out.merge(df, how="left", on="analysis_uid")
    merged["weight_unit"] = 1.0
    if "obligation_log1p" in merged.columns:
        merged["weight_log_obligation"] = merged["obligation_log1p"].fillna(0.0)
    elif "obligation_weight" in merged.columns:
        merged["weight_log_obligation"] = merged["obligation_weight"].fillna(0.0)
    elif "weight" in merged.columns:
        merged["weight_log_obligation"] = merged["weight"].fillna(0.0)
    else:
        merged["weight_log_obligation"] = 0.0
    if "obligation_nonneg" in merged.columns:
        merged["weight_raw_obligation"] = merged["obligation_nonneg"].fillna(0.0)
    elif "total_obligations" in merged.columns:
        merged["weight_raw_obligation"] = merged["total_obligations"].fillna(0.0)
    elif "obligation_raw" in merged.columns:
        merged["weight_raw_obligation"] = merged["obligation_raw"].clip(lower=0).fillna(0.0)
    else:
        merged["weight_raw_obligation"] = 0.0
    return merged[["analysis_uid", "weight_unit", "weight_log_obligation", "weight_raw_obligation"]]


def build_graph(
    edge_df: pd.DataFrame,
    semis: list[str],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from(edge_df[["src_uid", "dst_uid"]].itertuples(index=False, name=None))
    graph.add_node(SUPER_SOURCE)
    for semi in semis:
        if semi in graph:
            graph.add_edge(SUPER_SOURCE, semi)
    return graph


def compute_dominators(
    graph: nx.DiGraph,
    primes_df: pd.DataFrame,
    node_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reachable = nx.descendants(graph, SUPER_SOURCE) | {SUPER_SOURCE}
    subgraph = graph.subgraph(reachable).copy()
    idom = nx.immediate_dominators(subgraph, SUPER_SOURCE)

    prime_lookup = primes_df.set_index("analysis_uid")
    dom_count: dict[str, int] = {}
    dom_weight_log: dict[str, float] = {}
    dom_weight_raw: dict[str, float] = {}

    prime_rows: list[dict[str, Any]] = []
    for prime_uid, prow in prime_lookup.iterrows():
        if prime_uid not in idom:
            prime_rows.append(
                {
                    "analysis_uid": prime_uid,
                    "is_reachable_from_semis": False,
                    "dominator_path_nodes": 0,
                    "must_pass_intermediaries": 0,  # nosec B105 -- numeric threshold, not a password
                }
            )
            continue
        chain: list[str] = []
        current = prime_uid
        while current != SUPER_SOURCE:
            chain.append(current)
            current = idom[current]
        intermediaries = [node for node in chain[1:] if node != SUPER_SOURCE]
        for dom_uid in intermediaries:
            dom_count[dom_uid] = dom_count.get(dom_uid, 0) + 1
            dom_weight_log[dom_uid] = dom_weight_log.get(dom_uid, 0.0) + float(
                prow["weight_log_obligation"]
            )
            dom_weight_raw[dom_uid] = dom_weight_raw.get(dom_uid, 0.0) + float(
                prow["weight_raw_obligation"]
            )
        prime_rows.append(
            {
                "analysis_uid": prime_uid,
                "is_reachable_from_semis": True,
                "dominator_path_nodes": len(chain),
                "must_pass_intermediaries": len(intermediaries),
                "first_dominator": intermediaries[0] if intermediaries else None,
            }
        )

    node_rows: list[dict[str, Any]] = []
    node_meta_lookup = node_meta.set_index("analysis_uid")
    for uid, count in dom_count.items():
        meta = node_meta_lookup.loc[uid] if uid in node_meta_lookup.index else None
        node_rows.append(
            {
                "analysis_uid": uid,
                "name": None if meta is None else meta["name"],
                "entity_role": None if meta is None else meta["entity_role"],
                "is_semi_strict": False if meta is None else bool(meta["is_semi_strict"]),
                "is_tier1_prime": False if meta is None else bool(meta["is_tier1_prime"]),
                "dominated_primes_count": int(count),
                "dominated_primes_log_weight": float(dom_weight_log.get(uid, 0.0)),
                "dominated_primes_raw_weight": float(dom_weight_raw.get(uid, 0.0)),
            }
        )

    node_df = pd.DataFrame(node_rows)
    if not node_df.empty:
        node_df = node_df.sort_values(
            ["dominated_primes_log_weight", "dominated_primes_count", "analysis_uid"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        node_df["rank_dominated_primes_log_weight"] = (
            node_df["dominated_primes_log_weight"]
            .rank(method="min", ascending=False)
            .astype(np.int32)
        )
        node_df["rank_dominated_primes_count"] = (
            node_df["dominated_primes_count"].rank(method="min", ascending=False).astype(np.int32)
        )
    else:
        node_df = pd.DataFrame(
            columns=[
                "analysis_uid",
                "name",
                "entity_role",
                "is_semi_strict",
                "is_tier1_prime",
                "dominated_primes_count",
                "dominated_primes_log_weight",
                "dominated_primes_raw_weight",
                "rank_dominated_primes_log_weight",
                "rank_dominated_primes_count",
            ]
        )

    prime_df = pd.DataFrame(prime_rows).merge(primes_df, on="analysis_uid", how="left")
    return node_df, prime_df


def compute_mincut(
    graph: nx.DiGraph,
    primes_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    graph_cap = nx.DiGraph()
    for u, v in graph.edges():
        cap = 1.0
        if u == SUPER_SOURCE:
            cap = 1e9
        graph_cap.add_edge(u, v, capacity=cap)

    reachable = set(nx.descendants(graph_cap, SUPER_SOURCE))
    candidate_primes = (
        primes_df.sort_values(
            ["weight_log_obligation", "analysis_uid"], ascending=[False, True], kind="mergesort"
        )
        .head(top_n)["analysis_uid"]
        .astype(str)
        .tolist()
    )

    rows: list[dict[str, Any]] = []
    for i, prime_uid in enumerate(candidate_primes, start=1):
        if prime_uid not in graph_cap or prime_uid not in reachable:
            rows.append(
                {
                    "analysis_uid": prime_uid,
                    "is_reachable_from_semis": False,
                    "mincut_value": 0.0,
                    "cut_edge_count": 0,
                }
            )
            continue
        cut_value, partition = nx.minimum_cut(
            graph_cap,
            SUPER_SOURCE,
            prime_uid,
            capacity="capacity",
            flow_func=nx.algorithms.flow.shortest_augmenting_path,
        )
        src_side, dst_side = partition
        cut_edges = 0
        for u in src_side:
            for v in graph_cap.successors(u):
                if v in dst_side:
                    cut_edges += 1
        rows.append(
            {
                "analysis_uid": prime_uid,
                "is_reachable_from_semis": True,
                "mincut_value": float(cut_value),
                "cut_edge_count": int(cut_edges),
            }
        )
        if i % 5 == 0 or i == len(candidate_primes):
            print(f"[m2_3] mincut progress {i}/{len(candidate_primes)}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_config(cfg_path)

    snapshot = str(cfg.get("snapshot", "2025-06-09"))
    run_id = str(cfg.get("run_id", "ch4_unknown"))
    paths_cfg = cfg.get("paths", {})
    out_root = Path(str(paths_cfg.get("out_root", "artifacts/ch4/v2_fix01")))
    m0_dir = out_root / snapshot / "m0"
    m2_3_dir = out_root / snapshot / "m2_3"
    m2_3_dir.mkdir(parents=True, exist_ok=True)

    node_path = m0_dir / "node_table_contract.parquet"
    edge_path = m0_dir / "edge_table_contract.parquet"
    if not node_path.exists() or not edge_path.exists():
        raise FileNotFoundError("M0 contract outputs are required before M2.3")

    nodes = pd.read_parquet(node_path).reset_index(drop=True)
    edges = pd.read_parquet(edge_path).reset_index(drop=True)

    is_prime = bool_series_with_fallback(
        nodes,
        ["is_tier1_prime", "has_prime_vendor"],
        default=False,
    ).to_numpy(bool)
    is_semi = bool_series_with_fallback(nodes, ["is_semi_strict"], default=False).to_numpy(bool)
    prime_uids = nodes.loc[is_prime, "analysis_uid"].astype(str).to_numpy()
    semi_uids = nodes.loc[is_semi, "analysis_uid"].astype(str).tolist()

    weights_path = Path(
        str(paths_cfg.get("prime_weights", "artifacts/ch4/inputs/prime_weights.parquet"))
    )
    prime_weights = load_prime_weight_table(weights_path, prime_uids)
    primes_df = nodes.loc[is_prime, ["analysis_uid", "name", "entity_role"]].merge(
        prime_weights, on="analysis_uid", how="left"
    )
    node_meta = nodes[["analysis_uid", "name", "entity_role"]].copy()
    node_meta["is_semi_strict"] = bool_series_with_fallback(
        nodes, ["is_semi_strict"], default=False
    ).to_numpy(bool)
    node_meta["is_tier1_prime"] = is_prime

    m_cfg = cfg.get("m2_3", {})
    views = [str(v) for v in m_cfg.get("views", list(cfg.get("views", {}).keys()))]
    top_n_export = int(m_cfg.get("top_n_export", 100))
    top_n_mincut_primes = int(m_cfg.get("top_n_mincut_primes", 25))

    runtime_by_view: dict[str, float] = {}
    output_manifest: dict[str, dict[str, str]] = {}

    for view in views:
        t0 = time.perf_counter()
        include_any = [str(x) for x in cfg.get("views", {}).get(view, {}).get("include_any", [])]
        if not include_any:
            raise ValueError(f"view {view} include_any missing")

        mask = get_view_mask(edges, include_any)
        edge_view = build_view_edges(edges, mask)
        graph = build_graph(edge_view, semi_uids)

        dom_nodes, dom_primes = compute_dominators(
            graph=graph, primes_df=primes_df, node_meta=node_meta
        )
        dom_nodes.insert(0, "view", view)
        dom_primes.insert(0, "view", view)

        dom_nodes_out = m2_3_dir / f"node_dominator_scores_{view}.csv"
        dom_primes_out = m2_3_dir / f"prime_dominator_profiles_{view}.csv"
        top_dom_out = m2_3_dir / f"top_dominator_nodes_{view}.csv"
        dom_nodes.to_csv(dom_nodes_out, index=False)
        dom_primes.to_csv(dom_primes_out, index=False)
        dom_nodes.head(top_n_export).to_csv(top_dom_out, index=False)

        mincut = compute_mincut(graph=graph, primes_df=primes_df, top_n=top_n_mincut_primes)
        mincut = mincut.merge(primes_df, on="analysis_uid", how="left")
        mincut.insert(0, "view", view)
        mincut_out = m2_3_dir / f"mincut_prime_results_{view}.csv"
        mincut.to_csv(mincut_out, index=False)
        top_low_cut = mincut.sort_values(
            ["is_reachable_from_semis", "mincut_value", "analysis_uid"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        top_low_cut_out = m2_3_dir / f"top_critical_primes_by_low_cut_{view}.csv"
        top_low_cut.to_csv(top_low_cut_out, index=False)

        summary = pd.DataFrame(
            [
                {
                    "view": view,
                    "reachable_primes": int(dom_primes["is_reachable_from_semis"].sum()),
                    "total_primes": len(dom_primes),
                    "dominators_with_positive_count": int(
                        (dom_nodes["dominated_primes_count"] > 0).sum()
                    )
                    if len(dom_nodes)
                    else 0,
                    "median_dominated_primes_count": float(
                        dom_nodes["dominated_primes_count"].median()
                    )
                    if len(dom_nodes)
                    else 0.0,
                    "max_dominated_primes_count": int(dom_nodes["dominated_primes_count"].max())
                    if len(dom_nodes)
                    else 0,
                    "mincut_reachable_primes": int(mincut["is_reachable_from_semis"].sum()),
                    "mincut_min_value": float(
                        mincut.loc[mincut["is_reachable_from_semis"], "mincut_value"].min()
                    )
                    if int(mincut["is_reachable_from_semis"].sum()) > 0
                    else 0.0,
                    "mincut_median_value": float(
                        mincut.loc[mincut["is_reachable_from_semis"], "mincut_value"].median()
                    )
                    if int(mincut["is_reachable_from_semis"].sum()) > 0
                    else 0.0,
                    "mincut_max_value": float(
                        mincut.loc[mincut["is_reachable_from_semis"], "mincut_value"].max()
                    )
                    if int(mincut["is_reachable_from_semis"].sum()) > 0
                    else 0.0,
                }
            ]
        )
        summary_out = m2_3_dir / f"hard_chokepoint_summary_{view}.csv"
        summary.to_csv(summary_out, index=False)

        runtime_by_view[view] = float(time.perf_counter() - t0)
        output_manifest[view] = {
            "node_dominator_scores": str(dom_nodes_out),
            "prime_dominator_profiles": str(dom_primes_out),
            "top_dominator_nodes": str(top_dom_out),
            "mincut_prime_results": str(mincut_out),
            "top_critical_primes_by_low_cut": str(top_low_cut_out),
            "hard_chokepoint_summary": str(summary_out),
        }
        print(f"[m2_3] view={view} runtime_s={runtime_by_view[view]:.2f}", flush=True)

    run_metadata = {
        "module": "m2_3",
        "snapshot": snapshot,
        "run_id": run_id,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "settings": {
            "views": views,
            "top_n_export": top_n_export,
            "top_n_mincut_primes": top_n_mincut_primes,
            "super_source": SUPER_SOURCE,
        },
        "graph": {
            "n_nodes": len(nodes),
            "n_edges_total": len(edges),
            "n_primes": int(is_prime.sum()),
            "n_semis": int(is_semi.sum()),
        },
        "runtime_seconds_by_view": runtime_by_view,
    }
    run_meta_path = m2_3_dir / "run_metadata.json"
    run_meta_path.write_text(json.dumps(run_metadata, indent=2))

    manifest = {
        "module": "m2_3",
        "snapshot": snapshot,
        "run_id": run_id,
        "inputs": {
            "node_table_contract": str(node_path),
            "edge_table_contract": str(edge_path),
            "node_table_contract_sha256": file_sha256(node_path),
            "edge_table_contract_sha256": file_sha256(edge_path),
            "prime_weights": str(weights_path),
            "prime_weights_sha256": file_sha256(weights_path) if weights_path.exists() else None,
        },
        "outputs": output_manifest,
        "run_metadata": str(run_meta_path),
    }
    manifest_path = m2_3_dir / "manifest_m2_3.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
