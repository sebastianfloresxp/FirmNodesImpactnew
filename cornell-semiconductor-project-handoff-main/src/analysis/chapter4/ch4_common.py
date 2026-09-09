#!/usr/bin/env python3
"""Common data structures and algorithms for Chapter 4 vulnerability analysis.

Provides EntityGraph construction with identity collapse (UnionFind),
supply-edge and evidence-layer filtering, BFS reach and tier computation,
sparse transition matrices, and PageRank/HITS centrality utilities.
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DEFAULT_NODES = (
    "artifacts/ch3/network_upstream/dod_semiconductor_nodes_top5_d99_shipping_strict.parquet"
)
DEFAULT_EDGES = (
    "artifacts/ch3/network_upstream/dod_semiconductor_edges_top5_d99_shipping_strict.parquet"
)


EDGE_FLAG_COLS = [
    "is_contract",
    "is_maps_to_scr",
    "is_maps_to_factset",
    "is_disclosed",
    "is_predicted",
    "is_shipping",
]


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, obj: dict) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(obj, indent=2))


def coerce_edge_flags(edges: pd.DataFrame) -> pd.DataFrame:
    edges = edges.copy()
    for c in EDGE_FLAG_COLS:
        if c in edges.columns:
            edges[c] = edges[c].fillna(0).astype(np.int8)
    return edges


class UnionFind:
    __slots__ = ("parent", "size")

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        size = self.size
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        size[ra] += size[rb]


@dataclass(frozen=True)
class EntityGraph:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    node_map: pd.DataFrame  # node_uid -> entity_id/entity_uid


def _merge_pipe_separated(values: Sequence[str | None]) -> str | None:
    parts: set[str] = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        for p in s.split("|"):
            p = p.strip()
            if p:
                parts.add(p)
    if not parts:
        return None
    return "|".join(sorted(parts))


def collapse_identity_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> EntityGraph:
    """
    Collapse identity bridge edges (vendor<->SCR, vendor<->FactSet) into entity nodes.

    Mapping edges are treated as zero-length identity; after collapsing, mapping
    edges are dropped from the entity-edge set.
    """
    if "node_uid" not in nodes.columns:
        raise ValueError("nodes missing node_uid")
    if "src_uid" not in edges.columns or "dst_uid" not in edges.columns:
        raise ValueError("edges missing src_uid/dst_uid")

    nodes = nodes.copy()
    edges = coerce_edge_flags(edges)

    node_uids = nodes["node_uid"].astype(str).tolist()
    uid_to_idx = {uid: i for i, uid in enumerate(node_uids)}

    uf = UnionFind(len(node_uids))
    maps = edges[(edges["is_maps_to_scr"] == 1) | (edges["is_maps_to_factset"] == 1)][
        ["src_uid", "dst_uid"]
    ]
    for src_uid, dst_uid in maps.itertuples(index=False, name=None):
        si = uid_to_idx.get(str(src_uid))
        di = uid_to_idx.get(str(dst_uid))
        if si is None or di is None:
            continue
        uf.union(si, di)

    roots = np.fromiter(
        (uf.find(i) for i in range(len(node_uids))), dtype=np.int32, count=len(node_uids)
    )
    entity_id, _ = pd.factorize(roots, sort=True)
    nodes["entity_id"] = entity_id.astype(np.int32)
    nodes["entity_uid"] = "ent:" + nodes["entity_id"].astype(str)

    node_map = nodes[["node_uid", "entity_id", "entity_uid"]].copy()

    # Entity type flags (can be multi-typed due to identity resolution).
    nodes["is_dod"] = nodes["node_type"] == "dod_component"
    nodes["is_prime"] = nodes["node_type"] == "prime_vendor"
    nodes["is_scr"] = nodes["node_type"] == "scr_firm"
    nodes["is_factset"] = nodes["node_type"] == "factset_firm"

    g = nodes.groupby("entity_id", sort=False)
    entity = pd.DataFrame({"entity_id": g.size().index.astype(np.int32)})
    entity["entity_uid"] = "ent:" + entity["entity_id"].astype(str)
    # Important: pandas aligns Series to DataFrame by index label on assignment.
    # Since `entity_id` values are not guaranteed to match the row index order, we
    # must assign derived Series by *position* (via numpy arrays) in the same row order.
    eid = entity["entity_id"].to_numpy(np.int32, copy=False)
    entity["n_nodes_raw"] = g.size().to_numpy(np.int32)
    entity["has_dod_component"] = g["is_dod"].any().to_numpy(bool)
    entity["has_prime_vendor"] = g["is_prime"].any().to_numpy(bool)
    entity["has_scr_firm"] = g["is_scr"].any().to_numpy(bool)
    entity["has_factset_firm"] = g["is_factset"].any().to_numpy(bool)
    entity["n_dod_nodes"] = g["is_dod"].sum().to_numpy(np.int32)
    entity["n_prime_nodes"] = g["is_prime"].sum().to_numpy(np.int32)
    entity["n_scr_nodes"] = g["is_scr"].sum().to_numpy(np.int32)
    entity["n_factset_nodes"] = g["is_factset"].sum().to_numpy(np.int32)

    # Canonical role (for tiering and reporting).
    entity["entity_role"] = np.select(
        [
            entity["has_dod_component"],
            entity["has_prime_vendor"],
            entity["has_scr_firm"] | entity["has_factset_firm"],
        ],
        ["dod_component", "prime_vendor", "firm"],
        default="unknown",
    )

    # Prefer names and geography from vendor->SCR->FactSet order where possible.
    def pick_first_by_type(node_type: str, col: str) -> pd.Series:
        subset = nodes.loc[nodes["node_type"] == node_type, ["entity_id", col]].dropna(subset=[col])
        if subset.empty:
            return pd.Series(dtype=object)
        return subset.groupby("entity_id")[col].first()

    name_any = (
        nodes[["entity_id", "name"]].dropna(subset=["name"]).groupby("entity_id")["name"].first()
    )
    name_prime = pick_first_by_type("prime_vendor", "name")
    name_scr = pick_first_by_type("scr_firm", "name")
    name_factset = pick_first_by_type("factset_firm", "name")
    name_dod = pick_first_by_type("dod_component", "name")

    name = (
        (name_prime.reindex(eid).combine_first(name_scr.reindex(eid)))
        .combine_first(name_factset.reindex(eid))
        .combine_first(name_dod.reindex(eid))
    )
    name = name.combine_first(name_any.reindex(eid))
    entity["name"] = name.to_numpy(object, copy=False)

    # Representative identifiers for downstream joins / interpretation.
    if "vendor_key" in nodes.columns:
        rep_vendor_key = pick_first_by_type("prime_vendor", "vendor_key").reindex(eid)
        entity["rep_vendor_key"] = rep_vendor_key.to_numpy(object, copy=False)
    else:
        entity["rep_vendor_key"] = None
    if "scr_node_id" in nodes.columns:
        rep_scr = pick_first_by_type("scr_firm", "scr_node_id").reindex(eid)
        entity["rep_scr_node_id"] = rep_scr.to_numpy(object, copy=False)
    else:
        entity["rep_scr_node_id"] = None
    if "factset_entity_id" in nodes.columns:
        rep_fs = pick_first_by_type("factset_firm", "factset_entity_id").reindex(eid)
        entity["rep_factset_entity_id"] = rep_fs.to_numpy(object, copy=False)
    else:
        entity["rep_factset_entity_id"] = None

    # Geography fields.
    for geo_col in ["gr_country", "gr_region", "gr_continent"]:
        geo_scr = pick_first_by_type("scr_firm", geo_col)
        geo_factset = pick_first_by_type("factset_firm", geo_col)
        geo_prime = pick_first_by_type("prime_vendor", geo_col)
        geo_any = (
            nodes[["entity_id", geo_col]]
            .dropna(subset=[geo_col])
            .groupby("entity_id")[geo_col]
            .first()
        )
        geo = geo_scr.reindex(eid).combine_first(geo_factset.reindex(eid))
        geo = geo.combine_first(geo_prime.reindex(eid)).combine_first(geo_any.reindex(eid))
        entity[geo_col] = geo.to_numpy(object, copy=False)

    # Semiconductor flags and RBICS.
    if "is_semi_strict" in nodes.columns:
        nodes["is_semi_strict"] = nodes["is_semi_strict"].fillna(False).astype(bool)
    else:
        nodes["is_semi_strict"] = False
    semi_any = nodes.groupby("entity_id")["is_semi_strict"].any()
    entity["is_semi_strict"] = semi_any.reindex(eid).fillna(False).to_numpy(bool, copy=False)

    semis = nodes[nodes["is_semi_strict"]].copy()
    if not semis.empty:
        semi_source = (
            semis.groupby("entity_id")["semi_source"].apply(_merge_pipe_separated).reindex(eid)
        )
        rbics_l4_names = (
            semis.groupby("entity_id")["rbics_l4_names"].apply(_merge_pipe_separated).reindex(eid)
        )
        rbics_l4_ids = (
            semis.groupby("entity_id")["rbics_l4_ids"].apply(_merge_pipe_separated).reindex(eid)
        )
        entity["semi_source"] = semi_source.to_numpy(object, copy=False)
        entity["rbics_l4_names"] = rbics_l4_names.to_numpy(object, copy=False)
        entity["rbics_l4_ids"] = rbics_l4_ids.to_numpy(object, copy=False)
    else:
        entity["semi_source"] = None
        entity["rbics_l4_names"] = None
        entity["rbics_l4_ids"] = None

    # Collapse edges to entity-level endpoints; drop mapping edges (already used for collapse).
    uid_to_eid = node_map.set_index("node_uid")["entity_id"]
    edges = edges.copy()
    edges["src_eid"] = edges["src_uid"].astype(str).map(uid_to_eid).astype(np.int32)
    edges["dst_eid"] = edges["dst_uid"].astype(str).map(uid_to_eid).astype(np.int32)
    edges = edges[(edges["src_eid"] != edges["dst_eid"])].copy()

    nonmap = edges[(edges["is_maps_to_scr"] == 0) & (edges["is_maps_to_factset"] == 0)].copy()
    nonmap.drop(
        columns=[c for c in ["is_maps_to_scr", "is_maps_to_factset"] if c in nonmap.columns],
        inplace=True,
    )

    agg = {
        "is_contract": "max",
        "is_disclosed": "max",
        "is_predicted": "max",
        "is_shipping": "max",
        "fy_min": "min",
        "fy_max": "max",
        "disclosed_start_date": "min",
        "disclosed_duration_days": "max",
        "pred_score": "max",
        "pred_k": "min",
        "shipments_count": "sum",
        "ship_first_date": "min",
        "ship_last_date": "max",
    }
    agg = {k: v for k, v in agg.items() if k in nonmap.columns}
    entity_edges = nonmap.groupby(["src_eid", "dst_eid"], as_index=False).agg(agg)
    entity_edges["src_uid"] = "ent:" + entity_edges["src_eid"].astype(str)
    entity_edges["dst_uid"] = "ent:" + entity_edges["dst_eid"].astype(str)

    # Normalize flag dtypes.
    for c in ["is_contract", "is_disclosed", "is_predicted", "is_shipping"]:
        if c in entity_edges.columns:
            entity_edges[c] = entity_edges[c].fillna(0).astype(np.int8)

    return EntityGraph(nodes=entity, edges=entity_edges, node_map=node_map)


def supply_edge_mask(edges: pd.DataFrame) -> pd.Series:
    return (
        (edges.get("is_disclosed", 0) == 1)
        | (edges.get("is_predicted", 0) == 1)
        | (edges.get("is_shipping", 0) == 1)
    )


def layer_mask(edges: pd.DataFrame, layer: str) -> pd.Series:
    layer = layer.lower().strip()
    if layer == "disclosed":
        return edges.get("is_disclosed", 0) == 1
    if layer == "shipping":
        return edges.get("is_shipping", 0) == 1
    if layer == "predicted":
        return edges.get("is_predicted", 0) == 1
    if layer in {"highconf", "disclosed+shipping"}:
        return (edges.get("is_disclosed", 0) == 1) | (edges.get("is_shipping", 0) == 1)
    if layer in {"full", "any", "any_supply"}:
        return supply_edge_mask(edges)
    raise ValueError(f"Unknown layer: {layer}")


def evidence_weights(
    edges: pd.DataFrame,
    w_disclosed: float = 1.0,
    w_shipping: float = 0.8,
    w_predicted: float = 0.5,
) -> np.ndarray:
    d = edges.get("is_disclosed", 0).to_numpy(np.float64, copy=False)
    s = edges.get("is_shipping", 0).to_numpy(np.float64, copy=False)
    p = edges.get("is_predicted", 0).to_numpy(np.float64, copy=False)
    return np.maximum.reduce([w_disclosed * d, w_shipping * s, w_predicted * p])


def build_adj_lists(
    n: int, src: np.ndarray, dst: np.ndarray
) -> tuple[list[list[int]], list[list[int]]]:
    fwd: list[list[int]] = [[] for _ in range(n)]
    rev: list[list[int]] = [[] for _ in range(n)]
    for u, v in zip(src.tolist(), dst.tolist(), strict=False):
        fwd[u].append(v)
        rev[v].append(u)
    return fwd, rev


def bfs_reach(
    sources: Iterable[int], adj: list[list[int]], allowed: np.ndarray | None = None
) -> np.ndarray:
    n = len(adj)
    visited = np.zeros(n, dtype=bool)
    q: deque[int] = deque()
    if allowed is None:
        for s in sources:
            if not visited[s]:
                visited[s] = True
                q.append(s)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        return visited

    for s in sources:
        if allowed[s] and not visited[s]:
            visited[s] = True
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if allowed[v] and not visited[v]:
                visited[v] = True
                q.append(v)
    return visited


def bfs_tiers(
    seeds: Iterable[int], rev_adj: list[list[int]], allowed: np.ndarray | None = None
) -> np.ndarray:
    n = len(rev_adj)
    tiers = np.full(n, -1, dtype=np.int32)
    q: deque[int] = deque()
    if allowed is None:
        for s in seeds:
            if tiers[s] == -1:
                tiers[s] = 1
                q.append(s)
        while q:
            u = q.popleft()
            t = tiers[u]
            for v in rev_adj[u]:
                if tiers[v] == -1:
                    tiers[v] = t + 1
                    q.append(v)
        return tiers

    for s in seeds:
        if allowed[s] and tiers[s] == -1:
            tiers[s] = 1
            q.append(s)
    while q:
        u = q.popleft()
        t = tiers[u]
        for v in rev_adj[u]:
            if allowed[v] and tiers[v] == -1:
                tiers[v] = t + 1
                q.append(v)
    return tiers


def build_sparse_transition(
    n: int, src: np.ndarray, dst: np.ndarray, weight: np.ndarray
) -> tuple[sp.csr_matrix, np.ndarray]:
    out_strength = np.bincount(src, weights=weight, minlength=n).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        data = weight / out_strength[src]
    data[~np.isfinite(data)] = 0.0
    mat = sp.csr_matrix((data, (src, dst)), shape=(n, n), dtype=np.float64)
    dangling = out_strength == 0.0
    return mat, dangling


def pagerank_power(
    n: int,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray | None = None,
    alpha: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> np.ndarray:
    if weight is None:
        weight = np.ones_like(src, dtype=np.float64)
    weight = weight.astype(np.float64, copy=False)
    mat, dangling = build_sparse_transition(n, src, dst, weight)

    r = np.full(n, 1.0 / n, dtype=np.float64)
    v = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(max_iter):
        dangling_mass = alpha * r[dangling].sum()
        r_new = alpha * mat.T.dot(r) + (1.0 - alpha) * v + dangling_mass * v
        err = np.abs(r_new - r).sum()
        r = r_new
        if err < tol:
            break
    r_sum = r.sum()
    if r_sum > 0:
        r /= r_sum
    return r


def hits_power(
    n: int,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    if weight is None:
        weight = np.ones_like(src, dtype=np.float64)
    weight = weight.astype(np.float64, copy=False)
    a = sp.csr_matrix((weight, (src, dst)), shape=(n, n), dtype=np.float64)

    hub = np.full(n, 1.0 / math.sqrt(n), dtype=np.float64)
    auth = np.full(n, 1.0 / math.sqrt(n), dtype=np.float64)
    for _ in range(max_iter):
        auth_new = a.T.dot(hub)
        hub_new = a.dot(auth_new)

        auth_norm = np.linalg.norm(auth_new)
        hub_norm = np.linalg.norm(hub_new)
        if auth_norm > 0:
            auth_new /= auth_norm
        if hub_norm > 0:
            hub_new /= hub_norm

        err = np.abs(auth_new - auth).sum() + np.abs(hub_new - hub).sum()
        auth, hub = auth_new, hub_new
        if err < tol:
            break
    return hub, auth


def eigenvector_centrality_power(
    n: int,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray | None = None,
    undirected: bool = True,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> np.ndarray:
    if weight is None:
        weight = np.ones_like(src, dtype=np.float64)
    weight = weight.astype(np.float64, copy=False)
    a = sp.csr_matrix((weight, (src, dst)), shape=(n, n), dtype=np.float64)
    if undirected:
        a = a + a.T

    x = np.full(n, 1.0 / math.sqrt(n), dtype=np.float64)
    for _ in range(max_iter):
        x_new = a.dot(x)
        norm = np.linalg.norm(x_new)
        if norm > 0:
            x_new /= norm
        err = np.abs(x_new - x).sum()
        x = x_new
        if err < tol:
            break
    return x


LATEX_ESCAPE_RE = re.compile(r"([&_#%$])")


def latex_escape(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\", r"\textbackslash{}")
    s = LATEX_ESCAPE_RE.sub(r"\\\1", s)
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    return s


def safe_log10(x: float) -> float:
    if x <= 0:
        return float("nan")
    return math.log10(x)
