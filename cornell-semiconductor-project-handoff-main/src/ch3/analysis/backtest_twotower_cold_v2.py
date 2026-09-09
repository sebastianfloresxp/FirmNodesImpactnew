#!/usr/bin/env python3
"""
Backtest the cold-start Two-Tower configuration on SCR primes by zeroing structure.

This is a proxy validity check: use in-SCR primes, force structural features to zero,
score into a semis-restricted pool, and measure overlap with disclosed SCR edges.

Outputs:
  - figs/chapter3/twotower_cold_v2_backtest_topk_<pool>.tex
  - artifacts/ch3/twotower_cold_v2/summary_backtest_<pool>.json
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.twotower.run_twotower_eval import TwoTowerModel
from src.twotower.run_twotower_eval import load_features as tt_load_features


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def format_latex(df: pd.DataFrame, caption: str, label: str, float_cols: list[str]) -> str:
    df_fmt = df.copy()
    for col in float_cols:
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.3f}")
    return df_fmt.to_latex(index=False, caption=caption, label=label, escape=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest cold-start Two-Tower on SCR primes")
    ap.add_argument(
        "--pool",
        type=str,
        default="semis_adjacent",
        choices=["semis_core", "semis_adjacent", "semis_adjacent_hubs"],
    )
    ap.add_argument(
        "--geo-gated",
        action="store_true",
        help="Apply geo-gated candidate pools (country -> region -> global)",
    )
    ap.add_argument(
        "--geo-min-candidates",
        type=int,
        default=50,
        help="Minimum candidates to accept country/region pool before falling back",
    )
    ap.add_argument(
        "--semis-flags", type=Path, default=Path("artifacts/ch3/prediction/semis_flags.parquet")
    )
    ap.add_argument(
        "--tier1-primes", type=Path, default=Path("artifacts/ch3/network/tier1_primes_scr.parquet")
    )
    ap.add_argument(
        "--edges-disclosed",
        type=Path,
        default=Path("artifacts/ch3/network/edges_disclosed_scr.parquet"),
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
    ap.add_argument("--k-list", type=str, default="5,10,50")
    ap.add_argument("--batch-emb", type=int, default=131072)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--fig-root", type=Path, default=Path("figs/chapter3"))
    ap.add_argument("--out-root", type=Path, default=Path("artifacts/ch3/twotower_cold_v2"))
    args = ap.parse_args()

    fig_root = args.fig_root
    fig_root.mkdir(parents=True, exist_ok=True)
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    if not k_list:
        raise ValueError("--k-list must contain at least one K")
    k_max = max(k_list)

    device = torch.device(
        "cuda"
        if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    # Load model + features
    payload = torch.load(args.twotower_seed_dir / "model.pt", map_location=device)  # nosec B614 -- local pipeline artifacts
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

    # Pool definition
    semis = pd.read_parquet(args.semis_flags)
    core = semis[semis["is_semi_core"].astype(bool)]["node_id"].astype(int).to_numpy()
    adj = semis[semis["is_semi_adjacent"].astype(bool)]["node_id"].astype(int).to_numpy()
    dest_nodes = core
    if args.pool in {"semis_adjacent", "semis_adjacent_hubs"}:
        dest_nodes = np.unique(np.concatenate([core, adj]))
    if args.pool == "semis_adjacent_hubs":
        outd = np.load(args.deg_out)
        ind = np.load(args.deg_in)
        deg = outd.astype(np.int64) + ind.astype(np.int64)
        k = max(1, int(len(deg) * float(args.hub_top_frac)))
        idx = np.argpartition(deg, -k)[-k:]
        hubs = idx.astype(np.int64)
        dest_nodes = np.unique(np.concatenate([dest_nodes, hubs]))

    dest_nodes = dest_nodes.astype(np.int64)
    dest_nodes.sort()

    primes = pd.read_parquet(args.tier1_primes, columns=["node_id"])
    src_nodes = primes["node_id"].astype(np.int64).to_numpy()

    # Disclosed edges for base rate + hits
    df_known = pd.read_parquet(args.edges_disclosed, columns=["src_id", "dst_id"])
    df_known = df_known[df_known["src_id"].isin(src_nodes) & df_known["dst_id"].isin(dest_nodes)]
    df_known = df_known.drop_duplicates(subset=["src_id", "dst_id"])
    known_set = set(
        zip(df_known["src_id"].astype(int), df_known["dst_id"].astype(int), strict=False)
    )

    base_rate = (
        (len(known_set) / float(len(src_nodes) * len(dest_nodes))) if len(dest_nodes) else 0.0
    )

    # Build embeddings
    struct_v = feats.struct[dest_nodes]
    attr_codes_v = {k: v[dest_nodes] for k, v in feats.attr_codes.items()}
    ZV = compute_embeddings_side(
        model=model,
        side="v",
        struct=struct_v.astype(np.float32, copy=False),
        attr_codes={k: v.astype(np.int64, copy=False) for k, v in attr_codes_v.items()},
        device=device,
        batch_size=int(args.batch_emb),
    )

    struct_u = np.zeros((len(src_nodes), feats.struct.shape[1]), dtype=np.float32)
    attr_codes_u = {k: v[src_nodes] for k, v in feats.attr_codes.items()}
    ZU = compute_embeddings_side(
        model=model,
        side="u",
        struct=struct_u,
        attr_codes={k: v.astype(np.int64, copy=False) for k, v in attr_codes_u.items()},
        device=device,
        batch_size=int(args.batch_emb),
    )

    # Build geo indexes for gated candidate pools
    dest_country = feats.attr_codes["gr_country"].astype(np.int64, copy=False)
    dest_region = feats.attr_codes["gr_region"].astype(np.int64, copy=False)

    def _geo_index(nodes: np.ndarray, codes: np.ndarray) -> dict[int, np.ndarray]:
        d: dict[int, np.ndarray] = {}
        node_codes = codes[nodes]
        for code in np.unique(node_codes):
            d[int(code)] = nodes[node_codes == code]
        return d

    dest_by_country = _geo_index(dest_nodes, dest_country)
    dest_by_region = _geo_index(dest_nodes, dest_region)
    nan_country_code = (
        int(np.min(dest_country)) if np.min(dest_country) < 0 else int(np.max(dest_country) + 1)
    )
    nan_region_code = (
        int(np.min(dest_region)) if np.min(dest_region) < 0 else int(np.max(dest_region) + 1)
    )

    # Scoring + hits
    hit_counts = dict.fromkeys(k_list, 0)
    zv_full = ZV.to(device)
    for i, src_id in enumerate(src_nodes):
        zu = ZU[i].to(device)

        if args.geo_gated:
            src_country = int(feats.attr_codes["gr_country"][src_id])
            src_region = int(feats.attr_codes["gr_region"][src_id])
            cand = np.empty(0, dtype=np.int64)
            if src_country != nan_country_code:
                cand = dest_by_country.get(src_country, np.empty(0, dtype=np.int64))
            if cand.size < int(args.geo_min_candidates) and src_region != nan_region_code:
                cand = dest_by_region.get(src_region, np.empty(0, dtype=np.int64))
            if cand.size < int(args.geo_min_candidates):
                cand = dest_nodes
            zv = zv_full.index_select(0, torch.from_numpy(cand))
            scores = torch.mv(zv, zu)
            topk_idx = torch.topk(scores, min(k_max, scores.shape[0])).indices.cpu().numpy()
            topk_dst = cand[topk_idx]
        else:
            scores = torch.mv(zv_full, zu)
            topk_idx = torch.topk(scores, k_max).indices.cpu().numpy()
            topk_dst = dest_nodes[topk_idx]

        for k in k_list:
            for dst_id in topk_dst[:k]:
                if (int(src_id), dst_id) in known_set:
                    hit_counts[k] += 1

    rows = []
    for k in sorted(k_list):
        total = int(len(src_nodes) * k)
        hits = int(hit_counts[k])
        precision = hits / total if total else 0.0
        lift = precision / base_rate if base_rate else 0.0
        rows.append(
            {"K": k, "Predictions": total, "Known SCR": hits, "Precision": precision, "Lift": lift}
        )
    df = pd.DataFrame(rows)

    pool_tag = f"{args.pool}_geo" if args.geo_gated else args.pool
    tex_path = fig_root / f"twotower_cold_v2_backtest_topk_{pool_tag}.tex"
    tex_path.write_text(
        format_latex(
            df,
            caption=(
                "Cold-start backtest on SCR primes (structure zeroed). "
                f"Pool={pool_tag}. Base rate {base_rate * 100:.4f}\\%."
            ),
            label=f"tab:ch3_twotower_backtest_{pool_tag}",
            float_cols=["Precision", "Lift"],
        )
    )

    summary = {
        "created_at": _now_iso(),
        "pool": pool_tag,
        "k_list": k_list,
        "n_src": len(src_nodes),
        "n_dest": len(dest_nodes),
        "n_known": len(known_set),
        "base_rate": float(base_rate),
        "hits": hit_counts,
        "outputs": {"latex": str(tex_path)},
    }
    (out_root / f"summary_backtest_{args.pool}.json").write_text(json.dumps(summary, indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
