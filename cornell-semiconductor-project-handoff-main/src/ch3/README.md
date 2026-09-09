# Chapter 3 Pipeline (DoD Supply Network)

Utilities for building the DoD-anchored network for Chapter 3. Keep
intermediate artifacts under `artifacts/ch3/` and analysis-ready tables/plots
under `tables/` and `figs/`.

## Layout
- `config/` – timelines, operating policies, code lists
- `usaspending/` – DoD award ingestion and Tier-1 aggregation
- `matching/` – prime → FactSet matching (name+country)
- `prediction/` – candidate pools, scoring, meta join, Top-K selection
- `network/` – graph assembly (disclosed, predicted, shipping, pruning)
- `analysis/` – tables/figures for Chapter 3 writeup

## Canonical build (directionally correct, supplier→prime)
The canonical Chapter 3 build uses the **upstream** prediction orientation:
SCR edges are supplier→customer (`src_id`→`dst_id`), and the prediction layer is
constructed as **candidate supplier → DoD prime** with per-prime Top-K grouped
by `dst_id`.

Canonical outputs live in:
- `artifacts/ch3/prediction_upstream/` – predicted-layer artifacts (Top-K, diagnostics inputs)
- `artifacts/ch3/network_upstream/` – merged DoD network + strict semiconductor subgraph

Key final artifacts:
- Full DoD network (contracts + disclosed + predicted + shipping):
  - `artifacts/ch3/network_upstream/dod_network_edges_top5_d99_shipping.parquet`
  - `artifacts/ch3/network_upstream/dod_network_nodes_top5_d99_shipping.parquet`
- Strict semiconductor subgraph (Chapter 4 input):
  - `artifacts/ch3/network_upstream/dod_semiconductor_edges_top5_d99_shipping_strict.parquet`
  - `artifacts/ch3/network_upstream/dod_semiconductor_nodes_top5_d99_shipping_strict.parquet`
- Chapter 3 tabulars (LaTeX `tabular` only, for `\\input{...}`):
  - `tables/chapter3/tab_3.1_topk_precision_lift.tex`
  - `tables/chapter3/tab_3.2_table_deciles.tex`
  - `tables/chapter3/tab_3.3_model_agreement_precision.tex`
  - `tables/chapter3/tab_3.4_dod_semiconductor_network_summary.tex`

## Suggested rerun order (high level)
1) USAspending seed tables: `src/ch3/usaspending/01_build_dod_transactions.py`, `02_aggregate_primes.py`
2) Matching: `src/ch3/matching/match_primes.py` (outputs `prime_matches_thr90.parquet`)
3) Semi tags for candidate pools: `src/ch3/prediction/tag_semis.py` (writes `artifacts/ch3/reference/semis_flags.parquet`)
4) Candidate pool + scoring:
   - `src/ch3/prediction/build_candidate_pool.py`
   - `src/ch3/prediction/flip_candidates.py` (default writes sorted upstream candidates)
   - `src/ch3/prediction/run_scoring_pipeline.sh`
   - `src/ch3/prediction/build_meta_zero_slices.py`
   - `src/ensemble/apply_meta.py`
   - `src/ch3/prediction/select_topk.py --group-by dst_id`
5) Merge network layers + prune:
   - `src/ch3/network/package_scr_layers.py`
   - `src/ch3/network/package_dod_contract_layer.py`
   - `src/ch3/network/build_disclosed_baseline.py --prime-matches artifacts/ch3/matching/prime_matches_thr90.parquet --primes-agg artifacts/ch3/usaspending/dod_primes_fy2022-2025_asof.parquet --depth 99 --out-dir artifacts/ch3/network_upstream`
   - `src/ch3/network/package_dod_network_d99.py`
   - `src/ch3/network/build_shipping_observed_layer.py`
   - `src/ch3/network/package_dod_network_with_shipping.py`
   - Strict semis catalog: `src/ch3/prediction/build_semis_catalog_strict.py` (writes `artifacts/ch3/reference/semis_flags_strict.parquet`)
   - `src/ch3/network/build_semiconductor_subgraph.py --tag top5_d99_shipping_strict`

## Canonical build (copy/paste commands)
These commands reproduce the canonical upstream run (supplier→prime). Some steps require
FactSet SQL access (`.env`), and shipping/sector pulls will fail without DB credentials.

1) USAspending seed (FY2022–FY2025 as-of 2025-06-09)
   - `python src/ch3/usaspending/01_build_dod_transactions.py --fy-start 2022 --fy-end 2025 --as-of-date 2025-06-09`
     - writes:
       - `artifacts/ch3/usaspending/dod_transactions_fy2022-2025_asof.parquet`
       - `artifacts/ch3/usaspending/dod_transactions_fy2022-2025_post.parquet`
   - `python src/ch3/usaspending/02_aggregate_primes.py --transactions-asof artifacts/ch3/usaspending/dod_transactions_fy2022-2025_asof.parquet --transactions-post artifacts/ch3/usaspending/dod_transactions_fy2022-2025_post.parquet`
   - `python src/ch3/usaspending/03_report_usaspending_summary.py --transactions-asof artifacts/ch3/usaspending/dod_transactions_fy2022-2025_asof.parquet --transactions-post artifacts/ch3/usaspending/dod_transactions_fy2022-2025_post.parquet --primes-asof artifacts/ch3/usaspending/dod_primes_fy2022-2025_asof.parquet --primes-post artifacts/ch3/usaspending/dod_primes_fy2022-2025_post.parquet`

2) Tier-1 matching (name/DBA/parent + fuzzy; threshold 0.90)
   - `python src/ch3/matching/match_primes.py --primes artifacts/ch3/usaspending/dod_primes_fy2022-2025_asof.parquet --transactions artifacts/ch3/usaspending/dod_transactions_fy2022-2025_asof.parquet --fetch-sym-from-db --fuzzy-threshold 0.90 --out-matches artifacts/ch3/matching/prime_matches_thr90.parquet`

3) Semiconductor tags for candidate pools (core/adjacent)
   - `python src/ch3/prediction/tag_semis.py`
     - writes `artifacts/ch3/reference/semis_flags.parquet`

4) Candidate pool (prime→candidate) then flip to upstream (candidate→prime)
   - `python src/ch3/prediction/build_candidate_pool.py --prime-matches artifacts/ch3/matching/prime_matches_thr90.parquet --semis artifacts/ch3/reference/semis_flags.parquet`
     - writes `artifacts/ch3/prediction_upstream/candidates_prime_to_candidate.parquet`
   - `python src/ch3/prediction/flip_candidates.py --write-summary`
     - writes `artifacts/ch3/prediction_upstream/candidates_with_label_sorted.parquet`

5) Score candidates with trained Chapter 2 models (GPU if available)
   - `bash src/ch3/prediction/run_scoring_pipeline.sh`
     - writes staged per-model scores to `artifacts/ch3/prediction_upstream/ch3_scores/*_scores.parquet`

6) Build deployment-style meta inputs (zero slice features), apply meta model, select Top-K
   - `python src/ch3/prediction/build_meta_zero_slices.py --base artifacts/ch3/prediction_upstream/candidates_with_label_sorted.parquet --scores-root artifacts/ch3/prediction_upstream/ch3_scores --out-root artifacts/ch3/prediction_upstream --overwrite`
   - `python src/ensemble/apply_meta.py --model-config artifacts/ensemble/meta_ranker/meta_ranker_v4/meta_model.json --meta-input artifacts/ch3/prediction_upstream/meta_inputs_zero_slices.parquet --output artifacts/ch3/prediction_upstream/meta_scores_zero_slices.parquet`
   - `python src/ch3/prediction/select_topk.py --meta-scores artifacts/ch3/prediction_upstream/meta_scores_zero_slices.parquet --out-root artifacts/ch3/prediction_upstream --group-by dst_id --overwrite`

7) Merge disclosed + predicted + contracts + shipping, then prune to strict semis
   - `python src/ch3/network/package_scr_layers.py`
   - `python src/ch3/network/package_dod_contract_layer.py`
   - `python src/ch3/network/build_disclosed_baseline.py --prime-matches artifacts/ch3/matching/prime_matches_thr90.parquet --primes-agg artifacts/ch3/usaspending/dod_primes_fy2022-2025_asof.parquet --depth 99 --out-dir artifacts/ch3/network_upstream`
   - `python src/ch3/network/package_dod_network_d99.py`
   - `python src/ch3/network/build_shipping_observed_layer.py`
   - `python src/ch3/network/package_dod_network_with_shipping.py`
   - `python src/ch3/prediction/build_semis_catalog_strict.py`
   - `python src/ch3/network/build_semiconductor_subgraph.py --tag top5_d99_shipping_strict`

8) Chapter tables (tabular-only LaTeX)
   - `python src/ch3/analysis/build_ch3_tables.py`
   - `python src/ch3/analysis/build_semis_network_summary_table.py`

## Ablations and legacy artifacts
To keep the canonical pipeline readable:
- `artifacts/ch3/prediction_upstream_20k/` contains the pool-size (20k budget) ablation.
- Older/failed experiment outputs are intentionally deleted to reduce noise; regenerate into a new `--out-root` if needed.
