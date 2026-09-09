# Chapter 4 Module Contract (Reproducibility Standard)

## Purpose
This contract defines how Chapter 4 work is implemented, reviewed, and accepted
module-by-module so analysis remains transparent and reproducible.

Primary objective: quantify **structural severity if disrupted** (impact),
not disruption likelihood.

## Governance Rules
1. No new module implementation starts without explicit sign-off on the prior module.
2. Every module has fixed inputs, outputs, and acceptance checks.
3. Every run is versioned by `snapshot` and `run_id`.
4. All outputs are generated from scripts (no manual edits in artifacts).
5. Existing Chapter 4 outputs are retained as legacy unless explicitly removed.

## Module Order
`M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M7 -> M8 -> M9`

## M0 Graph Object and Direction Contract (Gatekeeper Module)
### Required decisions before any analysis module
1. **Identity handling**
   - Option A (recommended default): collapse `is_identity_bridge` edges into
     entity nodes and run supply analysis on entity graph.
   - Option B: keep raw graph and treat identity bridges as zero-length traversal
     only in path routines.
2. **Edge views**
   - `A (disclosed) = is_disclosed`
   - `A+B (observed) = is_disclosed OR is_observed_ship`
   - `A+B+C (full) = is_disclosed OR is_observed_ship OR is_predicted`
   - `is_contract_seed` remains metadata/anchoring and is excluded from
     structural vulnerability metrics.
3. **Direction semantics**
   - Validate upstream traversal from primes using reversed graph.
   - If inverted, flip traversal convention globally and log.

### M0 outputs
- `artifacts/ch4/v2/<snapshot>/m0/graph_contract.json`
- `artifacts/ch4/v2/<snapshot>/m0/node_table_contract.parquet`
- `artifacts/ch4/v2/<snapshot>/m0/edge_table_contract.parquet`
- `artifacts/ch4/v2/<snapshot>/m0/direction_sanity_report.csv`

### M0 acceptance checks
- Prime upstream traversal reaches expected semis/suppliers in sanity sample.
- View edge counts are monotone: `|E_disclosed| <= |E_observed| <= |E_full|`.
- Node/edge dedup checks pass on directed dyad key.

## M1 Baseline Topology
### Outputs
- `.../m1/graph_summary_<view>.csv`
- `.../m1/node_metrics_baseline_<view>.parquet`

### Acceptance checks
- Basic graph statistics match recomputed control totals.
- Degree distributions and giant component sizes are reproducible by seed.

## M9 Post-Analysis Semiconductor Relevance Lens (Reporting Layer)
### Execution order and entrypoints
- Run **after** structural modules (`M1`-`M7`, and `M8` when used).
- Module aliases (preferred naming):
  - `m9_1_code_universe.py` (alias of `m0_5_code_universe.py`)
  - `m9_2_factset_industry_enrichment.py` (alias of `m0_6_factset_industry_enrichment.py`)
  - `m9_3_semiconductor_lens.py` (alias of `m0_7_semiconductor_lens.py`)
  - `m9_0_semantic_lens_pipeline.py` (orchestrator for 9.1->9.3)
- Legacy implementation and artifact directories remain `m0_5`, `m0_6`,
  `m0_7` for backward compatibility.

### Purpose
- Apply a pre-registered RBICS/SIC codebook to classify nodes as:
  - `include` (semiconductor relevance lens),
  - `exclude`,
  - `uncertain`
- Preserve structural metrics from the locked Chapter 3 graph; this module only
  filters/report-ranks candidate nodes for interpretation and action.

### Inputs
- `.../m0_6/node_industry_enrichment_factset.parquet`
- `.../m0_6/node_code_membership_factset.parquet`
- `configs/ch4_semiconductor_codebook_v2_1.csv`
- `.../m0/edge_table_contract.parquet` (for empirical gate path checks)
- `.../m6_2/node_impacts_stratified_<view>.csv` (or `m6_1` fallback)

### Outputs
- `.../m0_7/node_semiconductor_lens_decisions.csv|parquet`
- `.../m0_7/node_semiconductor_lens_decisions_by_view.csv|parquet`
- `.../m0_7/node_semiconductor_lens_rule_hits.csv|parquet`
- `.../m0_7/node_semiconductor_lens_rule_hits_by_view.csv|parquet`
- `.../m0_7/codebook_rule_coverage.csv`
- `.../m0_7/lens_decision_summary.csv`
- `.../m0_7/lens_topn_coverage_by_view.csv`
- `.../m0_7/lens_topn_composition_by_rule_class.csv`
- `.../m0_7/top{N}_high_impact_semiconductor_lens_<view>.csv`
- `.../m0_7/top{N}_high_impact_semiconductor_lens_plus_uncertain_<view>.csv`
- `.../m0_7/top{N}_high_impact_semiconductor_value_chain_strict_<view>.csv`
- `.../m0_7/top{N}_high_impact_semiconductor_adjacent_uncertain_<view>.csv`
- `.../m0_7/top{N}_high_impact_digital_infrastructure_dependencies_<view>.csv`

### Acceptance checks
- Codebook joins are deterministic and fully auditable (rule hits exported).
- Evidence-gated conditional rules use **two-sided empirical path** checks
  (semi→node and node→prime) and are logged with pass/fail per node and view.
- Gate application is view-specific (`disclosed`, `observed`, `full`).
- Endpoint overrides are enforced: prime and semi anchors cannot be excluded.
- Filtered Top-N outputs are generated for all configured views.

## M2 Centrality Candidate Lists (Not Final Impact)
### Outputs
- `.../m2/node_centralities_<view>.parquet`
- `.../m2/seam_nodes_<view>.csv`

### Acceptance checks
- Approximation settings are logged (sampling sizes, seeds).
- Rank outputs are stable across rerun with same seed.

### M2.1 Seam Refinement (Exact Structural Split)
Use when full seam coverage is needed before moving to exposure/harm modules.

### Outputs
- `.../m2_refine/seam_nodes_refined_<view>.parquet`
- `.../m2_refine/seam_summary_<view>.csv`
- `.../m2_refine/top_seams_by_second_component_<view>.csv`

### Acceptance checks
- Exact split severity is computed for all articulation nodes and all bridges.
- Output includes second-component size distributions and prime-touch subsets.

## M3 DoD Exposure and Common-Mode Dependence
### Core definitions
- Prime endpoint weights from obligations table (`w_p`), unit weights fallback.
- `Up(p)` from upstream closure with hop cap `L`.
- Node `prime_reach(i)` weighted by reachable primes downstream from `i`.

### Outputs
- `.../m3/prime_upstream_sizes_<view>.csv`
- `.../m3/node_prime_reach_<view>.parquet`
- `.../m3/common_mode_top100_<view>.csv`

### Acceptance checks
- Weighted and unit-weight variants both executable.
- Prime reach totals reconcile with closure computations.

### M3.1 Supplier-Centric Refinement
Use after M3 when endpoint anchoring (`is_contract_seed`) is inflating common-mode
rankings for non-supplier roles.

### Core definitions
- Reuse M3 reach logic, but exclude configured edge flags (default:
  `is_contract_seed`) from reach calculations.
- Export supplier-centric rankings for all nodes, firm-only nodes, and semi-only
  nodes.

### Outputs
- `.../m3_refine/prime_upstream_sizes_suppliercentric_<view>.csv`
- `.../m3_refine/node_prime_reach_suppliercentric_<view>.parquet`
- `.../m3_refine/common_mode_top100_suppliercentric_firm_<view>.csv`
- `.../m3_refine/common_mode_top100_suppliercentric_semi_<view>.csv`

### Acceptance checks
- Edge exclusion policy is logged in run metadata.
- Prime reach reconciliation identity still holds after exclusions.

## M4 Semi->Prime Corridor Importance
### Core definitions
- Sources: `semi == 1`; targets: primes.
- Distances: `dist_from_semi`, `dist_to_prime`.
- Corridor condition: finite in both directions.

### Outputs
- `.../m4/corridor_nodes_<view>.parquet`
- `.../m4/top_corridor_nodes_<view>.csv`

### Acceptance checks
- Distance calculations validated on sampled hand-check paths.
- Corridor classification coverage reported.

## M4.1 Corridor Bottleneck Refinement
Use when M4 corridor scores are heavily tied (e.g., large SCC core) and require
additional structural discrimination.

### Core definitions
- SCC-normalized corridor coverage: `corridor_score / scc_size`.
- External boundary gate counts: inbound and outbound edges crossing SCC
  boundaries.
- Weighted shortest-path distances with predicted-only edge penalty in full view.
- Geodesic lane counts and local narrowness proxy.

### Outputs
- `.../m4_refine/corridor_nodes_refined_<view>.parquet`
- `.../m4_refine/top_corridor_nodes_refined_<view>.csv`
- `.../m4_refine/top_corridor_nodes_refined_intermediary_<view>.csv`
- `.../m4_refine/tie_diagnostics_<view>.csv`

### Acceptance checks
- Raw-score tie block size vs refined-score tie block size reported per view.
- Predicted-edge penalty parameter logged in run metadata.
- Refined rankings remain reproducible by fixed config and seed.

## M5 Redundancy / Substitution Proxy
### Core definitions
- `semi_support_count`: number of strict semis that can reach each prime.
- `entry_branch_count`: number of corridor-qualified immediate upstream entries
  into each prime (supplier-only by default).
- `entry_scc_count`: distinct SCC sources among those entries.
- `geodesic_entry_count`: entries lying on weighted shortest semi->prime paths.
- `redundancy_proxy = min(semi_support_count, entry_branch_count, entry_scc_count, geodesic_entry_count)`.

### Outputs
- `.../m5/prime_redundancy_<view>.csv`
- `.../m5/redundancy_single_point_primes_<view>.csv`
- `.../m5/redundancy_monotonicity_checks.csv`

### Acceptance checks
- Chosen proxy documented (k-shortest or disjoint approximation).
- Runtime and hop limits logged.

## M6 Disruption Severity Envelopes (Primary Chapter 4 Engine)
### Harm functions
- `H1`: DoD-weighted reachability loss
- `H2`: path-length growth and disconnect share
- `H3`: redundancy collapse

### Pilot protocol (recommended before full M6)
- Run `full` view only with candidate pool screening and reduced simulation budget.
- Suggested defaults: candidate pool `500`, deep reevaluation `100`,
  random repeats `10`, fractions `{0.1%, 0.5%, 1%}`, greedy `k={5,10}`.
- Pilot artifacts live under `.../m6_pilot/` to keep final M6 outputs clean.

### Outputs
- `.../m6/robustness_curves_<view>.csv`
- `.../m6/node_single_removal_impacts_<view>.parquet`
- `.../m6/top100_high_impact_<view>.csv`
- `.../m6/interdiction_sets_<view>.csv`
- `.../m6/interdiction_performance_<view>.csv`

### Acceptance checks
- Random removal confidence bands use configured repeat count.
- Targeted strategies include degree, PageRank, betweenness, prime_reach,
  corridor score.
- Greedy interdiction for `k in {5,10,25}` logged and reproducible.

## M7 Robustness to Evidence and Boundary
### Outputs
- `.../m7/stability_across_views.csv`
- `.../m7/high_impact_confidence_quadrants.csv`

### Acceptance checks
- Top-25 overlap and rank correlations reported across views.
- Boundary sensitivity variant status explicitly labeled:
  implemented vs constrained by fixed graph universe.

## M8 Semi-Upstream Disruption (Semiconductor Perspective)
### Purpose
- Recompute deny/delay severity from the semiconductor side by evaluating
  upstream node removals against semiconductor support and path structure.

### Outputs
- `.../m8/robustness_curves_<view>.csv`
- `.../m8/node_single_removal_impacts_<view>.parquet`
- `.../m8/top100_high_impact_<view>.csv`
- `.../m8/interdiction_sets_<view>.csv`
- `.../m8/interdiction_performance_<view>.csv`

### Acceptance checks
- Uses the same disruption controls as M6 (fractions, repeats, `k` budgets).
- Reports deny (`H1`) and delay (`H2`) effects with reproducible settings.

## M9.4 Action Matrix (Decision Translation)
### Purpose
- Translate impact and confidence outputs into decision buckets for policy use
  (`act_now`, `validate_then_act`, `monitor_routine`, `deprioritize`).

### Outputs
- `.../m9/action_matrix_nodes.csv`
- `.../m9/action_matrix_summary.csv`

### Acceptance checks
- Bucket assignment logic and thresholds are captured in run metadata.
- Inputs are traceable to M6/M7 outputs used for ranking and confidence.

## Reproducibility Requirements
1. Every module writes:
   - `manifest_<module>.json` with inputs, checksums, params, outputs
   - `run_metadata.json` with git commit hash, python/package versions, seed
2. File naming must include `snapshot` and `view`.
3. Randomized routines must read seed from config only.
4. Large outputs: Parquet; summary outputs: CSV.
5. Frozen defaults are recorded in `docs/chapter4_freeze.md`.

## Repository Layout (v2 workstream)
- Code:
  - `src/analysis/chapter4/ch4_common.py` (shared I/O and graph utilities)
  - `src/analysis/chapter4/modules/`
  - `src/analysis/chapter4/reporting/` (8 figure/table export scripts)
  - `src/analysis/chapter4/run_pipeline.sh` (M0→M9 + reporting orchestrator)
  - `src/analysis/chapter4/config/ch4_v2_fix01.yaml` (production config)
- Artifacts:
  - `artifacts/ch4/v2/<snapshot>/m0..m8/`
  - `artifacts/ch4/v2/<snapshot>/m0_5,m0_6,m0_7/` (M9 semantic lens layer)
  - `artifacts/ch4/v2/<snapshot>/m9/` (M9.4 action matrix)
- Tables/Figures:
  - `tables/chapter4/v2/`
  - `figs/chapter4/v2/`

## Legacy Handling Policy
- Keep current `artifacts/ch4/*` and `src/analysis/chapter4/0*_*.py` as
  legacy exploratory outputs.
- Do not delete until v2 results are accepted.
- If needed, mark legacy explicitly by adding `legacy_` prefixes later.
