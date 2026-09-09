# Chapter 4 Freeze Record

## Status
This file records the final configuration defaults for the Chapter 4 pipeline.
The goal is to prevent drift in methods and reporting.

Freeze date: 2026-02-18

## Canonical Configs
- `src/analysis/chapter4/config/ch4_v2.template.yaml`
- `src/analysis/chapter4/config/ch4_v2_fix01.yaml`

## Locked Defaults
1. Snapshot and run id
- `snapshot: 2025-06-09`
- Template `run_id: ch4_v2_final_freeze`
- Fix01 `run_id: ch4_v2_fix01_final_freeze`

2. Official H1 objective (M6.1 / M9 ranking)
- `primary_h1_key: log_obligation_any_support`
- Interdiction keys include:
  - `unit_support_count`
  - `log_obligation_any_support`

3. Effective-reach settings (M3.2)
- `hop_caps: [3, 5, 7, 10]`
- `cost_caps: [5, 10, 15, 20]`
- `predicted_only_edge_cost: 3.0`

4. Semantic lens defaults (M0.7 / M9.3)
- `codebook_path: configs/ch4_semiconductor_codebook_v2_1.csv`
- `primary_decision_view: observed`
- `gate_empirical_only: true`
- `apply_endpoint_overrides: true`
- Reporting headline stream: `semiconductor_value_chain_strict`

## Interpretation Guardrails (Locked Language)
1. Redundancy caveat (M5)
- `redundancy_proxy` is a conservative proxy:
  `min(semi_support_count, entry_branch_count, entry_scc_count, geodesic_entry_count)`.
- It is not equivalent to exact disjoint-path or min-cut redundancy.

2. Ratio reporting (M6 / M8)
- Report `target_vs_random_ratio_h1` with absolute values:
  - target mean H1 loss
  - random mean H1 loss
  - absolute delta
- Do not report ratios alone.

3. Scope claim
- Structural severity is measured on the locked dependency graph.
- Semantic lensing is a reporting layer and does not alter graph topology.

## Change Policy After Freeze
- No further codebook fine-tuning unless a rule-level error is found.
- Any post-freeze change requires:
  - updated run id,
  - explicit changelog entry in this file,
  - regenerated manifest for affected modules.
