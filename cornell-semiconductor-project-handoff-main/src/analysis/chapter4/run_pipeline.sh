#!/usr/bin/env bash
# =============================================================================
# Chapter 4 Analysis Pipeline Orchestrator
# =============================================================================
# Executes the full M0→M9 production module sequence plus reporting exports.
#
# Usage:
#   ./run_pipeline.sh                                  # default config
#   ./run_pipeline.sh --config path/to/config.yaml     # custom config
#   ./run_pipeline.sh --skip-m9-2                      # skip FactSet DB step
#   ./run_pipeline.sh --skip-reporting                 # modules only, no exports
#
# Run from the repository root:
#   bash src/analysis/chapter4/run_pipeline.sh --config src/analysis/chapter4/config/ch4_v2_fix01.yaml
#
# Module order follows the module contract: docs/chapter4_module_contract.md
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
CONFIG_PATH="src/analysis/chapter4/config/ch4_v2_fix01.yaml"
SKIP_M9_2=false
SKIP_REPORTING=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --skip-m9-2)
            SKIP_M9_2=true
            shift
            ;;
        --skip-reporting)
            SKIP_REPORTING=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--config path] [--skip-m9-2] [--skip-reporting]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

run_module() {
    local script="$1"
    shift
    local name
    name=$(basename "$script" .py)
    echo ""
    echo ">>> [$(ts)] $name — starting..."
    python "$script" --config "$CONFIG_PATH" "$@"
    echo "<<< [$(ts)] $name — done"
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Chapter 4 Pipeline"
echo "  Config:  $CONFIG_PATH"
echo "  Started: $(ts)"
echo "============================================================"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Check Ch3 input artifacts exist
NODES="artifacts/ch3/network_upstream/dod_semiconductor_nodes_top5_d99_shipping_strict.parquet"
EDGES="artifacts/ch3/network_upstream/dod_semiconductor_edges_top5_d99_shipping_strict.parquet"

if [[ ! -f "$NODES" ]]; then
    echo "ERROR: Ch3 node parquet not found: $NODES"
    echo "       Sync artifacts from Google Drive before running the pipeline."
    exit 1
fi
if [[ ! -f "$EDGES" ]]; then
    echo "ERROR: Ch3 edge parquet not found: $EDGES"
    echo "       Sync artifacts from Google Drive before running the pipeline."
    exit 1
fi

echo "Pre-flight: Ch3 inputs found."

# ---------------------------------------------------------------------------
# M0: Graph contract and prime weights
# ---------------------------------------------------------------------------
run_module src/analysis/chapter4/modules/m0_graph_contract.py
run_module src/analysis/chapter4/modules/m0_build_prime_weights.py

# ---------------------------------------------------------------------------
# M1: Baseline topology
# ---------------------------------------------------------------------------
run_module src/analysis/chapter4/modules/m1_topology.py

# ---------------------------------------------------------------------------
# M2: Centrality and structural refinements
# ---------------------------------------------------------------------------
run_module src/analysis/chapter4/modules/m2_centrality_candidates.py
run_module src/analysis/chapter4/modules/m2_1_seam_refinement.py
run_module src/analysis/chapter4/modules/m2_2_seam_dod_harm.py
run_module src/analysis/chapter4/modules/m2_3_hard_chokepoints.py

# ---------------------------------------------------------------------------
# M3: DoD exposure and supplier refinements
# ---------------------------------------------------------------------------
run_module src/analysis/chapter4/modules/m3_dod_exposure_common_mode.py
run_module src/analysis/chapter4/modules/m3_1_supplier_centric_refinement.py
run_module src/analysis/chapter4/modules/m3_2_effective_reach.py

# ---------------------------------------------------------------------------
# M4: Corridor importance
# ---------------------------------------------------------------------------
run_module src/analysis/chapter4/modules/m4_corridor_importance.py
run_module src/analysis/chapter4/modules/m4_1_corridor_refinement.py

# ---------------------------------------------------------------------------
# M5: Redundancy proxy
# ---------------------------------------------------------------------------
run_module src/analysis/chapter4/modules/m5_redundancy_proxy.py

# ---------------------------------------------------------------------------
# M6: Disruption severity (production engine: m6_1 weighted disruption)
# ---------------------------------------------------------------------------
run_module src/analysis/chapter4/modules/m6_1_weighted_disruption.py
run_module src/analysis/chapter4/modules/m6_2_stratified_impact.py
run_module src/analysis/chapter4/modules/m6_3_candidate_guardrails.py
run_module src/analysis/chapter4/modules/m6_4_effective_disruption.py
run_module src/analysis/chapter4/modules/m6_5_prime_exposure_profiles.py
run_module src/analysis/chapter4/modules/m6_6_mechanism_decomposition.py

# ---------------------------------------------------------------------------
# M7: Cross-view stability
# ---------------------------------------------------------------------------
run_module src/analysis/chapter4/modules/m7_stability.py
run_module src/analysis/chapter4/modules/m7_1_rank_stability_extended.py

# ---------------------------------------------------------------------------
# M8: Semiconductor-upstream disruption
# ---------------------------------------------------------------------------
run_module src/analysis/chapter4/modules/m8_semi_disruption.py

# ---------------------------------------------------------------------------
# M9: Semantic lens pipeline
# ---------------------------------------------------------------------------
M9_ARGS=()
if [[ "$SKIP_M9_2" == true ]]; then
    M9_ARGS+=("--skip-m9-2")
    echo ""
    echo "NOTE: Skipping M9.2 (FactSet enrichment). Using pre-existing M0_6 artifacts."
fi
run_module src/analysis/chapter4/modules/m9_0_semantic_lens_pipeline.py "${M9_ARGS[@]}"
run_module src/analysis/chapter4/modules/m9_action_matrix.py

# ---------------------------------------------------------------------------
# Reporting exports
# ---------------------------------------------------------------------------
if [[ "$SKIP_REPORTING" == true ]]; then
    echo ""
    echo "Skipping reporting exports (--skip-reporting)."
else
    echo ""
    echo "============================================================"
    echo "  Reporting Exports"
    echo "============================================================"
    run_module src/analysis/chapter4/reporting/export_figure_4_1_robustness_observed.py
    run_module src/analysis/chapter4/reporting/export_figure_4_2_mechanism_observed.py
    run_module src/analysis/chapter4/reporting/export_figure_4_3_numbers.py
    run_module src/analysis/chapter4/reporting/export_figure_4_4_case_study_network.py
    run_module src/analysis/chapter4/reporting/export_figure_4_5_schematic_corridor_vs_upstream.py
    run_module src/analysis/chapter4/reporting/export_table_4_1_interdiction_observed.py
    run_module src/analysis/chapter4/reporting/export_table_4_2_strict_top25.py
    run_module src/analysis/chapter4/reporting/export_appendix_a1_cross_view_stability.py
fi

echo ""
echo "============================================================"
echo "  Pipeline complete: $(ts)"
echo "============================================================"
