#!/usr/bin/env bash
# =============================================================================
# Chapter 3 Pipeline Orchestrator (DoD Supply Network)
# =============================================================================
# Executes Ch3 in frozen (default) or full mode.
#
# Usage:
#   bash src/ch3/run_ch3_pipeline.sh                # frozen mode (no DB needed)
#   bash src/ch3/run_ch3_pipeline.sh --frozen       # explicit frozen mode
#   bash src/ch3/run_ch3_pipeline.sh --full          # full mode (needs FactSet DB)
#   bash src/ch3/run_ch3_pipeline.sh --skip-tables   # skip table generation
#
# Run from the repository root.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODE="frozen"
SKIP_TABLES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --frozen)
            MODE="frozen"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --skip-tables)
            SKIP_TABLES=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--frozen|--full] [--skip-tables]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

run_step() {
    local label="$1"
    shift
    echo ""
    echo ">>> [$(ts)] $label — starting..."
    "$@"
    echo "<<< [$(ts)] $label — done"
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Chapter 3 Pipeline"
echo "  Mode:    $MODE"
echo "  Started: $(ts)"
echo "============================================================"

if [[ "$MODE" == "frozen" ]]; then
    REQUIRED_FILES=(
        "artifacts/ch3/matching/prime_matches_thr90.parquet"
        "artifacts/ch3/usaspending/dod_primes_fy2022-2025_asof.parquet"
        "artifacts/ch3/prediction_upstream/top5_pred.parquet"
        "artifacts/ch3/prediction_upstream/top10_pred.parquet"
        "artifacts/ch3/prediction_upstream/top50_pred.parquet"
        "artifacts/ch3/reference/semis_catalog_strict_entities.parquet"
        "artifacts/ch3/reference/semis_catalog_strict_l4.parquet"
        "data/processed/core/releases/core_v1/mapping/entity_map.parquet"
        "data/processed/core/releases/core_v1/splits/all_edges.parquet"
        "data/processed/core/releases/core_v1/splits/train_edges.parquet"
        "data/processed/core/releases/core_v1/splits/val_edges.parquet"
        "data/processed/core/releases/core_v1/splits/test_edges.parquet"
        "data/processed/core/releases/core_v1/features/node_features_T0.parquet"
        "data/processed/core/releases/core_v1/features/node_structural_v1.parquet"
        "data/processed/shipping/processed/shipping_edges_parent.parquet"
    )
    MISSING=0
    for f in "${REQUIRED_FILES[@]}"; do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: Required artifact not found: $f"
            MISSING=$((MISSING + 1))
        fi
    done
    if [[ $MISSING -gt 0 ]]; then
        echo ""
        echo "ERROR: $MISSING required artifact(s) missing."
        echo "       Sync frozen artifacts from Google Drive before running --frozen mode."
        echo "       See SETUP.md for rclone instructions."
        exit 1
    fi
    echo "Pre-flight: all $((${#REQUIRED_FILES[@]})) frozen artifacts found."

elif [[ "$MODE" == "full" ]]; then
    if [[ ! -f ".env" ]]; then
        echo "ERROR: .env file not found. Full mode requires FactSet DB credentials."
        echo "       See SETUP.md for configuration instructions."
        exit 1
    fi
    echo "Pre-flight: .env found."
fi

# =============================================================================
# FROZEN MODE — Groups 7-8 only (network assembly + tables)
# =============================================================================
if [[ "$MODE" == "frozen" ]]; then

    echo ""
    echo "============================================================"
    echo "  Group 7: Network assembly (frozen artifacts)"
    echo "============================================================"

    run_step "package_scr_layers" \
        python src/ch3/network/package_scr_layers.py

    run_step "package_dod_contract_layer" \
        python src/ch3/network/package_dod_contract_layer.py

    run_step "build_disclosed_baseline" \
        python src/ch3/network/build_disclosed_baseline.py \
            --prime-matches artifacts/ch3/matching/prime_matches_thr90.parquet \
            --primes-agg artifacts/ch3/usaspending/dod_primes_fy2022-2025_asof.parquet \
            --depth 99 \
            --out-dir artifacts/ch3/network_upstream

    run_step "package_dod_network_d99" \
        python src/ch3/network/package_dod_network_d99.py

    run_step "build_shipping_observed_layer" \
        python src/ch3/network/build_shipping_observed_layer.py

    run_step "package_dod_network_with_shipping" \
        python src/ch3/network/package_dod_network_with_shipping.py

    run_step "build_semis_catalog_strict (offline)" \
        python src/ch3/prediction/build_semis_catalog_strict.py \
            --catalog-entities artifacts/ch3/reference/semis_catalog_strict_entities.parquet \
            --catalog-l4 artifacts/ch3/reference/semis_catalog_strict_l4.parquet

    run_step "build_semiconductor_subgraph" \
        python src/ch3/network/build_semiconductor_subgraph.py \
            --tag top5_d99_shipping_strict

# =============================================================================
# FULL MODE — Groups 1-8 (all steps, requires DB)
# =============================================================================
elif [[ "$MODE" == "full" ]]; then

    echo ""
    echo "============================================================"
    echo "  Group 1: USAspending seed tables"
    echo "============================================================"

    run_step "01_build_dod_transactions" \
        python src/ch3/usaspending/01_build_dod_transactions.py \
            --fy-start 2022 --fy-end 2025 --as-of-date 2025-06-09

    run_step "02_aggregate_primes" \
        python src/ch3/usaspending/02_aggregate_primes.py \
            --transactions-asof artifacts/ch3/usaspending/dod_transactions_fy2022-2025_asof.parquet \
            --transactions-post artifacts/ch3/usaspending/dod_transactions_fy2022-2025_post.parquet

    run_step "03_report_usaspending_summary" \
        python src/ch3/usaspending/03_report_usaspending_summary.py \
            --transactions-asof artifacts/ch3/usaspending/dod_transactions_fy2022-2025_asof.parquet \
            --transactions-post artifacts/ch3/usaspending/dod_transactions_fy2022-2025_post.parquet \
            --primes-asof artifacts/ch3/usaspending/dod_primes_fy2022-2025_asof.parquet \
            --primes-post artifacts/ch3/usaspending/dod_primes_fy2022-2025_post.parquet

    echo ""
    echo "============================================================"
    echo "  Group 2: Tier-1 matching"
    echo "============================================================"

    run_step "match_primes" \
        python src/ch3/matching/match_primes.py \
            --primes artifacts/ch3/usaspending/dod_primes_fy2022-2025_asof.parquet \
            --transactions artifacts/ch3/usaspending/dod_transactions_fy2022-2025_asof.parquet \
            --fetch-sym-from-db --fuzzy-threshold 0.90 \
            --out-matches artifacts/ch3/matching/prime_matches_thr90.parquet

    echo ""
    echo "============================================================"
    echo "  Group 3: Semiconductor tags"
    echo "============================================================"

    run_step "tag_semis" \
        python src/ch3/prediction/tag_semis.py

    echo ""
    echo "============================================================"
    echo "  Group 4: Candidate pool + flip"
    echo "============================================================"

    run_step "build_candidate_pool" \
        python src/ch3/prediction/build_candidate_pool.py \
            --prime-matches artifacts/ch3/matching/prime_matches_thr90.parquet \
            --semis artifacts/ch3/reference/semis_flags.parquet

    run_step "flip_candidates" \
        python src/ch3/prediction/flip_candidates.py --write-summary

    echo ""
    echo "============================================================"
    echo "  Group 5: Score candidates"
    echo "============================================================"

    run_step "run_scoring_pipeline" \
        bash src/ch3/prediction/run_scoring_pipeline.sh

    echo ""
    echo "============================================================"
    echo "  Group 6: Meta model + Top-K selection"
    echo "============================================================"

    run_step "build_meta_zero_slices" \
        python src/ch3/prediction/build_meta_zero_slices.py \
            --base artifacts/ch3/prediction_upstream/candidates_with_label_sorted.parquet \
            --scores-root artifacts/ch3/prediction_upstream/ch3_scores \
            --out-root artifacts/ch3/prediction_upstream --overwrite

    run_step "apply_meta" \
        python src/ensemble/apply_meta.py \
            --model-config artifacts/ensemble/meta_ranker/meta_ranker_v4/meta_model.json \
            --meta-input artifacts/ch3/prediction_upstream/meta_inputs_zero_slices.parquet \
            --output artifacts/ch3/prediction_upstream/meta_scores_zero_slices.parquet

    run_step "select_topk" \
        python src/ch3/prediction/select_topk.py \
            --meta-scores artifacts/ch3/prediction_upstream/meta_scores_zero_slices.parquet \
            --out-root artifacts/ch3/prediction_upstream \
            --group-by dst_id --overwrite

    echo ""
    echo "============================================================"
    echo "  Group 7: Network assembly"
    echo "============================================================"

    run_step "package_scr_layers" \
        python src/ch3/network/package_scr_layers.py

    run_step "package_dod_contract_layer" \
        python src/ch3/network/package_dod_contract_layer.py

    run_step "build_disclosed_baseline" \
        python src/ch3/network/build_disclosed_baseline.py \
            --prime-matches artifacts/ch3/matching/prime_matches_thr90.parquet \
            --primes-agg artifacts/ch3/usaspending/dod_primes_fy2022-2025_asof.parquet \
            --depth 99 \
            --out-dir artifacts/ch3/network_upstream

    run_step "package_dod_network_d99" \
        python src/ch3/network/package_dod_network_d99.py

    run_step "build_shipping_observed_layer" \
        python src/ch3/network/build_shipping_observed_layer.py

    run_step "package_dod_network_with_shipping" \
        python src/ch3/network/package_dod_network_with_shipping.py

    run_step "build_semis_catalog_strict" \
        python src/ch3/prediction/build_semis_catalog_strict.py

    run_step "build_semiconductor_subgraph" \
        python src/ch3/network/build_semiconductor_subgraph.py \
            --tag top5_d99_shipping_strict
fi

# ---------------------------------------------------------------------------
# Group 8: Chapter tables
# ---------------------------------------------------------------------------
if [[ "$SKIP_TABLES" == true ]]; then
    echo ""
    echo "Skipping table generation (--skip-tables)."
else
    echo ""
    echo "============================================================"
    echo "  Group 8: Chapter 3 tables"
    echo "============================================================"

    run_step "build_ch3_tables" \
        python src/ch3/analysis/build_ch3_tables.py

    run_step "build_semis_network_summary_table" \
        python src/ch3/analysis/build_semis_network_summary_table.py
fi

echo ""
echo "============================================================"
echo "  Pipeline complete: $(ts)"
echo "============================================================"
