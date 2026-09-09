#!/usr/bin/env bash
# End-to-end scoring for the Chapter 3 candidate pool.
# Uses trained Chapter 2 models, writes per-model scores under artifacts/ch3/prediction_upstream/scores by default.
set -euo pipefail

# -------- Configuration -------- #
export PYTHONPATH="$(pwd)"

# Allow overrides so we can run ablations without editing this script.
# Example:
#   CANDIDATES=artifacts/ch3/prediction_upstream/candidates_with_label_sorted.parquet \
#   OUT_ROOT=artifacts/ch3/prediction_upstream/scores \
#   STAGE_OUT=artifacts/ch3/prediction_upstream/ch3_scores \
#   bash src/ch3/prediction/run_scoring_pipeline.sh
CANDIDATES="${CANDIDATES:-artifacts/ch3/prediction_upstream/candidates_with_label_sorted.parquet}"
OUT_ROOT="${OUT_ROOT:-artifacts/ch3/prediction_upstream/scores}"
STAGE_OUT="${STAGE_OUT:-artifacts/ch3/prediction_upstream/ch3_scores}"
ART_ROOT="${ART_ROOT:-artifacts}"

# Core_v1 paths (train snapshot)
ADJ="data/processed/core/releases/core_v1/adjacency/train_adj_T0.npz"
FEAT="data/processed/core/releases/core_v1/features/node_features_T0.parquet"
STRUCT="data/processed/core/releases/core_v1/features/node_structural_v1.parquet"
SPLITS="data/processed/core/releases/core_v1/splits"

# Single seed for consistency
SEED="42"
SPLIT_FLAG="--splits test"  # write a single production split

# Model tags (as stored under artifacts/<model>/<tag>)
TAG_GRAPH_SAGE="graphsage_production_v2"
TAG_NODE2VEC="node2vec_production_v1"
TAG_N2V_TEMPORAL="n2v_temporal_production_v2"
TAG_TWOTOWER="twotower_production_v2"
TAG_TGNN="tgnn_production_v2"
# Heuristics has no tagged subdir; we fall back to artifacts/heuristics
TAG_HEUR="heuristics"

mkdir -p "$OUT_ROOT"

log() { printf '%s %s\n' "$(date -Is)" "$*"; }

run_model() {
  local model=$1
  shift
  log "=== Starting $model ==="
  python -u src/ensemble/export_scores.py "$@" || { log "*** $model failed"; exit 1; }
  log "=== Finished $model ==="
}

# -------- GraphSAGE (CUDA) -------- #
run_model "graphsage" \
  --model graphsage \
  --tag "$TAG_GRAPH_SAGE" \
  --artifacts-root "$ART_ROOT" \
  --output-root "$OUT_ROOT" \
  --candidates-val "$CANDIDATES" \
  --candidates-test "$CANDIDATES" \
  --adj "$ADJ" \
  --features "$FEAT" \
  --seeds "$SEED" \
  --device cuda \
  --batch-size 2000000 \
  $SPLIT_FLAG \
  --overwrite

# -------- Node2Vec (CPU) -------- #
run_model "node2vec" \
  --model node2vec \
  --tag "$TAG_NODE2VEC" \
  --artifacts-root "$ART_ROOT" \
  --output-root "$OUT_ROOT" \
  --candidates-val "$CANDIDATES" \
  --candidates-test "$CANDIDATES" \
  --splits-root "$SPLITS" \
  --seeds "$SEED" \
  --device cpu \
  --batch-size 2000000 \
  $SPLIT_FLAG \
  --overwrite

# -------- Heuristics (CPU) -------- #
run_model "heuristics" \
  --model heuristics \
  --tag "$TAG_HEUR" \
  --artifacts-root "$ART_ROOT" \
  --output-root "$OUT_ROOT" \
  --candidates-val "$CANDIDATES" \
  --candidates-test "$CANDIDATES" \
  --adj "$ADJ" \
  --batch-size 2000000 \
  $SPLIT_FLAG \
  --overwrite

# -------- TwoTower (CUDA) -------- #
run_model "twotower" \
  --model twotower \
  --tag "$TAG_TWOTOWER" \
  --artifacts-root "$ART_ROOT" \
  --output-root "$OUT_ROOT" \
  --candidates-val "$CANDIDATES" \
  --candidates-test "$CANDIDATES" \
  --adj "$ADJ" \
  --features "$FEAT" \
  --struct-feats "$STRUCT" \
  --seeds "$SEED" \
  --device cuda \
  --batch-size 1000000 \
  $SPLIT_FLAG \
  --overwrite

# -------- N2V Temporal (EvolveGCN, CPU) -------- #
run_model "n2v_temporal" \
  --model n2v_temporal \
  --tag "$TAG_N2V_TEMPORAL" \
  --artifacts-root "$ART_ROOT" \
  --output-root "$OUT_ROOT" \
  --candidates-val "$CANDIDATES" \
  --candidates-test "$CANDIDATES" \
  --adj "$ADJ" \
  --features "$FEAT" \
  --splits-root "$SPLITS" \
  --seeds "$SEED" \
  --device cuda \
  --batch-size 500000 \
  $SPLIT_FLAG \
  --overwrite

# -------- TGNN (CUDA) -------- #
run_model "tgnn" \
  --model tgnn \
  --tag "$TAG_TGNN" \
  --artifacts-root "$ART_ROOT" \
  --output-root "$OUT_ROOT" \
  --candidates-val "$CANDIDATES" \
  --candidates-test "$CANDIDATES" \
  --adj "$ADJ" \
  --features "$FEAT" \
  --splits-root "$SPLITS" \
  --seeds "$SEED" \
  --device cuda \
  --batch-size 500000 \
  $SPLIT_FLAG \
  --overwrite

log "All models completed."

# -------- Stage Chapter 3 copies with clean names -------- #
mkdir -p "$STAGE_OUT"

copy_score() {
  local model=$1
  local tag=$2
  local seed_dir=$3
  local src="${OUT_ROOT}/${model}/${tag}/${seed_dir}/scores_test.parquet"
  local dst="${STAGE_OUT}/${model}_scores.parquet"
  if [ -f "$src" ]; then
    cp "$src" "$dst"
    log "Staged $model -> $dst"
  else
    log "WARNING: missing $src (skipping stage for $model)"
  fi
}

copy_score "graphsage" "$TAG_GRAPH_SAGE" "seed_${SEED}"
copy_score "node2vec" "$TAG_NODE2VEC" "seed_${SEED}"
copy_score "heuristics" "$TAG_HEUR" "seed_base"
copy_score "twotower" "$TAG_TWOTOWER" "seed_${SEED}"
copy_score "n2v_temporal" "$TAG_N2V_TEMPORAL" "seed_${SEED}"
copy_score "tgnn" "$TAG_TGNN" "seed_${SEED}"

log "Staging complete: $STAGE_OUT"
