#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ensemble_predict.sh [options]

Options:
  --tag TAG                 Output tag under results/ensemble/meta_ranker (default: meta_ranker_v3)
  --model PATH              Path to meta_model.json (default: artifacts/ensemble/meta_ranker_v3/meta_model.json)
  --meta-dataset DIR        Directory with meta_inputs_{val,test}.parquet (default: results/ensemble/meta_dataset_v3)
  --out-root DIR            Base directory for ensemble outputs (default: results/ensemble/meta_ranker)
  --predictions-root DIR    Base directory for ranked prediction exports (default: predictions)
  --precision FLOAT         Precision target for threshold selection (default: 0.90)
  --bin-width FLOAT         Histogram bin width passed to threshold_eval.py (default: 1e-4)
  --python CMD              Python executable to invoke (default: python)
  --skip-predictions        Skip writing predictions/<tag>/ranked_edges.parquet
  -h, --help                Show this help message
EOF
}

TAG="meta_ranker_v3"
MODEL="artifacts/ensemble/meta_ranker_v3/meta_model.json"
META_DATASET="results/ensemble/meta_dataset_v3"
OUT_ROOT="results/ensemble/meta_ranker"
PRED_ROOT="predictions"
PRECISION_TARGET="0.90"
BIN_WIDTH="1e-4"
PYTHON_BIN="python"
WRITE_PREDICTIONS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="$2"; shift 2 ;;
    --model)
      MODEL="$2"; shift 2 ;;
    --meta-dataset)
      META_DATASET="$2"; shift 2 ;;
    --out-root)
      OUT_ROOT="$2"; shift 2 ;;
    --predictions-root)
      PRED_ROOT="$2"; shift 2 ;;
    --precision)
      PRECISION_TARGET="$2"; shift 2 ;;
    --bin-width)
      BIN_WIDTH="$2"; shift 2 ;;
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --skip-predictions)
      WRITE_PREDICTIONS=0; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1 ;;
  esac
done

META_VAL="${META_DATASET}/meta_inputs_val.parquet"
META_TEST="${META_DATASET}/meta_inputs_test.parquet"
OUT_DIR="${OUT_ROOT}/${TAG}"
SCORES_VAL="${OUT_DIR}/scores_val.parquet"
SCORES_TEST="${OUT_DIR}/scores_test.parquet"
VAL_REPORT_JSON="${OUT_DIR}/threshold_report_val.json"
PRED_DIR="${PRED_ROOT}/${TAG}"
PRED_PARQUET="${PRED_DIR}/ranked_edges.parquet"
MODEL_NAME="ensemble"
BASELINES=(prob_graphsage prob_node2vec prob_heuristics prob_twotower prob_tgnn prob_n2v_temporal)

mkdir -p "${OUT_DIR}"

if [[ ! -f "${META_VAL}" ]]; then
  echo "Missing meta validation parquet: ${META_VAL}" >&2
  exit 1
fi
if [[ ! -f "${META_TEST}" ]]; then
  echo "Missing meta test parquet: ${META_TEST}" >&2
  exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "Missing meta model: ${MODEL}" >&2
  exit 1
fi

echo "[1/6] Applying meta model to validation set"
"${PYTHON_BIN}" -m src.ensemble.apply_meta \
  --model-config "${MODEL}" \
  --meta-input "${META_VAL}" \
  --output "${SCORES_VAL}" \
  --prob-column meta_prob \
  --logit-column meta_logit \
  --extra-columns label

echo "[2/6] Applying meta model to test set"
"${PYTHON_BIN}" -m src.ensemble.apply_meta \
  --model-config "${MODEL}" \
  --meta-input "${META_TEST}" \
  --output "${SCORES_TEST}" \
  --prob-column meta_prob \
  --logit-column meta_logit \
  --extra-columns label

BASELINE_ARGS=()
if [[ ${#BASELINES[@]} -gt 0 ]]; then
  BASELINE_ARGS+=("--baseline-prob-cols")
  for col in "${BASELINES[@]}"; do
    BASELINE_ARGS+=("${col}")
  done
fi

echo "[3/6] Sweeping validation thresholds"
"${PYTHON_BIN}" -m src.ensemble.threshold_eval \
  --scores "${SCORES_VAL}" \
  --meta-input "${META_VAL}" \
  --out-dir "${OUT_ROOT}" \
  --tag "${TAG}" \
  --split val \
  --precision-target "${PRECISION_TARGET}" \
  --hist-bin-width "${BIN_WIDTH}" \
  --model-name "${MODEL_NAME}" \
  "${BASELINE_ARGS[@]}"

if [[ ! -f "${VAL_REPORT_JSON}" ]]; then
  echo "Validation report not found: ${VAL_REPORT_JSON}" >&2
  exit 1
fi

VAL_THRESHOLD=$("${PYTHON_BIN}" -c "import json, sys
path, model = sys.argv[1:3]
with open(path, 'r', encoding='utf-8') as fh:
    payload = json.load(fh)
for row in payload.get('models', []):
    if row.get('model') == model:
        print(row.get('threshold'))
        break
else:
    raise SystemExit('Threshold not found in report')
" "${VAL_REPORT_JSON}" "${MODEL_NAME}")

echo "[4/6] Evaluating test split at τ=${VAL_THRESHOLD}"
"${PYTHON_BIN}" -m src.ensemble.threshold_eval \
  --scores "${SCORES_TEST}" \
  --meta-input "${META_TEST}" \
  --out-dir "${OUT_ROOT}" \
  --tag "${TAG}" \
  --split test \
  --precision-target "${PRECISION_TARGET}" \
  --hist-bin-width "${BIN_WIDTH}" \
  --model-name "${MODEL_NAME}" \
  --threshold "${VAL_THRESHOLD}" \
  "${BASELINE_ARGS[@]}"

echo "[5/6] Summarising selection policies"
"${PYTHON_BIN}" -m src.ensemble.eval_selection \
  --scores-dir "${OUT_DIR}" \
  --meta-dir "${META_DATASET}" \
  --out-dir "${OUT_ROOT}" \
  --tag "${TAG}" \
  --prediction-root "${PRED_ROOT}" \
  --export-threshold anchor_tau1=1.0 \
  --export-topk core_topk=K=10,floor=0.995,cap=50 \
  --export-topk discovery_topk=K=10,floor=0.98,cap=None

if [[ ${WRITE_PREDICTIONS} -eq 1 ]]; then
  echo "[6/6] Writing ranked predictions to ${PRED_PARQUET}"
  "${PYTHON_BIN}" - "${SCORES_TEST}" "${META_TEST}" "${VAL_THRESHOLD}" "${PRED_PARQUET}" <<'PY'
import sys
from pathlib import Path
import duckdb
import pandas as pd


def escape(path: Path) -> str:
    return path.as_posix().replace("'", "''")


scores_path = Path(sys.argv[1])
meta_path = Path(sys.argv[2])
threshold = float(sys.argv[3])
output_path = Path(sys.argv[4])
output_path.parent.mkdir(parents=True, exist_ok=True)

scores_sql = escape(scores_path)
meta_sql = escape(meta_path)

conn = duckdb.connect(database=":memory:")
query = f"""
    WITH base AS (
        SELECT s.src_id,
               s.dst_id,
               s.meta_prob,
               m.slice_CC,
               m.slice_CC3,
               m.slice_WW,
               m.slice_WW3,
               m.slice_deg_q1,
               m.slice_gt2hop
        FROM read_parquet('{scores_sql}') s
        JOIN read_parquet('{meta_sql}') m USING (src_id, dst_id)
        WHERE s.meta_prob >= {threshold}
    ),
    ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY src_id ORDER BY meta_prob DESC, dst_id) AS rank
        FROM base
    )
    SELECT * FROM ranked
"""
df = conn.execute(query).fetch_df()
conn.close()
df.to_parquet(output_path, index=False, compression='zstd')
PY
else
  echo "[6/6] Skipping ranked prediction export"
fi

echo "Ensemble prediction pipeline complete."
