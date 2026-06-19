#!/usr/bin/env bash
# =============================================================================
# run_qwen_all_eval.sh — CoxPH evaluation for all 5 Qwen3-8B cells.
#
# Run LOCALLY after transferring Qwen3-8B embeddings from server:
#   data/preprocessed/embeddings_qwen_text/    (S3 — disease prose)
#   data/preprocessed/embeddings_qwen_traj/    (S4 — decoupled trajectory)
#   data/preprocessed/embeddings_qwen_summary/ (S5 — unified prose)
#
# Cells covered:
#   s3_qwen_concat   S3 · Qwen3-8B · concat
#   s3_qwen_sum      S3 · Qwen3-8B · summation fusion
#   s4_qwen_concat   S4 · Qwen3-8B · concat
#   s4_qwen_sum      S4 · Qwen3-8B · summation fusion
#   s5_qwen          S5 · Qwen3-8B  (standalone, no raw features)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/preprocessed"
EVAL_DIR="${REPO_ROOT}/evaluation/embedding_results"

echo "===== Qwen3-8B CoxPH evaluations (all 5 cells) ====="
echo

echo "--- S3·Qwen·concat ---"
python "${REPO_ROOT}/evaluation/evaluate_embedding_survival.py" \
    --embedding-dir "${DATA_DIR}/embeddings_qwen_text" \
    --tag           patient \
    --method-name   s3_qwen_concat \
    --baseline-mode all \
    --fusion        concat \
    --pca-components 64 \
    --save-preds \
    --output-dir    "${EVAL_DIR}"
echo

echo "--- S3·Qwen·sum ---"
python "${REPO_ROOT}/evaluation/evaluate_embedding_survival.py" \
    --embedding-dir "${DATA_DIR}/embeddings_qwen_text" \
    --tag           patient \
    --method-name   s3_qwen_sum \
    --baseline-mode all \
    --fusion        sum \
    --save-preds \
    --output-dir    "${EVAL_DIR}"
echo

echo "--- S4·Qwen·concat ---"
python "${REPO_ROOT}/evaluation/evaluate_embedding_survival.py" \
    --embedding-dir "${DATA_DIR}/embeddings_qwen_traj" \
    --tag           trajectory \
    --method-name   s4_qwen_concat \
    --baseline-mode all \
    --fusion        concat \
    --pca-components 64 \
    --save-preds \
    --output-dir    "${EVAL_DIR}"
echo

echo "--- S4·Qwen·sum ---"
python "${REPO_ROOT}/evaluation/evaluate_embedding_survival.py" \
    --embedding-dir "${DATA_DIR}/embeddings_qwen_traj" \
    --tag           trajectory \
    --method-name   s4_qwen_sum \
    --baseline-mode all \
    --fusion        sum \
    --save-preds \
    --output-dir    "${EVAL_DIR}"
echo

echo "--- S5·Qwen (standalone, no raw features) ---"
python "${REPO_ROOT}/evaluation/evaluate_embedding_survival.py" \
    --embedding-dir "${DATA_DIR}/embeddings_qwen_summary" \
    --tag           patient \
    --method-name   s5_qwen \
    --baseline-mode none \
    --fusion        concat \
    --pca-components 64 \
    --save-preds \
    --output-dir    "${EVAL_DIR}"
echo

echo "===== All Qwen3-8B evaluations complete ====="
