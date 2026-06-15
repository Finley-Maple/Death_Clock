#!/usr/bin/env bash
# =============================================================================
# run_bcb_cox_eval.sh — Run CoxPH evaluation for all BCB embedding cells.
#
# Run LOCALLY (Mac) after transferring the BCB .npz files back from the server.
# Assumes the three embedding dirs are already populated:
#   data/preprocessed/embeddings_bcb_text/
#   data/preprocessed/embeddings_bcb_traj/
#   data/preprocessed/embeddings_bcb_summary/
#
# Usage:
#   conda activate delphi
#   bash scripts/run_bcb_cox_eval.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/preprocessed"
EVAL_DIR="${REPO_ROOT}/evaluation/embedding_results"

echo "===== BCB CoxPH evaluations ====="
echo

# ---------------------------------------------------------------------------
# Setting 3 · BCB · concat  (disease prose, concat fusion)
# ---------------------------------------------------------------------------
echo "--- S3·BCB·concat ---"
python "${REPO_ROOT}/evaluation/evaluate_embedding_survival.py" \
    --embedding-dir "${DATA_DIR}/embeddings_bcb_text" \
    --tag           patient \
    --method-name   s3_bcb_concat \
    --baseline-mode all \
    --fusion        concat \
    --pca-components 64 \
    --save-preds \
    --output-dir    "${EVAL_DIR}"

echo

# ---------------------------------------------------------------------------
# Setting 3 · BCB · sum  (disease prose, summation fusion)
# ---------------------------------------------------------------------------
echo "--- S3·BCB·sum ---"
python "${REPO_ROOT}/evaluation/evaluate_embedding_survival.py" \
    --embedding-dir "${DATA_DIR}/embeddings_bcb_text" \
    --tag           patient \
    --method-name   s3_bcb_sum \
    --baseline-mode all \
    --fusion        sum \
    --save-preds \
    --output-dir    "${EVAL_DIR}"

echo

# ---------------------------------------------------------------------------
# Setting 4 · BCB · concat  (decoupled trajectory, concat fusion)
# ---------------------------------------------------------------------------
echo "--- S4·BCB·concat ---"
python "${REPO_ROOT}/evaluation/evaluate_embedding_survival.py" \
    --embedding-dir "${DATA_DIR}/embeddings_bcb_traj" \
    --tag           trajectory \
    --method-name   s4_bcb_concat \
    --baseline-mode all \
    --fusion        concat \
    --pca-components 64 \
    --save-preds \
    --output-dir    "${EVAL_DIR}"

echo

# ---------------------------------------------------------------------------
# Setting 4 · BCB · sum  (decoupled trajectory, summation fusion)
# ---------------------------------------------------------------------------
echo "--- S4·BCB·sum ---"
python "${REPO_ROOT}/evaluation/evaluate_embedding_survival.py" \
    --embedding-dir "${DATA_DIR}/embeddings_bcb_traj" \
    --tag           trajectory \
    --method-name   s4_bcb_sum \
    --baseline-mode all \
    --fusion        sum \
    --save-preds \
    --output-dir    "${EVAL_DIR}"

echo

# ---------------------------------------------------------------------------
# Setting 5 · BCB  (unified prose, standalone — no raw features)
# ---------------------------------------------------------------------------
echo "--- S5·BCB (standalone, no raw features) ---"
python "${REPO_ROOT}/evaluation/evaluate_embedding_survival.py" \
    --embedding-dir "${DATA_DIR}/embeddings_bcb_summary" \
    --tag           patient \
    --method-name   s5_bcb \
    --baseline-mode none \
    --fusion        concat \
    --pca-components 64 \
    --save-preds \
    --output-dir    "${EVAL_DIR}"

echo

echo "===== All BCB CoxPH evaluations complete ====="
echo "Results saved to: ${EVAL_DIR}/"
echo "  s3_bcb_concat_results.json"
echo "  s3_bcb_sum_results.json"
echo "  s4_bcb_concat_results.json"
echo "  s4_bcb_sum_results.json"
echo "  s5_bcb_results.json"
echo
echo "Next: update evaluation/plot_comparison.py METHOD_CONFIG and re-run unified_evaluation.py"
