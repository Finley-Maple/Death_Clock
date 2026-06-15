#!/usr/bin/env bash
# =============================================================================
# run_qwen_concat_eval.sh — CoxPH evaluation for the missing Qwen3-8B cells.
#
# Run LOCALLY (Mac) using the existing Qwen3-8B embedding dirs.
# Default embedding dir names assumed below:
#   data/preprocessed/embeddings_qwen_text/    (Setting 3 — disease prose)
#   data/preprocessed/embeddings_qwen_traj/    (Setting 4 — decoupled trajectory)
#   data/preprocessed/embeddings_qwen_summary/ (Setting 5 — unified prose)
#
# NOTE: Check actual Qwen embedding directory names with
#   ls data/preprocessed/ | grep -i qwen
# before running — the directory names may differ from the defaults above.
#
# Cells covered (the three missing Qwen cells):
#   s3_qwen_concat  — Setting 3 · Qwen3 · concat fusion
#   s4_qwen_concat  — Setting 4 · Qwen3 · concat fusion
#   s5_qwen         — Setting 5 · Qwen3 · standalone (no raw features)
#
# The sum-fusion Qwen cells (s3_qwen_sum, s4_qwen_sum) already have results
# in evaluation/embedding_results/ and are NOT re-run here.
#
# Usage:
#   conda activate delphi
#   bash scripts/run_qwen_concat_eval.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/preprocessed"
EVAL_DIR="${REPO_ROOT}/evaluation/embedding_results"

echo "===== Qwen3-8B concat / standalone CoxPH evaluations ====="
echo
echo "NOTE: Verify embedding dirs with:"
echo "  ls ${DATA_DIR} | grep -i qwen"
echo

# ---------------------------------------------------------------------------
# Setting 3 · Qwen3 · concat  (disease prose, concat fusion)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Setting 4 · Qwen3 · concat  (decoupled trajectory, concat fusion)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Setting 5 · Qwen3  (unified prose, standalone — no raw features)
# ---------------------------------------------------------------------------
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

echo "===== All Qwen3 missing-cell CoxPH evaluations complete ====="
echo "Results saved to: ${EVAL_DIR}/"
echo "  s3_qwen_concat_results.json"
echo "  s4_qwen_concat_results.json"
echo "  s5_qwen_results.json"
echo
echo "Next: update evaluation/plot_comparison.py METHOD_CONFIG and re-run unified_evaluation.py"
