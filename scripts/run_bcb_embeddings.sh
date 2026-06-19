#!/usr/bin/env bash
# =============================================================================
# run_bcb_embeddings.sh — Generate BioClinicalBERT embeddings for all settings
#                          that use the BCB encoder (Settings 3, 4, 5).
#
# Run on the RTX 3090 server (port 31711) inside the 'delphi' conda env.
# Expected runtime: ~15–30 min per setting (BCB is small, ~110M params).
#
# Usage:
#   conda activate delphi
#   bash scripts/run_bcb_embeddings.sh
#
# Output (under data/preprocessed/):
#   embeddings_bcb_text/patient_embeddings.npz   — Setting 3 (prose)
#   embeddings_bcb_traj/trajectory_embeddings.npz — Setting 4 (decoupled token)
#   embeddings_bcb_summary/patient_embeddings.npz — Setting 5 (unified prose)
#
# After running, rsync/scp the three output dirs back to the Mac for evaluation.
# See transfer notes: use chunk-split for files >10 MB (featurize.cn truncation).
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/preprocessed"

echo "===== BioClinicalBERT embedding runs ====="
echo "REPO_ROOT: ${REPO_ROOT}"
echo "DATA_DIR:  ${DATA_DIR}"
echo

# ---------------------------------------------------------------------------
# Setting 3 — Isolated temporal prose  (text_before60.csv)
# ---------------------------------------------------------------------------
echo "--- Setting 3: disease-history prose (BioClinicalBERT) ---"
python "${REPO_ROOT}/embedding/clinical_bert_embedding.py" \
    --input-csv  "${DATA_DIR}/text_before60.csv" \
    --output-dir "${DATA_DIR}/embeddings_bcb_text" \
    --tag        patient \
    --batch-size 64

echo "Setting 3 done."
echo

# ---------------------------------------------------------------------------
# Setting 4 — Decoupled time & event  (trajectory_before60.csv → token+age)
# Embeds the ICD token strings with BCB and keeps the sinusoidal age encoding.
# Output dim = 128 (age) + 768 (BCB token) = 896.
# ---------------------------------------------------------------------------
echo "--- Setting 4: decoupled trajectory (BCB token encoder) ---"
python "${REPO_ROOT}/embedding/trajectory_embedding.py" \
    --input-csv    "${DATA_DIR}/trajectory_before60.csv" \
    --output-dir   "${DATA_DIR}/embeddings_bcb_traj" \
    --tag          trajectory \
    --token-mode   bioclinbert \
    --age-dim      128 \
    --pooling      mean \
    --text-col     trajectory_text

echo "Setting 4 done."
echo

# ---------------------------------------------------------------------------
# Setting 5 — Unified summary prose  (biomarker_text_before60.csv)
# NOTE: BCB truncates at 512 tokens — the disease-history tail may be lost
# for patients with long histories. This is flagged in the manuscript.
# ---------------------------------------------------------------------------
echo "--- Setting 5: unified summary prose (BioClinicalBERT) ---"
python "${REPO_ROOT}/embedding/clinical_bert_embedding.py" \
    --input-csv  "${DATA_DIR}/biomarker_text_before60.csv" \
    --output-dir "${DATA_DIR}/embeddings_bcb_summary" \
    --tag        patient \
    --batch-size 64

echo "Setting 5 done."
echo

echo "===== All BCB embedding runs complete ====="
echo "Output dirs:"
echo "  ${DATA_DIR}/embeddings_bcb_text/"
echo "  ${DATA_DIR}/embeddings_bcb_traj/"
echo "  ${DATA_DIR}/embeddings_bcb_summary/"
echo
echo "Next steps:"
echo "  1. Transfer output dirs back to the Mac (split large .npz files first):"
echo "     python scripts/split_npz.py <file> --chunk-mb 8"
echo "  2. Run CoxPH evaluation for BCB cells locally:"
echo "     bash scripts/run_bcb_cox_eval.sh"
