"""
Unified survival evaluation for embedding-based methods (text + trajectory).

Embeddings keyed by eid are merged with baseline features (configurable) and
evaluated with a shared CoxPH backend + harmonized metrics.

Usage:
    python evaluation/evaluate_embedding_survival.py \
        --embedding-dir data/preprocessed/embeddings_text \
        --tag patient \
        --method-name text_embedding
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import data_access, survival_eval  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "embedding_results"
SURVIVAL_CSV = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"
COHORT_SPLIT = PROJECT_ROOT / "evaluation" / "cohort_split.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embeddings(embedding_dir: Path, tag: str) -> dict:
    """Load {eid: np.ndarray} from a .npz file."""
    npz_path = embedding_dir / f"{tag}_embeddings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    return {int(k): np.array(data[k]) for k in data.files}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_embeddings(args):
    baseline_cfg = data_access.BaselineConfig(
        mode=args.baseline_mode,
        columns=args.baseline_cols,
    )
    matrices, baseline_cols, baseline_cov = survival_eval.load_survival_matrices(
        baseline_cfg,
        survival_csv=Path(args.survival_csv),
        cohort_json=Path(args.cohort_json),
    )

    print("Baseline coverage:")
    for split, cov in baseline_cov.items():
        print(f"  {split}: {cov}")

    print(f"Loading embeddings from {args.embedding_dir} (tag={args.tag})...")
    embeddings = load_embeddings(Path(args.embedding_dir), args.tag)
    print(f"  Loaded {len(embeddings)} embeddings")

    combined_matrices = {}
    emb_cov = {}
    for split, matrix in matrices.items():
        combined, cov = survival_eval.merge_with_embeddings(matrix, embeddings)
        combined_matrices[split] = combined
        emb_cov[split] = cov

    print("Embedding coverage after merge:")
    for split, cov in emb_cov.items():
        print(f"  {split}: {cov}")

    cox_cfg = survival_eval.CoxConfig(
        penalizer=args.penalizer,
        l1_ratio=args.l1_ratio,
        fallback_penalizer=args.fallback_penalizer,
        fallback_l1_ratio=args.fallback_l1_ratio,
    )

    preds_dir = Path(args.preds_dir) if args.save_preds else None
    results = survival_eval.run_cox_evaluation(
        combined_matrices,
        cox_cfg,
        horizons=None,
        save_preds_dir=preds_dir,
        method_name=args.method_name,
    )

    results["metadata"] = {
        "embedding_dir": str(args.embedding_dir),
        "embedding_tag": args.tag,
        "baseline_mode": args.baseline_mode,
        "baseline_columns": baseline_cols,
        "baseline_coverage": baseline_cov,
        "embedding_coverage": emb_cov,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{args.method_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate survival model on embedding + baseline features."
    )
    parser.add_argument("--embedding-dir", type=str, required=True,
                        help="Directory containing {tag}_embeddings.npz")
    parser.add_argument("--tag", type=str, default="patient",
                        help="Embedding file prefix (e.g. 'patient', 'trajectory')")
    parser.add_argument("--method-name", type=str, default="embedding",
                        help="Name for this method in results")
    parser.add_argument("--baseline-mode", type=str, default="all",
                        choices=["all", "none", "custom"],
                        help="Which baseline features to include.")
    parser.add_argument("--baseline-cols", type=str, nargs="*", default=None,
                        help="Explicit baseline columns when --baseline-mode=custom.")
    parser.add_argument("--survival-csv", type=str, default=str(SURVIVAL_CSV),
                        help="Path to autoprognosis_survival_dataset.csv.")
    parser.add_argument("--cohort-json", type=str, default=str(COHORT_SPLIT),
                        help="Path to cohort_split.json.")
    parser.add_argument("--penalizer", type=float, default=0.1)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--fallback-penalizer", type=float, default=1.0)
    parser.add_argument("--fallback-l1-ratio", type=float, default=0.9)
    parser.add_argument("--save-preds", action="store_true",
                        help="If set, save per-split predictions to --preds-dir.")
    parser.add_argument("--preds-dir", type=str,
                        default=str(DEFAULT_OUTPUT / "predictions"))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Directory to store results JSON.")
    args = parser.parse_args()

    evaluate_embeddings(args)


if __name__ == "__main__":
    main()
