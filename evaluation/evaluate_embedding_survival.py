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

    if args.fusion == "sum":
        if args.baseline_mode == "none":
            raise ValueError("--fusion sum requires --baseline-mode != none "
                             "(summation needs a raw feature block to add to).")
        print(f"Fusing by summation (proj_dim={args.proj_dim or 'baseline-width'})...")
        combined_matrices, emb_block_width, emb_cov = survival_eval.fuse_by_summation(
            matrices, embeddings, proj_dim=args.proj_dim,
        )
        # Already projected + standardized + summed -> plain matrix, no further PCA.
        pca_components, emb_block_width = None, None
    else:
        combined_matrices = {}
        emb_cov = {}
        for split, matrix in matrices.items():
            combined, cov = survival_eval.merge_with_embeddings(matrix, embeddings)
            combined_matrices[split] = combined
            emb_cov[split] = cov
        # emb_block_width: embedding dims come first in the combined [emb | baseline]
        # matrix. Pass this so PCA is applied only to the embedding block.
        sample_emb = next(iter(embeddings.values()))
        emb_block_width = int(np.asarray(sample_emb).ravel().shape[0]) if args.baseline_mode != "none" else None
        pca_components = args.pca_components

    print("Embedding coverage after fusion:")
    for split, cov in emb_cov.items():
        print(f"  {split}: {cov}")

    cox_cfg = survival_eval.CoxConfig(
        penalizer=args.penalizer,
        l1_ratio=args.l1_ratio,
        fallback_penalizer=args.fallback_penalizer,
        fallback_l1_ratio=args.fallback_l1_ratio,
        pca_components=pca_components,
        emb_block_width=emb_block_width,
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
        "fusion": args.fusion,
        "proj_dim": args.proj_dim,
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
    parser.add_argument("--pca-components", type=int, default=64,
                        help="Reduce embeddings to this many PCA dims before CoxPH "
                             "(avoids hours-long fits with 1024+ dim embeddings). "
                             "Set to 0 to disable PCA. Ignored when --fusion=sum.")
    parser.add_argument("--fusion", type=str, default="concat",
                        choices=["concat", "sum"],
                        help="How to combine embeddings with raw baseline features. "
                             "'concat' (default): [emb_PCA | raw_baseline] -> CoxPH. "
                             "'sum': project emb to baseline width, standardize both, "
                             "element-wise add -> CoxPH. Requires --baseline-mode != none.")
    parser.add_argument("--proj-dim", type=int, default=None,
                        help="Latent dimension for summation fusion projection "
                             "(default: baseline feature width). Only used when --fusion=sum.")
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

    if args.pca_components == 0:
        args.pca_components = None
    evaluate_embeddings(args)


if __name__ == "__main__":
    main()
