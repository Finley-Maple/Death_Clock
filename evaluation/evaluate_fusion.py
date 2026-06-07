"""
Fusion evaluation: combine text + trajectory embeddings with structured baseline.

Method 5: [disease_text_PCA(128) | trajectory_PCA(128) | baseline(39+indicators)]
          → CoxPH

Design decisions:
- PCA is applied INDEPENDENTLY per embedding block (fit on train eids only).
- Baseline structured features use the same median-imputation path as other methods.
- emb_block_width = text_pca + traj_pca so split-PCA in run_cox_evaluation
  does NOT re-apply PCA (pca_components=None).
- Biomarker text embedding intentionally excluded (redundant with structured baseline).

Usage:
    python evaluation/evaluate_fusion.py \\
        --text-embedding-dir data/preprocessed/embeddings_text \\
        --traj-embedding-dir data/preprocessed/embeddings_traj \\
        --text-pca-components 128 \\
        --traj-pca-components 128 \\
        --method-name fusion \\
        --output-dir evaluation/fusion_results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import data_access, survival_eval  # noqa: E402

SURVIVAL_CSV = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"
COHORT_SPLIT = PROJECT_ROOT / "evaluation" / "cohort_split.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "fusion_results"


def load_embeddings(embedding_dir: Path, tag: str) -> Dict[int, np.ndarray]:
    """Load {eid: ndarray} from {tag}_embeddings.npz."""
    npz_path = embedding_dir / f"{tag}_embeddings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    return {int(k): np.array(data[k]) for k in data.files}


def pca_embeddings_per_block(
    embedding_dicts: Sequence[Dict[int, np.ndarray]],
    splits: Dict[str, List[int]],
    n_components: int,
) -> Dict[str, np.ndarray]:
    """
    Fit PCA independently on each embedding dict using train eids, then
    transform all splits. Concatenate results across dicts.

    Args:
        embedding_dicts: list of {eid: ndarray} dicts, one per embedding source
        splits:          {"train": [...eids], "val": [...], "test": [...]}
        n_components:    PCA components per embedding block

    Returns:
        {"train": ndarray(n_train, n_components*len(dicts)), ...}
    """
    from sklearn.decomposition import PCA

    train_eids = splits["train"]
    all_split_arrays: Dict[str, List[np.ndarray]] = {k: [] for k in splits}

    for emb_dict in embedding_dicts:
        train_vecs = np.vstack([
            emb_dict[eid].ravel() for eid in train_eids if eid in emb_dict
        ])
        actual_n = min(n_components, train_vecs.shape[1], train_vecs.shape[0])
        pca = PCA(n_components=actual_n, random_state=42)
        pca.fit(train_vecs)
        print(f"  PCA block: {train_vecs.shape[1]} → {actual_n} dims "
              f"(var explained: {pca.explained_variance_ratio_.sum():.3f})")

        for split_name, eid_list in splits.items():
            vecs = np.vstack([
                emb_dict[eid].ravel() if eid in emb_dict
                else np.zeros(train_vecs.shape[1], dtype=np.float32)
                for eid in eid_list
            ])
            projected = pca.transform(vecs).astype(np.float32)
            all_split_arrays[split_name].append(projected)

    return {
        split_name: np.concatenate(arrays, axis=1)
        for split_name, arrays in all_split_arrays.items()
    }


def evaluate_fusion(args) -> Dict:
    baseline_cfg = data_access.BaselineConfig(mode="all")
    matrices, baseline_cols, baseline_cov = survival_eval.load_survival_matrices(
        baseline_cfg,
        survival_csv=Path(args.survival_csv),
        cohort_json=Path(args.cohort_json),
    )

    print(f"Loading text embeddings from {args.text_embedding_dir} (tag={args.text_tag})...")
    text_embs = load_embeddings(Path(args.text_embedding_dir), args.text_tag)
    print(f"Loading trajectory embeddings from {args.traj_embedding_dir} (tag={args.traj_tag})...")
    traj_embs = load_embeddings(Path(args.traj_embedding_dir), args.traj_tag)

    cohort = data_access.load_cohort(Path(args.cohort_json))
    split_eids = {
        "train": [e for e in cohort["train_eids"] if e in text_embs and e in traj_embs],
        "val":   [e for e in cohort["val_eids"]   if e in text_embs and e in traj_embs],
        "test":  [e for e in cohort["test_eids"]  if e in text_embs and e in traj_embs],
    }
    for split, eids in split_eids.items():
        print(f"  {split}: {len(eids)} eids with both embeddings")

    print(f"\nApplying PCA: text → {args.text_pca}, trajectory → {args.traj_pca} dims")
    if args.text_pca == args.traj_pca:
        pca_arrays = pca_embeddings_per_block(
            [text_embs, traj_embs], split_eids, n_components=args.text_pca
        )
    else:
        text_arrays = pca_embeddings_per_block([text_embs], split_eids, args.text_pca)
        traj_arrays = pca_embeddings_per_block([traj_embs], split_eids, args.traj_pca)
        pca_arrays = {s: np.concatenate([text_arrays[s], traj_arrays[s]], axis=1)
                      for s in split_eids}

    emb_block_width = args.text_pca + args.traj_pca

    combined_matrices = {}
    emb_cov = {}
    for split_name, matrix in matrices.items():
        pca_block = pca_arrays.get(split_name)
        if pca_block is None:
            continue
        pca_emb_dict = {eid: pca_block[i] for i, eid in enumerate(split_eids[split_name])}
        combined, cov = survival_eval.merge_with_embeddings(matrix, pca_emb_dict)
        combined_matrices[split_name] = combined
        emb_cov[split_name] = cov

    cox_cfg = survival_eval.CoxConfig(
        penalizer=args.penalizer,
        l1_ratio=args.l1_ratio,
        pca_components=None,
        emb_block_width=emb_block_width,
    )

    results = survival_eval.run_cox_evaluation(
        combined_matrices, cox_cfg,
        horizons=None, method_name=args.method_name,
    )
    results["metadata"] = {
        "text_embedding_dir": str(args.text_embedding_dir),
        "traj_embedding_dir": str(args.traj_embedding_dir),
        "text_pca": args.text_pca,
        "traj_pca": args.traj_pca,
        "emb_block_width": emb_block_width,
        "baseline_columns": baseline_cols,
        "baseline_coverage": baseline_cov,
        "embedding_coverage": emb_cov,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.method_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFusion results saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Late-fusion survival model.")
    parser.add_argument("--text-embedding-dir", type=str,
                        default=str(PROJECT_ROOT / "data/preprocessed/embeddings_text"))
    parser.add_argument("--text-tag", type=str, default="patient")
    parser.add_argument("--traj-embedding-dir", type=str,
                        default=str(PROJECT_ROOT / "data/preprocessed/embeddings_traj"))
    parser.add_argument("--traj-tag", type=str, default="trajectory")
    parser.add_argument("--text-pca-components", dest="text_pca", type=int, default=128)
    parser.add_argument("--traj-pca-components", dest="traj_pca", type=int, default=128)
    parser.add_argument("--method-name", type=str, default="fusion")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--survival-csv", type=str, default=str(SURVIVAL_CSV))
    parser.add_argument("--cohort-json", type=str, default=str(COHORT_SPLIT))
    parser.add_argument("--penalizer", type=float, default=0.1)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    args = parser.parse_args()
    evaluate_fusion(args)


if __name__ == "__main__":
    main()
