"""
Benchmarking survival model (baseline-only CoxPH) using the shared evaluation backend.

Usage:
    python evaluation/evaluate_benchmarking.py \
        --baseline-mode all \
        --output-dir evaluation/benchmarking_results
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import data_access, survival_eval  # noqa: E402

SURVIVAL_CSV = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"
COHORT_SPLIT = PROJECT_ROOT / "evaluation" / "cohort_split.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "benchmarking_results"


def evaluate_benchmarking(args):
    baseline_cfg = data_access.BaselineConfig(
        mode=args.baseline_mode,
        columns=args.baseline_cols,
    )
    matrices, baseline_cols, coverage = survival_eval.load_survival_matrices(
        baseline_cfg,
        survival_csv=Path(args.survival_csv),
        cohort_json=Path(args.cohort_json),
    )

    print("Baseline coverage:")
    for split, cov in coverage.items():
        print(f"  {split}: {cov}")

    cox_cfg = survival_eval.CoxConfig(
        penalizer=args.model_penalizer,
        l1_ratio=args.model_l1_ratio,
        fallback_penalizer=args.fallback_penalizer,
        fallback_l1_ratio=args.fallback_l1_ratio,
    )

    preds_dir = Path(args.preds_dir) if args.save_preds else None
    results = survival_eval.run_cox_evaluation(
        matrices,
        cox_cfg,
        horizons=None,
        save_preds_dir=preds_dir,
        method_name="benchmarking_cox",
    )

    results["metadata"] = {
        "baseline_mode": args.baseline_mode,
        "baseline_columns": baseline_cols,
        "baseline_coverage": coverage,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "benchmarking_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmarking on shared cohort.")
    parser.add_argument("--baseline-mode", type=str, default="all",
                        choices=["all", "none", "custom"],
                        help="Which baseline features to include.")
    parser.add_argument("--baseline-cols", type=str, nargs="*", default=None,
                        help="Explicit baseline columns when --baseline-mode=custom.")
    parser.add_argument("--survival-csv", type=str, default=str(SURVIVAL_CSV),
                        help="Path to autoprognosis_survival_dataset.csv.")
    parser.add_argument("--cohort-json", type=str, default=str(COHORT_SPLIT),
                        help="Path to cohort_split.json.")
    parser.add_argument("--model-penalizer", type=float, default=0.1,
                        help="Primary CoxPH penalizer.")
    parser.add_argument("--model-l1-ratio", type=float, default=0.5,
                        help="Primary CoxPH L1 ratio.")
    parser.add_argument("--fallback-penalizer", type=float, default=1.0,
                        help="Fallback penalizer if the main fit fails.")
    parser.add_argument("--fallback-l1-ratio", type=float, default=0.9,
                        help="Fallback L1 ratio if the main fit fails.")
    parser.add_argument("--save-preds", action="store_true",
                        help="If set, save per-split predictions to --preds-dir.")
    parser.add_argument("--preds-dir", type=str,
                        default=str(DEFAULT_OUTPUT / "predictions"))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Directory to store benchmarking_results.json.")
    args = parser.parse_args()

    evaluate_benchmarking(args)


if __name__ == "__main__":
    main()
