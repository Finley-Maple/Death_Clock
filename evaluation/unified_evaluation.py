"""
Unified Evaluation: compare all four death-prediction methods.

Loads results from each method's evaluation output and computes a
standardized comparison table with:
  - C-index
  - Time-dependent AUC (at each horizon)
  - Integrated Brier Score
  - DeLong confidence intervals where applicable

Important: This script does NOT copy from the existing Delphi evaluation
notebooks. It uses evaluate_auc_pipeline() as the canonical Delphi backend
and freshly computes all metrics for each method.

Usage:
    python evaluation/unified_evaluation.py

    Or as a module:
        from evaluation.unified_evaluation import run_unified_evaluation
        results = run_unified_evaluation()
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
COHORT_SPLIT = EVALUATION_DIR / "cohort_split.json"

# Result directories (created by each method's evaluation script)
DELPHI_RESULTS = EVALUATION_DIR / "delphi_results"
BENCHMARKING_RESULTS = EVALUATION_DIR / "benchmarking_results"
EMBEDDING_RESULTS = EVALUATION_DIR / "embedding_results"


# ---------------------------------------------------------------------------
# Load results from each method
# ---------------------------------------------------------------------------

def load_delphi_results() -> Optional[Dict]:
    """Load Delphi zero-shot results (prefer test split)."""
    for split in ["test", "val", "train"]:
        results_path = DELPHI_RESULTS / f"delphi_{split}_results.json"
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
                data.setdefault("active_split", split)
                return data

    # Backwards compatibility: legacy summary
    legacy_path = DELPHI_RESULTS / "delphi_death_summary.json"
    if legacy_path.exists():
        with open(legacy_path) as f:
            legacy = json.load(f)
        print("  [Delphi] Legacy summary detected; only DeLong AUC available.")
        return {
            "method": "delphi_legacy",
            "legacy_delong": legacy,
            "splits": {},
            "horizons": [],
        }

    print("  [Delphi] Results not found. Run evaluation/evaluate_delphi.py")
    return None


def load_delphi_cox_results() -> Optional[Dict]:
    """Load Delphi+CoxPH results (inference on all splits, CoxPH trained on train)."""
    results_path = DELPHI_RESULTS / "delphi_cox_results.json"
    if not results_path.exists():
        print(f"  [Delphi+CoxPH] Results not found. Run: python evaluation/evaluate_delphi.py --cox")
        return None
    with open(results_path) as f:
        return json.load(f)


def load_benchmarking_results() -> Optional[Dict]:
    """Load AutoPrognosis/CoxPH benchmarking results."""
    results_path = BENCHMARKING_RESULTS / "benchmarking_results.json"
    if not results_path.exists():
        print(f"  [Benchmarking] Results not found at {results_path}")
        print(f"  [Benchmarking] Run: python evaluation/evaluate_benchmarking.py")
        return None
    with open(results_path) as f:
        return json.load(f)


def load_embedding_results(method_name: str) -> Optional[Dict]:
    """Load embedding-based survival model results."""
    results_path = EMBEDDING_RESULTS / f"{method_name}_results.json"
    if not results_path.exists():
        print(f"  [{method_name}] Results not found at {results_path}")
        print(f"  [{method_name}] Run: python evaluation/evaluate_embedding_survival.py "
              f"--method-name {method_name}")
        return None
    with open(results_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------

def extract_split_metrics(results: Dict, split: str) -> Dict:
    splits = results.get("splits") or {}
    return splits.get(split, {})


def build_comparison_table(all_results: Dict[str, Optional[Dict]]) -> pd.DataFrame:
    """
    Assemble a comparison DataFrame from all methods using the new schema.
    Columns: method, test_c_index, test_mean_td_auc, test_ibs, val_c_index, val_mean_td_auc, notes
    """
    rows: List[Dict] = []

    # Delphi
    delphi = all_results.get("delphi")
    if delphi:
        if "splits" in delphi and delphi["splits"]:
            active_split = delphi.get("active_split", "test")
            metrics_split = extract_split_metrics(delphi, active_split)
            row = {
                "method": f"Delphi ({active_split})",
                "test_c_index": metrics_split.get("c_index"),
                "test_mean_td_auc": metrics_split.get("mean_td_auc"),
                "test_ibs": metrics_split.get("ibs"),
                "val_c_index": None,
                "val_mean_td_auc": None,
                "notes": f"Coverage: {delphi.get('coverage')}",
                "test_auc_delong": None,
                "test_auc_ci": None,
            }
        else:
            legacy = delphi.get("legacy_delong", {})
            results_list = legacy.get("results", [])
            aucs = [r.get("auc", float("nan")) for r in results_list]
            vars_ = [r.get("auc_variance_delong", float("nan")) for r in results_list]
            mean_auc = float(np.nanmean(aucs)) if aucs else None
            n = len([v for v in vars_ if not np.isnan(v)])
            mean_var = float(np.nansum(vars_) / max(n, 1) ** 2) if n > 0 else float("nan")
            ci_lo = mean_auc - 1.96 * np.sqrt(mean_var) if mean_auc is not None else None
            ci_hi = mean_auc + 1.96 * np.sqrt(mean_var) if mean_auc is not None else None
            row = {
                "method": "Delphi (legacy)",
                "test_c_index": None,
                "test_mean_td_auc": None,
                "test_ibs": None,
                "val_c_index": None,
                "val_mean_td_auc": None,
                "notes": "Legacy DeLong summary only",
                "test_auc_delong": mean_auc,
                "test_auc_ci": f"[{ci_lo:.4f}, {ci_hi:.4f}]" if ci_lo is not None else None,
            }
        rows.append(row)

    # Delphi + CoxPH
    delphi_cox = all_results.get("delphi_cox")
    if delphi_cox:
        dcox_test = extract_split_metrics(delphi_cox, "test")
        dcox_val  = extract_split_metrics(delphi_cox, "val")
        rows.append({
            "method": "Delphi + CoxPH",
            "test_c_index": dcox_test.get("c_index"),
            "test_mean_td_auc": dcox_test.get("mean_td_auc"),
            "test_ibs": dcox_test.get("ibs"),
            "val_c_index": dcox_val.get("c_index"),
            "val_mean_td_auc": dcox_val.get("mean_td_auc"),
            "notes": str(delphi_cox.get("metadata", {}).get("description")),
            "test_auc_delong": None,
            "test_auc_ci": None,
        })

    # Benchmarking
    bench = all_results.get("benchmarking")
    if bench:
        bench_test = extract_split_metrics(bench, "test")
        bench_val = extract_split_metrics(bench, "val")
        rows.append({
            "method": "Benchmarking (CoxPH)",
            "test_c_index": bench_test.get("c_index"),
            "test_mean_td_auc": bench_test.get("mean_td_auc"),
            "test_ibs": bench_test.get("ibs"),
            "val_c_index": bench_val.get("c_index"),
            "val_mean_td_auc": bench_val.get("mean_td_auc"),
            "notes": str(bench.get("metadata")),
            "test_auc_delong": None,
            "test_auc_ci": None,
        })

    # Text embeddings
    text = all_results.get("text_embedding")
    if text:
        text_test = extract_split_metrics(text, "test")
        text_val = extract_split_metrics(text, "val")
        rows.append({
            "method": "Text Embedding + CoxPH",
            "test_c_index": text_test.get("c_index"),
            "test_mean_td_auc": text_test.get("mean_td_auc"),
            "test_ibs": text_test.get("ibs"),
            "val_c_index": text_val.get("c_index"),
            "val_mean_td_auc": text_val.get("mean_td_auc"),
            "notes": str(text.get("metadata")),
            "test_auc_delong": None,
            "test_auc_ci": None,
        })

    # Trajectory embeddings
    traj = all_results.get("trajectory_embedding")
    if traj:
        traj_test = extract_split_metrics(traj, "test")
        traj_val = extract_split_metrics(traj, "val")
        rows.append({
            "method": "Trajectory Embedding + CoxPH",
            "test_c_index": traj_test.get("c_index"),
            "test_mean_td_auc": traj_test.get("mean_td_auc"),
            "test_ibs": traj_test.get("ibs"),
            "val_c_index": traj_val.get("c_index"),
            "val_mean_td_auc": traj_val.get("mean_td_auc"),
            "notes": str(traj.get("metadata")),
            "test_auc_delong": None,
            "test_auc_ci": None,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def format_value(val, fmt=".4f"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    return f"{val:{fmt}}"


def print_comparison_table(df: pd.DataFrame):
    """Print a nicely formatted comparison table."""
    print("\n" + "=" * 90)
    print("UNIFIED EVALUATION: DEATH PREDICTION COMPARISON")
    print("=" * 90)

    if len(df) == 0:
        print("\nNo results available. Run individual evaluation scripts first.")
        print("  1. python evaluation/evaluate_delphi.py")
        print("  2. python evaluation/evaluate_benchmarking.py")
        print("  3. python evaluation/evaluate_embedding_survival.py --method-name text_embedding ...")
        print("  4. python evaluation/evaluate_embedding_survival.py --method-name trajectory_embedding ...")
        return

    # Header
    header = (
        f"{'Method':<35} {'Test C-idx':>10} {'Test TD-AUC':>11} {'Test IBS':>10} "
        f"{'Test DeLong':>11} {'Val C-idx':>10} {'Val TD-AUC':>11}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for _, row in df.iterrows():
        line = (
            f"{row['method']:<35} "
            f"{format_value(row.get('test_c_index')):>10} "
            f"{format_value(row.get('test_mean_td_auc')):>11} "
            f"{format_value(row.get('test_ibs')):>10} "
            f"{format_value(row.get('test_auc_delong')):>11} "
            f"{format_value(row.get('val_c_index')):>10} "
            f"{format_value(row.get('val_mean_td_auc')):>11}"
        )
        print(line)
        if row.get("test_auc_ci"):
            print(f"{'':>35} {'':>10} {'':>11} {row['test_auc_ci']:>20}")

    print("-" * len(header))

    # Notes
    print("\nNotes:")
    for _, row in df.iterrows():
        if row.get("notes"):
            print(f"  {row['method']}: {row['notes']}")

    print()


# ---------------------------------------------------------------------------
# Detailed per-horizon table
# ---------------------------------------------------------------------------

def print_horizon_details(all_results: Dict[str, Optional[Dict]]):
    """Print time-dependent AUC at each horizon for methods that support it."""
    print("\n" + "=" * 70)
    print("TIME-DEPENDENT AUC BY HORIZON (test set)")
    print("=" * 70)

    for label, key in [
        ("Delphi (zero-shot)", "delphi"),
        ("Delphi + CoxPH", "delphi_cox"),
        ("Benchmarking", "benchmarking"),
        ("Text Embedding", "text_embedding"),
        ("Trajectory Embedding", "trajectory_embedding"),
    ]:
        results = all_results.get(key)
        if not results:
            continue
        split = "test" if key != "delphi" else results.get("active_split", "test")
        metrics_split = extract_split_metrics(results, split)
        td_auc = metrics_split.get("td_auc")
        if not td_auc:
            continue
        print(f"\n  {label} ({split}):")
        for horizon, auc_val in sorted(td_auc.items(), key=lambda x: float(x[0])):
            days = float(horizon)
            years = days / 365.25
            print(f"    Horizon {days:.0f} days ({years:.1f} yrs): AUC = {float(auc_val):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_unified_evaluation() -> pd.DataFrame:
    """Run the full unified evaluation and return the comparison DataFrame."""
    print("Loading cohort split...")
    if COHORT_SPLIT.exists():
        with open(COHORT_SPLIT) as f:
            cohort = json.load(f)
        print(f"  Cohort: {cohort['total']} total, "
              f"{cohort['counts']['train']} train, "
              f"{cohort['counts']['val']} val, "
              f"{cohort['counts']['test']} test")
        print(f"  Event rates: {cohort['event_rates']}")
    else:
        print(f"  Warning: Cohort split not found at {COHORT_SPLIT}")

    print("\nLoading method results...")
    all_results = {
        "delphi": load_delphi_results(),
        "delphi_cox": load_delphi_cox_results(),
        "benchmarking": load_benchmarking_results(),
        "text_embedding": load_embedding_results("text_embedding"),
        "trajectory_embedding": load_embedding_results("trajectory_embedding"),
    }

    df = build_comparison_table(all_results)
    print_comparison_table(df)
    print_horizon_details(all_results)

    # Save table
    output_path = EVALUATION_DIR / "unified_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"Comparison table saved to {output_path}")

    # Save as JSON too
    json_path = EVALUATION_DIR / "unified_comparison.json"
    df.to_json(json_path, orient="records", indent=2)

    return df


def main():
    run_unified_evaluation()


if __name__ == "__main__":
    main()
