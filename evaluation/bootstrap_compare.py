"""
Paired bootstrap comparison of survival C-index across methods.

Loads prediction npz files (produced by save_predictions() in survival_eval.py)
from a directory, aligns patients by eid, draws B paired bootstrap resamples,
and reports per-method point estimates + 95% CIs along with pairwise contrasts.

Usage
-----
python evaluation/bootstrap_compare.py \\
    --preds-dir evaluation/embedding_results/predictions \\
    --methods s1_baseline s2_delphi s3_qwen_concat s3_qwen_sum \\
              s4_qwen_concat s4_qwen_sum s5_qwen \\
    --n-bootstrap 1000 \\
    --output evaluation/bootstrap_results.json

The npz naming convention: {method_name}_test_preds.npz
Each npz must contain arrays: eids, risk_scores, durations, events.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# C-index helper (mirrors metrics.py line 100)
# ---------------------------------------------------------------------------

try:
    from lifelines.utils import concordance_index as _lifelines_ci

    def _c_index(durations: np.ndarray, risk_scores: np.ndarray, events: np.ndarray) -> float:
        """Harrell C-index: higher risk_score = higher predicted hazard."""
        return float(_lifelines_ci(durations, -risk_scores, events))

except ImportError:
    raise ImportError("lifelines is required (pip install lifelines)")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_method_preds(preds_dir: Path, method_name: str, split: str = "test") -> Dict[str, np.ndarray]:
    """
    Load prediction arrays from an npz file.

    Returns a dict with keys: eids, risk_scores, durations, events.
    Raises FileNotFoundError if the file is missing, or KeyError if required
    arrays are absent (e.g. old npz written before durations/events were added).
    """
    path = preds_dir / f"{method_name}_{split}_preds.npz"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    data = np.load(path)
    missing = [k for k in ("eids", "risk_scores", "durations", "events") if k not in data]
    if missing:
        raise KeyError(
            f"npz for method '{method_name}' is missing keys: {missing}. "
            "Re-run predictions with the updated save_predictions() that saves durations/events."
        )
    return {
        "eids": data["eids"].astype(np.int64),
        "risk_scores": data["risk_scores"].astype(np.float64),
        "durations": data["durations"].astype(np.float64),
        "events": data["events"].astype(np.int32),
    }


def align_predictions(
    method_data: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """
    Find the intersection of eids across all methods and reorder each
    method's arrays to the common sorted eid order.

    Returns
    -------
    common_eids : sorted array of eids in the intersection
    aligned     : method -> dict of aligned arrays (eids, risk_scores, durations, events)
    """
    # Intersect
    eid_sets = [set(d["eids"].tolist()) for d in method_data.values()]
    common_set: set = eid_sets[0]
    for s in eid_sets[1:]:
        common_set &= s
    common_eids = np.array(sorted(common_set), dtype=np.int64)
    n = len(common_eids)
    if n == 0:
        raise ValueError("No patients in common across all provided methods.")
    print(f"[align_predictions] Intersection: {n} patients across {len(method_data)} methods.")

    aligned: Dict[str, Dict[str, np.ndarray]] = {}
    for method, data in method_data.items():
        # Build eid -> position mapping
        eid_to_idx = {int(e): i for i, e in enumerate(data["eids"])}
        idx = np.array([eid_to_idx[int(e)] for e in common_eids], dtype=np.intp)
        aligned[method] = {
            "eids": common_eids,
            "risk_scores": data["risk_scores"][idx],
            "durations": data["durations"][idx],
            "events": data["events"][idx],
        }
    return common_eids, aligned


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_c_indices(
    aligned: Dict[str, Dict[str, np.ndarray]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Draw paired bootstrap resamples and compute C-index per method per resample.

    Returns dict method -> array of shape (n_bootstrap,).
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(aligned.values()))["eids"])
    boot_results: Dict[str, List[float]] = {m: [] for m in aligned}

    # Use durations/events from the first method (identical across aligned methods)
    ref = next(iter(aligned.values()))
    durations = ref["durations"]
    events = ref["events"]

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        d_b = durations[idx]
        e_b = events[idx]
        # Skip degenerate resamples (no events)
        if e_b.sum() == 0:
            for m in aligned:
                boot_results[m].append(np.nan)
            continue
        for method, data in aligned.items():
            rs_b = data["risk_scores"][idx]
            try:
                ci = _c_index(d_b, rs_b, e_b)
            except Exception:
                ci = np.nan
            boot_results[method].append(ci)

        if (b + 1) % 200 == 0:
            print(f"[bootstrap] {b + 1}/{n_bootstrap} resamples done", flush=True)

    return {m: np.array(v) for m, v in boot_results.items()}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_point_estimates(
    aligned: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, float]:
    """Compute C-index on the full intersection sample for each method."""
    ref = next(iter(aligned.values()))
    durations = ref["durations"]
    events = ref["events"]
    return {
        method: _c_index(durations, data["risk_scores"], events)
        for method, data in aligned.items()
    }


def summarise_bootstrap(
    boot_arrays: Dict[str, np.ndarray],
    point_estimates: Dict[str, float],
    contrasts: List[Tuple[str, str]],
) -> Dict:
    """
    Build the full results dict from bootstrap arrays.

    contrasts: list of (method_a, method_b) pairs — "improvement" = a - b > 0.
    """
    per_method: Dict[str, Dict] = {}
    for method, arr in boot_arrays.items():
        valid = arr[~np.isnan(arr)]
        per_method[method] = {
            "point_estimate": point_estimates[method],
            "ci_lower": float(np.percentile(valid, 2.5)) if len(valid) else None,
            "ci_upper": float(np.percentile(valid, 97.5)) if len(valid) else None,
            "n_bootstrap": len(arr),
            "n_valid": int(len(valid)),
        }

    pairwise: Dict[str, Dict] = {}
    for method_a, method_b in contrasts:
        if method_a not in boot_arrays or method_b not in boot_arrays:
            print(f"[summarise] Warning: skipping contrast {method_a}:{method_b} — method not found.")
            continue
        diffs = boot_arrays[method_a] - boot_arrays[method_b]
        valid_diffs = diffs[~np.isnan(diffs)]
        point_diff = point_estimates[method_a] - point_estimates[method_b]
        # One-sided p-value: fraction of resamples where diff <= 0 (null: a not better than b)
        p_one_sided = float((valid_diffs <= 0).mean()) if len(valid_diffs) else None
        key = f"{method_a}_vs_{method_b}"
        pairwise[key] = {
            "method_a": method_a,
            "method_b": method_b,
            "point_diff": float(point_diff),
            "ci_lower": float(np.percentile(valid_diffs, 2.5)) if len(valid_diffs) else None,
            "ci_upper": float(np.percentile(valid_diffs, 97.5)) if len(valid_diffs) else None,
            "p_one_sided": p_one_sided,
            "n_valid": int(len(valid_diffs)),
        }

    return {"per_method": per_method, "contrasts": pairwise}


def print_table(results: Dict, methods: List[str]) -> None:
    """Print a human-readable summary table to stdout."""
    pm = results["per_method"]
    print("\n" + "=" * 70)
    print(f"{'Method':<25} {'C-index':>8}  {'95% CI':^20}  {'n_valid':>7}")
    print("-" * 70)
    for m in methods:
        if m not in pm:
            continue
        r = pm[m]
        lo = f"{r['ci_lower']:.4f}" if r["ci_lower"] is not None else "  N/A "
        hi = f"{r['ci_upper']:.4f}" if r["ci_upper"] is not None else "  N/A "
        print(f"{m:<25} {r['point_estimate']:>8.4f}  [{lo}, {hi}]  {r['n_valid']:>7}")

    ct = results["contrasts"]
    if ct:
        print("\n" + "=" * 70)
        print(f"{'Contrast (A - B)':<30} {'Δ C-index':>10}  {'95% CI':^20}  {'p (one-sided)':>13}")
        print("-" * 70)
        for key, r in ct.items():
            lo = f"{r['ci_lower']:.4f}" if r["ci_lower"] is not None else "  N/A "
            hi = f"{r['ci_upper']:.4f}" if r["ci_upper"] is not None else "  N/A "
            p_str = f"{r['p_one_sided']:.4f}" if r["p_one_sided"] is not None else "  N/A "
            label = f"{r['method_a']} - {r['method_b']}"
            print(f"{label:<30} {r['point_diff']:>10.4f}  [{lo}, {hi}]  {p_str:>13}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_CONTRASTS = [
    ("s4_qwen_concat", "s3_qwen_concat"),
    ("s4_qwen_concat", "s1_baseline"),
    ("s5_qwen", "s1_baseline"),
]

DEFAULT_METHODS = [
    "s1_baseline",
    "s2_delphi",
    "s3_qwen_concat",
    "s3_qwen_sum",
    "s4_qwen_concat",
    "s4_qwen_sum",
    "s5_qwen",
]


def parse_contrast(s: str) -> Tuple[str, str]:
    """Parse a 'method_a:method_b' string into a (method_a, method_b) tuple."""
    parts = s.split(":", 1)
    if len(parts) != 2 or not all(p.strip() for p in parts):
        raise argparse.ArgumentTypeError(
            f"Invalid contrast '{s}': expected 'method_a:method_b'"
        )
    return parts[0].strip(), parts[1].strip()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap C-index comparison across survival-prediction methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preds-dir",
        type=Path,
        default=Path("evaluation/embedding_results/predictions"),
        help="Directory containing {method}_{split}_preds.npz files.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Method names to load (space-separated).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Data split to evaluate (must match the split suffix in npz filenames).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap.",
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        type=parse_contrast,
        default=DEFAULT_CONTRASTS,
        metavar="A:B",
        help=(
            "Pairwise contrasts to report as 'method_a:method_b' pairs "
            "(a - b is the improvement direction). "
            "Default: s4_qwen_concat:s3_qwen_concat s4_qwen_concat:s1_baseline s5_qwen:s1_baseline"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/bootstrap_results.json"),
        help="Path to write the JSON output.",
    )
    args = parser.parse_args(argv)

    preds_dir = args.preds_dir
    print(f"[bootstrap_compare] Loading predictions from: {preds_dir}")

    # Load each method's predictions
    method_data: Dict[str, Dict[str, np.ndarray]] = {}
    failed: List[str] = []
    for method in args.methods:
        try:
            method_data[method] = load_method_preds(preds_dir, method, split=args.split)
            n = len(method_data[method]["eids"])
            print(f"  [{method}] loaded {n} patients")
        except (FileNotFoundError, KeyError) as exc:
            print(f"  [{method}] SKIPPED — {exc}", file=sys.stderr)
            failed.append(method)

    loaded_methods = [m for m in args.methods if m not in failed]
    if len(loaded_methods) < 2:
        print(
            "ERROR: Need at least 2 methods with valid prediction files.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Align to intersection
    common_eids, aligned = align_predictions({m: method_data[m] for m in loaded_methods})

    n_common = len(next(iter(aligned.values()))["eids"])
    if n_common < 50:
        print(f"WARNING: Only {n_common} patients in eid intersection. Bootstrap CIs unreliable.", file=sys.stderr)

    # Point estimates on the full intersection
    print(f"\n[bootstrap_compare] Computing point estimates on {len(common_eids)} patients...")
    point_estimates = compute_point_estimates(aligned)
    for method, ci in point_estimates.items():
        print(f"  {method}: C-index = {ci:.4f}")

    # Bootstrap
    print(f"\n[bootstrap_compare] Running {args.n_bootstrap} bootstrap resamples (seed={args.seed})...")
    boot_arrays = bootstrap_c_indices(aligned, n_bootstrap=args.n_bootstrap, seed=args.seed)

    # Summarise
    # Filter contrasts to loaded methods only
    valid_contrasts = [
        (a, b) for a, b in args.contrasts
        if a in aligned and b in aligned
    ]
    results = summarise_bootstrap(boot_arrays, point_estimates, valid_contrasts)

    # Print table
    print_table(results, loaded_methods)

    # Write JSON
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bootstrap_compare] Results written to: {out_path}")


if __name__ == "__main__":
    main()
