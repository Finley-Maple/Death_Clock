"""
Shared helpers for loading the survival dataset, cohort split, and baseline
feature matrices for the aligned death-prediction evaluations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SURVIVAL_CSV = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"
COHORT_JSON = PROJECT_ROOT / "evaluation" / "cohort_split.json"

TARGET_COLUMNS = ["eid", "event_flag", "duration_days"]


@dataclass
class BaselineConfig:
    """Configuration for selecting baseline (non-embedding) features."""

    mode: str = "all"  # {"all", "none", "custom"}
    columns: Optional[Sequence[str]] = None

    def normalized_mode(self) -> str:
        mode = (self.mode or "all").lower()
        if mode not in {"all", "none", "custom"}:
            raise ValueError(f"Unsupported baseline mode '{self.mode}'.")
        if mode == "custom" and not self.columns:
            raise ValueError("baseline mode 'custom' requires explicit columns.")
        return mode


@lru_cache(maxsize=4)
def _load_survival_cached(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    df["eid"] = df["eid"].astype(int)
    return df


def load_survival_dataframe(path: Path = SURVIVAL_CSV) -> pd.DataFrame:
    """Load the survival dataset (cached) and return a copy."""
    return _load_survival_cached(str(path)).copy()


def load_cohort(path: Path = COHORT_JSON) -> Dict:
    """Load cohort_split.json."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cohort split not found at {path}. Run evaluation/cohort_split.py first."
        )
    with open(path) as f:
        return json.load(f)


def assert_dataset_matches(cohort: Dict, survival_csv: Path) -> None:
    """Validate that the survival CSV matches the dataset metadata stored in the cohort split."""
    meta = cohort.get("dataset_meta")
    if not meta:
        # Backwards compatibility: no metadata recorded yet.
        return
    expected_sha = meta.get("sha256")
    if not expected_sha:
        return
    actual_sha = _sha256_file(survival_csv)
    if actual_sha != expected_sha:
        raise ValueError(
            "Survival dataset has changed since cohort split was generated.\n"
            f"  cohort file: {survival_csv}\n"
            f"  expected sha256: {expected_sha}\n"
            f"  actual sha256:   {actual_sha}\n"
            "Regenerate evaluation/cohort_split.py to realign the cohort."
        )


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_baseline_columns(df: pd.DataFrame, config: BaselineConfig) -> List[str]:
    """Return the list of baseline columns based on the config."""
    mode = config.normalized_mode()
    if mode == "none":
        return []
    if mode == "custom":
        missing = [c for c in config.columns or [] if c not in df.columns]
        if missing:
            raise ValueError(f"Baseline columns not found: {missing}")
        return list(config.columns or [])

    # Default: all numeric non-target columns.
    exclude = set(TARGET_COLUMNS)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def align_split_dataframe(df: pd.DataFrame, eids: Sequence[int]) -> Tuple[pd.DataFrame, List[int]]:
    """
    Reindex the survival dataframe to match the provided eid order.

    Returns:
        aligned_df: dataframe with rows ordered by eids (missing rows dropped)
        missing_eids: list of cohort eids missing from the dataframe
    """
    if not len(df):
        return pd.DataFrame(columns=df.columns), list(eids)

    aligned = df.set_index("eid").reindex(eids)
    missing_mask = aligned["event_flag"].isna() & aligned["duration_days"].isna()
    missing = [int(eid) for eid in aligned.index[missing_mask]]
    aligned = aligned[~missing_mask].reset_index()
    aligned.rename(columns={"index": "eid"}, inplace=True)
    aligned["eid"] = aligned["eid"].astype(int)
    return aligned, missing


def coverage_report(expected: Sequence[int], available: Sequence[int]) -> Dict:
    """Return coverage stats for diagnostic logging."""
    exp_set = list(expected)
    avail_set = list(available)
    expected_n = len(exp_set)
    avail_n = len(avail_set)
    missing = sorted(set(exp_set) - set(avail_set))
    coverage = avail_n / expected_n if expected_n else 0.0
    return {
        "expected": expected_n,
        "available": avail_n,
        "coverage": coverage,
        "missing_count": len(missing),
        "missing_examples": missing[:5],
    }
