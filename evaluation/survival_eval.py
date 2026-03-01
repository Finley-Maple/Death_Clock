"""
Reusable survival evaluation utilities (feature loading, CoxPH training,
metrics, and prediction export) for the death-prediction pipelines.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np
import pandas as pd

from . import data_access
from . import metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SURVIVAL_CSV = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"
COHORT_JSON = PROJECT_ROOT / "evaluation" / "cohort_split.json"


@dataclass
class FeatureMatrix:
    """Container for aligned features, targets, and eid order."""

    X: np.ndarray
    durations: np.ndarray
    events: np.ndarray
    eids: List[int]

    def subset(self, indices: Sequence[int]) -> "FeatureMatrix":
        if not indices:
            return FeatureMatrix(
                np.zeros((0, self.X.shape[1]), dtype=np.float32),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.int32),
                [],
            )
        idx_arr = np.array(indices, dtype=int)
        return FeatureMatrix(
            self.X[idx_arr],
            self.durations[idx_arr],
            self.events[idx_arr],
            [self.eids[i] for i in idx_arr],
        )


@dataclass
class CoxConfig:
    penalizer: float = 0.1
    l1_ratio: float = 0.5
    fallback_penalizer: float = 1.0
    fallback_l1_ratio: float = 0.9


def load_survival_matrices(
    baseline_config: data_access.BaselineConfig,
    survival_csv: Path = SURVIVAL_CSV,
    cohort_json: Path = COHORT_JSON,
    splits: Sequence[str] = ("train", "val", "test"),
) -> Tuple[Dict[str, FeatureMatrix], List[str], Dict[str, Dict]]:
    """
    Build baseline-only feature matrices for the requested splits.

    Returns:
        matrices: dict split -> FeatureMatrix
        baseline_cols: list of baseline feature names used
        coverage: split -> coverage diagnostics
    """
    cohort = data_access.load_cohort(cohort_json)
    survival_df = data_access.load_survival_dataframe(survival_csv)
    data_access.assert_dataset_matches(cohort, survival_csv)
    baseline_cols = data_access.resolve_baseline_columns(survival_df, baseline_config)

    matrices: Dict[str, FeatureMatrix] = {}
    coverage: Dict[str, Dict] = {}

    for split in splits:
        eid_list = cohort.get(f"{split}_eids")
        if eid_list is None:
            continue
        split_df, missing = data_access.align_split_dataframe(survival_df, eid_list)
        coverage[split] = data_access.coverage_report(eid_list, split_df["eid"].tolist())

        if missing:
            print(f"[load_survival_matrices] Warning: {len(missing)} {split} eids missing from survival CSV.")

        if baseline_cols:
            feature_matrix = split_df[baseline_cols].to_numpy(dtype=np.float32)
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        else:
            feature_matrix = np.zeros((len(split_df), 0), dtype=np.float32)

        matrices[split] = FeatureMatrix(
            feature_matrix,
            split_df["duration_days"].to_numpy(dtype=np.float64),
            split_df["event_flag"].to_numpy(dtype=np.int32),
            split_df["eid"].astype(int).tolist(),
        )

    return matrices, baseline_cols, coverage


def merge_with_embeddings(
    baseline_matrix: FeatureMatrix,
    embeddings: Mapping[int, np.ndarray],
    drop_missing: bool = True,
) -> Tuple[FeatureMatrix, Dict]:
    """
    Append embedding vectors to the baseline matrix (embedding first, then baseline).

    Args:
        baseline_matrix: FeatureMatrix containing baseline features (can be zero-width).
        embeddings: mapping eid -> embedding vector.
        drop_missing: if True, rows without embeddings are dropped.

    Returns:
        (combined FeatureMatrix, coverage diagnostics)
    """
    emb_rows: List[np.ndarray] = []
    base_rows: List[np.ndarray] = []
    durations: List[float] = []
    events: List[int] = []
    eids: List[int] = []
    missing: List[int] = []

    for idx, eid in enumerate(baseline_matrix.eids):
        emb = embeddings.get(eid)
        if emb is None:
            missing.append(eid)
            if drop_missing:
                continue
            else:
                raise ValueError(f"Embedding missing for eid={eid}")
        emb_rows.append(np.asarray(emb, dtype=np.float32).ravel())
        base_rows.append(baseline_matrix.X[idx])
        durations.append(float(baseline_matrix.durations[idx]))
        events.append(int(baseline_matrix.events[idx]))
        eids.append(eid)

    if not emb_rows:
        raise ValueError("No overlapping embeddings found for the provided cohort.")

    emb_array = np.vstack(emb_rows).astype(np.float32)
    base_array = np.vstack(base_rows).astype(np.float32) if baseline_matrix.X.size else np.zeros(
        (len(emb_rows), 0), dtype=np.float32
    )

    combined = np.concatenate([emb_array, base_array], axis=1) if base_array.size else emb_array

    coverage = {
        "expected": len(baseline_matrix.eids),
        "available": len(eids),
        "coverage": len(eids) / max(len(baseline_matrix.eids), 1),
        "missing_count": len(missing),
        "missing_examples": missing[:5],
    }

    return (
        FeatureMatrix(
            combined,
            np.array(durations, dtype=np.float64),
            np.array(events, dtype=np.int32),
            eids,
        ),
        coverage,
    )


def standardize_features(
    train: FeatureMatrix, splits: Dict[str, FeatureMatrix]
) -> Tuple[FeatureMatrix, Dict[str, FeatureMatrix], Dict[str, np.ndarray]]:
    """Standardize features using train statistics."""
    if train.X.size == 0:
        return train, splits, {"mean": np.array([]), "std": np.array([])}

    mean = train.X.mean(axis=0)
    std = train.X.std(axis=0)
    std[std < 1e-8] = 1.0

    def _transform(matrix: FeatureMatrix) -> FeatureMatrix:
        if matrix.X.size == 0:
            return matrix
        X = (matrix.X - mean) / std
        return FeatureMatrix(X, matrix.durations, matrix.events, matrix.eids)

    transformed = {name: _transform(mat) for name, mat in splits.items()}
    return _transform(train), transformed, {"mean": mean, "std": std}


def save_predictions(
    output_dir: Path,
    method_name: str,
    split: str,
    eids: Sequence[int],
    risk_scores: np.ndarray,
    horizons: Sequence[float],
    survival_probs: Optional[np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{method_name}_{split}_preds.npz"
    surv_array = (
        np.asarray(survival_probs, dtype=np.float64)
        if survival_probs is not None
        else np.empty((0, 0), dtype=np.float64)
    )
    np.savez_compressed(
        path,
        eids=np.array(eids, dtype=np.int64),
        risk_scores=np.asarray(risk_scores, dtype=np.float64),
        horizons=np.asarray(horizons, dtype=np.float64),
        survival_probs=surv_array,
    )
    print(f"[save_predictions] Saved {split} predictions to {path}")


def run_cox_evaluation(
    matrices: Dict[str, FeatureMatrix],
    cox_config: CoxConfig,
    horizons: Optional[Sequence[float]] = None,
    save_preds_dir: Optional[Path] = None,
    method_name: str = "cox",
) -> Dict:
    """Train CoxPH on train split and evaluate on the remaining splits."""
    if "train" not in matrices:
        raise ValueError("Train split is required for Cox evaluation.")
    from lifelines import CoxPHFitter

    train_matrix = matrices["train"]
    other_splits = {k: v for k, v in matrices.items() if k != "train"}

    train_matrix_std, other_splits_std, stats = standardize_features(train_matrix, other_splits)
    col_names = [f"f{i}" for i in range(train_matrix_std.X.shape[1])]

    df_train = pd.DataFrame(train_matrix_std.X, columns=col_names)
    df_train["duration"] = train_matrix_std.durations
    df_train["event"] = train_matrix_std.events

    cph = CoxPHFitter(penalizer=cox_config.penalizer, l1_ratio=cox_config.l1_ratio)
    try:
        cph.fit(df_train, duration_col="duration", event_col="event", show_progress=False)
    except Exception as exc:
        print(f"[run_cox_evaluation] Primary Cox fit failed ({exc}); retrying with fallback penalizer.")
        cph = CoxPHFitter(penalizer=cox_config.fallback_penalizer, l1_ratio=cox_config.fallback_l1_ratio)
        cph.fit(df_train, duration_col="duration", event_col="event", show_progress=False)

    if horizons is None or len(horizons) == 0:
        horizons = metrics.derive_time_horizons(train_matrix_std.durations)

    train_struct = metrics.to_structured(train_matrix_std.durations, train_matrix_std.events)
    results = {
        "method": method_name,
        "horizons": horizons,
        "stats": {"feature_mean": stats["mean"].tolist(), "feature_std": stats["std"].tolist()},
        "splits": {},
    }

    all_splits = {"train": train_matrix_std, **other_splits_std}
    for split_name, matrix in all_splits.items():
        df_features = pd.DataFrame(matrix.X, columns=col_names)
        risk_scores = cph.predict_partial_hazard(df_features).values.flatten()
        survival_probs = None
        if horizons:
            surv_df = cph.predict_survival_function(df_features, times=horizons)
            survival_probs = surv_df.T.values if surv_df.size else None

        eval_struct = metrics.to_structured(matrix.durations, matrix.events)
        split_metrics = metrics.compute_metrics(
            train_struct, eval_struct, risk_scores, horizons, survival_probs
        )
        split_metrics["size"] = len(matrix.eids)
        split_metrics["event_rate"] = float(matrix.events.mean()) if len(matrix.events) else 0.0
        results["splits"][split_name] = split_metrics

        if save_preds_dir is not None:
            save_predictions(save_preds_dir, method_name, split_name, matrix.eids, risk_scores, horizons, survival_probs)

    return results
