"""
Common survival metrics (C-index, time-dependent AUC, Integrated Brier Score)
shared by benchmarking, embedding, and Delphi evaluations.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from lifelines.utils import concordance_index

    LIFELINES_AVAILABLE = True
except Exception:  # pragma: no cover - lifelines optional
    concordance_index = None
    LIFELINES_AVAILABLE = False

try:
    from sksurv.metrics import (
        cumulative_dynamic_auc,
        concordance_index_ipcw,
        integrated_brier_score,
    )
    SKSURV_AVAILABLE = True
except Exception:  # pragma: no cover - sksurv optional
    SKSURV_AVAILABLE = False


def to_structured(durations: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Convert arrays to the structured format required by sksurv."""
    return np.array(
        [(bool(e), float(t)) for e, t in zip(events, durations)],
        dtype=[("event", bool), ("duration", float)],
    )


def derive_time_horizons(
    durations: np.ndarray,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    cap_quantile: float = 0.80,
) -> List[float]:
    """Derive evaluation horizons from training durations.

    Horizons are capped at the *cap_quantile* of training durations so that
    the longest horizon stays within a range with enough at-risk patients.
    """
    if len(durations) == 0:
        return []
    cap = float(np.quantile(durations, cap_quantile))
    qs = np.quantile(durations, quantiles)
    horizons = sorted({int(q) for q in qs if 0 < q <= cap})
    return horizons


def rmst(survival_probs_dense: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Compute per-patient Restricted Mean Survival Time (RMST) in years.

    The RMST is restricted to tau = grid[-1] (the maximum time point in the
    dense grid), which should be set to the 80th percentile of training
    durations to ensure enough at-risk patients throughout the range.

    Args:
        survival_probs_dense: array of shape (n_samples, len(grid)) with
            survival probabilities evaluated at each grid point.
        grid: 1-D array of time points in days.

    Returns:
        Per-patient RMST values in years, shape (n_samples,).
    """
    return np.trapz(survival_probs_dense, grid, axis=1) / 365.25


def compute_metrics(
    train_structured: np.ndarray,
    eval_structured: np.ndarray,
    risk_scores: np.ndarray,
    horizons: Sequence[float],
    survival_probs: Optional[np.ndarray] = None,
    survival_probs_dense: Optional[np.ndarray] = None,
    dense_grid: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute survival metrics for a split.

    Args:
        train_structured: structured (event, duration) array for the training split.
        eval_structured: structured array for the evaluated split.
        risk_scores: higher = more risk (partial hazards).
        horizons: sequence of time horizons (days) for TD-AUC / IBS.
        survival_probs: optional array of shape (n_samples, len(horizons)) with survival
                        probabilities at the specified horizons.
        survival_probs_dense: optional array of shape (n_samples, len(dense_grid)) with
                        survival probabilities on a dense grid, used to compute RMST.
        dense_grid: optional 1-D array of time points in days for the dense grid.
    """
    result = {
        "c_index": None,
        "c_index_ipcw": None,
        "td_auc": None,
        "mean_td_auc": None,
        "ibs": None,
        "rmst_mean": None,
    }

    if not LIFELINES_AVAILABLE:
        raise ImportError("lifelines is required for survival metrics (pip install lifelines)")

    if len(risk_scores) == 0:
        return result

    durations = eval_structured["duration"]
    events = eval_structured["event"].astype(int)

    # Harrell C-index (lifelines) — kept for backward compatibility.
    result["c_index"] = float(concordance_index(durations, -risk_scores, events))

    # IPCW C-index (sksurv) — preferred under heavy censoring.
    if SKSURV_AVAILABLE:
        try:
            tau = float(min(durations.max(), train_structured["duration"].max()))
            _, ipcw_c = concordance_index_ipcw(train_structured, eval_structured, risk_scores, tau=tau)
            result["c_index_ipcw"] = float(ipcw_c)
        except Exception as exc:
            logger.warning("IPCW C-index computation failed: %s", exc)

    if SKSURV_AVAILABLE and horizons:
        train_max = float(train_structured["duration"].max())
        eval_max = float(durations.max())
        valid_horizons = [h for h in horizons if h < min(train_max, eval_max)]
        dropped = len(horizons) - len(valid_horizons)
        if dropped:
            logger.warning(
                "Dropped %d/%d horizons exceeding min(train_max=%.0f, eval_max=%.0f) days",
                dropped, len(horizons), train_max, eval_max,
            )
        if valid_horizons:
            aucs, mean_auc = cumulative_dynamic_auc(
                train_structured, eval_structured, risk_scores, valid_horizons
            )
            result["td_auc"] = {str(int(h)): float(a) for h, a in zip(valid_horizons, aucs)}
            result["mean_td_auc"] = float(mean_auc)
        else:
            result["td_auc"] = {}
            result["mean_td_auc"] = None
    else:
        result["td_auc"] = None
        result["mean_td_auc"] = None

    if SKSURV_AVAILABLE and survival_probs is not None and horizons:
        try:
            ibs_val = integrated_brier_score(
                train_structured, eval_structured, survival_probs, horizons
            )
            result["ibs"] = float(ibs_val)
        except Exception as exc:
            logger.warning("IBS computation failed: %s", exc)
            result["ibs"] = None
    else:
        result["ibs"] = None

    if survival_probs_dense is not None and dense_grid is not None:
        try:
            result["rmst_mean"] = float(rmst(survival_probs_dense, dense_grid).mean())
        except Exception as exc:
            logger.warning("RMST computation failed: %s", exc)

    return result
