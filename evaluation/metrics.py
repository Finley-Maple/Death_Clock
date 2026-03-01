"""
Common survival metrics (C-index, time-dependent AUC, Integrated Brier Score)
shared by benchmarking, embedding, and Delphi evaluations.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np

try:
    from lifelines.utils import concordance_index

    LIFELINES_AVAILABLE = True
except Exception:  # pragma: no cover - lifelines optional
    concordance_index = None
    LIFELINES_AVAILABLE = False

try:
    from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

    SKSURV_AVAILABLE = True
except Exception:  # pragma: no cover - sksurv optional
    SKSURV_AVAILABLE = False


def to_structured(durations: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Convert arrays to the structured format required by sksurv."""
    return np.array(
        [(bool(e), float(t)) for e, t in zip(events, durations)],
        dtype=[("event", bool), ("duration", float)],
    )


def derive_time_horizons(durations: np.ndarray, quantiles: Sequence[float] = (0.25, 0.5, 0.75)) -> List[float]:
    """Derive evaluation horizons from training durations."""
    if len(durations) == 0:
        return []
    qs = np.quantile(durations, quantiles)
    horizons = sorted({int(q) for q in qs if q > 0})
    return horizons


def compute_metrics(
    train_structured: np.ndarray,
    eval_structured: np.ndarray,
    risk_scores: np.ndarray,
    horizons: Sequence[float],
    survival_probs: Optional[np.ndarray] = None,
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
    """
    result = {
        "c_index": None,
        "td_auc": None,
        "mean_td_auc": None,
        "ibs": None,
    }

    if not LIFELINES_AVAILABLE:
        raise ImportError("lifelines is required for survival metrics (pip install lifelines)")

    if len(risk_scores) == 0:
        return result

    durations = eval_structured["duration"]
    events = eval_structured["event"].astype(int)
    result["c_index"] = float(concordance_index(durations, -risk_scores, events))

    if SKSURV_AVAILABLE and horizons:
        valid_horizons = [h for h in horizons if h < durations.max()]
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
        except Exception:
            result["ibs"] = None
    else:
        result["ibs"] = None

    return result
