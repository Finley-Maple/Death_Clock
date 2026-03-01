import numpy as np
import pytest

from evaluation import metrics

pytestmark = pytest.mark.skipif(
    not metrics.LIFELINES_AVAILABLE,
    reason="lifelines is required for survival metrics tests",
)


def test_derive_time_horizons_basic():
    durations = np.array([10, 20, 30, 40], dtype=float)
    horizons = metrics.derive_time_horizons(durations)
    assert horizons == [20, 30], "Expected quantile-based horizons"


def test_compute_metrics_monotonic_risk():
    durations_train = np.array([5, 10, 15, 20], dtype=float)
    events_train = np.array([1, 1, 0, 1], dtype=int)
    durations_eval = np.array([6, 12, 25, 30], dtype=float)
    events_eval = np.array([1, 0, 1, 0], dtype=int)
    risk_scores = np.array([0.9, 0.4, 0.7, 0.1], dtype=float)
    horizons = [5, 15, 25]

    train_struct = metrics.to_structured(durations_train, events_train)
    eval_struct = metrics.to_structured(durations_eval, events_eval)
    result = metrics.compute_metrics(train_struct, eval_struct, risk_scores, horizons, survival_probs=None)

    assert 0.0 <= result["c_index"] <= 1.0
    if metrics.SKSURV_AVAILABLE and horizons:
        assert isinstance(result["td_auc"], dict)
    else:
        assert result["td_auc"] in (None, {})
