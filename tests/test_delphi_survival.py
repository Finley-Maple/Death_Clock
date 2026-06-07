import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_build_survival_curve_shape():
    from evaluation.evaluate_delphi import build_survival_curve
    logits = np.zeros((3, 10, 2), dtype=np.float32)
    ages = np.tile(np.linspace(50.0, 75.0, 10), (3, 1)).astype(np.float32)
    horizons = [365, 1825, 3650]
    surv, risk = build_survival_curve(logits, ages, horizons)
    assert surv.shape == (3, 3)
    assert risk.shape == (3,)


def test_build_survival_curve_monotone():
    from evaluation.evaluate_delphi import build_survival_curve
    # High hazard after 60 → survival decreases with horizon
    logits = np.full((1, 10, 1), 3.0, dtype=np.float32)  # sigmoid(3) ≈ 0.95
    ages = np.linspace(58.0, 68.0, 10).reshape(1, -1).astype(np.float32)
    horizons = [365, 1825, 3650]
    surv, _ = build_survival_curve(logits, ages, horizons)
    assert surv[0, 0] >= surv[0, 1] >= surv[0, 2], "Survival must be non-increasing"


def test_build_survival_curve_probs_in_range():
    from evaluation.evaluate_delphi import build_survival_curve
    logits = np.random.randn(5, 20, 2).astype(np.float32)
    ages = np.tile(np.linspace(55.0, 80.0, 20), (5, 1)).astype(np.float32)
    horizons = [365, 3650]
    surv, risk = build_survival_curve(logits, ages, horizons)
    assert np.all(surv >= 0) and np.all(surv <= 1)
    assert np.all(risk >= 0) and np.all(risk <= 1)


def test_build_survival_curve_no_post60_events():
    from evaluation.evaluate_delphi import build_survival_curve
    # All positions are before age 60 → survival stays 1.0
    logits = np.full((2, 5, 1), 5.0, dtype=np.float32)
    ages = np.tile(np.linspace(30.0, 59.0, 5), (2, 1)).astype(np.float32)
    horizons = [365, 3650]
    surv, risk = build_survival_curve(logits, ages, horizons)
    assert np.allclose(surv, 1.0), "No post-60 events → survival should be 1.0"
    assert np.allclose(risk, 0.0)
