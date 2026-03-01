import numpy as np
import pytest

from evaluation import metrics, survival_eval

pytestmark = pytest.mark.skipif(
    not metrics.LIFELINES_AVAILABLE,
    reason="lifelines is required for survival evaluation tests",
)


def make_matrix(values, durations, events, eids):
    return survival_eval.FeatureMatrix(
        X=np.array(values, dtype=np.float32),
        durations=np.array(durations, dtype=np.float64),
        events=np.array(events, dtype=np.int32),
        eids=eids,
    )


def test_run_cox_evaluation_smoke():
    train = make_matrix(
        values=[[1.0, 0.0], [0.5, 1.0], [0.2, 0.3]],
        durations=[5, 10, 15],
        events=[1, 0, 1],
        eids=[1, 2, 3],
    )
    val = make_matrix(
        values=[[0.8, 0.1], [0.3, 0.7]],
        durations=[7, 12],
        events=[1, 0],
        eids=[4, 5],
    )
    test = make_matrix(
        values=[[0.6, 0.2], [0.4, 0.5]],
        durations=[9, 18],
        events=[0, 1],
        eids=[6, 7],
    )

    matrices = {"train": train, "val": val, "test": test}
    cfg = survival_eval.CoxConfig(penalizer=0.1, l1_ratio=0.0)
    results = survival_eval.run_cox_evaluation(matrices, cfg, method_name="unit_test")

    assert "splits" in results
    assert "val" in results["splits"]
    assert "test" in results["splits"]
    assert results["splits"]["val"]["size"] == 2


def test_merge_with_embeddings_appends_features():
    baseline = make_matrix(
        values=[[0.0], [1.0], [2.0]],
        durations=[5, 6, 7],
        events=[1, 0, 1],
        eids=[10, 11, 12],
    )
    embeddings = {
        10: np.array([0.1, 0.2], dtype=np.float32),
        11: np.array([0.3, 0.4], dtype=np.float32),
        12: np.array([0.5, 0.6], dtype=np.float32),
    }

    combined, coverage = survival_eval.merge_with_embeddings(baseline, embeddings)
    assert combined.X.shape[1] == 3  # 2 embedding dims + 1 baseline dim
    assert coverage["coverage"] == 1.0
