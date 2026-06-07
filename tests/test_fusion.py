import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_pca_per_embedding_shape():
    """PCA applied independently to two embedding blocks gives correct output dims."""
    from evaluation.evaluate_fusion import pca_embeddings_per_block
    rng = np.random.RandomState(42)
    emb_a = {i: rng.randn(1024).astype(np.float32) for i in range(100)}
    emb_b = {i: rng.randn(1024).astype(np.float32) for i in range(100)}
    train_eids = list(range(70))
    val_eids = list(range(70, 85))
    test_eids = list(range(85, 100))
    splits = {"train": train_eids, "val": val_eids, "test": test_eids}

    result = pca_embeddings_per_block(
        [emb_a, emb_b], splits, n_components=16
    )
    assert result["train"].shape == (70, 32), f"Expected (70,32), got {result['train'].shape}"
    assert result["val"].shape == (15, 32)
    assert result["test"].shape == (15, 32)


def test_pca_fit_on_train_only():
    """Val/test transform uses train PCA — not refit."""
    from evaluation.evaluate_fusion import pca_embeddings_per_block
    rng = np.random.RandomState(7)
    train_embs = {i: rng.randn(64).astype(np.float32) * 10 for i in range(80)}
    val_embs = {i: rng.randn(64).astype(np.float32) * 0.01 for i in range(80, 100)}
    all_embs = {**train_embs, **val_embs}
    splits = {"train": list(range(80)), "val": list(range(80, 100))}
    result = pca_embeddings_per_block([all_embs], splits, n_components=4)
    val_var = result["val"].var()
    train_var = result["train"].var()
    assert train_var > val_var * 5, "Val variance should be much smaller (train PCA applied)"
