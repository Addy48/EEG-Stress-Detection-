import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from eeg_stress.evaluate import evaluate, make_splits


def _toy(n_groups=20, per_group=30, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per_group)
    y = np.tile([0, 1], len(groups) // 2)
    X = rng.normal(size=(len(groups), 4)) + y[:, None] * 0.5
    return X, y, groups


def test_si_folds_share_no_group():
    X, y, groups = _toy()
    for tr, te in make_splits("si", X, y, groups):
        assert not set(groups[tr]) & set(groups[te])


def test_sd_folds_are_stratified_and_cover_all_rows():
    X, y, _ = _toy()
    splits = make_splits("sd", X, y)
    assert sorted(np.concatenate([te for _, te in splits])) == list(range(len(y)))
    for _, te in splits:
        assert abs(y[te].mean() - 0.5) < 0.02


def test_si_needs_groups():
    X, y, _ = _toy()
    with pytest.raises(ValueError):
        make_splits("si", X, y, None)


def test_evaluate_returns_five_folds_of_every_metric():
    X, y, groups = _toy()
    res = evaluate(LogisticRegression(max_iter=200), X, y, groups, "si")
    assert all(len(v) == 5 for v in res.fold_metrics.values())
    assert 0.0 <= res.mean["accuracy"] <= 1.0
    assert len(res.y_true) == len(y)
    assert res.confusion.sum() == len(y)
    assert all(a + b == 20 for a, b in zip(res.n_train_groups, res.n_test_groups))
