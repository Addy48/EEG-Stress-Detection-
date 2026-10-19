"""Voting, Bagging and Stacking as configured in notebooks/02_ensembles.ipynb.

Stacking here follows the notebook: the logistic-regression meta-learner is trained
on the base models' predictions for the *same* rows those base models were fitted on
(in-sample). That is not the usual out-of-fold stacking and tends to over-trust the
base models; it is kept as-is so the code matches the saved results. See
docs/PROVENANCE.md. Stacking under SI uses a different script
(notebooks/01_baseline_and_extensions.ipynb, Phase 3) and is not reimplemented here.
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .evaluate import make_splits

ENSEMBLE_NAMES = ("Voting", "Bagging", "Stacking")


def _voting_base(protocol: str):
    if protocol == "si":
        return [
            ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=4, random_state=42)),
            ("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, n_jobs=2, random_state=42, eval_metric="logloss")),
            ("AdaBoost", AdaBoostClassifier(n_estimators=100, learning_rate=0.8, random_state=42)),
            ("CatBoost", CatBoostClassifier(iterations=100, depth=6, random_state=42, verbose=False, thread_count=4)),
        ]
    return [
        ("RF", RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=2, random_state=42)),
        ("XGB", XGBClassifier(n_estimators=50, max_depth=5, n_jobs=1, random_state=42, eval_metric="logloss")),
        ("ADA", AdaBoostClassifier(n_estimators=50, learning_rate=0.8, random_state=42)),
        ("CAT", CatBoostClassifier(iterations=50, depth=5, random_state=42, verbose=False, thread_count=2)),
    ]


def _bagging():
    return BaggingClassifier(
        estimator=AdaBoostClassifier(n_estimators=30, random_state=42),
        n_estimators=8, max_samples=0.7, max_features=0.8, random_state=42, n_jobs=2,
    )


def _stacking_fold(X_tr, y_tr, X_te):
    bases = [
        RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=2, random_state=42),
        XGBClassifier(n_estimators=50, max_depth=5, n_jobs=1, random_state=42, eval_metric="logloss"),
        AdaBoostClassifier(n_estimators=50, learning_rate=0.8, random_state=42),
    ]
    meta_tr, meta_te = [], []
    for base in bases:
        base.fit(X_tr, y_tr)
        meta_tr.append(base.predict_proba(X_tr)[:, 1])  # in-sample, as in the notebook
        meta_te.append(base.predict_proba(X_te)[:, 1])
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(np.column_stack(meta_tr), y_tr)
    return meta.predict(np.column_stack(meta_te))


def evaluate_ensemble(method: str, X, y, groups=None, protocol: str = "si", n_splits: int = 5) -> dict:
    """Mean and std of fold accuracy for one ensemble method."""
    if method not in ENSEMBLE_NAMES:
        raise ValueError(f"method must be one of {ENSEMBLE_NAMES}")
    if method == "Stacking" and protocol == "si":
        raise NotImplementedError("Stacking under SI lives in notebooks/01_baseline_and_extensions.ipynb (Phase 3)")
    accs = []
    for tr, te in make_splits(protocol, X, y, groups, n_splits):
        scaler = StandardScaler()
        X_tr, X_te = scaler.fit_transform(X[tr]), scaler.transform(X[te])
        if method == "Voting":
            model = VotingClassifier(estimators=_voting_base(protocol), voting="soft", n_jobs=1)
            pred = clone(model).fit(X_tr, y[tr]).predict(X_te)
        elif method == "Bagging":
            pred = _bagging().fit(X_tr, y[tr]).predict(X_te)
        else:
            pred = _stacking_fold(X_tr, y[tr], X_te)
        accs.append(float(accuracy_score(y[te], pred)))
    return {"Accuracy": float(np.mean(accs)), "Std": float(np.std(accs)), "folds": accs}
