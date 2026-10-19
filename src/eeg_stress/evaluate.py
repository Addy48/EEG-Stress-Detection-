"""Cross-validation protocols and metrics.

Subject-independent (SI) runs use five-fold ``GroupKFold``. Subject-dependent (SD)
runs use five-fold shuffled ``StratifiedKFold`` (random_state=42), which lets rows
from the same recording appear in both train and test. In both protocols the
``StandardScaler`` is fitted on the training fold only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROTOCOLS = ("si", "sd")
METRICS = ("accuracy", "precision", "recall", "f1", "auc")


@dataclass
class EvalResult:
    fold_metrics: dict[str, list[float]]
    y_true: np.ndarray
    y_pred: np.ndarray
    n_train_groups: list[int] = field(default_factory=list)
    n_test_groups: list[int] = field(default_factory=list)

    @property
    def mean(self) -> dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.fold_metrics.items()}

    @property
    def std(self) -> dict[str, float]:
        return {k: float(np.std(v)) for k, v in self.fold_metrics.items()}

    @property
    def confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_true, self.y_pred)

    @property
    def report(self) -> dict:
        return classification_report(
            self.y_true, self.y_pred, target_names=["Relax", "Task"],
            output_dict=True, zero_division=0,
        )


def make_splits(protocol: str, X, y, groups=None, n_splits: int = 5):
    if protocol == "si":
        if groups is None:
            raise ValueError("The SI protocol needs groups")
        return list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    if protocol == "sd":
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y))
    raise ValueError(f"protocol must be one of {PROTOCOLS}")


def _scores(model, X_test, y_test, y_pred):
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, proba),
    }


def evaluate(model, X, y, groups=None, protocol: str = "si", n_splits: int = 5, verbose: bool = False) -> EvalResult:
    """Fit ``model`` on each training fold and score it on the held-out fold."""
    fold_metrics = {m: [] for m in METRICS}
    y_true, y_pred = [], []
    n_train_groups, n_test_groups = [], []
    for fold, (tr, te) in enumerate(make_splits(protocol, X, y, groups, n_splits), 1):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        fitted = clone(model).fit(X_tr, y[tr])
        pred = fitted.predict(X_te)
        for name, value in _scores(fitted, X_te, y[te], pred).items():
            fold_metrics[name].append(float(value))
        y_true.extend(y[te])
        y_pred.extend(pred)
        if groups is not None:
            n_train_groups.append(len(np.unique(groups[tr])))
            n_test_groups.append(len(np.unique(groups[te])))
        if verbose:
            print(f"  fold {fold}/{n_splits}: accuracy={fold_metrics['accuracy'][-1]:.4f}")
    return EvalResult(fold_metrics, np.asarray(y_true), np.asarray(y_pred), n_train_groups, n_test_groups)
