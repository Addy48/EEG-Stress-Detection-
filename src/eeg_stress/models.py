"""The seven baseline classifiers, with the settings used for the reported results."""
from __future__ import annotations

from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

MODEL_NAMES = (
    "XGBoost",
    "LinearSVC",
    "CatBoost",
    "AdaBoost",
    "LogisticRegression",
    "KNN",
    "RandomForest",
)


def build_models(n_jobs: int = 4) -> dict:
    """Fresh, unfitted instances of the seven baselines."""
    return {
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
            n_jobs=n_jobs, eval_metric="logloss", tree_method="hist",
        ),
        "LinearSVC": LinearSVC(max_iter=2000, random_state=42),
        "CatBoost": CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1, random_state=42,
            verbose=False, thread_count=n_jobs,
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42, n_jobs=n_jobs),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs),
    }
