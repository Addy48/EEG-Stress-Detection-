import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from src.eeg_classification import evaluate_subject_dependent, evaluate_subject_independent

def test_evaluations_synthetic():
    np.random.seed(42)
    # 3 subjects, each having 10 Relax (0) and 10 Task (1) samples = 60 samples
    sub_groups = []
    y_list = []
    for s in range(3):
        sub_groups.extend([f"sub_{s}"] * 20)
        y_list.extend([0] * 10 + [1] * 10)
    
    n_samples = len(y_list)
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = np.array(y_list)
    groups = np.array(sub_groups)
    
    model = LogisticRegression(max_iter=200)
    
    # Test SD (Stratified KFold)
    avg_sd, std_sd, cm_sd = evaluate_subject_dependent(X, y, "LogReg", model, n_splits=3)
    assert 0.0 <= avg_sd['accuracy'] <= 1.0
    assert cm_sd.shape == (2, 2)
    
    # Test SI (Group KFold)
    avg_si, std_si, cm_si = evaluate_subject_independent(X, y, groups, "LogReg", model, n_splits=3)
    assert 0.0 <= avg_si['accuracy'] <= 1.0
    assert cm_si.shape == (2, 2)
