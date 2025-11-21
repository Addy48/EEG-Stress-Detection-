import numpy as np
import pytest
from src.feature_extraction import (
    extract_channel_features,
    _shannon_entropy,
    extract_trial_features,
    build_feature_matrix
)

def test_extract_channel_features():
    signal = np.linspace(-1, 1, 128)
    features = extract_channel_features(signal)
    expected_keys = {
        'mean', 'variance', 'std', 'energy', 'rms',
        'peak_to_peak', 'mad', 'kurtosis', 'skewness', 'entropy'
    }
    assert expected_keys.issubset(set(features.keys()))
    assert features['peak_to_peak'] == pytest.approx(2.0, rel=1e-3)
    assert np.isfinite(features['entropy'])

def test_extract_trial_features():
    trial = np.random.randn(14, 128)
    feat_dict = extract_trial_features(trial)
    # 14 channels x 10 features = 140 features
    assert len(feat_dict) == 140
    assert 'AF3_mean' in feat_dict
    assert 'AF4_entropy' in feat_dict

def test_build_feature_matrix():
    trials = [np.random.randn(14, 128) for _ in range(6)]
    labels = [0, 0, 0, 1, 1, 1]
    X, y, names = build_feature_matrix(trials, labels)
    assert X.shape == (6, 140)
    assert len(y) == 6
    assert len(names) == 140
