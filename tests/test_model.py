import pytest
import numpy as np

from app.models_ml import predict_features


class DummyLogisticModel:
    """Imite sklearn LogisticRegression: predict_proba -> numpy array (n,2)."""
    def __init__(self, proba_pos: float):
        self.proba_pos = float(proba_pos)

    def predict_proba(self, X):
        p = self.proba_pos
        # sklearn retourne un numpy array pour supporter [:, 1]
        return np.array([[1.0 - p, p] for _ in range(len(X))], dtype=float)


def test_predict_features_returns_types_and_bounds():
    artifacts = {"model": DummyLogisticModel(0.7), "threshold": 0.5}
    features = [0.0] * 61

    label, proba = predict_features(artifacts, features)

    assert isinstance(label, int)
    assert isinstance(proba, float)
    assert 0.0 <= proba <= 1.0


def test_predict_features_above_threshold():
    artifacts = {"model": DummyLogisticModel(0.7), "threshold": 0.5}
    features = [0.0] * 61

    label, proba = predict_features(artifacts, features)

    assert proba == pytest.approx(0.7)
    assert label == 1


def test_predict_features_below_threshold():
    artifacts = {"model": DummyLogisticModel(0.3), "threshold": 0.5}
    features = [0.0] * 61

    label, proba = predict_features(artifacts, features)

    assert proba == pytest.approx(0.3)
    assert label == 0
