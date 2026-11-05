import numpy as np
import pytest
from app.services.prediction_service import PredictionService


class FakeManager:
    required_input_columns = ["adults", "children"]
    model = type("Model", (), {"classes_": np.array([0, 1])})()

    def predict(self, data):
        return np.array([1] * len(data))

    def predict_proba(self, data):
        return np.array([[0.2, 0.8]] * len(data))


def test_prediction_service():
    result = PredictionService(FakeManager()).predict([{"adults": 2, "children": None}])
    assert result == [{"cancelada": 1, "probabilidade_cancelamento": 0.8}]


def test_missing_field():
    with pytest.raises(ValueError, match="children"):
        PredictionService(FakeManager()).predict([{"adults": 2}])
