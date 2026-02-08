import pytest
from pydantic import ValidationError

from app.schemas import PredictionRequest, PredictFromDBRequest


def test_prediction_request_valid():
    PredictionRequest(features=[0.0] * 61)


def test_prediction_request_invalid_length():
    with pytest.raises(ValidationError):
        PredictionRequest(features=[0.0] * 60)


def test_prediction_request_invalid_type():
    with pytest.raises(ValidationError):
        PredictionRequest(features=["x"] * 61)


def test_predict_from_db_request_valid():
    PredictFromDBRequest(client_id=123)


def test_predict_from_db_request_missing_client_id():
    with pytest.raises(ValidationError):
        PredictFromDBRequest()
