import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.models.predict import predict


SAMPLE_INPUT = {
    "person_age": 28.0,
    "person_income": 45000.0,
    "person_home_ownership": "RENT",
    "person_emp_length": 3.0,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10000.0,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.22,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 3.0,
}


def _mock_model(prob: float):
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - prob, prob]])
    model.predict.return_value = np.array([int(prob >= 0.5)])
    return model


def _mock_preprocessor():
    prep = MagicMock()
    prep.transform.return_value = np.zeros((1, 20))
    return prep


@patch("src.models.predict._model", _mock_model(0.12))
@patch("src.models.predict._preprocessor", _mock_preprocessor())
def test_predict_low_risk():
    result = predict(SAMPLE_INPUT)
    assert result["loan_status"] == 0
    assert result["risk_level"] == "low"
    assert 0.0 <= result["default_probability"] <= 1.0


@patch("src.models.predict._model", _mock_model(0.45))
@patch("src.models.predict._preprocessor", _mock_preprocessor())
def test_predict_medium_risk():
    result = predict(SAMPLE_INPUT)
    assert result["risk_level"] == "medium"


@patch("src.models.predict._model", _mock_model(0.75))
@patch("src.models.predict._preprocessor", _mock_preprocessor())
def test_predict_high_risk():
    result = predict(SAMPLE_INPUT)
    assert result["loan_status"] == 1
    assert result["risk_level"] == "high"


def test_predict_no_model_raises():
    with patch("src.models.predict._model", None), \
         patch("src.models.predict._preprocessor", None), \
         patch("src.models.predict._load_artifacts", side_effect=FileNotFoundError("нет модели")):
        with pytest.raises(FileNotFoundError):
            predict(SAMPLE_INPUT)
