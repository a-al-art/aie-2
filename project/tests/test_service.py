import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.service.app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "person_age": 28,
    "person_income": 45000,
    "person_home_ownership": "RENT",
    "person_emp_length": 3.0,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.22,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 3.0,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_endpoint_valid_input():
    mock_result = {
        "loan_status": 0,
        "default_probability": 0.12,
        "risk_level": "low",
    }
    with patch("src.service.app.predict", return_value=mock_result):
        response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "loan_status" in data
    assert "default_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ("low", "medium", "high")


def test_predict_endpoint_invalid_age():
    payload = VALID_PAYLOAD.copy()
    payload["person_age"] = 15
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_missing_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "loan_amnt"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_model_not_found():
    with patch("src.service.app.predict", side_effect=FileNotFoundError("Модель не найдена")):
        response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
