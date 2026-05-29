import joblib
import pandas as pd
from pathlib import Path

from src.utils import get_logger, load_config
from src.features import (
    add_features,
    ALL_NUMERIC_FEATURES,
    ALL_CATEGORICAL_FEATURES,
)

logger = get_logger(__name__)

_model = None
_preprocessor = None


def _load_artifacts(cfg: dict):
    global _model, _preprocessor
    if _model is None:
        model_path = cfg["artifacts"]["model_path"]
        prep_path = cfg["artifacts"]["preprocessor_path"]
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}"
            )
        _model = joblib.load(model_path)
        _preprocessor = joblib.load(prep_path)
        logger.info("Модель и препроцессор загружены")


def predict(features: dict, config_path: str = "configs/config.yaml") -> dict:
    cfg = load_config(config_path)
    _load_artifacts(cfg)

    df = pd.DataFrame([features])
    df = add_features(df)

    X = df[ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES]
    X_prep = _preprocessor.transform(X)

    prob = float(_model.predict_proba(X_prep)[0, 1])
    label = int(_model.predict(X_prep)[0])

    if prob < 0.3:
        risk_level = "low"
    elif prob < 0.6:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "loan_status": label,
        "default_probability": round(prob, 4),
        "risk_level": risk_level,
    }