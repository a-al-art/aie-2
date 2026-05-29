import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import predict, _load_artifacts
from src.utils import get_logger, load_config

logger = get_logger(__name__, level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск сервиса")
    cfg = load_config("configs/config.yaml")
    try:
        _load_artifacts(cfg)
        logger.info("Сервис готов")
    except FileNotFoundError as e:
        logger.warning(f"Модель не найдена {e}")
    yield
    logger.info("Сервис остановлен")


app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Сервис оценки кредитного риска"
        "Предсказывает вероятность дефолта по профилю заявителя"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ApplicantFeatures(BaseModel):
    person_age: float = Field(..., ge=18, le=100, example=28, description="Возраст заявителя")
    person_income: float = Field(..., gt=0, example=45000, description="Годовой доход ($)")
    person_home_ownership: str = Field(..., example="RENT", description="RENT | OWN | MORTGAGE | OTHER")
    person_emp_length: float = Field(..., ge=0, le=60, example=3.0, description="Стаж работы (лет)")
    loan_intent: str = Field(..., example="PERSONAL", description="PERSONAL | EDUCATION | MEDICAL | VENTURE | HOMEIMPROVEMENT | DEBTCONSOLIDATION")
    loan_grade: str = Field(..., example="B", description="A | B | C | D | E | F | G")
    loan_amnt: float = Field(..., gt=0, example=10000, description="Сумма кредита ($)")
    loan_int_rate: float = Field(..., gt=0, le=40, example=11.5, description="Процентная ставка (%)")
    loan_percent_income: float = Field(..., ge=0, le=1, example=0.22, description="Доля кредита от дохода")
    cb_person_default_on_file: str = Field(..., example="N", description="Y | N - был ли дефолт ранее")
    cb_person_cred_hist_length: float = Field(..., ge=0, example=3.0, description="Длина кредитной истории (лет)")


class PredictionResponse(BaseModel):
    loan_status: int = Field(..., description="0 - выплачен кредит, 1 - не выплачен")
    default_probability: float = Field(..., description="Вероятность дефолта [0..1]")
    risk_level: str = Field(..., description="low | medium | high")


@app.get("/health", summary="Проверка работоспособности")
def health():
    from src.models.predict import _model
    return {
        "status": "ok",
        "model_loaded": _model is not None,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Оценка кредитного риска")
def predict_risk(applicant: ApplicantFeatures):
    logger.info(f"Запрос /predict: age={applicant.person_age}, income={applicant.person_income}, loan={applicant.loan_amnt}")
    try:
        result = predict(applicant.model_dump())
        logger.info(f"Результат: {result}")
        return result
    except FileNotFoundError as e:
        logger.error(f"Модель не загружена: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {e}")
