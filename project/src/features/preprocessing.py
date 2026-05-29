import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.utils import get_logger

logger = get_logger(__name__)


NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

ENGINEERED_NUMERIC = [
    "debt_to_monthly_income",
    "loan_grade_numeric",
    "grade_rate_risk",
    "is_homeowner",
    "had_default",
]

ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + ENGINEERED_NUMERIC
ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES  

TARGET = "loan_status"

_GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_len = len(df)

    df = df.drop_duplicates()
    logger.info(f"Удалено дубликатов: {initial_len - len(df)}")

    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"] <= 60]
    logger.info(f"После удаления выбросов: {len(df)} строк")

    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())

    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["monthly_payment_approx"] = df["loan_amnt"] / 36
    df["debt_to_monthly_income"] = df["monthly_payment_approx"] / (df["person_income"] / 12)
    df["loan_grade_numeric"] = df["loan_grade"].map(_GRADE_MAP)
    df["grade_rate_risk"] = df["loan_grade_numeric"] * df["loan_int_rate"]
    df["is_homeowner"] = df["person_home_ownership"].isin(["OWN", "MORTGAGE"]).astype(int)
    df["had_default"] = (df["cb_person_default_on_file"] == "Y").astype(int)

    logger.info(f"добавлено {len(ENGINEERED_NUMERIC)} новых признаков")
    return df


def save_processed(df: pd.DataFrame, path: str = "data/processed_dataset.csv") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Обработанный датасет сохранён: {out} ({len(df)} строк, {df.shape[1]} колонок)")


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, ALL_NUMERIC_FEATURES),
        ("cat", categorical_pipeline, ALL_CATEGORICAL_FEATURES),
    ])

    return preprocessor


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y
