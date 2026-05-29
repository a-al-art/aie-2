import pytest
import pandas as pd
import numpy as np

from src.features.preprocessing import (
    clean_data,
    build_preprocessor,
    split_features_target,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)


@pytest.fixture
def sample_df():
    data = {
        "person_age": [28, 35, 144, 45, 22],
        "person_income": [45000, 60000, 30000, 80000, 25000],
        "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "RENT", "RENT"],
        "person_emp_length": [3.0, 10.0, 5.0, 123.0, 1.0],
        "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "PERSONAL", "VENTURE"],
        "loan_grade": ["B", "A", "C", "D", "B"],
        "loan_amnt": [10000, 20000, 5000, 15000, 8000],
        "loan_int_rate": [11.5, 7.9, 14.2, 16.0, 10.5],
        "loan_percent_income": [0.22, 0.33, 0.17, 0.19, 0.32],
        "cb_person_default_on_file": ["N", "N", "Y", "N", "N"],
        "cb_person_cred_hist_length": [3.0, 8.0, 2.0, 10.0, 1.0],
        "loan_status": [0, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_clean_data_removes_outliers(sample_df):
    cleaned = clean_data(sample_df)
    assert cleaned["person_age"].max() <= 100, "Возраст > 100 не удалён"
    assert cleaned["person_emp_length"].max() <= 60, "Стаж > 60 не удалён"


def test_clean_data_removes_duplicates():
    data = {col: [1] * 3 for col in ["person_age", "person_income", "person_emp_length",
                                       "loan_amnt", "loan_int_rate", "loan_percent_income",
                                       "cb_person_cred_hist_length"]}
    data.update({
        "person_home_ownership": ["RENT"] * 3,
        "loan_intent": ["PERSONAL"] * 3,
        "loan_grade": ["B"] * 3,
        "cb_person_default_on_file": ["N"] * 3,
        "loan_status": [0] * 3,
    })
    df = pd.DataFrame(data)
    cleaned = clean_data(df)
    assert len(cleaned) == 1, "Дубликаты не удалены"

