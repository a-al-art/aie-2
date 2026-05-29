import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, classification_report,
)
from catboost import CatBoostClassifier

from src.data import load_dataset
from src.features import (
    clean_data,
    add_features,
    save_processed,
    build_preprocessor,
    split_features_target,
)
from src.utils import get_logger, load_config

logger = get_logger(__name__)

def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "roc_auc":    round(roc_auc_score(y_true, y_prob), 4),
        "f1":         round(f1_score(y_true, y_pred), 4),
        "precision":  round(precision_score(y_true, y_pred), 4),
        "recall":     round(recall_score(y_true, y_pred), 4),
        "accuracy":   round(accuracy_score(y_true, y_pred), 4),
    }

def train(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)

    df = load_dataset(cfg["data"]["raw_path"])
    df = clean_data(df)
    df = add_features(df)
    save_processed(df, cfg["data"].get("processed_path", "data/processed_dataset.csv"))

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}, дефолт: {y_train.mean():.2%}")

    preprocessor = build_preprocessor()
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep  = preprocessor.transform(X_test)

    all_metrics = {}

    # Logistic Regression
    lr_cfg = cfg["models"]["logistic_regression"]
    lr = LogisticRegression(
        C=lr_cfg["C"], max_iter=lr_cfg["max_iter"],
        class_weight=lr_cfg["class_weight"], random_state=lr_cfg["random_state"],
    )
    lr.fit(X_train_prep, y_train)
    lr_pred = lr.predict(X_test_prep)
    lr_prob = lr.predict_proba(X_test_prep)[:, 1]
    all_metrics["logistic_regression"] = evaluate(y_test, lr_pred, lr_prob)
    logger.info(f"LR metrics: {all_metrics['logistic_regression']}")
    logger.info("\n" + classification_report(y_test, lr_pred, target_names=["no_default", "default"]))

    # CatBoost
    cb_cfg = cfg["models"]["catboost"]
    cb = CatBoostClassifier(
        iterations=cb_cfg["iterations"], learning_rate=cb_cfg["learning_rate"],
        depth=cb_cfg["depth"], l2_leaf_reg=cb_cfg["l2_leaf_reg"],
        scale_pos_weight=3.0,
        random_seed=cb_cfg["random_seed"],
        verbose=cb_cfg["verbose"],
    )
    cb.fit(X_train_prep, y_train)
    cb_pred = cb.predict(X_test_prep)
    cb_prob = cb.predict_proba(X_test_prep)[:, 1]
    all_metrics["catboost"] = evaluate(y_test, cb_pred, cb_prob)
    logger.info(f"CatBoost metrics: {all_metrics['catboost']}")
    logger.info("\n" + classification_report(y_test, cb_pred, target_names=["no_default", "default"]))

    # Кросс-валидация
    cv = StratifiedKFold(n_splits=cfg["training"]["cv_folds"], shuffle=True, random_state=42)
    cv_scores = cross_val_score(cb, X_train_prep, y_train, cv=cv, scoring="roc_auc")
    all_metrics["catboost"]["cv_roc_auc_mean"] = round(cv_scores.mean(), 4)
    all_metrics["catboost"]["cv_roc_auc_std"]  = round(cv_scores.std(), 4)
    logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} +- {cv_scores.std():.4f}")

    # артефакты
    Path(cfg["artifacts"]["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cb, cfg["artifacts"]["model_path"])
    joblib.dump(preprocessor, cfg["artifacts"]["preprocessor_path"])
    with open(cfg["artifacts"]["metrics_path"], "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Модель: {cfg['artifacts']['model_path']}")
    logger.info(f"Препроцессор: {cfg['artifacts']['preprocessor_path']}")
    logger.info(f"Метрики: {cfg['artifacts']['metrics_path']}")

if __name__ == "__main__":
    train()
