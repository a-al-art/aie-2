# Артефакты проекта

В эту папку можно сохранять **артефакты**, получаемые в процессе работы проекта:


| Файл | Что содержит |
|------|-------------|
| `model.pkl` | Обученная модель CatBoost - финальная модель для инференса |
| `preprocessor.pkl` | Sklearn-препроцессор: StandardScaler для числовых признаков + OneHotEncoder для категориальных |
| `metrics.json` | Метрики обеих моделей на тестовой выборке: ROC-AUC, F1, Precision, Recall, Accuracy; для CatBoost также CV ROC-AUC |
| `class_balance.png` | График баланса классов: соотношение дефолт (кредит не погашен) / нет дефолта (погашен) в датасете |
| `numeric_distributions.png` | Гистограммы распределений всех числовых признаков |
| `correlation_matrix.png` | Корреляционная матрица числовых признаков с целевой переменной |
| `categorical_default_rate.png` | Доля дефолтов по категориальным признакам (home ownership, loan intent, grade, default on file) |
| `model_comparison.png` | Столбчатый график сравнения метрик Logistic Regression vs CatBoost |
| `roc_curves.png` | ROC-кривые обеих моделей на тестовой выборке |
| `confusion_matrices.png` | Матрицы ошибок (confusion matrix) для обеих моделей |
| `feature_importance.png` | Топ-10 признаков по важности согласно CatBoost |
