# Тесты проекта

Юнит-тесты на `pytest`, покрывающие три модуля: препроцессинг, инференс и HTTP-эндпоинты сервиса.

## Запуск

```bash
# Из папки project/
pytest tests -v
```

---

## `test_preprocessing.py` - тесты препроцессинга

Проверяют корректность функций из `src/features/preprocessing.py`.

| Тест | Что проверяет | Зачем |
|------|--------------|-------|
| `test_clean_data_removes_outliers` | После `clean_data()` нет строк с возрастом > 100 и стажем > 60 | Убеждаемся, что нереалистичные значения удаляются |
| `test_clean_data_removes_duplicates` | После `clean_data()` датафрейм из 3 одинаковых строк становится 1 строкой | Проверяем дедупликацию |

---

## `test_predict.py` - тесты логики предсказания

Проверяют функцию `predict()` из `src/models/predict.py`. Модель и препроцессор подменяются моками - тесты работают без обученных артефактов.

| Тест | Что проверяет | Зачем |
|------|--------------|-------|
| `test_predict_low_risk` | При вероятности дефолта 0.12 - `loan_status=0`, `risk_level="low"` | Проверяем правильную классификацию низкого риска |
| `test_predict_medium_risk` | При вероятности 0.45 - `risk_level="medium"` | Проверяем средний уровень риска |
| `test_predict_high_risk` | При вероятности 0.75 - `loan_status=1`, `risk_level="high"` | Проверяем высокий риск |
| `test_predict_no_model_raises` | Если модель не загружена - выбрасывается `FileNotFoundError` | Убеждаемся, что ошибка не упускается |

---

## `test_service.py` - тесты HTTP-эндпоинтов

Проверяют FastAPI-приложение из `src/service/app.py` через `TestClient`. Реальная модель не нужна - используются моки.

| Тест | Что проверяет | Зачем |
|------|--------------|-------|
| `test_health_endpoint` | GET `/health` возвращает 200 и поля `status`, `model_loaded` | Проверяем работоспособность эндпоинта здоровья |
| `test_predict_endpoint_valid_input` | POST `/predict` с корректными данными возвращает 200 и поля `loan_status`, `default_probability`, `risk_level` | Основной happy-path тест |
| `test_predict_endpoint_invalid_age` | POST `/predict` с возрастом 15 (< 18) возвращает 422 | Проверяем Pydantic-валидацию входных данных |
| `test_predict_endpoint_missing_field` | POST `/predict` без поля `loan_amnt` возвращает 422 | Проверяем обязательность всех полей |
| `test_predict_endpoint_model_not_found` | Если `predict()` бросает `FileNotFoundError` - сервис возвращает 503 | Проверяем корректную обработку ошибки отсутствия модели |
