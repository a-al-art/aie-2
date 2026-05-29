# Исходный код проекта

В этой папке размещается **основной код проекта**, который используется для:

- подготовки данных;
- обучения моделей;
- инференса (получения предсказаний);
- запуска сервисов (API/CLI/скриптов).

## Структура

```
src/
├── data/
│   ├── __init__.py
│   └── loader.py - Загрузка датасета из CSV, сохранение sample
│
├── features/
│   ├── __init__.py
│   └── preprocessing.py - Очистка данных, feature engineering,
│
├── models/
│   ├── __init__.py
│   ├── train.py    - Полный пайплайн обучения: LR + CatBoost, метрики, сохранение артефактов
│   └── predict.py  - Инференс: загрузка модели, предсказание, уровень риска
│
├── service/
│   ├── __init__.py
│   ├── __main__.py - Точка входа для python -m src.service
│   └── app.py      - FastAPI-приложение: /health, /predict
│
├── utils/
│   ├── __init__.py
│   ├── logger.py   - Настроенный логгер
│   └── config.py   - Загрузка config.yaml
│
├── __init__.py
├── train.py        - Точка входа для python -m src.train
└── service.py      - Запуск uvicorn через load_config
```

---

## Основные команды

```bash
# Обучение модели
python -m src.train

# Запуск сервиса
python -m src.service
```

---

## Описание модулей

### `src/data/loader.py`
Загружает CSV-файл датасета.

### `src/features/preprocessing.py`
Содержит три основных этапа подготовки данных:
- `clean_data()` - удаление дубликатов, фильтрация выбросов, заполнение пропусков медианой
- `add_features()` - создание новых признаков
- `save_processed()` - сохранение обработанного датасета в `data/processed_dataset.csv`
- `build_preprocessor()` - sklearn ColumnTransformer: StandardScaler для числовых, OneHotEncoder для категориальных
- `split_features_target()` - разбивка на X и y

### `src/models/train.py`
Полный пайплайн обучения: загрузка - очистка - feature engineering - сохранение - препроцессинг - SMOTE - обучение LR и CatBoost - кросс-валидация - сохранение модели, препроцессора и метрик.

### `src/models/predict.py`
Инференс для одного заявителя. Модель и препроцессор загружаются один раз при старте (кешируются в памяти). Возвращает `loan_status`, `default_probability` и `risk_level` (low / medium / high).

### `src/service/app.py`
FastAPI-приложение. При старте автоматически загружает модель. Валидирует входные данные через Pydantic (`ApplicantFeatures`): проверяет диапазоны возраста, дохода, суммы кредита и других полей. Логирует каждый запрос.

### `src/utils/logger.py`
Возвращает настроенный `logging.Logger` с форматом `время | уровень | модуль | сообщение`. Используется во всех модулях.

### `src/utils/config.py`
Загружает `configs/config.yaml` и возвращает словарь параметров.
