# Данные проекта

## Источник

Датасет: [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle, автор: laotse).

Симулирует данные кредитного бюро: 32 581 строка, 12 колонок.

## Файлы в папке

| Файл | Описание |
|------|----------|
| `credit_risk_dataset.csv` | Полный датасет |
| `processed_dataset.csv` | Датасет после очистки и добавления новых признаков (создаётся при обучении) |



## Структура датасета

| Колонка | Тип | Описание |
|---------|-----|----------|
| `person_age` | float | Возраст заявителя |
| `person_income` | float | Годовой доход ($) |
| `person_home_ownership` | str | Тип владения жильём: RENT - аренда, OWN - собственность, MORTGAGE - ипотека, OTHER - другое |
| `person_emp_length` | float | Стаж работы (лет) |
| `loan_intent` | str | Цель кредита: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION |
| `loan_grade` | str | Кредитный рейтинг: A (низкий риск) - G (высокий риск) |
| `loan_amnt` | float | Запрашиваемая сумма кредита (долл.) |
| `loan_int_rate` | float | Процентная ставка по кредиту (%) |
| `loan_percent_income` | float | Доля суммы кредита от годового дохода (0–1) |
| `cb_person_default_on_file` | str | Наличие дефолта в кредитной истории: Y - был, N - не было |
| `cb_person_cred_hist_length` | float | Длина кредитной истории (лет) |
| `loan_status` | int | **Целевая переменная**: 0 - кредит выплачен, 1 - дефолт |

### Новые признаки (добавляются при обработке)

| Колонка | Описание |
|---------|----------|
| `monthly_payment_approx` | Приближённый ежемесячный платёж (сумма / 36 месяцев) |
| `debt_to_monthly_income` | Долговая нагрузка: платёж / месячный доход |
| `interest_amount` | Абсолютная сумма переплаты по процентам |
| `income_per_emp_year` | Доход на год стажа - показатель стабильности |
| `work_start_age` | Возраст начала трудовой деятельности |
| `emp_to_age_ratio` | Доля жизни в трудоустройстве |
| `loan_grade_numeric` | Числовой рейтинг: A=1, B=2, ..., G=7 |
| `grade_rate_risk` | Риск-скор: числовой рейтинг * процентная ставка |
| `is_homeowner` | 1 - владеет жильём (OWN или MORTGAGE), 0 - нет |
| `had_default` | 1 - был дефолт в истории, 0 - не было |