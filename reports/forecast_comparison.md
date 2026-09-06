# Forecasting: Prophet vs XGBoost vs LSTM (walk-forward CV)

Rolling-origin CV, expanding window, 1-year horizon, pooled across all eligible (drug, state) series.

| model   |   n_folds |   MAPE |     RMSE |
|:--------|----------:|-------:|---------:|
| prophet |       128 |   3.09 |  71128.8 |
| xgboost |       128 |   5.26 | 155589   |
| lstm    |       128 |   5.71 | 111898   |

**Winner (lower MAPE): `prophet`** — served by the API and dashboard.


Excluded (too few annual points / heavily suppressed): **0** (drug, state) pairs: []

