# Forecasting: Prophet vs XGBoost (walk-forward CV)

Rolling-origin CV, expanding window, 1-year horizon, pooled across all eligible (drug, state) series.

| model   |   n_folds |   MAPE |   RMSE |
|:--------|----------:|-------:|-------:|
| prophet |       112 |  18.18 | 1452.7 |
| xgboost |       112 |  22.71 | 1899.9 |

**Winner (lower MAPE): `prophet`** — served by the API and dashboard.


Excluded (too few annual points / heavily suppressed): **0** (drug, state) pairs: []

