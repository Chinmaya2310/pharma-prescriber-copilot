"""Compare Prophet vs XGBoost with rolling-origin (walk-forward) cross-validation,
pick the winner, refit on all history, and persist forecasts to SQLite.

Why walk-forward and not a single split: with a short annual series a single
train/test split can crown a model by luck. Rolling-origin CV re-evaluates the
model at every feasible origin (expanding window, 1-year horizon) and averages
the error, which is the standard honest way to validate a time-series model.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common import config  # noqa: E402
from src.common.db import write_table  # noqa: E402
from src.forecasting import series as S  # noqa: E402
from src.forecasting.models import get_forecaster  # noqa: E402

MODELS = ["prophet", "xgboost"]
MIN_TRAIN = 4          # smallest training window before we start scoring
FORECAST_HORIZON = 2   # years to project beyond the last observed year


def walk_forward_series(hist: pd.DataFrame, model_name: str) -> list[dict]:
    """Expanding-window, 1-step-ahead CV over a single series."""
    folds = []
    n = len(hist)
    for origin in range(MIN_TRAIN, n):
        train = hist.iloc[:origin]
        actual = float(hist.iloc[origin]["claims"])
        try:
            fc = get_forecaster(model_name).fit(train)
            pred = float(fc.predict(1)[0])
        except Exception:
            continue
        folds.append({"year": int(hist.iloc[origin]["Year"]), "actual": actual, "pred": pred})
    return folds


def evaluate(series: pd.DataFrame, keys: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for model_name in MODELS:
        all_actual, all_pred = [], []
        for drug, state in keys:
            hist = S.get_series(series, drug, state)
            for fold in walk_forward_series(hist, model_name):
                all_actual.append(fold["actual"])
                all_pred.append(fold["pred"])
        if all_actual:
            rows.append({
                "model": model_name,
                "n_folds": len(all_actual),
                "MAPE": round(S.mape(np.array(all_actual), np.array(all_pred)), 2),
                "RMSE": round(S.rmse(np.array(all_actual), np.array(all_pred)), 1),
            })
    return pd.DataFrame(rows)


def _save_comparison_report(cv: pd.DataFrame, winner: str, excluded: list) -> None:
    path = config.REPORTS_DIR / "forecast_comparison.md"
    lines = [
        "# Forecasting: Prophet vs XGBoost (walk-forward CV)\n",
        "Rolling-origin CV, expanding window, 1-year horizon, pooled across all "
        "eligible (drug, state) series.\n",
        cv.to_markdown(index=False),
        f"\n**Winner (lower MAPE): `{winner}`** — served by the API and dashboard.\n",
        f"\nExcluded (too few annual points / heavily suppressed): "
        f"**{len(excluded)}** (drug, state) pairs: {excluded}\n",
    ]
    path.write_text("\n".join(lines) + "\n")


def _plot_example(series: pd.DataFrame, keys, winner: str) -> None:
    if not keys:
        return
    drug, state = keys[0]
    hist = S.get_series(series, drug, state)
    fc = get_forecaster(winner).fit(hist)
    future = fc.predict(FORECAST_HORIZON)
    fut_years = [int(hist["Year"].max()) + i for i in range(1, FORECAST_HORIZON + 1)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["Year"], hist["claims"], marker="o", label="history")
    ax.plot(fut_years, future, marker="s", ls="--", label=f"{winner} forecast")
    ax.set(title=f"Demand forecast — {drug} / {state}", xlabel="Year", ylabel="total claims")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "forecast_example.png", dpi=120)
    plt.close(fig)


def run(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_parquet(config.PROCESSED_PARQUET)
    series = S.build_and_save(df)
    keep, drop = S.eligible_series(series)

    cv = evaluate(series, keep)
    if cv.empty:
        raise RuntimeError("No eligible series produced CV folds; check the data window.")
    winner = cv.sort_values("MAPE").iloc[0]["model"]

    # --- refit winner on full history and persist forward forecasts ---
    out_rows = []
    for drug, state in keep:
        hist = S.get_series(series, drug, state)
        fc = get_forecaster(winner).fit(hist)
        future = fc.predict(FORECAST_HORIZON)
        last_year = int(hist["Year"].max())
        for i, yhat in enumerate(future, start=1):
            out_rows.append({
                "gnrc_name": drug, "state": state, "year": last_year + i,
                "forecast_claims": round(float(yhat), 1), "model": winner,
            })
    forecast_df = pd.DataFrame(out_rows)
    write_table(forecast_df, "drug_region_forecast")

    _save_comparison_report(cv, winner, drop)
    _plot_example(series, keep, winner)

    from src.mlops.registry import save_model
    save_model(
        {"winner": winner, "horizon": FORECAST_HORIZON},
        name="forecasting",
        metrics={"winner": winner, **cv.set_index("model")["MAPE"].to_dict()},
    )
    return {
        "winner": winner,
        "cv": cv.to_dict("records"),
        "n_series": len(keep),
        "n_excluded": len(drop),
    }


def main() -> None:
    res = run()
    print("Forecast comparison (walk-forward CV):")
    for row in res["cv"]:
        print(f"  {row['model']:<8} MAPE={row['MAPE']}%  RMSE={row['RMSE']}  "
              f"(folds={row['n_folds']})")
    print(f"  winner: {res['winner']}")
    print(f"  series forecast: {res['n_series']}, excluded: {res['n_excluded']}")
    print(f"  report: {config.REPORTS_DIR / 'forecast_comparison.md'}")


if __name__ == "__main__":
    main()
