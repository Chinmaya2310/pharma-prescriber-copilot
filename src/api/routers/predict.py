"""/predict endpoints — next-period growth-class prediction per prescriber."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import GrowthPrediction
from src.common.db import read_sql

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.get("/summary")
def prediction_summary() -> list[dict]:
    """Counts per predicted class — for the dashboard."""
    try:
        df = read_sql(
            "SELECT predicted_class, COUNT(*) AS n FROM prescriber_growth_prediction "
            "GROUP BY predicted_class ORDER BY n DESC"
        )
    except Exception as exc:
        raise HTTPException(
            503, f"Predictions not available — run training first ({exc}).") from exc
    return df.to_dict("records")


@router.get("/{provider_id}", response_model=GrowthPrediction)
def get_prediction(provider_id: int) -> GrowthPrediction:
    """Predicted next-period growth class (+ probabilities) for one prescriber NPI."""
    try:
        df = read_sql(
            "SELECT * FROM prescriber_growth_prediction WHERE prscrbr_npi = :npi",
            params={"npi": provider_id},
        )
    except Exception as exc:
        raise HTTPException(
            503, f"Predictions not available — run training first ({exc}).") from exc
    if df.empty:
        raise HTTPException(404, f"No growth prediction for prescriber NPI {provider_id}.")
    return GrowthPrediction(**df.iloc[0].to_dict())


@router.get("")
def list_predictions(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    df = read_sql(
        "SELECT prscrbr_npi, predicted_class, prob_growing, prob_declining, model "
        "FROM prescriber_growth_prediction ORDER BY prob_growing DESC LIMIT :lim",
        params={"lim": limit},
    )
    return df.to_dict("records")
