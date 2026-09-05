"""/forecast endpoints — drug/region demand forecasts."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import ForecastPoint, ForecastResponse
from src.common.db import read_sql

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/{drug}/{region}", response_model=ForecastResponse)
def get_forecast(drug: str, region: str) -> ForecastResponse:
    """Forward demand forecast for one generic drug in one state.

    `drug` matches Gnrc_Name case-insensitively (e.g. 'metformin hcl'); `region`
    is the 2-letter state code.
    """
    try:
        df = read_sql(
            "SELECT gnrc_name, state, year, forecast_claims, model "
            "FROM drug_region_forecast "
            "WHERE LOWER(gnrc_name) = LOWER(:drug) AND UPPER(state) = UPPER(:region) "
            "ORDER BY year",
            params={"drug": drug, "region": region},
        )
    except Exception as exc:
        raise HTTPException(503, f"Forecasts not available — run training first ({exc}).") from exc
    if df.empty:
        raise HTTPException(404, f"No forecast for drug='{drug}', region='{region}'.")
    points = [ForecastPoint(**r) for r in df.to_dict("records")]
    return ForecastResponse(drug=points[0].gnrc_name, region=points[0].state, points=points)


@router.get("")
def all_forecasts() -> list[dict]:
    df = read_sql(
        "SELECT gnrc_name, state, year, forecast_claims, model "
        "FROM drug_region_forecast ORDER BY gnrc_name, state, year"
    )
    return df.to_dict("records")
