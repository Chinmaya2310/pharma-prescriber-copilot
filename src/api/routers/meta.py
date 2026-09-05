"""/kpis and reference-data endpoints for the dashboard."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import KpiResponse
from src.common.db import list_tables, read_sql

router = APIRouter(tags=["meta"])


@router.get("/kpis", response_model=KpiResponse)
def kpis() -> KpiResponse:
    """Top-line numbers for the dashboard overview page."""
    tables = list_tables()
    if "prescribers" not in tables:
        raise HTTPException(503, "Warehouse not loaded — run the pipeline first.")

    agg = read_sql(
        "SELECT SUM(Tot_Clms) AS total_claims, "
        "COUNT(DISTINCT Prscrbr_NPI) AS n_prescribers FROM prescribers"
    ).iloc[0]
    years = read_sql("SELECT DISTINCT Year FROM prescribers ORDER BY Year")["Year"].tolist()
    states = read_sql(
        "SELECT DISTINCT Prscrbr_State_Abrvtn AS s FROM prescribers ORDER BY s"
    )["s"].tolist()
    drugs = read_sql("SELECT DISTINCT Gnrc_Name AS d FROM prescribers ORDER BY d")["d"].tolist()

    n_segments = 0
    if "prescriber_segments" in tables:
        n_segments = int(read_sql(
            "SELECT COUNT(DISTINCT segment_id) AS n FROM prescriber_segments"
        ).iloc[0]["n"])

    forecast_model = None
    if "drug_region_forecast" in tables:
        row = read_sql("SELECT model FROM drug_region_forecast LIMIT 1")
        if not row.empty:
            forecast_model = row.iloc[0]["model"]

    return KpiResponse(
        total_claims=int(agg["total_claims"] or 0),
        n_prescribers=int(agg["n_prescribers"] or 0),
        n_segments=n_segments,
        years=[int(y) for y in years],
        states=states,
        drugs=drugs,
        forecast_model=forecast_model,
    )


@router.get("/tables")
def tables() -> list[str]:
    return list_tables()
