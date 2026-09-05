"""/segments endpoints — prescriber behavioural segments."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import SegmentResponse
from src.common.db import read_sql

router = APIRouter(prefix="/segments", tags=["segments"])


@router.get("/summary")
def segment_summary() -> list[dict]:
    """Segment sizes + average profile — used by the dashboard overview."""
    try:
        df = read_sql(
            "SELECT segment_id, segment_name, COUNT(*) AS n_prescribers, "
            "ROUND(AVG(volume),1) AS avg_volume, ROUND(AVG(growth),3) AS avg_growth "
            "FROM prescriber_segments GROUP BY segment_id, segment_name ORDER BY segment_id"
        )
    except Exception as exc:
        raise HTTPException(503, f"Segments not available — run training first ({exc}).") from exc
    return df.to_dict("records")


@router.get("/{provider_id}", response_model=SegmentResponse)
def get_segment(provider_id: int) -> SegmentResponse:
    """Return the segment assignment + profile for one prescriber NPI."""
    try:
        df = read_sql(
            "SELECT * FROM prescriber_segments WHERE prscrbr_npi = :npi",
            params={"npi": provider_id},
        )
    except Exception as exc:
        raise HTTPException(503, f"Segments not available — run training first ({exc}).") from exc
    if df.empty:
        raise HTTPException(404, f"No segment for prescriber NPI {provider_id}.")
    return SegmentResponse(**df.iloc[0].to_dict())


@router.get("")
def list_segments(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    df = read_sql(
        "SELECT prscrbr_npi, state, specialty, segment_id, segment_name, volume "
        "FROM prescriber_segments ORDER BY volume DESC LIMIT :lim",
        params={"lim": limit},
    )
    return df.to_dict("records")
