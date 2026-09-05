"""Pydantic request/response models for the API (typed, self-documenting)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SegmentResponse(BaseModel):
    prscrbr_npi: int
    state: str
    specialty: str
    volume: float
    growth: float
    n_drugs: int
    avg_cost_per_claim: float
    segment_id: int
    segment_name: str


class ForecastPoint(BaseModel):
    gnrc_name: str
    state: str
    year: int
    forecast_claims: float
    model: str


class ForecastResponse(BaseModel):
    drug: str
    region: str
    points: list[ForecastPoint]


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=500,
        examples=["Which specialty prescribed the most Metformin in CA last year?"],
    )
    max_attempts: int = Field(3, ge=1, le=5)


class AttemptModel(BaseModel):
    n: int
    sql: str
    ok: bool
    error: str | None = None
    n_rows: int | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sql: str | None
    rows: list[dict]
    success: bool
    attempts: list[AttemptModel]


class KpiResponse(BaseModel):
    total_claims: int
    n_prescribers: int
    n_segments: int
    years: list[int]
    states: list[str]
    drugs: list[str]
    forecast_model: str | None
