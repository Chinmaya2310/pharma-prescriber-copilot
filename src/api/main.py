"""FastAPI application entrypoint for the Prescriber Analytics Copilot.

Run:  uvicorn src.api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI

from src.api.routers import ask, forecast, meta, predict, segments

app = FastAPI(
    title="Pharma Prescriber Analytics & Forecasting Copilot",
    description=(
        "Serves prescriber segments, drug-demand forecasts, and an agentic "
        "text-to-SQL assistant over Medicare Part D prescriber data."
    ),
    version="1.0.0",
)

app.include_router(meta.router)
app.include_router(segments.router)
app.include_router(forecast.router)
app.include_router(predict.router)
app.include_router(ask.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "pharma-prescriber-copilot",
        "docs": "/docs",
        "endpoints": ["/kpis", "/segments/{npi}", "/forecast/{drug}/{region}",
                      "/predict/{npi}", "/ask"],
    }
