"""API tests via FastAPI TestClient against a temp warehouse."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_kpis(temp_warehouse):
    r = client.get("/kpis")
    assert r.status_code == 200
    body = r.json()
    assert body["n_prescribers"] == 4
    assert body["total_claims"] == 500 + 120 + 800 + 60
    assert set(body["states"]) == {"CA", "TX"}


def test_segments_missing_returns_503(temp_warehouse):
    # temp warehouse has no prescriber_segments table -> graceful 503, not a 500.
    r = client.get("/segments/summary")
    assert r.status_code == 503


def test_ask_without_api_key_returns_503(temp_warehouse, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = client.post("/ask", json={"question": "how many claims in CA?"})
    assert r.status_code == 503
    assert "GROQ_API_KEY" in r.json()["detail"]


def test_ask_validation_rejects_short_question():
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 422  # pydantic min_length
