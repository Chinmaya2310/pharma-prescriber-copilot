"""Tests for the drift/monitoring check — untested monitoring silently rots."""
from __future__ import annotations

import json

from src.common import config
from src.mlops import monitoring


class TestEvaluateDrift:
    def test_degraded_mape_fires_flag(self):
        # current MAPE 30% vs historical avg 3% -> 10x -> must flag
        flags = monitoring.evaluate_drift(forecast_mape=30.0, stability_ari=0.7,
                                          hist_mape_avg=3.0)
        assert any("FORECAST DRIFT" in f for f in flags)

    def test_healthy_mape_no_flag(self):
        flags = monitoring.evaluate_drift(forecast_mape=3.2, stability_ari=0.7,
                                          hist_mape_avg=3.0)
        assert not any("FORECAST DRIFT" in f for f in flags)

    def test_low_ari_fires_flag(self):
        flags = monitoring.evaluate_drift(forecast_mape=3.0, stability_ari=0.30,
                                          hist_mape_avg=3.0)
        assert any("SEGMENT INSTABILITY" in f for f in flags)

    def test_healthy_ari_no_flag(self):
        flags = monitoring.evaluate_drift(forecast_mape=3.0, stability_ari=0.70,
                                          hist_mape_avg=3.0)
        assert flags == []

    def test_no_history_is_safe(self):
        # first-ever run: no historical average -> no MAPE flag, no crash
        flags = monitoring.evaluate_drift(forecast_mape=99.0, stability_ari=0.7,
                                          hist_mape_avg=None)
        assert not any("FORECAST DRIFT" in f for f in flags)


def test_check_drift_writes_log_and_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(monitoring.registry, "REGISTRY_LOG", tmp_path / "registry.jsonl")

    # two healthy prior forecasting runs + a current one (last is excluded as "current")
    def fc(mape):
        return {"name": "forecasting", "timestamp": "t",
                "metrics": {"winner": "prophet", "prophet": mape}}
    # two healthy priors + a current run (last is excluded as "current")
    entries = [fc(3.0), fc(3.2), fc(40.0)]
    (tmp_path / "registry.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    # current run is degraded (MAPE 40 vs hist avg ~3.1) -> must flag
    result = monitoring.check_drift(forecast_mape=40.0, stability_ari=0.7,
                                    forecast_winner="prophet")
    assert result["flags"], "expected a drift flag on a degraded run"
    log = (tmp_path / "monitoring_log.md").read_text()
    assert "FORECAST DRIFT" in log
