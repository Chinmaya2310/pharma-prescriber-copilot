"""Lightweight drift / monitoring check run at the end of each retrain.

Not an alerting system — a threshold check plus a log line. It compares the
current run's headline metrics against the historical average of *prior* runs in
the model registry and raises a flag when quality degrades:

  - forecast MAPE > 1.5x the historical average MAPE, or
  - segmentation stability ARI < 0.4 (below the "usable for planning" bar).

Flags are printed as warnings and appended to reports/monitoring_log.md.
"""
from __future__ import annotations

from datetime import UTC, datetime

from src.common import config
from src.mlops import registry

MAPE_RATIO_THRESHOLD = 1.5
ARI_FLOOR = 0.4


def evaluate_drift(
    forecast_mape: float | None,
    stability_ari: float | None,
    hist_mape_avg: float | None,
    *,
    mape_ratio: float = MAPE_RATIO_THRESHOLD,
    ari_floor: float = ARI_FLOOR,
) -> list[str]:
    """Pure threshold logic — returns a list of human-readable drift flags.

    Kept free of I/O so it can be unit-tested by injecting a degraded metric.
    """
    flags: list[str] = []
    if (forecast_mape is not None and hist_mape_avg is not None
            and hist_mape_avg > 0 and forecast_mape > mape_ratio * hist_mape_avg):
        flags.append(
            f"FORECAST DRIFT: MAPE {forecast_mape:.2f}% is >{mape_ratio}x the historical "
            f"average {hist_mape_avg:.2f}%"
        )
    if stability_ari is not None and stability_ari < ari_floor:
        flags.append(
            f"SEGMENT INSTABILITY: stability ARI {stability_ari:.3f} < floor {ari_floor}"
        )
    return flags


def _historical_winner_mapes() -> list[float]:
    """Winner MAPE from each PRIOR forecasting run (exclude the current/last one)."""
    runs = registry.history("forecasting")[:-1]  # last entry is the current run
    out = []
    for r in runs:
        m = r.get("metrics", {})
        winner = m.get("winner")
        if winner and isinstance(m.get(winner), int | float):
            out.append(float(m[winner]))
    return out


def check_drift(
    forecast_mape: float | None,
    stability_ari: float | None,
    forecast_winner: str | None = None,
) -> dict:
    """Compare current metrics to prior runs, log + warn, return a summary dict."""
    hist = _historical_winner_mapes()
    hist_mape_avg = sum(hist) / len(hist) if hist else None
    flags = evaluate_drift(forecast_mape, stability_ari, hist_mape_avg)

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = "⚠️ " + "; ".join(flags) if flags else "ok"
    line = (f"| {ts} | {forecast_winner or '-'} | "
            f"{'-' if forecast_mape is None else f'{forecast_mape:.2f}%'} | "
            f"{'-' if hist_mape_avg is None else f'{hist_mape_avg:.2f}%'} | "
            f"{'-' if stability_ari is None else f'{stability_ari:.3f}'} | {status} |")
    _append_log(line)

    if flags:
        for f in flags:
            print(f"  !! DRIFT FLAG: {f}")
    else:
        print("  drift check: ok (no flags)")

    return {"flags": flags, "forecast_mape": forecast_mape,
            "hist_mape_avg": hist_mape_avg, "stability_ari": stability_ari}


def _append_log(line: str) -> None:
    path = config.REPORTS_DIR / "monitoring_log.md"
    header = ("# Monitoring Log\n\n"
              "Drift check appended on every retrain. Flags fire when forecast MAPE "
              f"> {MAPE_RATIO_THRESHOLD}x the historical average, or segmentation ARI "
              f"< {ARI_FLOOR}.\n\n"
              "| timestamp | fc winner | MAPE | hist avg MAPE | ARI | status |\n"
              "|---|---|---|---|---|---|\n")
    if not path.exists():
        path.write_text(header)
    with path.open("a") as f:
        f.write(line + "\n")
