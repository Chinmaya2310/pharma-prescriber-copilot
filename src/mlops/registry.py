"""Minimal model registry: timestamped artifacts + an append-only metrics log.

Not production MLflow — just enough to demonstrate the concept: every training run
writes a versioned artifact `models/<name>_<timestamp>.joblib`, updates a
`models/<name>_latest.joblib` pointer, appends a row to `models/registry.jsonl`,
and prunes to the last N versions per model.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from src.common import config

REGISTRY_LOG = config.MODELS_DIR / "registry.jsonl"
KEEP_LAST_N = 5


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def save_model(obj: Any, name: str, metrics: dict | None = None) -> Path:
    ts = _timestamp()
    versioned = config.MODELS_DIR / f"{name}_{ts}.joblib"
    joblib.dump(obj, versioned)

    latest = config.MODELS_DIR / f"{name}_latest.joblib"
    joblib.dump(obj, latest)

    entry = {
        "name": name,
        "timestamp": ts,
        "artifact": versioned.name,
        "metrics": metrics or {},
    }
    with REGISTRY_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    _prune(name)
    return versioned


def load_latest(name: str) -> Any:
    latest = config.MODELS_DIR / f"{name}_latest.joblib"
    if not latest.exists():
        raise FileNotFoundError(f"No trained model '{name}'. Run the training step first.")
    return joblib.load(latest)


def _prune(name: str) -> None:
    versions = sorted(config.MODELS_DIR.glob(f"{name}_*.joblib"))
    versions = [p for p in versions if not p.name.endswith("_latest.joblib")]
    for old in versions[:-KEEP_LAST_N]:
        old.unlink(missing_ok=True)


def history(name: str | None = None) -> list[dict]:
    if not REGISTRY_LOG.exists():
        return []
    rows = [json.loads(line) for line in REGISTRY_LOG.read_text().splitlines() if line.strip()]
    return [r for r in rows if name is None or r["name"] == name]


def main() -> None:
    for row in history():
        print(f"{row['timestamp']}  {row['name']:<22} {row['metrics']}")


if __name__ == "__main__":
    main()
