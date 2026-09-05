# Pharma Prescriber Analytics & Forecasting Copilot

An end-to-end analytics pipeline on **real CMS Medicare Part D prescriber data**:
it cleans and integrates the data, explores it, segments prescribers by behaviour,
forecasts drug-level demand, serves everything through a FastAPI service and a
Streamlit dashboard, and adds an **agentic text-to-SQL assistant** so a sales-ops
user can ask plain-English questions and get back **real, auditable numbers** —
no hallucination, no RAG.

> **Why no RAG?** The questions this data invites ("which specialty prescribes the
> most X in CA", "how did claims change year over year") are structured
> aggregations — a `GROUP BY`, not a document search. So the assistant writes SQL,
> runs it read-only, and answers from the actual rows. See
> [`DECISIONS.md`](DECISIONS.md#2-why-text-to-sql-and-not-rag-for-the-assistant).

---

## ⚠️ Data access note (read once)

`data.cms.gov` blocks some datacenter IPs with an Akamai 403. The real download
script works from a normal network; where CMS is unreachable, a **clearly-labelled
synthetic fixture** (identical schema, real suppression behaviour) lets the whole
pipeline run and CI pass. Synthetic-derived numbers are labelled as such
everywhere. Full story in [`DECISIONS.md` §0](DECISIONS.md#0-data-access-blocker--the-synthetic-fixture-read-this-first).

---

## Architecture

```
CMS raw CSV ──▶ [1] Ingestion & Cleaning ──▶ prescribers.parquet ──▶ SQLite warehouse
                                                     │
        ┌────────────────────────────┬──────────────┴───────────────┐
        ▼                            ▼                               ▼
 [2] EDA (saved charts)   [3] Segmentation (KMeans)      [4] Forecasting (Prophet vs
                          → prescriber_segments             XGBoost, walk-forward CV)
                                                          → drug_region_forecast
        └────────────────────────────┴──────────────┬───────────────┘
                                                     ▼
                                     [5] FastAPI  ──▶  [6] Streamlit dashboard
                                        /segments/{npi}
                                        /forecast/{drug}/{region}
                                        /ask   ← [7] agentic text-to-SQL
                                                     ▲
                                     [8] MLOps: retrain.py · model registry · GitHub Actions
```

## Quickstart (under 10 minutes)

```bash
# 1. Environment (Python 3.11). Using conda:
conda create -y -n pharma python=3.11 && conda activate pharma
pip install -r requirements.txt
# xgboost needs an OpenMP runtime:
#   mac/conda:  conda install -c conda-forge llvm-openmp
#   ubuntu:     sudo apt-get install -y libgomp1
# Prophet needs CmdStan (one-time compile):
python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.33.1')"

# 2. Get data + run the whole pipeline (ingest → clean → EDA → segment → forecast)
python -m src.mlops.retrain --download     # real CMS data (needs CMS-reachable network)
#   or, if CMS is blocked / for a quick local run:
python -m src.mlops.retrain --synthetic    # labelled synthetic fixture

# 3. Serve the API
uvicorn src.api.main:app --reload          # docs at http://127.0.0.1:8000/docs

# 4. Dashboard (new terminal, same env)
streamlit run dashboard/app.py

# 5. (Optional) the text-to-SQL assistant needs an Anthropic key
cp .env.example .env    # then add ANTHROPIC_API_KEY
python -m src.text2sql.demo                # writes reports/text2sql_transcript.md
```

## Running the pieces individually

| Step | Command |
|---|---|
| Download real CMS data | `python -m src.ingestion.download_cms` |
| Generate synthetic fixture | `python -m src.ingestion.make_synthetic` |
| Clean | `python -m src.ingestion.clean` |
| Load warehouse | `python -m src.ingestion.load_db` |
| EDA (charts + summary) | `python -m src.eda.eda` |
| Segmentation | `python -m src.segmentation.train_kmeans` |
| Forecast compare + persist | `python -m src.forecasting.compare` |
| Model registry log | `python -m src.mlops.registry` |
| Tests | `pytest -q` |
| Lint | `ruff check src tests` |

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/` | liveness / index |
| GET | `/kpis` | top-line numbers for the dashboard |
| GET | `/segments/{npi}` | one prescriber's segment + profile |
| GET | `/segments/summary` | segment sizes & average profiles |
| GET | `/forecast/{drug}/{region}` | forward demand forecast |
| POST | `/ask` | agentic text-to-SQL (answer + SQL + rows + trace) |

## What's produced

- `data/processed/prescribers.parquet` — cleaned fact table
- `data/processed/warehouse.db` — SQLite warehouse (facts, segments, forecasts)
- `reports/data_quality.md`, `reports/eda_summary.md`,
  `reports/forecast_comparison.md`, `reports/text2sql_transcript.md`
- `reports/figures/*.png` — EDA, silhouette, forecast charts
- `models/*.joblib` + `models/registry.jsonl` — versioned models & metrics

## Documentation

- [`DECISIONS.md`](DECISIONS.md) — every design choice + why (the interview file)
- [`BUSINESS_CASE.md`](BUSINESS_CASE.md) — territory / incentive recommendation
- reports listed above — the data-driven outputs

## Tech
Python 3.11 · pandas/pyarrow · scikit-learn · Prophet + XGBoost · SQLAlchemy/SQLite ·
FastAPI · Streamlit · Anthropic API · sqlglot · pytest · ruff · GitHub Actions

## Note on Tableau/Power BI
The brief targets Tableau/Power BI; those aren't scriptable in this repo, so the
dashboard is **Streamlit** as a functional, reproducible stand-in that reads
through the same API an external BI tool would use.
