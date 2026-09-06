# Pharma Prescriber Analytics & Forecasting Copilot

An end-to-end analytics pipeline on **real CMS Medicare Part D prescriber data**:
it cleans and integrates the data, explores it, segments prescribers by behaviour,
**classifies** each prescriber's likely next-period growth, **forecasts** drug-level
demand (Prophet vs XGBoost vs LSTM, walk-forward CV), serves everything through a
FastAPI service and a Streamlit dashboard, and adds an **agentic text-to-SQL
assistant** so a sales-ops user can ask plain-English questions and get back
**real, auditable numbers** — no hallucination, no RAG.

> **Why no RAG?** The questions this data invites ("which specialty prescribes the
> most X in CA", "how did claims change year over year") are structured
> aggregations — a `GROUP BY`, not a document search. So the assistant writes SQL,
> runs it read-only, and answers from the actual rows. See
> [`DECISIONS.md`](DECISIONS.md#2-why-text-to-sql-and-not-rag-for-the-assistant).

---

## Data

Built on the **real CMS Medicare Part D Prescribers** public data, using two
datasets each at the grain its task needs:

- **by Geography and Drug** → state-level demand series, **2013–2024**, for forecasting
- **by Provider and Drug** → prescriber-level detail, **2022–2024**, for segmentation

Scope: 4 drugs (Metformin, Atorvastatin, Amoxicillin, Anastrozole) × 4 states
(CA, TX, NY, FL). Real cleaned volume: **518,946 provider-drug-year rows, 187,436
prescribers, 69.7M claims.**

> **Note on access:** `data.cms.gov` geo-restricts to US IPs (Akamai 403 otherwise).
> From a US network the download just works. From elsewhere, set `CMS_PROXY_LIST` to
> US HTTP proxies (the downloader rotates through them; TLS stays end-to-end). CI has
> no US egress, so it runs on a clearly-labelled **synthetic fixture** — any report
> showing a "SYNTHETIC" banner came from that path, not real data. Full story in
> [`DECISIONS.md` §0](DECISIONS.md#0-data-access-real-cms-data-and-the-geo-block-we-worked-around).

---

## Architecture

```
CMS by-Geography ─▶ clean_geography ─▶ forecast_series.parquet ─┐
CMS by-Provider  ─▶ clean ─▶ prescribers.parquet ─▶ SQLite ─────┤
                                     │                          │
        ┌────────────────────────────┼──────────────┐          │
        ▼                            ▼               ▼          ▼
 [2] EDA (saved charts)   [3] Segmentation    [4] Forecasting (Prophet vs
                          (KMeans)             XGBoost, walk-forward CV)
                          → prescriber_segments → drug_region_forecast
        └────────────────────────────┴──────────────┬───────────┘
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
python -m src.mlops.retrain --download     # real CMS data (US network; downloads
                                           # by-Geography all years + by-Provider recent years)
#   from a geo-blocked (non-US) network, set US proxies first:
#   export CMS_PROXY_LIST="ip:port,ip:port,..."
#   or, for a quick offline/CI run on the labelled synthetic fixture:
python -m src.mlops.retrain --synthetic

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
| Download real provider data | `python -m src.ingestion.download_cms --years 2022 2023 2024` |
| Download real geography data | `python -m src.ingestion.download_geography` |
| Generate synthetic fixture | `python -m src.ingestion.make_synthetic` |
| Clean provider | `python -m src.ingestion.clean` |
| Build forecast series (geo) | `python -m src.ingestion.clean_geography` |
| Load warehouse | `python -m src.ingestion.load_db` |
| EDA (charts + summary) | `python -m src.eda.eda` |
| Segmentation | `python -m src.segmentation.train_kmeans` |
| Forecast compare (Prophet/XGBoost/LSTM) | `python -m src.forecasting.compare` |
| Growth classification | `python -m src.classification.train` |
| Model registry log | `python -m src.mlops.registry` |
| Export deploy data | `python -m scripts.export_deploy_data` |
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
| GET | `/predict/{npi}` | predicted next-period growth class (+ probabilities) |
| POST | `/ask` | agentic text-to-SQL (answer + SQL + rows + trace) |

## What's produced

- `data/processed/prescribers.parquet` — cleaned provider fact table
- `data/processed/forecast_series.parquet` — state-level annual demand series (by-Geography)
- `data/processed/warehouse.db` — SQLite warehouse: `prescribers`, `geo_drug_year`,
  `prescriber_segments`, `drug_region_forecast`
- `reports/data_quality.md`, `reports/eda_summary.md`,
  `reports/forecast_comparison.md`, `reports/text2sql_transcript.md`
- `reports/figures/*.png` — EDA, silhouette, forecast charts
- `models/*.joblib` + `models/registry.jsonl` — versioned models & metrics

## Live deployment

The API is deployed on Render's free tier (serving-only image; builds the warehouse
from committed ~14 MB parquets at start — no torch/prophet at runtime):

- **Live URL:** _pending first deploy_ (see `render.yaml`; will be filled in once live)
- Local development instructions above still work unchanged.

Deploy config: [`render.yaml`](render.yaml) · serving deps: `requirements-api.txt` ·
data rebuild: `scripts/build_deploy_db.py` (from `data/deploy/*.parquet`).

## Monitoring

Every retrain appends a drift check to [`reports/monitoring_log.md`](reports/monitoring_log.md):
it flags if forecast MAPE exceeds 1.5× the historical average or segmentation ARI
drops below 0.4 (`src/mlops/monitoring.py`).

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
