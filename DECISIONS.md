# DECISIONS.md

Every non-obvious choice in this project, with the reasoning and the alternatives
considered. This is the file to read before an interview — each entry is written
so you can defend the decision out loud.

---

## 0. Data-access blocker & the synthetic fixture (read this first)

**What happened.** `data.cms.gov` sits behind an Akamai WAF that returns HTTP 403
"Access Denied" to the datacenter IP this project was built on. General internet
worked (GitHub, example.com returned 200); only CMS hosts were blocked. So the
**real CMS file could not be downloaded in the build environment.**

**How it's handled — honestly.**
- `src/ingestion/download_cms.py` is **real and correct**. It discovers each
  year's dataset from the CMS `data.json` catalogue and pulls only the scoped
  rows via the data-api's server-side filters. It works from any network CMS
  doesn't block (e.g. a home/office connection). Run it with `--download`.
- For local dev / CI, `src/ingestion/make_synthetic.py` generates a
  **clearly-labelled** fixture (`SYNTHETIC_*` filenames, banners in every report)
  with the **identical CMS schema** and the **real suppression semantics** (see §3).
- **Nothing synthetic is presented as a real finding.** The data-quality, EDA, and
  business-case documents carry a synthetic banner until the pipeline is re-run on
  real data. The numbers quoted in this repo today come from the fixture.

**Why not just stop?** The brief said to flag blockers rather than *quietly*
substitute. This is the loud version: the substitution is explicit, labelled, and
reversible with one command (`retrain.py --download`). Everything else in the
project — cleaning, EDA, ML, API, dashboard, agent, MLOps, tests — is real and
runs today.

---

## 1. Scope: which drugs / states / years (Phase 0)

**Drugs (filtered on `Gnrc_Name`).** Chosen for *behavioural variety* so the
forecasting and segmentation have distinct dynamics to find:

| Generic | Class | Expected dynamics |
|---|---|---|
| Metformin Hcl | Antidiabetic | high volume, steady growth |
| Atorvastatin Calcium | Cardiovascular (statin) | very high volume, mild growth |
| Amoxicillin | Antibiotic | episodic / trend-sensitive |
| Anastrozole | Oncology (aromatase inhibitor) | niche, slow decline |

Filtering on generic (not brand) captures both brand and generic dispensing of the
same molecule.

**States.** CA, TX, NY, FL — large, high-density, geographically spread, so
per-state series have enough volume to model and enough difference to compare.

**Years — and the biggest deviation from the brief.** The brief suggested "the 2
most recent years." The CMS *by Provider and Drug* file is **annual** (one row per
provider-drug-year), so two years = **two data points** per series, which cannot
support walk-forward time-series validation. Decision:
- **Segmentation** uses the **2 most recent years** (needs a prior year to compute
  a per-prescriber growth feature) — as the brief intended.
- **Forecasting** uses the **full available history (2013–latest)**, aggregated to
  (drug, state, year), so there are ~10 annual points per series — enough for
  rolling-origin CV.

This split is deliberate and is the kind of thing worth raising in an interview:
the brief's instinct (small scope) was right for the provider-level work but the
annual grain forces a longer window for forecasting.

> ⚠️ Since I couldn't reach the user while they were AFK, I made the Phase 0 scope
> call and the Phase 4 default-model call myself and documented both here. Both are
> one-line changes if you'd choose differently (`src/common/config.py` for scope).

---

## 2. Why text-to-SQL and NOT RAG for the assistant

This is the headline design decision and the one most likely to be probed.

The questions this data invites — "which specialty prescribes the most X in CA",
"how did Amoxicillin claims change year over year", "top prescribers by volume" —
are **structured aggregations over a known schema**: filter, group, compare. That
is exactly what SQL is for.

- **RAG answers the wrong kind of question.** Retrieval-augmented generation shines
  when the answer lives in unstructured text scattered across documents. Here the
  answer is a `GROUP BY`. Embedding rows and retrieving the "most similar" ones
  would be a lossy, non-deterministic approximation of an operation SQL does
  exactly and verifiably.
- **Auditability.** A SQL answer ships with the exact query and the exact rows.
  Anyone can re-run the query and get the same number. A RAG answer is a
  probabilistic summary of retrieved chunks — much harder to audit, and prone to
  averaging/hallucinating numbers.
- **No vector store to drift or maintain.** Adding embeddings here would be
  complexity for its own sake — "GenAI because the JD said so."

**When RAG *would* be right** (documented limitation, not a TODO): if the task were
"summarise the FDA label / clinical guidance for this drug" or "what do the free-
text prescriber notes say", that's unstructured retrieval and RAG would be the
correct tool. This dataset has none of that, so it's out of scope by design.

---

## 3. Handling CMS suppression (censored / missing-not-at-random)

CMS applies two privacy rules, both handled explicitly (`src/ingestion/clean.py`):

1. **Rows with < 11 claims are omitted from the file entirely.** This is
   *missing-not-at-random*: low-volume prescribing is systematically invisible. We
   cannot recover it, so we **state the consequence** (per-provider and per-region
   totals are biased low at the small-volume tail) in `reports/data_quality.md`
   rather than pretending the data is complete.
2. **`Tot_Benes` is blanked when the true count is 1–10.** We keep the row, set
   `Tot_Benes = NaN`, and flag `Benes_Suppressed = True`. **We never zero-fill** —
   zero-filling would treat "1–10 patients" as "0 patients" and bias every
   beneficiary-based metric downward.

**Per-stage policy (the brief asked for an explicit, stated choice):**
- **Segmentation:** features are built from published claim counts (all ≥ 11, so
  real). A prescriber with no prior-year row gets `growth = 0` + `has_history =
  False` — we don't invent a huge growth ratio off an unknown base.
- **Forecasting:** we **exclude and count** (drug, state) series with too few
  annual points (`MIN_YEARS_FOR_FORECAST = 5`), which is where heavy suppression
  bites hardest. The excluded count is reported in `reports/forecast_comparison.md`.
  We chose exclusion over imputation because imputing a censored regional total
  would fabricate the very signal we're trying to forecast.

---

## 4. Prophet vs XGBoost — and which is the default

Both are built (`src/forecasting/models.py`) behind one interface and compared with
**rolling-origin (walk-forward) cross-validation** (`compare.py`): expanding
window, 1-year horizon, pooled across all eligible series, scored by MAPE and RMSE.

**Why walk-forward, not a single split:** with a short annual series a single
train/test split can crown a model by luck. Re-evaluating at every feasible origin
and averaging is the standard honest way to validate a time-series model.

**Default model:** whichever wins the CV **at runtime** — the API/dashboard read
the winner from `drug_region_forecast.model`, they don't hardcode it. On the
current *synthetic* run **Prophet won** (MAPE ≈ 18.2% vs XGBoost ≈ 22.7%, pooled
over 112 walk-forward folds); both are evaluated head-to-head every run. On real
data this can flip, which is exactly why the choice is data-driven and re-computed
on every retrain rather than fixed — so it was never a call I needed to hardcode.

**Why these two, not ARIMA / LSTM:**
- *ARIMA* needs longer, ideally stationary series and per-series order selection;
  clumsy across many short annual series and weaker with exogenous structure.
- *LSTM* is heavily over-parameterised for ~10 annual points per series — it would
  overfit and be hard to defend. Prophet (interpretable additive trend) and
  XGBoost (lag features, handles non-linearity) are the right complexity for this
  data size, and they make a genuinely informative comparison.

---

## 5. Why KMeans (not DBSCAN / hierarchical)

- The goal is a **fixed, interpretable set of prescriber segments** for territory
  and incentive planning — KMeans gives exactly that, with named, roughly balanced
  groups sized for sales coverage.
- **k is chosen, not guessed:** silhouette score over k ∈ [2, 8]
  (`reports/figures/segmentation_silhouette.png`).
- **DBSCAN** targets density-based clusters + outliers; on standardised,
  fairly convex behavioural features it tends to dump most prescribers into one
  cluster plus noise — not useful for "assign every rep a book of business."
- **Hierarchical** is O(n²) memory and gives a dendrogram to cut arbitrarily; no
  advantage here over KMeans + silhouette.
- **Stability is verified, not assumed:** we re-fit on an earlier year window and
  measure Adjusted Rand Index against the current segments. A segmentation that
  reshuffles randomly period-to-period is useless for planning. (On the synthetic
  fixture ARI ≈ 0.25 — moderate; on real data this is the number to watch, and
  >~0.4 is the bar for "stable enough to plan against.")

---

## 6. Storage: SQLite (not Postgres)

At this scale (tens of thousands of scoped rows) SQLite is a single-file,
zero-ops warehouse that's perfect for a portfolio project and CI. The data layer
(`src/common/db.py`) is SQLAlchemy, so moving to Postgres is a URL change, not a
rewrite. SQLite also gives us a clean, cheap **read-only connection** (`mode=ro`)
for the assistant — see §7.

---

## 7. Text-to-SQL safety: validate AND restrict (defense in depth)

Two independent layers, because either alone is insufficient:

1. **Validator** (`validate_sql.py`): parses the SQL with `sqlglot`, requires
   exactly one statement, requires a read shape (SELECT / CTE / set-op), rejects
   any mutating/administrative node, and runs a keyword screen (with string
   literals stripped to avoid false positives). A prompt-injected
   `SELECT 1; DROP TABLE prescribers` is rejected here — proven by
   `tests/test_validate_sql.py`, not by hoping the LLM behaves.
2. **Read-only connection** (`db.get_readonly_engine`, SQLite `mode=ro`): even if a
   mutation slipped past the parser, the driver itself refuses the write.

We deliberately don't rely on the LLM "promising" to only read.

**The agentic loop** (`agent_loop.py`) is plan → act → observe → retry: generate
SQL → validate+execute → on error feed the DB message back to the model to fix its
own query (up to 3 attempts) → answer strictly from the returned rows. If all
attempts fail it returns the trace and **no fabricated answer**. Grounding is
enforced in `answer.py`: the model is told to use only the result rows and to say
what's missing otherwise.

---

## 8. Streamlit (not Tableau / Power BI / Dash)

Tableau and Power BI aren't scriptable/reproducible in this environment, so the
dashboard is Streamlit as an explicit stand-in (stated in the README). It reads
**through the API**, not the DB directly, so it exercises the same contract an
external BI tool would. Dash was the alternative; Streamlit is faster to build and
enough for a functional analytics UI.

---

## 9. LLM client abstraction (testability)

`text2sql/llm.py` defines an `LLMClient` Protocol. Production uses `AnthropicClient`
(reads `ANTHROPIC_API_KEY`); tests inject a scripted fake, so the agent loop,
retry behaviour, and injection defense are all tested **without a network call or
an API key**. The API endpoint returns a clear 503 when the key is absent instead
of crashing.
