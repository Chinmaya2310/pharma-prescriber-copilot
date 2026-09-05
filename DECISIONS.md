# DECISIONS.md

Every non-obvious choice in this project, with the reasoning and the alternatives
considered. This is the file to read before an interview — each entry is written
so you can defend the decision out loud.

---

## 0. Data access: real CMS data (and the geo-block we worked around)

**All numbers in this repo are from the real CMS data** (data years 2013–2024).
Getting it took some work worth knowing about:

`data.cms.gov` sits behind an Akamai WAF that **geo-restricts to US IPs** — from a
non-US IP it returns HTTP 403 "Access Denied" regardless of User-Agent (confirmed:
the 403 was identical for our custom UA and a full Chrome UA; general internet was
fine). Because the build ran on a non-US connection, the download was initially
blocked. It was unblocked by routing requests through **US HTTP proxies** (the
`CMS_PROXY_LIST` env var enables proxy rotation in `download_cms.py`); since CMS is
HTTPS, a CONNECT proxy only tunnels encrypted bytes, so TLS stays end-to-end and
the data can't be read or tampered with in transit. **From a US network none of
this is needed** — `retrain.py --download` just works.

**The synthetic fixture stays — but only for CI.** `src/ingestion/make_synthetic.py`
generates a clearly-labelled fixture (`SYNTHETIC_*` filenames) with the identical
CMS schema and real suppression semantics. CI has no US egress and shouldn't depend
on flaky public proxies, so the GitHub Actions pipeline runs on the synthetic
fixture to exercise the code end-to-end. **Every report regenerated from real data
has its synthetic banner removed;** if you ever see a "SYNTHETIC" banner in a
report, that report was produced by the CI/fixture path, not real data.

**Reproducing the real run:** from a US network, `python -m src.mlops.retrain
--download`. From a geo-blocked network, set `CMS_PROXY_LIST=ip:port,ip:port,...`
to US proxies first. Provider data is large (~600k scoped rows for 3 years), so the
download is chunked and resumable.

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

**States.** CA, TX, NY, FL — large, high-density, geographically spread. Real
scoped volume is substantial: **518,946 cleaned provider-drug-year rows, 187,436
distinct prescribers, 69.7M total claims.**

**Years — the biggest deviation from the brief, and why it's right.** The brief
suggested "the 2 most recent years." The CMS files are **annual**, so two years =
**two data points** per series — far too few for walk-forward time-series CV. The
resolution is the two-dataset design in §1a: forecasting uses the full **2013–2024**
history at state grain; segmentation uses the **most recent years** at provider
grain (2023–2024 for features; 2022 added so an *earlier* window exists for the
stability check). This isn't a shortcut — two annual points genuinely cannot
support rolling-origin validation, and CMS only publishes annually.

---

## 1a. Two datasets, each at the grain its task needs (key design choice)

Rather than pull ~1M provider rows and aggregate them up for forecasting, the
project uses **two different CMS datasets**, each at the natural grain of the task:

| Task | CMS dataset | Grain | Why |
|---|---|---|---|
| **Forecasting** | Part D Prescribers **by Geography and Drug** | state × drug × year, **2013–2024** | CMS already publishes state-level totals; this is exactly the demand series to forecast, and gives a full 12-year annual history for walk-forward CV. Tiny to download (~192 rows). |
| **Segmentation** | Part D Prescribers **by Provider and Drug** | prescriber × drug × year, **2022–2024** | Segmentation needs per-prescriber behaviour, which only the provider file has. Only recent years are needed, so we don't pull the full multi-GB history. |

This is the single most defensible design decision in the project: *use the
aggregate dataset for the aggregate question and the granular dataset for the
granular question.* It also makes the download ~10× smaller and the forecast series
cleaner (state totals aren't distorted by the provider-level <11-claim omission).

> Phase 0 scope (drugs/states) and the Phase 4 default model were decided
> autonomously (the user was AFK) and are documented here; both are one-line config
> changes. The forecast winner is chosen from CV at runtime, so it was never
> hardcoded.

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

On the real data, **25.9% of provider-drug rows have a suppressed `Tot_Benes`** —
a large fraction, which is exactly why zero-filling would have been so damaging.

**Per-stage policy (the brief asked for an explicit, stated choice):**
- **Segmentation:** features are built from published claim counts (all ≥ 11, so
  real). A prescriber with no prior-year row gets `growth = 0` + `has_history =
  False` — we don't invent a huge growth ratio off an unknown base.
- **Forecasting:** uses the **by-Geography** state totals, where the <11-claim
  provider omission barely matters (state totals are in the millions). Series with
  too few annual points are still **excluded and counted**
  (`MIN_YEARS_FOR_FORECAST = 5`, reported in `reports/forecast_comparison.md`); on
  the real data **0 of 16** (drug, state) series were excluded — all have the full
  12-year history. Exclusion beats imputation because imputing a censored regional
  total would fabricate the signal we're forecasting.
- **Brand aggregation:** CMS lists a separate row per `Brnd_Name` (e.g. "Metformin
  Hcl" and "Metformin Hcl Er" under one `Gnrc_Name`). Cleaning **sums** claims/
  fills/supply/cost across formulations to the (provider, generic, year) grain —
  dropping duplicates would undercount high-volume prescribers. (Real data collapsed
  607,018 raw rows → 518,946.)

---

## 4. Prophet vs XGBoost — and which is the default

Both are built (`src/forecasting/models.py`) behind one interface and compared with
**rolling-origin (walk-forward) cross-validation** (`compare.py`): expanding
window, 1-year horizon, pooled across all eligible series, scored by MAPE and RMSE.

**Why walk-forward, not a single split:** with a short annual series a single
train/test split can crown a model by luck. Re-evaluating at every feasible origin
and averaging is the standard honest way to validate a time-series model.

**Default model:** whichever wins the CV **at runtime** — the API/dashboard read
the winner from `drug_region_forecast.model`, they don't hardcode it. On the **real
data, Prophet won: MAPE 3.09% vs XGBoost 5.26%** (RMSE 71,129 vs 155,589), pooled
over **128 walk-forward folds across 16 (drug, state) series, 2013–2024**. The low
MAPE reflects that real state-level annual demand is smooth and strongly trending,
which Prophet's additive trend captures well; XGBoost's lag features are at a
disadvantage on short annual series. The winner is re-computed every retrain, so it
was never hardcoded — on the synthetic CI fixture the gap is different but Prophet
also wins there.

**Why these two, not ARIMA / LSTM:**
- *ARIMA* needs longer, ideally stationary series and per-series order selection;
  clumsy across many short annual series and weaker with exogenous structure.
- *LSTM* is heavily over-parameterised for ~12 annual points per series — it would
  overfit and be hard to defend. Prophet (interpretable additive trend) and
  XGBoost (lag features, handles non-linearity) are the right complexity for this
  data size, and they make a genuinely informative comparison.

---

## 5. Why KMeans (not DBSCAN / hierarchical)

- The goal is a **fixed, interpretable set of prescriber segments** for territory
  and incentive planning — KMeans gives exactly that, with named groups sized for
  sales coverage. On the real data: **k = 8** chosen by silhouette (score 0.566),
  over **165,894 prescribers** (latest two years, 2023–2024).
- **k is chosen, not guessed:** silhouette score over k ∈ [2, 8]
  (`reports/figures/segmentation_silhouette.png`). Because silhouette is O(n²), it
  is scored on a 10k random sample while KMeans still fits on all 165k prescribers
  — the standard scalable approach.
- **DBSCAN** targets density-based clusters + outliers; on standardised,
  fairly convex behavioural features it tends to dump most prescribers into one
  cluster plus noise — not useful for "assign every rep a book of business."
- **Hierarchical** is O(n²) memory and gives a dendrogram to cut arbitrarily; no
  advantage here over KMeans + silhouette.
- **Stability is verified, not assumed:** we re-fit segmentation on an earlier
  window (2022–2023) and measure Adjusted Rand Index against the current (2023–2024)
  segments. **On the real data ARI = 0.704** — comfortably above the ~0.4 bar for
  "stable enough to plan against," so a rep's book built on these segments won't
  reshuffle randomly next year. (This is exactly why the 2022 provider year was
  pulled in addition to 2023–2024: without a third year there is no earlier window
  to test against.)

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
