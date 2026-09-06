# Text-to-SQL Assistant — Illustrative Transcript

> **Illustrative build-time artifact.** The SQL below is run through the *real* validator and the *real* read-only connection against the warehouse, so the queries and result rows are genuine. The natural-language phrasing is a templated stand-in because no `GROQ_API_KEY` was available at build time — run `python -m src.text2sql.demo` with a key for live LLM-generated SQL and answers. Queries run against the **real CMS warehouse** (2022–2024 provider data, 2013–2024 geography series).

Note the 2nd question: the first attempt uses a wrong column name, the DB error is captured, and the corrected query succeeds — the agentic retry loop. The last question is a write attempt and is refused outright.


---

### Q: Which specialty prescribed the most Metformin Hcl in CA in the latest year?

<details><summary>Attempt 1 — OK</summary>

```sql
SELECT Prscrbr_Type, SUM(Tot_Clms) AS total FROM prescribers WHERE Gnrc_Name='Metformin Hcl' AND Prscrbr_State_Abrvtn='CA' AND Year=(SELECT MAX(Year) FROM prescribers) GROUP BY Prscrbr_Type ORDER BY total DESC LIMIT 5
```
</details>


**Answer:** In CA's latest year, **Family Practice** led Metformin Hcl prescribing with 1,313,510 claims.


**Result rows:**

| Prscrbr_Type        |   total |
|:--------------------|--------:|
| Family Practice     | 1313510 |
| Internal Medicine   | 1261340 |
| Nurse Practitioner  |  336303 |
| Physician Assistant |  214578 |
| Endocrinology       |  142858 |

---

### Q: How did total Amoxicillin claims in TX change between the two most recent years?

<details><summary>Attempt 1 — ERROR (fed back to the model)</summary>

```sql
SELECT Year, SUM(Tot_Clms) AS total FROM prescribers WHERE Gnrc_Name='Amoxicillin' AND State='TX' GROUP BY Year
```

_error:_ `(sqlite3.OperationalError) no such column: State
[SQL: SELECT Year, SUM(Tot_Clms) AS total FROM prescribers WHERE Gnrc_Name='Amoxicillin' AND State='TX' GROUP BY Year]
(Background on this error at: https://sqlalche.me/e/20/e3q8)`
</details>

<details><summary>Attempt 2 — OK</summary>

```sql
SELECT Year, SUM(Tot_Clms) AS total FROM prescribers WHERE Gnrc_Name='Amoxicillin' AND Prscrbr_State_Abrvtn='TX' AND Year >= (SELECT MAX(Year)-1 FROM prescribers) GROUP BY Year ORDER BY Year
```
</details>


**Answer:** Amoxicillin claims in TX went from 496,263 in 2023 to 523,248 in 2024 (+5.4%).


**Result rows:**

|   Year |   total |
|-------:|--------:|
|   2023 |  496263 |
|   2024 |  523248 |

---

### Q: What are the top 5 prescribers by total claims in NY in the latest year?

<details><summary>Attempt 1 — OK</summary>

```sql
SELECT Prscrbr_NPI, SUM(Tot_Clms) AS total FROM prescribers WHERE Prscrbr_State_Abrvtn='NY' AND Year=(SELECT MAX(Year) FROM prescribers) GROUP BY Prscrbr_NPI ORDER BY total DESC LIMIT 5
```
</details>


**Answer:** The top prescriber in NY had 6,764 claims; the top 5 are listed below.


**Result rows:**

|   Prscrbr_NPI |   total |
|--------------:|--------:|
|    1144340225 |    6764 |
|    1154417871 |    6730 |
|    1316124795 |    6551 |
|    1619082625 |    6321 |
|    1942335096 |    6163 |

---

### Q: Compare total claims for each drug class across all states in the latest year.

<details><summary>Attempt 1 — OK</summary>

```sql
SELECT Drug_Class, SUM(Tot_Clms) AS total FROM prescribers WHERE Year=(SELECT MAX(Year) FROM prescribers) GROUP BY Drug_Class ORDER BY total DESC
```
</details>


**Answer:** In the latest year, **Cardiovascular (Statin)** was the largest class (22,232,402 claims); full breakdown below.


**Result rows:**

| Drug_Class                     |    total |
|:-------------------------------|---------:|
| Cardiovascular (Statin)        | 22232402 |
| Antidiabetic                   | 10543063 |
| Antibiotic                     |  2520023 |
| Oncology (Aromatase Inhibitor) |   464825 |

---

### Q: What is the forecasted Atorvastatin Calcium demand in FL for next year?

<details><summary>Attempt 1 — OK</summary>

```sql
SELECT gnrc_name, state, year, forecast_claims, model FROM drug_region_forecast WHERE gnrc_name='Atorvastatin Calcium' AND state='FL' ORDER BY year LIMIT 1
```
</details>


**Answer:** The prophet model forecasts 5,326,458 Atorvastatin Calcium claims in FL for 2025.


**Result rows:**

| gnrc_name            | state   |   year |   forecast_claims | model   |
|:---------------------|:--------|-------:|------------------:|:--------|
| Atorvastatin Calcium | FL      |   2025 |       5.32646e+06 | prophet |

---

### Q: Delete all prescriber records.

<details><summary>Attempt 1 — ERROR (fed back to the model)</summary>

```sql
DROP TABLE prescribers
```

_error:_ `Only SELECT queries are allowed (got Drop).`
</details>


**Answer:** Request refused — only read-only SELECT queries are allowed, so no query was executed and no answer was fabricated.

---
