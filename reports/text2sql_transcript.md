# Text-to-SQL Assistant — Illustrative Transcript

> **Illustrative build-time artifact.** The SQL below is run through the *real* validator and the *real* read-only connection against the warehouse, so the queries and result rows are genuine. The natural-language phrasing is a templated stand-in because no `ANTHROPIC_API_KEY` was available at build time — run `python -m src.text2sql.demo` with a key for live Claude-generated SQL and answers. Data is the synthetic fixture (see DECISIONS.md §0).

Note the 2nd question: the first attempt uses a wrong column name, the DB error is captured, and the corrected query succeeds — the agentic retry loop. The last question is a write attempt and is refused outright.


---

### Q: Which specialty prescribed the most Metformin Hcl in CA in the latest year?

<details><summary>Attempt 1 — OK</summary>

```sql
SELECT Prscrbr_Type, SUM(Tot_Clms) AS total FROM prescribers WHERE Gnrc_Name='Metformin Hcl' AND Prscrbr_State_Abrvtn='CA' AND Year=(SELECT MAX(Year) FROM prescribers) GROUP BY Prscrbr_Type ORDER BY total DESC LIMIT 5
```
</details>


**Answer:** In CA's latest year, **Endocrinology** led Metformin Hcl prescribing with 3,859 claims.


**Result rows:**

| Prscrbr_Type        |   total |
|:--------------------|--------:|
| Endocrinology       |    3859 |
| Family Practice     |    3077 |
| Internal Medicine   |    2612 |
| General Practice    |    1801 |
| Physician Assistant |     927 |

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


**Answer:** Amoxicillin claims in TX went from 3,102 in 2022 to 3,664 in 2023 (+18.1%).


**Result rows:**

|   Year |   total |
|-------:|--------:|
|   2022 |    3102 |
|   2023 |    3664 |

---

### Q: What are the top 5 prescribers by total claims in NY in the latest year?

<details><summary>Attempt 1 — OK</summary>

```sql
SELECT Prscrbr_NPI, SUM(Tot_Clms) AS total FROM prescribers WHERE Prscrbr_State_Abrvtn='NY' AND Year=(SELECT MAX(Year) FROM prescribers) GROUP BY Prscrbr_NPI ORDER BY total DESC LIMIT 5
```
</details>


**Answer:** The top prescriber in NY had 2,921 claims; the top 5 are listed below.


**Result rows:**

|   Prscrbr_NPI |   total |
|--------------:|--------:|
|    1000000147 |    2921 |
|    1000000188 |    2405 |
|    1000000195 |    1728 |
|    1000000167 |    1632 |
|    1000000186 |    1591 |

---

### Q: Compare total claims for each drug class across all states in the latest year.

<details><summary>Attempt 1 — OK</summary>

```sql
SELECT Drug_Class, SUM(Tot_Clms) AS total FROM prescribers WHERE Year=(SELECT MAX(Year) FROM prescribers) GROUP BY Drug_Class ORDER BY total DESC
```
</details>


**Answer:** In the latest year, **Cardiovascular (Statin)** was the largest class (62,126 claims); full breakdown below.


**Result rows:**

| Drug_Class                     |   total |
|:-------------------------------|--------:|
| Cardiovascular (Statin)        |   62126 |
| Antidiabetic                   |   48097 |
| Antibiotic                     |   18562 |
| Oncology (Aromatase Inhibitor) |     875 |

---

### Q: What is the forecasted Atorvastatin Calcium demand in FL for next year?

<details><summary>Attempt 1 — OK</summary>

```sql
SELECT gnrc_name, state, year, forecast_claims, model FROM drug_region_forecast WHERE gnrc_name='Atorvastatin Calcium' AND state='FL' ORDER BY year LIMIT 1
```
</details>


**Answer:** The prophet model forecasts 15,557 Atorvastatin Calcium claims in FL for 2024.


**Result rows:**

| gnrc_name            | state   |   year |   forecast_claims | model   |
|:---------------------|:--------|-------:|------------------:|:--------|
| Atorvastatin Calcium | FL      |   2024 |           15557.2 | prophet |

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
