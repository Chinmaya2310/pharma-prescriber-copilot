# BUSINESS_CASE.md — Territory Design & Incentive Compensation

All figures below are from the **real CMS Medicare Part D data** (provider file
2022–2024; geography file 2013–2024; scope: Metformin, Atorvastatin, Amoxicillin,
Anastrozole across CA, TX, NY, FL).

## The question a sales-ops leader is actually asking

> "I have a fixed field force. Which prescribers should each rep own, and how do I
> pay reps so their effort goes where the growth is — not just where the volume
> already is?"

This project answers that with two artifacts it produces from real data:
**behavioural segments** (who prescribes, how much, and whether they're growing)
and **demand forecasts** (where each drug's market is heading by state).

## Step 1 — Segments give you the unit of territory design

KMeans over 165,894 prescribers (2023–2024) yields 8 interpretable segments.
**The segmentation is stable (Adjusted Rand Index 0.704 vs the 2022–2023 window)**,
so a book of business built on it won't reshuffle next year — the precondition for
using it to plan territories at all.

| Segment | Prescribers | Avg claims/yr | Avg growth | Play |
|---|---|---|---|---|
| High-volume broad statins-leaning | 14,011 | ~1,061 | +11% | **Flagship accounts** — protect & deepen |
| High-volume focused statins-leaning | 58,177 | ~246 | +12% | **Core** — largest cohort, the volume base |
| High-volume focused statins (fast risers) | 5,014 | ~217 | **+266%*** | **Invest** — explosive growth off a low base |
| High-volume focused diabetes-leaning | 24 | ~309 | +15% | Niche high-volume diabetes |
| Low-volume focused antibiotics-leaning | 43,594 | ~48 | +9% | **Develop** — broad, low-touch |
| Low-volume focused diabetes-leaning | 5,738 | ~130 | +4% | Develop (diabetes) |
| Low-volume focused oncology-leaning | 4,856 | ~86 | +15% | **Specialist** — oncology niche |
| Low-volume focused statins (steady) | 34,480 | ~65 | ~0% | **Maintain** — low priority |

\*growth ratio capped at +500%; this cohort is ramping from a small prior-year base.

**Why this matters:** volume alone would send every rep to the ~1,061-claims/yr
flagship accounts. Segments separate **volume** from **momentum**, so books can be
balanced across *protect* (flagship/core) and *capture* (fast-riser/develop)
accounts instead of one rep inheriting all the easy volume.

## Step 2 — Forecasts tell you where to move capacity

`drug_region_forecast` (Prophet, walk-forward CV **MAPE 3.09%**) projects each
(drug, state) forward. The dominant, still-growing market is **atorvastatin**
(statins), e.g. California: **7.84M claims in 2024 → 8.19M forecast for 2025
(+4.4%)**, on top of **+165% growth since 2013**. Amoxicillin shows a clear
**2020 COVID dip** then recovery — a real, explainable pattern a quota model must
not mistake for a trend break. Metformin grows steadily (+37% since 2013);
anastrozole (oncology) is a smaller, steadily-growing niche (+51%).

**Implication:** statins deserve the largest and rising quota allocation; the
oncology niche is best covered by a shared specialist rather than dedicated field
time; antibiotic quotas should be set off the post-2020 recovery baseline, not the
2019 peak.

## Step 3 — The recommendation

**A. Territory redesign — segment-balanced books.**
Build each rep's book to balance total *opportunity* (current volume + forecasted
growth), not raw current volume. Every book gets a mix of Flagship/Core (protect
the ~72k high-volume statin prescribers) and Fast-riser/Develop accounts (the ~5k
explosive risers + ~54k low-volume growers). This prevents "lucky" territories that
are just inherited volume and distributes developmental work fairly.

**B. Incentive compensation — pay for momentum, not just level.**
- ~70% on attainment vs a **forecast-based quota** (per drug×state from
  `drug_region_forecast`, so targets track where the market is actually going);
- ~30% on **growth within the Fast-riser + Develop segments**, so reps are paid to
  convert the low-volume-growing and rapidly-ramping prescribers — where the
  *incremental* scripts come from, not the saturated base.

## Step 4 — Impact sizing (transparent)

- The **Fast-riser** segment alone is ~5,014 prescribers growing at triple-digit
  rates; the **Develop** segments add ~54k low-volume growers. Together that's the
  marginal-script engine a growth-oriented comp plan targets.
- Reallocating even a modest share of field effort from the **Maintain** cohort
  (~34k prescribers, ~0% growth) toward these growing segments shifts effort from
  ~0% marginal return to the highest-growth cohorts in the panel.
- Because the forecast is validated at **3.09% MAPE**, quotas built on it are
  credible to the field — a quota nobody believes is a quota nobody chases.

**How to validate for real:** hold out 2024, rebuild quotas from the 2013–2023
forecast, and check whether a forecast-quota + growth-segment comp plan would have
steered effort toward the prescribers/states that actually grew — using the same
walk-forward discipline already in `forecasting/compare.py`.

## Why this is the piece most candidates skip

Segments and forecasts are means, not ends. The deliverable a pharma-analytics
employer cares about is the **decision**: who each rep owns and how they're paid.
This closes that loop with real numbers and states exactly how it would be
back-tested.
