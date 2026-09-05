# BUSINESS_CASE.md — Territory Design & Incentive Compensation

> ⚠️ The specific figures below come from the **synthetic fixture** (the real CMS
> download was blocked in the build environment — see `DECISIONS.md` §0). The
> **methodology and recommendation structure are the deliverable**; re-run
> `retrain.py --download` to refresh every number from real data. Where a number
> appears, it is pulled from the current warehouse, not invented.

## The question a sales-ops leader is actually asking

> "I have a fixed field force. Which prescribers should each rep own, and how do I
> pay reps so their effort goes where the growth is — not just where the volume
> already is?"

This project answers that with two artifacts it produces: **behavioural segments**
(who prescribes, how, and whether they're growing) and **demand forecasts** (where
the market is heading by drug and state).

## Step 1 — Segments give you the unit of territory design

The KMeans segmentation groups every prescriber into a named, interpretable
segment. On the current run the segments are, e.g.:

| Segment | Prescribers | Avg volume | Avg growth | Read |
|---|---|---|---|---|
| High-volume broad statins-leaning (growing) | 16 | ~2,286 | +0.64 | **Flagship accounts** |
| High-volume focused diabetes-leaning (growing) | 12 | ~594 | +4.2* | **Fast risers** |
| High-volume focused statins-leaning (steady) | 49 | ~570 | ~0 | **Maintain** |
| High-volume focused diabetes-leaning (steady) | 62 | ~448 | ~0 | **Maintain** |
| Low-volume broad statins-leaning (growing) | 51 | ~420 | +0.44 | **Develop** |
| Low-volume focused antibiotics-leaning (declining) | 47 | ~169 | −0.16 | **Low priority** |
| Low-volume focused oncology-leaning (steady) | 23 | ~38 | ~0 | **Specialist niche** |

\*capped growth ratio — a handful of fast risers off a small prior-year base.

**Why this matters for territory design:** volume alone would tell every rep to
camp on the big "steady" accounts. Segments separate **volume** from **momentum**,
so you can build balanced books of business that each contain a mix of *maintain*
(protect the base) and *develop/fast-riser* (capture growth) accounts, rather than
one rep inheriting all the easy volume and another all the hard developmental work.

**Stability check (why you can trust it):** the segmentation is re-fit on an
earlier year window and compared with Adjusted Rand Index. If prescribers didn't
keep their segment period-to-period, the segments couldn't anchor annual territory
plans. That number is reported on every run (`models/registry.jsonl`).

## Step 2 — Forecasts tell you where to *move* capacity

`drug_region_forecast` projects claims per (drug, state) for the next years using
the CV-winning model. Territory quotas and headcount should follow the forecast,
not last year's actuals. Example pattern from the current data: states with a
rising Metformin/statin trajectory warrant quota (and possibly headcount) increases,
while a declining Anastrozole niche should be covered by a shared specialist rather
than dedicated field time.

## Step 3 — The recommendation

**A. Territory redesign — segment-balanced books.**
Assign each rep a book balanced across segments so total *opportunity* (current
volume + forecasted growth) is roughly equal, instead of balancing on raw current
volume. This stops the classic problem of "lucky" territories that are just
inherited volume, and it distributes developmental accounts fairly.

**B. Incentive compensation — pay for momentum, not just level.**
Split the incentive:
- ~70% on attainment vs a **forecast-based quota** (so the target reflects where
  the market is going, computed per drug×state from `drug_region_forecast`);
- ~30% on **growth within Develop / Fast-riser segments** specifically, so reps are
  paid to convert low-volume-growing and fast-rising prescribers, which is where
  incremental scripts actually come from.

## Step 4 — Estimating impact (transparent, defensible)

We don't claim a precise ROI from a model on annual data. We frame it as a sizing
argument a leader can sanity-check:

- The **Develop** + **Fast-riser** segments hold ~63 prescribers on this run.
- Their combined current volume is a small share of the total book, but their
  positive growth rates mean they are the **marginal scripts** a growth-oriented
  comp plan targets.
- Reallocating even a modest fraction of field effort from saturated **Maintain**
  accounts (near-zero growth) toward these growing segments moves effort from ~0%
  marginal return to the highest-growth cohort in the panel.

**How you'd validate it for real:** run the pipeline on real CMS data, hold out the
most recent year, and back-test whether a forecast-quota + growth-segment comp plan
would have directed effort toward the prescribers that actually grew — using the
same walk-forward discipline already in `forecasting/compare.py`.

## Why this is the piece most candidates skip

Segments and forecasts are means, not ends. The deliverable a pharma-analytics
employer cares about is the **decision**: who each rep owns and how they're paid.
This document closes that loop and states exactly how it would be validated on real
data.
