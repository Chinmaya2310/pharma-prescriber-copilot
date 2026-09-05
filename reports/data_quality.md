# Data Quality Summary

> ⚠️ **SYNTHETIC DATA** — these figures come from the local fixture, not the real CMS file. Regenerate after running the real download.


- Rows read (raw): **4,609**
- Rows after cleaning + scope filter: **4,609**
- Distinct prescribers: **280**
- Years: **[2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]**
- States: **['CA', 'FL', 'NY', 'TX']**
- Drugs (generic): **['Amoxicillin', 'Anastrozole', 'Atorvastatin Calcium', 'Metformin Hcl']**
- Total claims (scoped): **992,312**
- Rows with suppressed Tot_Benes (true count 1-10): **7.03%** — kept as NaN + `Benes_Suppressed=True`, never zero-filled.

## Known censoring (missing-not-at-random)

CMS **omits any provider-drug row with fewer than 11 claims** from the source file. Those low-volume prescribing events are therefore invisible to this analysis. Consequence: per-provider and per-region totals are biased *low* at the small-volume tail, and any prescriber who only ever prescribes a drug a handful of times will not appear for that drug at all. This is a property of the data source, not a cleaning choice. See DECISIONS.md for how segmentation and forecasting each account for it.
