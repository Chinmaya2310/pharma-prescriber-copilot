# Data Quality Summary


- Rows read (raw): **893,997**
- Rows after cleaning + scope filter: **765,841**
- Distinct prescribers: **204,044**
- Years: **[2022, 2023, 2024]**
- States: **['CA', 'FL', 'NY', 'TX']**
- Drugs (generic): **['Amoxicillin', 'Anastrozole', 'Atorvastatin Calcium', 'Metformin Hcl']**
- Total claims (scoped): **102,019,574**
- Rows with suppressed Tot_Benes (true count 1-10): **26.13%** — kept as NaN + `Benes_Suppressed=True`, never zero-filled.

## Known censoring (missing-not-at-random)

CMS **omits any provider-drug row with fewer than 11 claims** from the source file. Those low-volume prescribing events are therefore invisible to this analysis. Consequence: per-provider and per-region totals are biased *low* at the small-volume tail, and any prescriber who only ever prescribes a drug a handful of times will not appear for that drug at all. This is a property of the data source, not a cleaning choice. See DECISIONS.md for how segmentation and forecasting each account for it.
