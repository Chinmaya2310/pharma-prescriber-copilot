# EDA Summary

> ⚠️ **SYNTHETIC DATA** — illustrative patterns only; regenerate from real CMS data before quoting.


![trend](figures/eda_trend_by_class.png)

![state](figures/eda_by_state.png)

![specialty](figures/eda_by_specialty.png)

## Patterns found

1. **Divergent class trajectories** (2013→2023):
  - **Antibiotic**: +66% from 2013 to 2023
  - **Antidiabetic**: +143% from 2013 to 2023
  - **Cardiovascular (Statin)**: +125% from 2013 to 2023
  - **Oncology (Aromatase Inhibitor)**: -36% from 2013 to 2023

2. **Specialty concentration**: `Internal Medicine` is the single largest prescribing specialty in 2023 (~20% of scoped claims), confirming prescribing is concentrated in a handful of specialties — relevant for targeting and territory design.

3. **Geographic spread**: claim volumes differ markedly across CA, FL, NY, TX, so demand forecasts and territory quotas should be built per-state, not nationally.
