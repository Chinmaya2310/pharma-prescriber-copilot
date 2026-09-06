# Growth Classification — logistic regression vs XGBoost

**Task:** predict each prescriber's next-period growth class (declining ≤ -10%, growing ≥ +10%, else stable).

**Time-honest by construction:** features are computed from the 2022–2023 window (as-of end-2023); the label is the 2023→2024 class. No feature uses 2024 data, so there is no leakage from the label period. Train/test is a stratified 70/30 split over prescribers.

![growth histogram](figures/classification_growth_hist.png)

**Class balance (train+test):** {'growing': 47531, 'stable': 37035, 'declining': 35598} — mildly imbalanced (growing dominates); logistic regression uses `class_weight='balanced'`, and we report macro-F1 + per-class metrics rather than accuracy.

**Split sizes:** train 84,114 / test 36,050.

## Results (held-out test)

| model | macro-F1 | ROC-AUC (OVR) | F1[declining] | F1[stable] | F1[growing] |
|---|---|---|---|---|---|
| logreg | 0.430 | 0.615 | 0.41 | 0.49 | 0.40 |
| xgboost | 0.484 | 0.678 | 0.39 | 0.50 | 0.55 |

**Winner (macro-F1): `xgboost`.**

Per-class precision/recall for the winner:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| declining | 0.44 | 0.36 | 0.39 | 10680 |
| stable | 0.52 | 0.48 | 0.50 | 11111 |
| growing | 0.51 | 0.61 | 0.55 | 14259 |
