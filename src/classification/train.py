"""Train + compare growth classifiers (logistic regression vs XGBoost).

Mirrors the Prophet-vs-XGBoost forecasting pattern: two models, one fair
evaluation, pick the winner, persist. Evaluation uses per-class precision/recall/
F1 and one-vs-rest ROC-AUC (not just accuracy) because the classes are mildly
imbalanced (growing ~43% dominates). Winner is chosen on macro-F1.

The stored `prescriber_growth_prediction` rows are genuine *forward* predictions:
the winning model, trained on (as-of-2023 features -> 2023->2024 class), is applied
to as-of-2024 features to predict each prescriber's next-period (2024->2025) class.
"""
from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.classification.labels import build_dataset  # noqa: E402
from src.common import config  # noqa: E402
from src.segmentation.features import FEATURE_COLS, build_features  # noqa: E402

RANDOM_STATE = 42
CLASSES = config.GROWTH_CLASSES
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}


def _models():
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),  # multinomial by default
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0,
        objective="multi:softprob", num_class=len(CLASSES), eval_metric="mlogloss",
    )
    return {"logreg": logreg, "xgboost": xgb}


def _evaluate(model, Xtr, ytr, Xte, yte) -> dict:
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    pred = proba.argmax(axis=1)
    macro_f1 = f1_score(yte, pred, average="macro")
    # OVR macro ROC-AUC (guard against a class missing from yte)
    try:
        auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro",
                            labels=list(range(len(CLASSES))))
    except ValueError:
        auc = float("nan")
    report = classification_report(
        yte, pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    return {"model": model, "macro_f1": macro_f1, "roc_auc_ovr": auc,
            "report": report, "pred": pred}


def _forward_predictions(df: pd.DataFrame, model, feature_names: list[str]) -> pd.DataFrame:
    """Apply the winner to as-of-latest features to predict the NEXT period."""
    years = sorted(df["Year"].unique())
    cur = build_features(df, years=years[-2:])   # as-of-2024 features
    cur = cur[cur["has_history"]]
    top = cur["specialty"].value_counts().head(12).index
    cur["specialty_grp"] = cur["specialty"].where(cur["specialty"].isin(top), "Other")
    spec = pd.get_dummies(cur["specialty_grp"], prefix="spec")
    X = pd.concat([cur[FEATURE_COLS].reset_index(drop=True), spec.reset_index(drop=True)], axis=1)
    X = X.reindex(columns=feature_names, fill_value=0)  # align to training columns
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)
    out = pd.DataFrame({
        "prscrbr_npi": cur["Prscrbr_NPI"].values,
        "predicted_class": [CLASSES[i] for i in pred],
    })
    for i, c in enumerate(CLASSES):
        out[f"prob_{c}"] = proba[:, i].round(4)
    return out


def _write_report(results: dict, winner: str, balance: pd.Series, n_train: int, n_test: int):
    lines = [
        "# Growth Classification — logistic regression vs XGBoost\n",
        "**Task:** predict each prescriber's next-period growth class "
        f"(declining ≤ {config.GROWTH_DECLINE_THRESHOLD:+.0%}, growing ≥ "
        f"{config.GROWTH_GROW_THRESHOLD:+.0%}, else stable).\n",
        "**Time-honest by construction:** features are computed from the 2022–2023 "
        "window (as-of end-2023); the label is the 2023→2024 class. No feature uses "
        "2024 data, so there is no leakage from the label period. Train/test is a "
        "stratified 70/30 split over prescribers.\n",
        "![growth histogram](figures/classification_growth_hist.png)\n",
        f"**Class balance (train+test):** {balance.to_dict()} — mildly imbalanced "
        "(growing dominates); logistic regression uses `class_weight='balanced'`, "
        "and we report macro-F1 + per-class metrics rather than accuracy.\n",
        f"**Split sizes:** train {n_train:,} / test {n_test:,}.\n",
        "## Results (held-out test)\n",
        "| model | macro-F1 | ROC-AUC (OVR) | " + " | ".join(f"F1[{c}]" for c in CLASSES) + " |",
        "|---|---|---|" + "---|" * len(CLASSES),
    ]
    for name, r in results.items():
        f1s = " | ".join(f"{r['report'][c]['f1-score']:.2f}" for c in CLASSES)
        lines.append(f"| {name} | {r['macro_f1']:.3f} | {r['roc_auc_ovr']:.3f} | {f1s} |")
    lines += [
        f"\n**Winner (macro-F1): `{winner}`.**\n",
        "Per-class precision/recall for the winner:\n",
        "| class | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    wr = results[winner]["report"]
    for c in CLASSES:
        lines.append(f"| {c} | {wr[c]['precision']:.2f} | {wr[c]['recall']:.2f} | "
                     f"{wr[c]['f1-score']:.2f} | {int(wr[c]['support'])} |")
    (config.REPORTS_DIR / "classification_report.md").write_text("\n".join(lines) + "\n")


def train(df: pd.DataFrame | None = None) -> dict:
    from src.common.db import write_table
    from src.mlops.registry import save_model

    if df is None:
        df = pd.read_parquet(config.PROCESSED_PARQUET)

    X, y, feature_names = build_dataset(df)
    balance = y.value_counts()
    yi = y.map(CLASS_TO_INT).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(
        X[feature_names], yi, test_size=0.30, random_state=RANDOM_STATE, stratify=yi
    )

    results = {name: _evaluate(m, Xtr, ytr, Xte, yte) for name, m in _models().items()}
    winner = max(results, key=lambda k: results[k]["macro_f1"])

    # confusion matrix for the winner
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        yte, results[winner]["pred"], display_labels=CLASSES, ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix — {winner}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "classification_confusion.png", dpi=120)
    plt.close(fig)

    # refit winner on ALL labeled rows, then forward-predict next period for everyone
    win_model = _models()[winner]
    win_model.fit(X[feature_names], yi)
    preds = _forward_predictions(df, win_model, feature_names)
    preds["model"] = winner
    write_table(preds, "prescriber_growth_prediction")

    _write_report(results, winner, balance, len(Xtr), len(Xte))
    save_model(
        {"model": win_model, "feature_names": feature_names, "classes": CLASSES, "winner": winner},
        name="growth_classifier",
        metrics={"winner": winner,
                 **{k: round(v["macro_f1"], 3) for k, v in results.items()},
                 "roc_auc_winner": round(results[winner]["roc_auc_ovr"], 3)},
    )
    return {
        "winner": winner,
        "balance": balance.to_dict(),
        "macro_f1": {k: round(v["macro_f1"], 3) for k, v in results.items()},
        "roc_auc": {k: round(v["roc_auc_ovr"], 3) for k, v in results.items()},
        "n_predictions": len(preds),
    }


def main() -> None:
    res = train()
    print("Classification complete:")
    print(f"  class balance : {res['balance']}")
    print(f"  macro-F1      : {res['macro_f1']}")
    print(f"  ROC-AUC (OVR) : {res['roc_auc']}")
    print(f"  winner        : {res['winner']}")
    print(f"  forward preds : {res['n_predictions']:,} -> prescriber_growth_prediction")


if __name__ == "__main__":
    main()
