"""KMeans prescriber segmentation with a defensible choice of k and a stability check.

Pipeline:
  1. Build features (latest window) -> standardise.
  2. Choose k by silhouette score over k in [2, 8]; save the curve as a figure.
  3. Fit final KMeans, assign + name segments in business terms.
  4. Stability check: re-fit on an *earlier* year window and measure how much
     prescribers keep their segment (Adjusted Rand Index). A segmentation that
     reshuffles randomly period-to-period is useless for territory planning, so
     this number is reported either way (see DECISIONS.md).
Outputs: segment table -> SQLite, model + scaler -> models/, figures -> reports/.
"""
from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.common import config  # noqa: E402
from src.segmentation.features import FEATURE_COLS, build_features  # noqa: E402

K_RANGE = range(2, 9)
RANDOM_STATE = 42


def _choose_k(X) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
    best_k = max(scores, key=scores.get)
    return best_k, scores


def _save_silhouette_plot(scores: dict[int, float], best_k: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(scores), list(scores.values()), marker="o")
    ax.axvline(best_k, color="red", ls="--", label=f"chosen k={best_k}")
    ax.set(xlabel="k (clusters)", ylabel="silhouette score",
           title="Segmentation: choosing k by silhouette")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "segmentation_silhouette.png", dpi=120)
    plt.close(fig)


def _name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Heuristic business names from each cluster's centroid profile."""
    names: dict[int, str] = {}
    vol_median = profile["volume"].median()
    for cid, row in profile.iterrows():
        high_vol = row["volume"] >= vol_median
        # dominant class share
        shares = {
            "diabetes": row["share_antidiabetic"],
            "statins": row["share_cardio"],
            "antibiotics": row["share_antibiotic"],
            "oncology": row["share_oncology"],
        }
        focus = max(shares, key=shares.get)
        breadth = "broad" if row["n_drugs"] >= 2.5 else "focused"
        tier = "High-volume" if high_vol else "Low-volume"
        if row["growth"] > 0.03:
            grow = "growing"
        elif row["growth"] < -0.03:
            grow = "declining"
        else:
            grow = "steady"
        names[cid] = f"{tier} {breadth} {focus}-leaning ({grow})"
    return names


def _fit(feats: pd.DataFrame, k: int):
    scaler = StandardScaler()
    X = scaler.fit_transform(feats[FEATURE_COLS])
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    return scaler, km, labels


def train(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_parquet(config.PROCESSED_PARQUET)

    feats = build_features(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(feats[FEATURE_COLS])

    best_k, scores = _choose_k(X)
    _save_silhouette_plot(scores, best_k)

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    feats["segment_id"] = km.fit_predict(X)

    profile = feats.groupby("segment_id")[FEATURE_COLS].mean()
    names = _name_segments(profile)
    feats["segment_name"] = feats["segment_id"].map(names)

    # ---------------- stability check on an earlier window ----------------
    ari = None
    years = sorted(df["Year"].unique())
    if len(years) >= config.SEGMENTATION_YEARS + 1:
        early_years = years[: config.SEGMENTATION_YEARS]
        early_feats = build_features(df, years=early_years)
        _, _, early_labels = _fit(early_feats, best_k)
        early_feats = early_feats.assign(segment_id_early=early_labels)
        merged = feats.merge(
            early_feats[["Prscrbr_NPI", "segment_id_early"]], on="Prscrbr_NPI", how="inner"
        )
        if len(merged) > 10:
            ari = float(adjusted_rand_score(merged["segment_id_early"], merged["segment_id"]))

    _persist(feats, scaler, km, profile, names, best_k, scores, ari)
    return {
        "k": best_k,
        "silhouette": round(scores[best_k], 3),
        "segments": names,
        "n_prescribers": len(feats),
        "stability_ari": None if ari is None else round(ari, 3),
    }


def _persist(feats, scaler, km, profile, names, best_k, scores, ari) -> None:
    from src.common.db import write_table
    from src.mlops.registry import save_model

    seg_table = feats[[
        "Prscrbr_NPI", "state", "specialty", "volume", "growth", "n_drugs",
        "avg_cost_per_claim", "segment_id", "segment_name",
    ]].rename(columns={"Prscrbr_NPI": "prscrbr_npi"})
    write_table(seg_table, "prescriber_segments")

    profile_out = profile.copy()
    profile_out["segment_name"] = [names[i] for i in profile_out.index]
    profile_out.reset_index().to_parquet(
        config.ARTIFACTS_DIR / "segment_profiles.parquet", index=False)

    save_model(
        {"model": km, "scaler": scaler, "feature_cols": FEATURE_COLS, "names": names},
        name="kmeans_segmentation",
        metrics={
            "k": best_k,
            "silhouette": scores[best_k],
            "stability_ari": ari,
        },
    )


def main() -> None:
    res = train()
    print("Segmentation complete:")
    print(f"  chosen k          : {res['k']} (silhouette={res['silhouette']})")
    print(f"  prescribers       : {res['n_prescribers']:,}")
    print(f"  stability (ARI)   : {res['stability_ari']}  "
          "(1.0=identical, 0=random; >0.4 is usable for territory planning)")
    print("  segments:")
    for cid, name in res["segments"].items():
        print(f"    {cid}: {name}")


if __name__ == "__main__":
    main()
