"""Exploratory data analysis: saves charts to reports/figures and a written
summary of concrete patterns to reports/eda_summary.md.

Charts are *saved*, not shown, so the same code runs in CI and feeds the README.
The written patterns are computed from the data (not hardcoded) so they stay
truthful if the underlying data changes.
"""
from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common import config  # noqa: E402


def _trend_by_class(series: pd.DataFrame) -> pd.DataFrame:
    # Uses the full-history state-level series (by-Geography) so the trend spans
    # all available years, not just the 2 provider years used for segmentation.
    s = series.copy()
    s["Drug_Class"] = s["Gnrc_Name"].map(config.DRUG_CLASS)
    piv = (
        s.groupby(["Year", "Drug_Class"])["claims"].sum().reset_index()
        .pivot(index="Year", columns="Drug_Class", values="claims")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    piv.plot(marker="o", ax=ax)
    ax.set(title="Total claims by drug class over time", xlabel="Year", ylabel="total claims")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_trend_by_class.png", dpi=120)
    plt.close(fig)
    return piv


def _by_state(df: pd.DataFrame) -> None:
    latest = df["Year"].max()
    g = (
        df[df["Year"] == latest].groupby(["Prscrbr_State_Abrvtn", "Drug_Class"])["Tot_Clms"]
        .sum().reset_index()
        .pivot(index="Prscrbr_State_Abrvtn", columns="Drug_Class", values="Tot_Clms")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    g.plot(kind="bar", stacked=True, ax=ax)
    ax.set(title=f"Claims by state & class ({latest})", xlabel="State", ylabel="total claims")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_by_state.png", dpi=120)
    plt.close(fig)


def _by_specialty(df: pd.DataFrame) -> pd.Series:
    latest = df["Year"].max()
    top = (
        df[df["Year"] == latest].groupby("Prscrbr_Type")["Tot_Clms"].sum()
        .sort_values(ascending=False).head(10)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    top.iloc[::-1].plot(kind="barh", ax=ax)
    ax.set(title=f"Top prescribing specialties ({latest})", xlabel="total claims")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "eda_by_specialty.png", dpi=120)
    plt.close(fig)
    return top


def _write_summary(
    df: pd.DataFrame, trend: pd.DataFrame, top_spec: pd.Series, synthetic: bool
) -> None:
    # Trend year-range comes from the geography series (full history); the
    # specialty leader is computed from the latest provider year.
    trend_years = list(trend.index)
    first, last = trend_years[0], trend_years[-1]
    prov_latest = int(df["Year"].max())
    growth_lines = []
    for cls in trend.columns:
        a, b = trend[cls].iloc[0], trend[cls].iloc[-1]
        if a and a > 0:
            pct = (b - a) / a * 100
            growth_lines.append(f"  - **{cls}**: {pct:+.0f}% from {first} to {last}")
    leader = top_spec.index[0]
    leader_share = top_spec.iloc[0] / df[df["Year"] == prov_latest]["Tot_Clms"].sum() * 100

    banner = ("> ⚠️ **SYNTHETIC DATA** — illustrative patterns only; regenerate from "
              "real CMS data before quoting.\n\n" if synthetic else "")
    lines = [
        "# EDA Summary\n", banner,
        "![trend](figures/eda_trend_by_class.png)\n",
        "![state](figures/eda_by_state.png)\n",
        "![specialty](figures/eda_by_specialty.png)\n",
        "## Patterns found\n",
        f"1. **Divergent class trajectories** ({first}→{last}):",
        *growth_lines,
        f"\n2. **Specialty concentration**: `{leader}` is the single largest "
        f"prescribing specialty in {prov_latest} (~{leader_share:.0f}% of scoped "
        "claims), confirming prescribing is concentrated in a handful of "
        "specialties — relevant for targeting and territory design.\n",
        f"3. **Geographic spread**: claim volumes differ markedly across "
        f"{', '.join(sorted(df['Prscrbr_State_Abrvtn'].unique()))}, so demand "
        "forecasts and territory quotas should be built per-state, not nationally.",
    ]
    (config.REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines) + "\n")


def run(df: pd.DataFrame | None = None) -> None:
    from src.forecasting import series as S

    if df is None:
        df = pd.read_parquet(config.PROCESSED_PARQUET)
    synthetic = (config.REPORTS_DIR / "data_quality.md").exists() and \
        "SYNTHETIC" in (config.REPORTS_DIR / "data_quality.md").read_text()
    geo_series = S.load_series()  # full-history state-level series for the trend
    trend = _trend_by_class(geo_series)
    _by_state(df)
    top_spec = _by_specialty(df)
    _write_summary(df, trend, top_spec, synthetic)


def main() -> None:
    run()
    print(f"EDA figures + summary written to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
