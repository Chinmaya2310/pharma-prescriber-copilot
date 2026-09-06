"""Streamlit dashboard for the Prescriber Analytics Copilot.

Reads everything from the FastAPI service (not the DB directly) so it exercises
the same API an external BI tool would. This is a functional stand-in for
Tableau / Power BI, which aren't scriptable in this environment (see README).

Run:  streamlit run dashboard/app.py   (API must be running first)
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PLOTLY_TEMPLATE = "plotly_white"
ACCENT = "#0E7C7B"
CLASS_COLORS = {"growing": "#2E9E5B", "stable": "#C9A227", "declining": "#C0392B"}


def get_api_base_url() -> str:
    """Resolve the API base URL: env var (Render/local .env) -> Streamlit Cloud
    secret -> localhost. Env var first so st.secrets is never touched locally."""
    env = os.getenv("API_BASE_URL")
    if env:
        return env
    try:
        return st.secrets["API_BASE_URL"]
    except Exception:
        return "http://127.0.0.1:8000"


API = get_api_base_url()

st.set_page_config(
    page_title="Prescriber Analytics Copilot",
    layout="wide",
    initial_sidebar_state="auto",  # auto-collapses on mobile
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      [data-testid="stMetricLabel"] {opacity: 0.75;}
      .pill {display:inline-block; padding:2px 10px; border-radius:999px;
             font-size:0.8rem; font-weight:600;}

      /* --- mobile: stack columns full-width instead of cramming side-by-side --- */
      @media (max-width: 640px) {
        .block-container {padding: 1rem 0.8rem 2rem;}
        [data-testid="stHorizontalBlock"] {flex-wrap: wrap; gap: 0.4rem;}
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
          flex: 1 1 100% !important; min-width: 100% !important;
        }
        [data-testid="stMetricValue"] {font-size: 1.35rem;}
        h1 {font-size: 1.5rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------- API helpers (cached) --------------------------- #
@st.cache_data(ttl=600, show_spinner=False)
def _cached_get(path: str):
    r = requests.get(f"{API}{path}", timeout=60)
    r.raise_for_status()
    return r.json()


def api_get(path: str, quiet: bool = False):
    try:
        return _cached_get(path)
    except Exception as exc:
        if not quiet:
            st.error(f"API error on GET {path}: {exc}")
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=120)
        if r.status_code == 503:
            st.warning(r.json().get("detail", "Service unavailable."))
            return None
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error on POST {path}: {exc}")
        return None


def human(n) -> str:
    """Compact number formatting: 102019574 -> '102.0M'."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:,.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


# ------------------------------- Sidebar ------------------------------------ #
with st.sidebar:
    st.markdown("## Prescriber Copilot")
    st.caption(
        "Analytics, forecasting & an agentic text-to-SQL assistant over real CMS "
        "Medicare Part D prescriber data."
    )
    health = api_get("/health", quiet=True)
    if health and health.get("status") == "ok":
        st.markdown(
            '<span class="pill" style="background:#E6F4EA;color:#1E7E34;">API online</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="pill" style="background:#FDECEA;color:#C0392B;">'
            'API waking / offline</span>',
            unsafe_allow_html=True,
        )
        st.caption("Free tier cold start can take ~30–60s. Give it a moment and rerun.")
    st.divider()
    st.markdown(
        "**Stack**\n\n"
        "- KMeans segmentation\n- Prophet vs XGBoost vs LSTM forecasting\n"
        "- Growth classification (XGBoost)\n- Groq (Llama/GPT-OSS) text-to-SQL"
    )
    st.divider()
    st.caption(f"API: `{API}`")


# ------------------------------- Header ------------------------------------- #
st.title("Pharma Prescriber Analytics & Forecasting Copilot")

kpis = api_get("/kpis", quiet=True)
if kpis is None:
    st.info(
        "Connecting to the API. On the free tier the service sleeps after ~15 min "
        "idle and takes ~30–60s to wake on the first request. Please wait a moment, "
        "then rerun (press **R**)."
    )
    if st.button("Retry now"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

st.caption(
    f"**{len(kpis['drugs'])} drugs** · **{len(kpis['states'])} states** · "
    f"years **{min(kpis['years'])}–{max(kpis['years'])}** · "
    f"forecast model: **{kpis['forecast_model'] or '—'}**"
)

tab_overview, tab_seg, tab_fc, tab_pred, tab_ask = st.tabs(
    ["Overview", "Segmentation", "Forecasts", "Growth prediction", "Ask (text-to-SQL)"]
)

# ------------------------------ Overview ------------------------------------ #
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total claims", human(kpis["total_claims"]))
    c2.metric("Prescribers", f"{kpis['n_prescribers']:,}")
    c3.metric("Behavioural segments", kpis["n_segments"])
    c4.metric("Forecast model", (kpis["forecast_model"] or "—").title())

    st.markdown("#### Forecasted demand by drug & state")
    fc = api_get("/forecast")
    if fc:
        df = pd.DataFrame(fc)
        latest = df[df["year"] == df["year"].min()]  # nearest forecast year
        fig = px.bar(
            latest, x="state", y="forecast_claims", color="gnrc_name",
            barmode="group", template=PLOTLY_TEMPLATE,
            labels={"forecast_claims": "forecast claims", "gnrc_name": "drug",
                    "state": "state"},
        )
        fig.update_layout(legend_title_text="", margin=dict(t=10, b=0), height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Next-year ({int(latest['year'].min())}) forecast · Prophet, "
                   "walk-forward CV MAPE ≈ 3.1%.")

# ---------------------------- Segmentation ---------------------------------- #
with tab_seg:
    st.markdown("#### Prescriber segments")
    summ = api_get("/segments/summary")
    if summ:
        sdf = pd.DataFrame(summ).sort_values("n_prescribers", ascending=False)
        fig = px.bar(
            sdf, x="n_prescribers", y="segment_name", orientation="h",
            template=PLOTLY_TEMPLATE, color="avg_growth",
            color_continuous_scale="Teal",
            labels={"n_prescribers": "prescribers", "segment_name": "",
                    "avg_growth": "avg YoY growth"},
        )
        fig.update_layout(height=430, margin=dict(t=10, b=0), yaxis={"autorange": "reversed"})
        st.plotly_chart(fig, use_container_width=True)
        disp = sdf.copy()
        disp["avg_growth"] = disp["avg_growth"] * 100  # show as percent
        st.dataframe(
            disp, use_container_width=True, hide_index=True,
            column_config={
                "segment_id": st.column_config.NumberColumn("ID", width="small"),
                "segment_name": st.column_config.TextColumn("Segment"),
                "n_prescribers": st.column_config.NumberColumn("Prescribers", format="%d"),
                "avg_volume": st.column_config.NumberColumn("Avg claims/yr", format="%.0f"),
                "avg_growth": st.column_config.NumberColumn("Avg YoY growth %", format="%.1f"),
            },
        )

    st.markdown("#### Look up a prescriber")
    npi = st.text_input("Prescriber NPI", placeholder="e.g. 1043286479", key="seg_npi")
    if npi.strip().isdigit():
        seg = api_get(f"/segments/{npi.strip()}", quiet=True)
        if seg:
            a, b, c = st.columns(3)
            a.metric("Segment", seg["segment_name"])
            b.metric("Claims / yr", f"{seg['volume']:,.0f}")
            c.metric("YoY growth", f"{seg['growth']*100:+.1f}%")
            d, e, f = st.columns(3)
            d.metric("State", seg["state"])
            e.metric("Specialty", seg["specialty"])
            f.metric("Avg $/claim", f"${seg['avg_cost_per_claim']:,.2f}")
        else:
            st.info("No prescriber found with that NPI in the segmented set.")

# ------------------------------ Forecasts ----------------------------------- #
with tab_fc:
    st.markdown("#### Drug demand forecast by state")
    col1, col2 = st.columns(2)
    drug = col1.selectbox("Drug (generic)", kpis["drugs"])
    region = col2.selectbox("State", kpis["states"])
    fcp = api_get(f"/forecast/{drug}/{region}", quiet=True)
    if fcp:
        pts = pd.DataFrame(fcp["points"])
        fig = px.line(
            pts, x="year", y="forecast_claims", markers=True,
            template=PLOTLY_TEMPLATE, labels={"forecast_claims": "forecast claims"},
        )
        fig.update_traces(line_color=ACCENT, line_width=3, marker_size=9)
        fig.update_layout(height=380, margin=dict(t=10, b=0),
                          xaxis=dict(dtick=1, title="year"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            pts, use_container_width=True, hide_index=True,
            column_config={
                "gnrc_name": st.column_config.TextColumn("Drug"),
                "state": st.column_config.TextColumn("State", width="small"),
                "year": st.column_config.NumberColumn("Year", format="%d"),
                "forecast_claims": st.column_config.NumberColumn("Forecast claims", format="%.0f"),
                "model": st.column_config.TextColumn("Model", width="small"),
            },
        )
    else:
        st.info("No forecast available for that drug/state combination.")

# --------------------------- Growth prediction ------------------------------ #
with tab_pred:
    st.markdown("#### Predicted next-period growth class")
    st.caption(
        "A classifier (logistic regression vs XGBoost, time-honest split) predicts "
        "whether each prescriber will be declining / stable / growing next period, "
        "from features known as of the prior year."
    )
    summ = api_get("/predict/summary")
    if summ:
        pdf = pd.DataFrame(summ)
        order = ["declining", "stable", "growing"]
        pdf["predicted_class"] = pd.Categorical(pdf["predicted_class"], order, ordered=True)
        pdf = pdf.sort_values("predicted_class")
        fig = px.bar(
            pdf, x="predicted_class", y="n", color="predicted_class",
            template=PLOTLY_TEMPLATE, color_discrete_map=CLASS_COLORS,
            labels={"predicted_class": "", "n": "prescribers"},
        )
        fig.update_layout(showlegend=False, height=360, margin=dict(t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Look up a prescriber's prediction")
    npi_p = st.text_input("Prescriber NPI", placeholder="e.g. 1043286479", key="pred_npi")
    if npi_p.strip().isdigit():
        pr = api_get(f"/predict/{npi_p.strip()}", quiet=True)
        if pr:
            color = CLASS_COLORS.get(pr["predicted_class"], ACCENT)
            st.markdown(
                f'Predicted: <span class="pill" style="background:{color}22;color:{color};">'
                f'{pr["predicted_class"].upper()}</span>  '
                f'<span style="opacity:.6">(model: {pr["model"]})</span>',
                unsafe_allow_html=True,
            )
            probs = pd.DataFrame(
                {"class": ["declining", "stable", "growing"],
                 "probability": [pr["prob_declining"], pr["prob_stable"], pr["prob_growing"]]}
            )
            fig = px.bar(
                probs, x="probability", y="class", orientation="h",
                template=PLOTLY_TEMPLATE, color="class", color_discrete_map=CLASS_COLORS,
                range_x=[0, 1],
            )
            fig.update_layout(showlegend=False, height=200, margin=dict(t=6, b=0),
                              yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction found with that NPI.")

# -------------------------------- Ask --------------------------------------- #
with tab_ask:
    st.markdown("#### Ask a question in plain English")
    st.caption(
        "The assistant writes read-only SQL, runs it, and answers from the actual "
        "rows. It shows you the SQL and the data."
    )

    EXAMPLES = [
        "Which specialty prescribed the most Metformin Hcl in CA in the latest year?",
        "How did total Amoxicillin claims in TX change between the two most recent years?",
        "Compare total claims for each drug class across all states in the latest year.",
    ]
    st.session_state.setdefault("ask_q", EXAMPLES[0])
    st.write("Try an example:")
    ex_cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.ask_q = ex

    q = st.text_input("Question", key="ask_q")
    if st.button("Ask", type="primary"):
        with st.spinner("Thinking (plan → SQL → run → answer)…"):
            res = api_post("/ask", {"question": q})
        if res:
            (st.success if res["success"] else st.warning)(res["answer"])
            if res["rows"]:
                st.dataframe(pd.DataFrame(res["rows"]), use_container_width=True,
                             hide_index=True)
            with st.expander("SQL used"):
                st.code(res["sql"] or "(none)", language="sql")
            with st.expander("Attempt trace (agentic retry loop)"):
                for a in res["attempts"]:
                    st.markdown(f"**Attempt {a['n']}** — {'OK' if a['ok'] else 'FAILED'}")
                    st.code(a["sql"], language="sql")
                    if a["error"]:
                        st.text(f"error: {a['error']}")
