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


def get_api_base_url() -> str:
    """Resolve the API base URL across all three deploy targets.

    Order: env var (Render env / local .env) -> Streamlit Cloud secret -> localhost.
    Env var is checked FIRST so that locally (where API_BASE_URL is in .env)
    st.secrets is never accessed at all — that avoids Streamlit's "No secrets found"
    message on machines with no secrets.toml. st.secrets is only read on Streamlit
    Cloud, where no env var is set, and is wrapped since accessing it raises when
    no secrets file exists.
    """
    env = os.getenv("API_BASE_URL")
    if env:
        return env
    try:
        return st.secrets["API_BASE_URL"]
    except Exception:
        return "http://127.0.0.1:8000"


API = get_api_base_url()

st.set_page_config(page_title="Prescriber Analytics Copilot", layout="wide")


def api_get(path: str):
    try:
        r = requests.get(f"{API}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
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


st.title("💊 Pharma Prescriber Analytics & Forecasting Copilot")

kpis = api_get("/kpis")
if kpis is None:
    st.stop()

tab_overview, tab_seg, tab_fc, tab_pred, tab_ask = st.tabs(
    ["Overview", "Segmentation", "Forecasts", "Growth prediction", "Ask (text-to-SQL)"]
)

# ------------------------------ Overview ------------------------------ #
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total claims", f"{kpis['total_claims']:,}")
    c2.metric("Prescribers", f"{kpis['n_prescribers']:,}")
    c3.metric("Segments", kpis["n_segments"])
    c4.metric("Forecast model", kpis["forecast_model"] or "—")
    st.caption(
        f"Years {kpis['years']} · States {kpis['states']} · Drugs {kpis['drugs']}"
    )
    fc = api_get("/forecast")
    if fc:
        df = pd.DataFrame(fc)
        fig = px.bar(df, x="state", y="forecast_claims", color="gnrc_name",
                     barmode="group", title="Forecasted claims by drug & state (next years)")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Segmentation ---------------------------- #
with tab_seg:
    st.subheader("Segment sizes & profiles")
    summ = api_get("/segments/summary")
    if summ:
        sdf = pd.DataFrame(summ)
        st.dataframe(sdf, use_container_width=True)
        fig = px.bar(sdf, x="segment_name", y="n_prescribers",
                     title="Prescribers per segment")
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Look up a prescriber")
    npi = st.text_input("Prescriber NPI")
    if npi.strip().isdigit():
        seg = api_get(f"/segments/{npi.strip()}")
        if seg:
            st.json(seg)

# ------------------------------ Forecasts ----------------------------- #
with tab_fc:
    drug = st.selectbox("Drug (generic)", kpis["drugs"])
    region = st.selectbox("State", kpis["states"])
    if st.button("Get forecast"):
        fc = api_get(f"/forecast/{drug}/{region}")
        if fc:
            pts = pd.DataFrame(fc["points"])
            st.dataframe(pts, use_container_width=True)
            fig = px.line(pts, x="year", y="forecast_claims", markers=True,
                          title=f"Forecast — {drug} / {region}")
            st.plotly_chart(fig, use_container_width=True)

# --------------------------- Growth prediction ------------------------ #
with tab_pred:
    st.subheader("Predicted next-period growth class")
    st.caption(
        "A classifier (logistic regression vs XGBoost, time-honest split) predicts "
        "whether each prescriber will be declining / stable / growing next period, "
        "from features known as of the prior year. Shows the class + probabilities."
    )
    summ = api_get("/predict/summary")
    if summ:
        pdf = pd.DataFrame(summ)
        fig = px.bar(pdf, x="predicted_class", y="n", color="predicted_class",
                     title="Prescribers per predicted growth class")
        st.plotly_chart(fig, use_container_width=True)
    npi_p = st.text_input("Prescriber NPI (growth prediction)")
    if npi_p.strip().isdigit():
        pr = api_get(f"/predict/{npi_p.strip()}")
        if pr:
            st.metric("Predicted class", pr["predicted_class"])
            st.json({k: pr[k] for k in ["prob_declining", "prob_stable", "prob_growing", "model"]})

# -------------------------------- Ask --------------------------------- #
with tab_ask:
    st.subheader("Ask a question in plain English")
    st.caption(
        "The assistant writes read-only SQL, runs it, and answers from the actual "
        "rows. It shows you the SQL and the data."
    )
    q = st.text_input(
        "Question",
        value="Which specialty prescribed the most Metformin Hcl in CA in the latest year?",
    )
    if st.button("Ask"):
        with st.spinner("Thinking (plan → SQL → run → answer)…"):
            res = api_post("/ask", {"question": q})
        if res:
            if res["success"]:
                st.success(res["answer"])
            else:
                st.warning(res["answer"])
            with st.expander("SQL used"):
                st.code(res["sql"] or "(none)", language="sql")
            if res["rows"]:
                st.dataframe(pd.DataFrame(res["rows"]), use_container_width=True)
            with st.expander("Attempt trace (agentic retry loop)"):
                for a in res["attempts"]:
                    status = "✅" if a["ok"] else "❌"
                    st.markdown(f"**Attempt {a['n']}** {status}")
                    st.code(a["sql"], language="sql")
                    if a["error"]:
                        st.text(f"error: {a['error']}")
