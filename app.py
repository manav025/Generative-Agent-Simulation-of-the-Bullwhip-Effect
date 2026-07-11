"""
app.py
Streamlit UI for the Generative-Agent Bullwhip Effect Simulator.

Deploy this on Streamlit Community Cloud (free) directly from GitHub.
Set your Hugging Face token as a secret named HF_TOKEN in the Streamlit
Cloud dashboard (Settings -> Secrets) — never commit it to the repo.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from simulation import run_simulation, TIERS
from metrics import bullwhip_ratios

st.set_page_config(page_title="GenAI Bullwhip Simulator", layout="wide")

st.title("🔗 Generative-Agent Bullwhip Effect Simulator")
st.caption(
    "A 4-tier supply chain (Retailer → Distributor → Manufacturer → Supplier) "
    "run by LLM agents, compared against the classical order-up-to-S policy."
)

with st.sidebar:
    st.header("Simulation Settings")
    total_weeks = st.slider("Number of weeks", 10, 40, 20)
    base_demand = st.slider("Base customer demand (units/week)", 10, 100, 40)
    shock_week = st.slider("Demand shock week", 1, total_weeks - 1, total_weeks // 3)
    shock_size = st.slider("Shock size (extra units)", 0, 100, 30)
    noise = st.slider("Demand noise (std dev)", 0, 20, 5)

    groq_key = st.text_input(
        "Groq API Key (optional — leave blank to use fallback heuristic)",
        type="password",
        help="Get a free key at console.groq.com/keys — no credit card required. "
             "Without it, LLM-mode agents fall back to a simple heuristic.",
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    run_button = st.button("Run Simulation", type="primary")


def make_demand_series(total_weeks, base_demand, shock_week, shock_size, noise):
    rng = np.random.default_rng(42)
    series = []
    for w in range(total_weeks):
        d = base_demand + (shock_size if w >= shock_week else 0)
        d += rng.normal(0, noise)
        series.append(max(0, round(d)))
    return series


if run_button:
    demand_series = make_demand_series(total_weeks, base_demand, shock_week, shock_size, noise)

    with st.spinner("Running classical order-up-to-S simulation..."):
        classical_state = run_simulation(total_weeks, "classical", demand_series)

    with st.spinner("Running LLM-agent simulation (calling Groq)..."):
        llm_state = run_simulation(total_weeks, "llm", demand_series)

    st.subheader("📈 Orders Placed by Each Tier")
    tab1, tab2 = st.tabs(["Classical Policy", "LLM Agents"])

    for tab, state, label in [(tab1, classical_state, "Classical"), (tab2, llm_state, "LLM Agent")]:
        with tab:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=demand_series, name="Customer Demand", line=dict(dash="dot", color="black")))
            for tier in TIERS:
                fig.add_trace(go.Scatter(y=state["order_history"][tier], name=f"{tier.capitalize()} orders"))
            fig.update_layout(title=f"{label} Policy — Orders by Tier", xaxis_title="Week", yaxis_title="Units")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌀 Bullwhip Ratios (Var(orders) / Var(demand))")
    classical_ratios = bullwhip_ratios(demand_series, classical_state["order_history"])
    llm_ratios = bullwhip_ratios(demand_series, llm_state["order_history"])

    df = pd.DataFrame({
        "Tier": [t.capitalize() for t in TIERS],
        "Classical Policy": [classical_ratios[t] for t in TIERS],
        "LLM Agents": [llm_ratios[t] for t in TIERS],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig2 = go.Figure(data=[
        go.Bar(name="Classical Policy", x=df["Tier"], y=df["Classical Policy"]),
        go.Bar(name="LLM Agents", x=df["Tier"], y=df["LLM Agents"]),
    ])
    fig2.update_layout(barmode="group", title="Bullwhip Amplification by Tier", yaxis_title="Bullwhip Ratio")
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "A ratio above 1 means that tier is over-reacting to demand changes "
        "relative to the customer — the classic bullwhip signature. Compare "
        "how LLM agents amplify (or dampen) this versus the textbook policy."
    )

    with st.expander("🧠 See LLM agent reasoning log"):
        reasoning_df = pd.DataFrame(llm_state["reasoning_log"])
        st.dataframe(reasoning_df, use_container_width=True, hide_index=True)

else:
    st.write("👈 Set your parameters in the sidebar and click **Run Simulation** to begin.")
