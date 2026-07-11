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
import llm_client

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

    # Prefer a key configured in Streamlit Cloud secrets (shared by the app
    # owner) so visitors don't need their own Groq account. Falls back to
    # a manual input field if no secret is configured.
    groq_key = st.secrets.get("GROQ_API_KEY", None)

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("✅ Groq API connected (shared key)")
    else:
        manual_key = st.text_input(
            "Groq API Key (optional — leave blank to use fallback heuristic)",
            type="password",
            help="Get a free key at console.groq.com/keys — no credit card required. "
                 "Without it, LLM-mode agents fall back to a simple heuristic.",
        )
        if manual_key:
            os.environ["GROQ_API_KEY"] = manual_key

    include_informed = st.checkbox(
        "Also test 'informed' LLM agents (full demand visibility)",
        value=True,
        help="Roughly doubles API calls/run time, since it runs a third simulation.",
    )

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
    import time
    demand_series = make_demand_series(total_weeks, base_demand, shock_week, shock_size, noise)

    t0 = time.time()
    with st.spinner("Running classical order-up-to-S simulation..."):
        classical_state = run_simulation(total_weeks, "classical", demand_series)
    classical_time = time.time() - t0

    t0 = time.time()
    with st.spinner("Running LLM-agent simulation (isolated - calling Groq)..."):
        llm_client.reset_fallback_counter()
        llm_state = run_simulation(total_weeks, "llm", demand_series)
        isolated_fallbacks = llm_client.fallback_call_count
        if isolated_fallbacks > 0:
            st.warning(
                f"⚠️ Isolated LLM run: {isolated_fallbacks} of {total_weeks} weekly calls "
                f"fell back to the simple heuristic (likely rate limits)."
            )
    llm_time = time.time() - t0

    informed_state = None
    informed_time = 0.0
    if include_informed:
        t0 = time.time()
        with st.spinner("Running LLM-agent simulation (informed - calling Groq)..."):
            llm_client.reset_fallback_counter()
            informed_state = run_simulation(total_weeks, "llm_informed", demand_series)
            informed_fallbacks = llm_client.fallback_call_count
            if informed_fallbacks > 0:
                st.warning(
                    f"⚠️ Informed LLM run: {informed_fallbacks} of {total_weeks} weekly calls "
                    f"fell back to the simple heuristic (likely rate limits)."
                )
        informed_time = time.time() - t0

    st.caption(
        f"⏱️ Classical: {classical_time:.1f}s | LLM (isolated): {llm_time:.1f}s"
        + (f" | LLM (informed): {informed_time:.1f}s" if include_informed else "")
    )

    st.subheader("📈 Orders Placed by Each Tier")
    tab_labels = ["Classical Policy", "LLM Agents (Isolated)"]
    tab_states = [classical_state, llm_state]
    tab_titles = ["Classical", "LLM Agent (Isolated)"]
    if include_informed:
        tab_labels.append("LLM Agents (Informed)")
        tab_states.append(informed_state)
        tab_titles.append("LLM Agent (Informed)")

    tabs = st.tabs(tab_labels)
    for tab, state, label in zip(tabs, tab_states, tab_titles):
        with tab:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=demand_series, name="Customer Demand", line=dict(dash="dot", color="black")))
            for tier in TIERS:
                fig.add_trace(go.Scatter(y=state["order_history"][tier], name=f"{tier.capitalize()} orders"))
            fig.update_layout(title=f"{label} Policy — Orders by Tier", xaxis_title="Week", yaxis_title="Units")
            st.plotly_chart(fig, width='stretch')

    st.subheader("🌀 Bullwhip Ratios (Var(orders) / Var(demand))")
    classical_ratios = bullwhip_ratios(demand_series, classical_state["order_history"])
    llm_ratios = bullwhip_ratios(demand_series, llm_state["order_history"])

    ratio_data = {
        "Tier": [t.capitalize() for t in TIERS],
        "Classical Policy": [classical_ratios[t] for t in TIERS],
        "LLM Agents (Isolated)": [llm_ratios[t] for t in TIERS],
    }
    if include_informed:
        informed_ratios = bullwhip_ratios(demand_series, informed_state["order_history"])
        ratio_data["LLM Agents (Informed)"] = [informed_ratios[t] for t in TIERS]

    df = pd.DataFrame(ratio_data)
    st.dataframe(df, width='stretch', hide_index=True)

    bars = [
        go.Bar(name="Classical Policy", x=df["Tier"], y=df["Classical Policy"]),
        go.Bar(name="LLM Agents (Isolated)", x=df["Tier"], y=df["LLM Agents (Isolated)"]),
    ]
    if include_informed:
        bars.append(go.Bar(name="LLM Agents (Informed)", x=df["Tier"], y=df["LLM Agents (Informed)"]))

    fig2 = go.Figure(data=bars)
    fig2.update_layout(barmode="group", title="Bullwhip Amplification by Tier", yaxis_title="Bullwhip Ratio")
    st.plotly_chart(fig2, width='stretch')

    st.info(
        "A ratio above 1 means that tier is over-reacting to demand changes "
        "relative to the customer — the classic bullwhip signature. 'Isolated' "
        "agents only see the order from the tier below them (a distorted, "
        "secondhand signal). 'Informed' agents also see the real customer "
        "demand history directly — this tests whether information sharing "
        "reduces bullwhip amplification for LLM agents, and whether they can "
        "use that visibility to distinguish real trend shifts from noise "
        "better than a rigid formula can."
    )

    with st.expander("🧠 See LLM agent reasoning log (isolated)"):
        reasoning_df = pd.DataFrame(llm_state["reasoning_log"])
        st.dataframe(reasoning_df, width='stretch', hide_index=True)

    if include_informed:
        with st.expander("🧠 See LLM agent reasoning log (informed)"):
            informed_reasoning_df = pd.DataFrame(informed_state["reasoning_log"])
            st.dataframe(informed_reasoning_df, width='stretch', hide_index=True)

else:
    st.write("👈 Set your parameters in the sidebar and click **Run Simulation** to begin.")
