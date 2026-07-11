# 🔗 Generative-Agent Bullwhip Effect Simulator

A multi-agent Generative AI system that simulates the **bullwhip effect** —
the well-known phenomenon in supply chain management where small demand
fluctuations amplify as they move upstream (Retailer → Distributor →
Manufacturer → Supplier).

Instead of modeling this with fixed formulas alone, each tier of the supply
chain is run by an **LLM agent** (open-source, via Hugging Face's free
Inference API) that reasons in natural language about its inventory,
backlog, and incoming demand signal, and decides how much to order — the
same way a real human planner would. The system then quantifies how much
bullwhip amplification the LLM agents produce compared to the textbook
**order-up-to-S** operations-research policy.

## Why this project

This sits at the intersection of:
- **Operations Research / Supply Chain Management** — the bullwhip effect
  is a core, quantifiable concept taught in every SCM course
- **Multi-agent Generative AI** — LangGraph orchestrates 4 autonomous LLM
  agents that negotiate a real economic system, not just chat
- **Open-source LLMs** — no paid API required; runs on free Hugging Face
  Inference API models (e.g., Mistral-7B-Instruct)

## Architecture

```
Customer demand ──▶ Retailer ──▶ Distributor ──▶ Manufacturer ──▶ Supplier
                        │              │               │              │
                        ▼              ▼               ▼              ▼
                 LLM decides order quantity based on local state
                 (inventory, backlog, demand signal, order history)
```

- `simulation.py` — LangGraph state graph orchestrating the 4-tier chain, week by week
- `llm_client.py` — Hugging Face Inference API wrapper (open-source LLM calls)
- `classical_policy.py` — baseline order-up-to-S policy for comparison
- `metrics.py` — bullwhip ratio calculation: `Var(tier orders) / Var(customer demand)`
- `app.py` — Streamlit UI to configure demand shocks and visualize results

## Running locally

```bash
git clone https://github.com/YOUR_USERNAME/bullwhip-genai.git
cd bullwhip-genai
pip install -r requirements.txt
streamlit run app.py
```

Set your free Hugging Face token as an environment variable (or paste it
into the sidebar of the app):

```bash
export HF_TOKEN=your_token_here
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Without a token, the app still runs using a fallback heuristic so the
simulation never breaks.

## Key result

The app shows, for a given demand shock, how much each tier over-reacts
relative to the customer — and lets you directly compare an LLM-agent-run
supply chain against the classical OR baseline.

## Tech stack

- LangGraph (multi-agent orchestration)
- Hugging Face Inference API — Mistral-7B-Instruct-v0.3 (open source)
- Streamlit + Plotly (UI and visualization)
- NumPy / Pandas (simulation + metrics)

## License

MIT
