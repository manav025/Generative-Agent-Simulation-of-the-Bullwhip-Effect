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

### Sample results (23-week run, demand shock at week 10, +63 units)

| Tier | Classical Policy | LLM Agents |
|---|---|---|
| Retailer | 1.289 | **0.777** |
| Distributor | 2.038 | 3.537 |
| Manufacturer | 4.231 | 24.414 |
| Supplier | 9.968 | **132.216** |

**Finding:** the LLM agents are actually *more disciplined* than the
classical order-up-to-S formula at the retailer level (0.777 vs 1.289) —
they don't overreact to a signal they can see clearly. But upstream, where
each tier only sees an already-distorted order signal from the tier below
it (not the real customer demand), the LLM agents' local overreactions
compound tier-over-tier and blow past the classical policy's fixed ceiling
— reaching **13x more amplification than the classical baseline** by the
supplier. The classical formula can't overreact more than its fixed
multiplier allows; the LLM has no such ceiling once bad signals compound.

This mirrors a real, well-documented failure mode in human-run supply
chains: planners upstream overreact to already-distorted information they
have no way of correcting for — and it emerged here without being
hand-coded into the agents.

## Tech stack

- LangGraph (multi-agent orchestration)
- Hugging Face Inference API — Mistral-7B-Instruct-v0.3 (open source)
- Streamlit + Plotly (UI and visualization)
- NumPy / Pandas (simulation + metrics)

## License

MIT
