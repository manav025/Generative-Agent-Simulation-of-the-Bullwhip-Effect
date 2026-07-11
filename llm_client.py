"""
llm_client.py
Wraps calls to a free, open-source model (Meta's Llama 3.1 8B Instruct)
via Groq's API. Groq hosts open-weight models on dedicated fast hardware
and offers a genuinely free tier (14,400 requests/day, no credit card) -
far more reliable for a live demo than Hugging Face's shared routed
inference credits, which are easy to exhaust in a single simulation run.

Get a free key at https://console.groq.com/keys
"""

import json
import os
import re
from groq import Groq

# Candidate models, tried in order, all open-weight and free on Groq.
MODEL_CANDIDATES = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None
        _client = Groq(api_key=api_key)
    return _client


AGENT_SYSTEM_PROMPT = """You are {role} in a 4-tier supply chain
(Retailer -> Distributor -> Manufacturer -> Supplier).
Each week you see the order placed by the tier below you (your "demand
signal"), your current inventory, and your current backlog (unfilled
orders). You must decide how many units to order from the tier above you.

Rules of thumb real supply-chain planners use (and often overreact with):
- If demand looks like it's rising, order extra as a buffer (this is what
  causes the bullwhip effect - be realistic, don't be a perfect optimizer).
- If you have a backlog, order more to catch up.
- If inventory is piling up, order less.

Respond ONLY with compact JSON, no other text:
{{"order_quantity": <integer>, "reasoning": "<one short sentence>"}}
"""


def get_llm_order_decision(role, demand_signal, inventory, backlog, order_history):
    """
    Calls Groq to get an order-quantity decision + short reasoning from an
    open-source LLM. Tries each model in MODEL_CANDIDATES until one
    succeeds. Falls back to a simple heuristic only if all fail (e.g. no
    API key set, or rate limit hit) so the simulation never crashes.
    """
    history_str = ", ".join(str(x) for x in order_history[-5:]) or "none yet"
    user_prompt = (
        f"This week's incoming demand signal: {demand_signal} units.\n"
        f"Your current inventory: {inventory} units.\n"
        f"Your current backlog: {backlog} units.\n"
        f"Your last 5 order quantities: {history_str}.\n"
        f"Decide your order quantity for this week."
    )

    client = get_client()
    if client is None:
        fallback_qty = max(0, demand_signal + backlog // 2)
        return fallback_qty, "[fallback heuristic - no GROQ_API_KEY set]"

    last_error = None

    for model_id in MODEL_CANDIDATES:
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT.format(role=role)},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=120,
                temperature=0.7,
            )
            text = completion.choices[0].message.content
            match = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(match.group(0))
            qty = int(parsed.get("order_quantity", demand_signal))
            reasoning = parsed.get("reasoning", "")
            return max(0, qty), f"[{model_id}] {reasoning}"
        except Exception as e:
            last_error = f"{model_id} -> {type(e).__name__}: {str(e)[:150]}"
            continue

    # All candidates failed - fallback heuristic so the sim keeps running.
    fallback_qty = max(0, demand_signal + backlog // 2)
    return fallback_qty, f"[fallback heuristic - all models failed. Last error: {last_error}]"
