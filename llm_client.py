"""
llm_client.py
Wraps calls to a free, open-source Hugging Face model (Mistral-7B-Instruct)
so each supply-chain agent can reason in natural language and output an
order quantity as structured JSON.

Uses the huggingface_hub InferenceClient (serverless, free tier) exactly
like the Aarav sales-agent project — same integration pattern, new use case.
"""

import json
import os
import re
from huggingface_hub import InferenceClient

# You can swap this for any other free instruct model on HF, e.g.:
# "HuggingFaceH4/zephyr-7b-beta", "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

_client = None


def get_client():
    global _client
    if _client is None:
        token = os.environ.get("HF_TOKEN")
        _client = InferenceClient(model=MODEL_ID, token=token)
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
    Calls the HF model to get an order-quantity decision + short reasoning.
    Falls back to a simple heuristic if the API call fails (e.g. no token,
    rate limit, or model cold-start) so the simulation never crashes.
    """
    history_str = ", ".join(str(x) for x in order_history[-5:]) or "none yet"
    user_prompt = (
        f"This week's incoming demand signal: {demand_signal} units.\n"
        f"Your current inventory: {inventory} units.\n"
        f"Your current backlog: {backlog} units.\n"
        f"Your last 5 order quantities: {history_str}.\n"
        f"Decide your order quantity for this week."
    )

    try:
        client = get_client()
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": AGENT_SYSTEM_PROMPT.format(role=role)},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=120,
            temperature=0.7,
        )
        text = response.choices[0].message.content
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group(0))
        qty = int(parsed.get("order_quantity", demand_signal))
        reasoning = parsed.get("reasoning", "")
        return max(0, qty), reasoning
    except Exception as e:
        # Fallback heuristic: naive order-up-to-ish behavior so the sim
        # keeps running even without a valid HF_TOKEN.
        fallback_qty = max(0, demand_signal + backlog // 2)
        return fallback_qty, f"[fallback heuristic used - {type(e).__name__}]"
