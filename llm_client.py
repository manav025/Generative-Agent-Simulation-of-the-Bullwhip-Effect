"""
llm_client.py
Wraps calls to a single, fixed open-source model (Meta's Llama 3.1 8B
Instruct) via Groq's API, so every tier in a given run is answered by the
SAME model - no silent mid-run swaps to a different model, which would
confound the comparison between the classical policy and "the LLM agent."

Groq hosts open-weight models on dedicated fast hardware and offers a
genuinely free tier (14,400 requests/day, no credit card).
Get a free key at https://console.groq.com/keys
"""

import json
import os
import re
import time
from groq import Groq

# Single fixed model for the whole run - keeps the experiment clean.
# (Previously this tried a second, different-strength model as a
# fallback, which meant some weeks were secretly answered by a
# different model than others - a confound. Retrying the SAME model
# with backoff is the correct way to handle transient rate limits.)
MODEL_ID = "llama-3.1-8b-instant"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

_client = None
fallback_call_count = 0  # exposed so the UI can warn if this run degraded


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

Base your decision ONLY on the actual numbers given below - do not assume
demand is continuing at a past average if the current demand signal says
otherwise. If demand signal is 0, treat that as real information.

Rules of thumb real supply-chain planners use (and often overreact with):
- If demand looks like it's rising, order extra as a buffer (this is what
  causes the bullwhip effect - be realistic, don't be a perfect optimizer).
- If you have a backlog, order more to catch up.
- If inventory is piling up, order less.

Respond ONLY with compact JSON, no other text:
{{"order_quantity": <integer>, "reasoning": "<one short sentence>"}}
"""


def reset_fallback_counter():
    global fallback_call_count
    fallback_call_count = 0


def get_llm_order_decision(role, demand_signal, inventory, backlog, order_history):
    """
    Calls Groq to get an order-quantity decision + short reasoning from the
    fixed model. Retries the SAME model up to MAX_RETRIES times (with a
    short delay) on transient errors like rate limits, so a brief hiccup
    doesn't silently swap in a different model's behavior. Only falls back
    to the simple heuristic if every retry fails (e.g. no API key, or a
    sustained outage) - and that fallback is counted so the UI can flag it.
    """
    global fallback_call_count

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
        fallback_call_count += 1
        fallback_qty = max(0, demand_signal + backlog // 2)
        return fallback_qty, "[fallback heuristic - no GROQ_API_KEY set]"

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL_ID,
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
            return max(0, qty), f"[{MODEL_ID}] {reasoning}"
        except Exception as e:
            last_error = f"attempt {attempt + 1} -> {type(e).__name__}: {str(e)[:150]}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            continue

    # All retries on the SAME model failed - fallback heuristic, counted.
    fallback_call_count += 1
    fallback_qty = max(0, demand_signal + backlog // 2)
    return fallback_qty, f"[fallback heuristic - {MODEL_ID} failed after {MAX_RETRIES} tries. {last_error}]"
