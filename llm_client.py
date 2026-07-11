"""
llm_client.py
Wraps calls to a single, fixed open-source model (Meta's Llama 3.1 8B
Instruct) via Groq's API.

IMPORTANT DESIGN CHOICE: all 4 tiers' decisions for a given week are
requested in ONE API call (not four separate calls). The model reasons
through the chain in order - Retailer's order becomes Distributor's demand
signal, and so on - inside a single prompt/response. This cuts API calls
(and tokens, and run time) by 4x versus calling once per tier.

TWO LLM MODES:
- "isolated": each tier only ever sees the order placed by the tier below
  it - never the real customer demand (this is what causes bullwhip: a
  distorted, secondhand signal, like a game of telephone).
- "informed": every tier ALSO sees the real customer demand history
  directly, alongside its local signal. This tests whether the classic
  supply-chain finding "information sharing reduces the bullwhip effect"
  holds for LLM agents too - and whether an LLM, unlike a rigid formula,
  can use that visibility to actually distinguish real trend from noise
  rather than just overreacting less.

Get a free key at https://console.groq.com/keys
"""

import json
import os
import re
import time
from groq import Groq

MODEL_ID = "llama-3.1-8b-instant"
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2
TIER_ORDER = ["retailer", "distributor", "manufacturer", "supplier"]

_client = None
fallback_call_count = 0  # exposed so the UI can warn if a run degraded


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None
        _client = Groq(api_key=api_key)
    return _client


def reset_fallback_counter():
    global fallback_call_count
    fallback_call_count = 0


BATCH_SYSTEM_PROMPT_ISOLATED = """You are simulating one week of decisions
across a 4-tier supply chain: Retailer -> Distributor -> Manufacturer -> Supplier.

Demand signal flow this week (process in this exact order):
1. Retailer's demand signal = customer demand (given below).
2. Distributor's demand signal = the order quantity YOU decide for Retailer in step 1.
3. Manufacturer's demand signal = the order quantity YOU decide for Distributor in step 2.
4. Supplier's demand signal = the order quantity YOU decide for Manufacturer in step 3.

Each tier can ONLY see the demand signal described above - not the real
customer demand directly (except the Retailer, who sees it first-hand).

For each tier, decide an order quantity using this logic:
- Base decisions ONLY on the actual numbers given - do not assume a trend
  the numbers don't support. If a demand signal is 0, treat that as real.
- If a tier's demand signal is rising, it's realistic to order extra as a
  buffer (this tendency is what causes the bullwhip effect).
- If a tier has a backlog, it should order more to catch up.
- If a tier's inventory is piling up, it should order less.

Current state per tier:
{tier_details}

Customer demand this week: {customer_demand} units.

Respond ONLY with compact JSON in exactly this shape, no other text:
{{"retailer": {{"order_quantity": <int>, "reasoning": "<short sentence>"}},
 "distributor": {{"order_quantity": <int>, "reasoning": "<short sentence>"}},
 "manufacturer": {{"order_quantity": <int>, "reasoning": "<short sentence>"}},
 "supplier": {{"order_quantity": <int>, "reasoning": "<short sentence>"}}}}
"""

BATCH_SYSTEM_PROMPT_INFORMED = """You are simulating one week of decisions
across a 4-tier supply chain: Retailer -> Distributor -> Manufacturer -> Supplier.

UNLIKE a normal supply chain, every tier here has full visibility into the
REAL customer demand history (given below) - not just the order from the
tier below it. Use this to tell the difference between real trend shifts
and short-term noise, instead of just reacting to whatever the tier below
you ordered.

For each tier, decide an order quantity using this logic:
- Look at the real customer demand history/trend to judge whether a change
  is a genuine shift or just noise, and calibrate your order accordingly.
- Avoid overreacting to a single week's fluctuation if the broader trend
  is flat.
- If a tier has a backlog, it should order more to catch up.
- If a tier's inventory is piling up, it should order less.
- Base decisions on the actual numbers given - do not invent a trend the
  numbers don't support.

Current state per tier:
{tier_details}

Real customer demand history (most recent last): {demand_history}
Customer demand this week: {customer_demand} units.

Respond ONLY with compact JSON in exactly this shape, no other text:
{{"retailer": {{"order_quantity": <int>, "reasoning": "<short sentence>"}},
 "distributor": {{"order_quantity": <int>, "reasoning": "<short sentence>"}},
 "manufacturer": {{"order_quantity": <int>, "reasoning": "<short sentence>"}},
 "supplier": {{"order_quantity": <int>, "reasoning": "<short sentence>"}}}}
"""


def _fallback_all(customer_demand, tier_states, note=None):
    """Simple heuristic fallback for all 4 tiers, used only if the batch
    API call fails after retries (no key, sustained outage, etc.)."""
    result = {}
    demand_signal = customer_demand
    tag = "[fallback heuristic]" if not note else f"[fallback heuristic - {note}]"
    for tier in TIER_ORDER:
        backlog = tier_states[tier]["backlog"]
        qty = max(0, demand_signal + backlog // 2)
        result[tier] = (qty, tag)
        demand_signal = qty  # next tier's signal is this tier's order
    return result


def get_all_tier_decisions(customer_demand, tier_states, informed=False, demand_history=None):
    """
    tier_states: dict tier_name -> {"inventory": int, "backlog": int, "order_history": list}
    informed: if True, every tier also sees the real customer demand history
        (tests whether information sharing reduces bullwhip for LLM agents).
    demand_history: list of past customer demand values, required if informed=True.
    Returns dict tier_name -> (order_qty, reasoning_str) for all 4 tiers,
    from a SINGLE Groq API call.
    """
    global fallback_call_count

    client = get_client()
    if client is None:
        fallback_call_count += 1
        return _fallback_all(customer_demand, tier_states, note="no GROQ_API_KEY set")

    tier_lines = []
    for tier in TIER_ORDER:
        st = tier_states[tier]
        hist = ", ".join(str(x) for x in st["order_history"][-5:]) or "none yet"
        tier_lines.append(
            f"- {tier.capitalize()}: inventory={st['inventory']}, "
            f"backlog={st['backlog']}, last 5 orders=[{hist}]"
        )
    tier_details = "\n".join(tier_lines)

    if informed:
        hist_str = ", ".join(str(x) for x in (demand_history or [])[-10:]) or "none yet"
        prompt = BATCH_SYSTEM_PROMPT_INFORMED.format(
            tier_details=tier_details, customer_demand=customer_demand, demand_history=hist_str
        )
    else:
        prompt = BATCH_SYSTEM_PROMPT_ISOLATED.format(
            tier_details=tier_details, customer_demand=customer_demand
        )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7,
            )
            text = completion.choices[0].message.content
            match = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(match.group(0))
            result = {}
            for tier in TIER_ORDER:
                t = parsed.get(tier, {})
                qty = max(0, int(t.get("order_quantity", 0)))
                reasoning = t.get("reasoning", "")
                result[tier] = (qty, f"[{MODEL_ID}] {reasoning}")
            return result
        except Exception as e:
            last_error = f"attempt {attempt + 1} -> {type(e).__name__}: {str(e)[:150]}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            continue

    fallback_call_count += 1
    return _fallback_all(customer_demand, tier_states, note=f"batch call failed. {last_error}")
