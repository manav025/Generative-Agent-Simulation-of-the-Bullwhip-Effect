"""
metrics.py
Computes the bullwhip ratio for each tier:
    bullwhip_ratio(tier) = Var(tier's order quantities) / Var(customer demand)

A ratio > 1 means that tier is amplifying demand variability (the bullwhip
effect). The further up the chain, the higher this usually gets — that
amplification pattern is the entire point of the demo.
"""

import numpy as np


def bullwhip_ratios(customer_demand_series, order_history):
    demand_var = np.var(customer_demand_series)
    ratios = {}
    for tier, orders in order_history.items():
        tier_var = np.var(orders)
        ratios[tier] = round(tier_var / demand_var, 3) if demand_var > 0 else float("nan")
    return ratios
