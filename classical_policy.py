"""
classical_policy.py
The textbook "order-up-to" inventory policy used in every SCM/OR course.
This is the baseline we compare the LLM-agent chain against, so we can
quantify whether generative agents amplify or dampen the bullwhip effect
versus standard operations-research heuristics.
"""


def order_up_to_decision(demand_signal, inventory, backlog, pipeline, safety_factor=1.5):
    """
    Classic order-up-to-S policy:
    Target level S = safety_factor * expected demand.
    Inventory position = on-hand inventory + orders already in transit - backlog.
    Order = S - inventory_position
    (Including the pipeline is essential - without it, the policy double-orders
    for stock that's already on its way, causing unrealistic oscillation.)
    """
    target_level = safety_factor * demand_signal
    inventory_position = inventory + sum(pipeline) - backlog
    order_qty = target_level - inventory_position
    return max(0, round(order_qty))
