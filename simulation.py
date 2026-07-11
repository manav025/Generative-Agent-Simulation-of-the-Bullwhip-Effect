"""
simulation.py
Core simulation engine. Uses LangGraph to orchestrate a 4-tier supply chain
(Retailer -> Distributor -> Manufacturer -> Supplier) week by week.

Each week:
  1. Customer demand hits the Retailer.
  2. Each tier observes the order from the tier below (its "demand signal"),
     decides how much to order from the tier above, ships what it can from
     inventory, and carries forward any backlog.
  3. We record every tier's order quantity so we can compute the bullwhip
     ratio at the end: Var(tier's orders) / Var(customer demand).

Two modes:
  - "llm"       -> each tier's decision comes from llm_client.get_llm_order_decision
  - "classical" -> each tier's decision comes from classical_policy.order_up_to_decision
"""

from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END

from classical_policy import order_up_to_decision
from llm_client import get_llm_order_decision

TIERS = ["retailer", "distributor", "manufacturer", "supplier"]
LEAD_TIME = 2  # weeks for an order to arrive as usable inventory


class SimState(TypedDict):
    week: int
    total_weeks: int
    mode: Literal["llm", "classical"]
    customer_demand_series: List[int]

    inventory: dict          # tier -> current on-hand inventory
    backlog: dict            # tier -> unfilled orders owed downstream
    pipeline: dict           # tier -> list of orders in transit (length = LEAD_TIME)
    order_history: dict      # tier -> list of all order quantities placed
    demand_signal_history: dict  # tier -> list of demand signals it received
    reasoning_log: List[dict]    # for the UI: {week, tier, reasoning}


def _init_state(total_weeks, mode, customer_demand_series) -> SimState:
    return SimState(
        week=0,
        total_weeks=total_weeks,
        mode=mode,
        customer_demand_series=customer_demand_series,
        inventory={t: 50 for t in TIERS},
        backlog={t: 0 for t in TIERS},
        pipeline={t: [0] * LEAD_TIME for t in TIERS},
        order_history={t: [] for t in TIERS},
        demand_signal_history={t: [] for t in TIERS},
        reasoning_log=[],
    )


def _tier_node_factory(tier_name, tier_index):
    """Builds a LangGraph node function for a given supply-chain tier."""

    def node(state: SimState) -> SimState:
        week = state["week"]

        # Demand signal = customer demand for retailer, else the order the
        # tier below just placed.
        if tier_index == 0:
            demand_signal = state["customer_demand_series"][week]
        else:
            downstream = TIERS[tier_index - 1]
            demand_signal = state["order_history"][downstream][-1] if state["order_history"][downstream] else 0

        inventory = state["inventory"][tier_name]
        backlog = state["backlog"][tier_name]

        if state["mode"] == "llm":
            qty, reasoning = get_llm_order_decision(
                role=tier_name.capitalize(),
                demand_signal=demand_signal,
                inventory=inventory,
                backlog=backlog,
                order_history=state["order_history"][tier_name],
            )
        else:
            qty = order_up_to_decision(demand_signal, inventory, backlog, state["pipeline"][tier_name])
            reasoning = "order-up-to-S heuristic"

        # Fulfil this tier's demand signal from current inventory.
        shipped = min(inventory, demand_signal + backlog)
        new_inventory = inventory - shipped
        new_backlog = max(0, (demand_signal + backlog) - shipped)

        # Receive the oldest in-transit order into inventory (lead time).
        arriving = state["pipeline"][tier_name][0]
        new_inventory += arriving
        new_pipeline = state["pipeline"][tier_name][1:] + [qty]

        state["inventory"][tier_name] = new_inventory
        state["backlog"][tier_name] = new_backlog
        state["pipeline"][tier_name] = new_pipeline
        state["order_history"][tier_name].append(qty)
        state["demand_signal_history"][tier_name].append(demand_signal)
        state["reasoning_log"].append(
            {"week": week, "tier": tier_name, "order_qty": qty, "reasoning": reasoning}
        )
        return state

    return node


def _advance_week(state: SimState) -> SimState:
    state["week"] += 1
    return state


def build_graph():
    graph = StateGraph(SimState)

    for i, tier in enumerate(TIERS):
        graph.add_node(tier, _tier_node_factory(tier, i))

    graph.add_node("advance_week", _advance_week)

    graph.set_entry_point(TIERS[0])
    for i in range(len(TIERS) - 1):
        graph.add_edge(TIERS[i], TIERS[i + 1])
    graph.add_edge(TIERS[-1], "advance_week")
    graph.add_edge("advance_week", END)

    return graph.compile()


def run_simulation(total_weeks, mode, customer_demand_series):
    """
    Runs the full multi-week simulation by invoking the compiled LangGraph
    once per week (LangGraph handles one 4-tier pass per invocation here;
    we loop weeks in Python for clarity and to keep state simple).
    """
    app = build_graph()
    state = _init_state(total_weeks, mode, customer_demand_series)

    for _ in range(total_weeks):
        state = app.invoke(state)

    return state
