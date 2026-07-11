"""
simulation.py
Core simulation engine. Uses LangGraph to orchestrate a 4-tier supply chain
(Retailer -> Distributor -> Manufacturer -> Supplier) week by week.

Each week:
  1. In "llm" mode: ONE batched API call decides all 4 tiers' order
     quantities together (see llm_client.get_all_tier_decisions) - this
     keeps run time and token usage low (1 call/week instead of 4).
     In "classical" mode: each tier applies the order-up-to-S formula.
  2. Each tier ships what it can from inventory, carries forward backlog,
     and receives inventory arriving from its own earlier order (lead time).
  3. We record every tier's order quantity so we can compute the bullwhip
     ratio at the end: Var(tier's orders) / Var(customer demand).
"""

from typing import TypedDict, List, Literal, Optional
from langgraph.graph import StateGraph, END

from classical_policy import order_up_to_decision
from llm_client import get_all_tier_decisions

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
    batch_decisions: Optional[dict]  # this week's LLM decisions, set once per week


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
        batch_decisions=None,
    )


def _llm_batch_decide(state: SimState) -> SimState:
    """Runs once per week, before the tier nodes, when mode == 'llm'.
    Makes a SINGLE API call that decides all 4 tiers' order quantities,
    instead of one call per tier."""
    if state["mode"] != "llm":
        return state

    week = state["week"]
    customer_demand = state["customer_demand_series"][week]
    tier_states = {
        t: {
            "inventory": state["inventory"][t],
            "backlog": state["backlog"][t],
            "order_history": state["order_history"][t],
        }
        for t in TIERS
    }
    state["batch_decisions"] = get_all_tier_decisions(customer_demand, tier_states)
    return state


def _tier_node_factory(tier_name, tier_index):
    """Builds a LangGraph node function for a given supply-chain tier."""

    def node(state: SimState) -> SimState:
        week = state["week"]

        # Demand signal = customer demand for retailer, else the order the
        # tier below just placed (used for logging/classical policy).
        if tier_index == 0:
            demand_signal = state["customer_demand_series"][week]
        else:
            downstream = TIERS[tier_index - 1]
            demand_signal = state["order_history"][downstream][-1] if state["order_history"][downstream] else 0

        inventory = state["inventory"][tier_name]
        backlog = state["backlog"][tier_name]

        if state["mode"] == "llm":
            qty, reasoning = state["batch_decisions"][tier_name]
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

    graph.add_node("llm_batch_decide", _llm_batch_decide)
    for i, tier in enumerate(TIERS):
        graph.add_node(tier, _tier_node_factory(tier, i))
    graph.add_node("advance_week", _advance_week)

    graph.set_entry_point("llm_batch_decide")
    graph.add_edge("llm_batch_decide", TIERS[0])
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
