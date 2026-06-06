"""Compiled LangGraph agent wiring — async edition.

Graph topology
--------------
                          ┌──────────────────┐
                          │  intent_router   │
                          └────────┬─────────┘
          ┌────────────┬───────────┼──────────┬──────────────┐
          ▼            ▼           ▼           ▼              ▼
   tx/insight/  financial_  anomaly_    general →       [future]
   fin_advice   health      check      response_node
          │            │           │
          ▼            ▼           ▼
   transactions_node (async + structured output)
          │
          ▼
    insights_node (async)
          │
    ┌─────┼──────────────┬──────────────────┐
    ▼     ▼              ▼                  ▼
 rag_   response_  financial_health_  anomalies_
 node   node       node               node
    │            │                  │
    └────────────┴──────────────────┘
                 │
           response_node → END

financial_advice flows through the full pipeline then enriches with RAG.
financial_health and anomaly_check branch after insights_node.
Errors in any node are caught by the graph's built-in exception handler
and routed to a graceful error response.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from graph.nodes.anomalies_node import anomalies_node
from graph.nodes.financial_health_node import financial_health_node
from graph.nodes.insights_node import insights_node
from graph.nodes.intent_router import intent_router
from graph.nodes.rag_node import rag_node
from graph.nodes.response_node import response_node
from graph.nodes.transactions_node import transactions_node
from graph.state import AgentState

_log = logging.getLogger("agent.graph.agent_graph")


def route_intent(state: AgentState) -> str:
    """Route after intent classification.

    All intents that need transaction data go to transactions_node first.
    Unclear intent and general / meta queries skip directly to response_node.
    """
    intent = state.get("intent", "general")
    if intent in {
        "transaction_query", "insight_request", "financial_advice",
        "financial_health", "anomaly_check",
    }:
        return "transactions_node"
    return "response_node"


def route_after_insights(state: AgentState) -> str:
    """After aggregating insights, route to the appropriate specialised node."""
    intent = state.get("intent", "general")
    if intent == "financial_advice":
        return "rag_node"
    if intent == "financial_health":
        return "financial_health_node"
    if intent == "anomaly_check":
        return "anomalies_node"
    return "response_node"


def compile_graph():
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("intent_router", intent_router)
    workflow.add_node("transactions_node", transactions_node)
    workflow.add_node("insights_node", insights_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("financial_health_node", financial_health_node)
    workflow.add_node("anomalies_node", anomalies_node)
    workflow.add_node("response_node", response_node)

    workflow.set_entry_point("intent_router")

    # intent_router → either transactions pipeline or direct response
    workflow.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "transactions_node": "transactions_node",
            "response_node": "response_node",
        },
    )

    workflow.add_edge("transactions_node", "insights_node")

    # insights_node branches based on intent
    workflow.add_conditional_edges(
        "insights_node",
        route_after_insights,
        {
            "rag_node": "rag_node",
            "financial_health_node": "financial_health_node",
            "anomalies_node": "anomalies_node",
            "response_node": "response_node",
        },
    )

    workflow.add_edge("rag_node", "response_node")
    workflow.add_edge("financial_health_node", "response_node")
    workflow.add_edge("anomalies_node", "response_node")
    workflow.add_edge("response_node", END)

    graph = workflow.compile()
    _log.info({"event": "graph_compiled", "nodes": 7})
    return graph


GRAPH = compile_graph()
