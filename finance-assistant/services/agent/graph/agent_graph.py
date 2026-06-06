"""Compiled LangGraph agent wiring.

Graph topology
--------------
                          ┌──────────────────┐
                          │  intent_router   │
                          └────────┬─────────┘
               ┌──────────────────┼─────────────────┐
               ▼                  ▼                  ▼
    transaction_query /    financial_advice       general
      insight_request
               │                  │
               ▼                  ▼
       transactions_node  transactions_node   ──► response_node
               │                  │
               ▼                  ▼
        insights_node      insights_node
               │                  │
               ▼                  ▼
        response_node ◄──── rag_node
                                  │
                            response_node

financial_advice flows through the full transaction pipeline first so the
LLM has access to the user's actual spending data when giving advice,
then enriches with curated knowledge via rag_node before the final answer.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

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
    Only truly general / meta queries skip directly to response_node.
    """
    intent = state.get("intent", "general")
    if intent in {"transaction_query", "insight_request", "financial_advice"}:
        return "transactions_node"
    return "response_node"


def route_after_insights(state: AgentState) -> str:
    """After aggregating insights, advice queries enrich with RAG before
    responding; data-only queries go straight to the response node.
    """
    if state.get("intent") == "financial_advice":
        return "rag_node"
    return "response_node"


def compile_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("intent_router", intent_router)
    workflow.add_node("transactions_node", transactions_node)
    workflow.add_node("insights_node", insights_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("response_node", response_node)

    workflow.set_entry_point("intent_router")

    workflow.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "transactions_node": "transactions_node",
            "response_node": "response_node",
        },
    )

    workflow.add_edge("transactions_node", "insights_node")

    # insights_node branches: advice → rag_node → response_node
    #                         queries → response_node directly
    workflow.add_conditional_edges(
        "insights_node",
        route_after_insights,
        {
            "rag_node": "rag_node",
            "response_node": "response_node",
        },
    )

    workflow.add_edge("rag_node", "response_node")
    workflow.add_edge("response_node", END)

    graph = workflow.compile()
    _log.info({"event": "graph_compiled"})
    return graph


GRAPH = compile_graph()

