import logging

from langchain_core.messages import HumanMessage

from graph.state import AgentState
from rag.retriever import get_retriever

_log = logging.getLogger("agent.graph.nodes.rag_node")

# Map intent → relevant document topics for metadata-filtered retrieval.
# Topics match the `topic` field stored as chunk metadata at ingest time.
_INTENT_TOPICS: dict[str, list[str]] = {
    "financial_advice": [
        "50_30_20_rule",
        "budgeting_strategies",
        "saving_techniques",
        "financial_literacy",
        "emergency_fund_sizing",
        "debt_payoff_strategies",
        "investment_basics",
        "tax_saving_accounts",
        "spending_benchmarks",
        "subscription_audit",
    ],
}

# Keywords in the user query → narrow the topic filter further.
_KEYWORD_TOPICS: list[tuple[tuple[str, ...], list[str]]] = [
    (("emergency fund", "emergency"), ["emergency_fund_sizing", "saving_techniques"]),
    (("budget", "50/30/20", "50 30 20"), ["50_30_20_rule", "budgeting_strategies"]),
    (("debt", "loan", "credit card", "payoff", "avalanche", "snowball"), ["debt_payoff_strategies"]),
    (("invest", "index fund", "stock", "compound"), ["investment_basics", "financial_literacy"]),
    (("credit score", "fico", "utilization"), ["credit_score_factors"]),
    (("401k", "ira", "hsa", "roth", "retirement"), ["tax_saving_accounts"]),
    (("subscription", "cancel", "streaming"), ["subscription_audit"]),
    (("benchmark", "average spending", "typical"), ["spending_benchmarks"]),
    (("save", "saving"), ["saving_techniques", "emergency_fund_sizing"]),
]


def _topic_filter(user_text: str, intent: str) -> list[str] | None:
    """Return a topic whitelist for Chroma filtering, or None for a full search."""
    base = _INTENT_TOPICS.get(intent)
    if not base:
        return None  # general / transaction queries: search all topics

    lowered = user_text.lower()
    for keywords, topics in _KEYWORD_TOPICS:
        if any(kw in lowered for kw in keywords):
            # Narrow to the most relevant topics while keeping all as fallback
            narrowed = [t for t in topics if t in base]
            if narrowed:
                return narrowed

    return base  # use full advice topic list


def _latest_user(messages) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def rag_node(state: AgentState):
    retriever = get_retriever()

    user_text = _latest_user(list(state.get("messages", []) or []))
    intent = state.get("intent", "general")
    query = user_text or "personal finance guidance"

    topics = _topic_filter(query, intent)
    snippets = retriever.retrieve(query, top_k=4, topic_filter=topics)

    _log.info({"event": "rag_context_built", "snippets": len(snippets), "topics": topics})

    return {"rag_context": snippets}
