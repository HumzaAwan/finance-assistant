import logging

from langchain_core.messages import HumanMessage

from graph.state import AgentState
from rag.retriever import RAGRetriever

_log = logging.getLogger("agent.graph.nodes.rag_node")


def _latest_user(messages) -> str:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def rag_node(state: AgentState):

    retriever = RAGRetriever()

    user_text = _latest_user(list(state.get("messages", []) or []))

    snippets = retriever.retrieve(user_text or "personal finance guidance", top_k=3)

    _log.info({"event": "rag_context_built", **{"snippets": len(snippets)}})

    return {"rag_context": snippets}
