from typing import Any, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    user_id: str
    session_id: str
    intent: str
    transaction_data: Optional[dict]
    insights: Optional[dict]
    rag_context: Optional[List[dict]]
    final_response: str
    route_hint: Optional[str]
    # Populated once in agent_app.py before graph invocation so nodes never
    # call memory.get_history() themselves (which would read a stale snapshot
    # because messages are only persisted after GRAPH.invoke returns).
    memory_snapshot: Optional[List[Any]]
    # When True, response_node skips the LLM call and returns the assembled
    # prompt context so the /chat/stream endpoint can stream it token-by-token.
    streaming_mode: Optional[bool]
    streaming_context: Optional[dict]
