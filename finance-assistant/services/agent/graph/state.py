from typing import List, Optional, TypedDict

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
