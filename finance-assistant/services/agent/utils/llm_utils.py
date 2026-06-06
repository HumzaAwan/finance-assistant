"""Retry-aware LLM invocation helpers.

All graph nodes should call ``safe_invoke`` / ``safe_ainvoke`` instead of
``llm.invoke`` directly so transient Ollama errors (connection reset,
model-loading cold-start) are automatically retried with backoff.
"""

from __future__ import annotations

import logging
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_log = logging.getLogger("agent.utils.llm_utils")

# Retry on any exception coming from the LLM layer.
# 3 attempts, 2 s → 4 s exponential backoff.
_RETRY = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    before_sleep=before_sleep_log(_log, logging.WARNING),
    reraise=True,
)


@_RETRY
def safe_invoke(llm: Any, messages: list) -> Any:
    """Invoke ``llm`` with automatic retry on transient errors."""
    return llm.invoke(messages)


def safe_invoke_or_none(llm: Any, messages: list) -> Any | None:
    """Like ``safe_invoke`` but returns ``None`` instead of raising on final failure."""
    try:
        return safe_invoke(llm, messages)
    except Exception as exc:
        _log.warning({"event": "llm_invoke_failed_all_retries", "detail": repr(exc)})
        return None
