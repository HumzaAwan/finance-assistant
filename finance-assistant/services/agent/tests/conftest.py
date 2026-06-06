"""Pytest configuration and shared fixtures for the agent test suite."""
from __future__ import annotations

import os

import pytest

# Set minimal environment variables required by node imports so tests can
# run without a full .env / running services.
os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
os.environ.setdefault("OLLAMA_MODEL", "llama3.2")
os.environ.setdefault("OLLAMA_EMBED_MODEL", "nomic-embed-text")
os.environ.setdefault("CHROMA_COLLECTION", "finance_knowledge")
os.environ.setdefault("BANKING_API_URL", "http://127.0.0.1:8001")
os.environ.setdefault("DEFAULT_USER_ID", "user_001")
os.environ.setdefault("LOG_LEVEL", "WARNING")
