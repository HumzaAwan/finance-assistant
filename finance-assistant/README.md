# AI-Powered Personal Finance Assistant

A locally runnable demo: Streamlit chat → LangGraph agent → mock banking API, Ollama for LLM + embeddings, and embedded Chroma persistence for RAG. **No Docker required.**

**Setup (short steps):** **`SETUP_GUIDE.md`**. Architecture: **`PROJECT_OVERVIEW.md`**.

## Prerequisites

Python **3.11–3.13**, **[Ollama](https://ollama.com)** with `llama3.2` and `nomic-embed-text`. See **`SETUP_GUIDE.md`** for venv, `.env`, and `run_local.py`.

## Run everything

From `finance-assistant/`:

```bash
python run_local.py
```

Then open:

- Frontend: http://127.0.0.1:8501  
- Agent API: http://127.0.0.1:8000/docs  
- Mock banking API: http://127.0.0.1:8001/docs  

**Flags:** `--skip-ingest` (reuse Chroma index), `--no-ui` (no Streamlit), `--free-ports` (Windows port cleanup).

Stop with Ctrl+C.

**pip / Windows builds:** upgrade pip and use pinned versions in `requirements-local.txt`; Chroma 1.5+ ships wheels.

### Optional Redis

Set `REDIS_USE_FAKEREDIS=false` in `.env` and point `REDIS_URL` at a reachable Redis (`redis://127.0.0.1:6379`).

### Optional standalone Chroma HTTP server

If you operate a separate Chroma service, set `CHROMA_MODE=http`, `CHROMA_HOST`, and `CHROMA_PORT` accordingly.

### Optional n8n

Not started by `run_local.py`. Install n8n via Node (`npm install -g n8n`), run it separately, import `services/n8n/workflows/finance_workflow.json`, and point HTTP Request nodes at `http://127.0.0.1:8000/chat`.

## Architecture (same logical flow)

```
User → Streamlit (8501) → Agent API (8000)
                              ├ Mock bank API (8001)
                              ├ Ollama (11434)
                              ├ Embedded Chroma (local_data/)
                              └ RedisMemory (fakeredis unless you enable Redis)
```

For a full walkthrough (LangChain, LangGraph, Chroma, Redis, Streamlit, n8n, mock API): see **`PROJECT_OVERVIEW.md`**.

## Example queries

- **Transactions:** “List my recent debits,” “What did I spend on dining this week?”
- **Insights:** “Summarize my cash flow,” “What drives most spending?”
- **RAG / advice:** “How do I build a starter emergency fund?” “Explain 50/30/20 simply.”

## Reset local data

- Delete **`services/mock-api/data/`** (SQLite DB + JSON export).
- Delete `local_data/chroma/` to clear the vector DB
- Set `REDIS_USE_FAKEREDIS=false` and flush Redis keys if using real Redis
