# AI-Powered Personal Finance Assistant — Build Plan

This document summarizes **what we are building**, **how we will implement it**, and **what still needs confirmation** before implementation starts.

---

## 1. Goal (confirmed)

Deliver a **fully local** personal finance assistant that starts with:

```bash
docker-compose up --build
```

No separate Python virtualenv or host installs required for normal use (Docker Desktop + enough RAM).

**Architecture rule:**  
- Root `docker-compose.yml` orchestrates everything.  
- **Every custom service has its own Dockerfile** (`services/ollama`, `services/mock-api`, `services/agent`, `services/frontend`, `services/n8n`).  
- **Redis** and **ChromaDB** use official images only (no custom Dockerfile).

---

## 2. Repository layout vs your machine

Your spec names the repo root **`finance-assistant/`**.  
Current Cursor workspace folder is **`AI-powered Personal Finance Assistant`**.

**Planned approach (unless you say otherwise):**  
Create the structure **inside this workspace**, matching your tree exactly (`docker-compose.yml`, `.env`, `services/`, etc.). Optionally we rename the outer folder later or you clone into `finance-assistant` — functionality is unchanged.

---

## 3. What we build (inventory)

| Area | Responsibility |
|------|----------------|
| **docker-compose.yml** | All services, named network, volumes, `depends_on` + `condition: service_healthy` per your spec |
| **.env** | Inter-service URLs (Docker DNS names), Ollama/Chroma/Redis, n8n auth, defaults |
| **Ollama** | Custom image: extend `ollama/ollama`, start server, pull `llama3.2` + `nomic-embed-text`, health on `/api/tags` |
| **mock-api** | FastAPI SQLite mock bank: seed 150 txs (idempotent), JSON export, filters + summary endpoints |
| **agent** | FastAPI async: LangGraph (intent router → transactional path OR RAG path OR direct response), Ollama LLM/embeddings, Chroma RAG, Redis chat memory |
| **frontend** | Streamlit chat UI → `AGENT_API_URL/chat` |
| **n8n** | Custom image + workflow webhook → optionally call agent |

**Ports (host)**  
11434 · 6379 · 8002 · 8001 · 8000 · 8501 · 5678 — as specified.

---

## 4. How we will implement (phased execution)

Implementation will follow dependency order — same mental model as Compose health ordering.

### Phase A — Skeleton & Compose

1. Create directory tree (`services/…` folders, `documents/`, `workflows/`).
2. Add root `docker-compose.yml`, `.env`, `.dockerignore` where helpful.
3. Wire volumes and healthchecks; fix any mismatches discovered at run time (e.g. image healthcheck quirks).

### Phase B — Infrastructure services (no custom code)

1. Confirm Redis/Chroma compose definitions match running images.

### Phase C — Ollama service

1. Dockerfile + model pull logic (aligned: either **`pull_models.sh` as single entrypoint** after `ollama serve`, or inlined script — avoid dead `pull_models.sh`).
2. Ensure first boot allows **long enough** `start_period` so pulls complete before agent depends on healthy Ollama.

### Phase D — Mock banking API

1. SQLAlchemy models + `/app/data/finance.db`.
2. Idempotent seed (Faker, patterns you specified, skip if seeded).
3. Export `data/mock_transactions.json`.
4. Routers: list, get by id, summary; `/health`.
5. Async FastAPI handlers + structured logging module-wide.

### Phase E — Agent: RAG

1. Author three Markdown docs (~600+ words each per spec).
2. `ingest.py`: splitter, `OllamaEmbeddings`, Chroma collection recreate, metadata.
3. `retriever.py`: `RAGRetriever.retrieve`.

### Phase F — Agent: LangGraph + API

1. `AgentState`, nodes (intent via LLM only; httpx → mock-api; insights math; RAG node; response node merging context + Redis history).
2. `agent_graph.py` with edges exactly as specified.
3. `main.py`: `POST /chat`, history GET/DELETE, `/health`; load/save Redis memory; **`async`** endpoints throughout.

### Phase G — Frontend

1. Streamlit layout, session state, suggestion buttons, httpx calls to agent.

### Phase H — n8n

1. Dockerfile (workflow JSON in place; permissions).
2. Valid `finance_workflow.json` with nodes/branches described.

### Phase I — README

1. Prerequisites, commands, URLs table, ASCII architecture, examples, teardown.

---

## 5. Spec alignment & technical notes (resolved during build)

These are **not blockers**; we handle them explicitly in implementation:

| Topic | Resolution |
|--------|------------|
| **`/health` on mock-api** | Single canonical endpoint in **`main.py`** (router duplication avoided unless you insist both). Healthcheck stays `GET /health`. |
| **Ollama entrypoint vs `pull_models.sh`** | Use **one consistent path**: e.g. entrypoint invokes script that waits for API then pulls — matches your FILES 2 & 3 without leaving unused script. |
| **n8n sends `route_hint`** | Extend `POST /chat` body with **optional** `route_hint` (ignored or lightly biases prompts) so n8n does not break validation. |
| **LangChain / LangGraph pins** | Start from your pinned versions; if import/API drift appears at build time, bump minimally for Python 3.11 compatibility — no placeholders. |

---

## 6. Verification checklist (`docker-compose up --build`)

- [ ] All services reach **healthy** (or running where no healthcheck).
- [ ] Frontend chat returns sensible answers for sample intents.
- [ ] Mock API docs and seeded data reachable.
- [ ] Agent `/docs` shows async routes; RAG ingests on startup without duplicate explosions.
- [ ] n8n webhook reachable and returns merged response structure.

---

## 7. Questions — please confirm for 100% alignment

Answer these so implementation matches your expectations exactly:

1. **Root folder name** — Keep content in the current workspace (`AI-powered Personal Finance Assistant`), or do you require the root directory to literally be **`finance-assistant`** (rename/move)?

2. **SQLite / JSON location** — Spec uses `/app/data/finance.db` inside the container. OK to ensure **`services/mock-api/data/`** exists with a `.gitkeep` and optional empty JSON overwritten on seed?

3. **n8n “import on startup”** — Copying `finance_workflow.json` into `.n8n/workflows/` **does not always register** workflows in newer n8n versions. Prefer: **A)** Dockerfile copy only (manual activate in UI), **B)** add a small startup script/API import (more moving parts), or **C)** document “import JSON from UI” while still versioning the JSON in-repo?

4. **`route_hint` semantics** — Should **`advice` / `data`** from n8n **force** subgraph routing (skip LLM intent), **soft-hint** in the prompt, or **ignored** (strict LangGraphClassifier-only behavior)?

5. **Secrets in `.env`** — `admin123` in repo `.env` is fine for **local demos**; should we add **`.env.example`** (same keys, dummy passwords) and gitignore `.env`, or ship **committed `.env`** as in your spec for true one-command clone?

6. **`GET /transactions/summary`** — FastAPI resolves routes in order; path **`/transactions/summary`** must be declared **before** **`/transactions/{id}`** to avoid `{id}` capturing `"summary"`. Confirm **no alternate path** naming required.

Once you reply to §7, execution can proceed **file-by-file** with **no TODOs**, ending in **`docker-compose up --build`** as the single user command.

---

## 8. After your confirmation

Next step: scaffold the repo and implement phases A→I, then run Compose locally to validate health and a short E2E chat path.
