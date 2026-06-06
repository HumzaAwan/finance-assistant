# AI-Powered Personal Finance Assistant

A **locally runnable** AI finance assistant — Streamlit chat + live dashboard, LangGraph agent, hybrid RAG (BM25 + vector + RRF), SSE streaming, budget tracking, and Prometheus observability. No Docker, no cloud API key required.

Full architecture → **`PROJECT_OVERVIEW.md`** · Audit trail → **`BACKEND_AUDIT_ROADMAP.md`**

---

## System Architecture

```mermaid
graph TB
    subgraph Browser["Browser"]
        ST["Streamlit :8501<br/>💬 Chat  |  📊 Dashboard"]
    end

    subgraph AgentSvc["Agent API  :8000  (FastAPI + LangGraph)"]
        direction TB
        EP_CHAT["POST /chat<br/>POST /chat/stream (SSE)"]
        EP_META["GET /health<br/>GET /metrics"]
        LG["LangGraph Graph"]
        MEM["FakeRedis / Redis<br/>Session Memory"]
        RAG["Hybrid RAG<br/>BM25 + Chroma + RRF"]
        OLL["Ollama :11434<br/>llama3.2 · nomic-embed-text"]

        EP_CHAT --> LG
        LG --> MEM
        LG --> RAG
        LG --> OLL
        RAG --> OLL
    end

    subgraph MockSvc["Mock Banking API  :8001  (FastAPI + SQLite)"]
        direction TB
        TXN["GET /transactions<br/>GET /transactions/summary"]
        ACC["GET /accounts<br/>GET /accounts/balance"]
        BUD["GET /budgets<br/>PUT /budgets"]
        PROM2["GET /metrics"]
        DB[("SQLite<br/>600 txns · 2 users<br/>accounts · budgets")]

        TXN & ACC & BUD --> DB
    end

    ST -->|"POST /chat<br/>POST /chat/stream"| EP_CHAT
    ST -->|"/metrics /accounts<br/>/budgets /summary"| MockSvc
    LG -->|"fetch transactions<br/>+ budget targets"| MockSvc
    EP_META --> AgentSvc
```

---

## LangGraph Agent Flow

```mermaid
flowchart TD
    A([User Message]) --> B[intent_router\nChatOllama classification\n+ heuristic fallback]

    B -->|transaction_query\ninsight_request\nfinancial_advice| C[transactions_node\nLLM extracts date·category·limit\nfetch /transactions + /budgets]
    B -->|general| G

    C --> D[insights_node\nPure-Python aggregation\ntotals · by-category · week-over-week\nbudget vs actual comparison]

    D -->|financial_advice| E[rag_node\nTopic filter by intent+keywords\nBM25 + vector search\nRRF fusion → top-4 chunks]
    D -->|transaction_query\ninsight_request| G

    E --> G[response_node\nAssemble system prompt\nInsights JSON + RAG lines\n+ memory snapshot\n→ ChatOllama synthesis]

    G -->|streaming_mode=False| H([Final response\nstored in Redis])
    G -->|streaming_mode=True| I([Prompt context returned\nto /chat/stream endpoint\n→ llm.astream tokens])
```

---

## RAG Pipeline

```mermaid
flowchart LR
    subgraph Ingest["Ingest  (run once)"]
        direction TB
        DOCS["11 Markdown\ndocuments"]
        MHS["MarkdownHeaderTextSplitter\nsection metadata → chunks"]
        RCS["RecursiveCharacterTextSplitter\nchunk_size=800  overlap=160"]
        EMB["OllamaEmbeddings\nnomic-embed-text"]
        CHR[("Chroma\nlocal_data/chroma/")]

        DOCS --> MHS --> RCS --> EMB --> CHR
    end

    subgraph Retrieve["Retrieve  (per query)"]
        direction TB
        Q["User Query"]
        VEC["Dense: embed query\n→ Chroma similarity\n→ ranked list A"]
        BM["Sparse: BM25Okapi\n→ keyword scores\n→ ranked list B"]
        RRF["Reciprocal Rank Fusion\nscore = Σ 1 / 60 + rank\n→ fused ranked list"]
        THR["Score threshold ≥ 0.20\n→ top-4 chunks"]

        Q --> VEC & BM
        VEC & BM --> RRF --> THR
    end

    subgraph Filter["Topic Filter  (rag_node)"]
        direction TB
        MAP["intent + keywords\n→ topic whitelist"]
        WH["where topic IN list\napplied to Chroma + BM25"]
        MAP --> WH
    end

    CHR -.->|"build BM25 index\nat startup"| BM
    CHR --> VEC
    Filter --> VEC
    Filter --> BM
```

---

## Prerequisites

- Python **3.11–3.13**
- [Ollama](https://ollama.com) running locally

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

---

## Setup and run

```bash
cd "finance-assistant/"

python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
pip install --upgrade pip
pip install -r requirements-local.txt

python run_local.py
```

| Flag | Effect |
|---|---|
| `--skip-ingest` | Reuse existing Chroma index (fast restart) |
| `--no-ui` | APIs only, no Streamlit |
| `--free-ports` | Windows: kill stale processes on 8000/8001/8501 |

Stop with **Ctrl+C**.

---

## URLs

| Service | URL |
|---|---|
| Streamlit — Chat + Dashboard | http://127.0.0.1:8501 |
| Agent API docs | http://127.0.0.1:8000/docs |
| Mock Banking API docs | http://127.0.0.1:8001/docs |
| Agent Prometheus metrics | http://127.0.0.1:8000/metrics |
| Mock API Prometheus metrics | http://127.0.0.1:8001/metrics |

---

## Example questions

**Transactions / Insights**
- "List my recent spending"
- "What did I spend on food this month?"
- "Compare my spending this week vs last week"
- "Which category am I overspending in?"

**Financial advice (triggers RAG)**
- "How much should I have in my emergency fund?"
- "Explain the 50/30/20 rule with my income in mind"
- "Should I use the debt avalanche or snowball method?"
- "What's the difference between a Roth IRA and a 401k?"
- "How does my spending compare to the average American?"

---

## Resetting local data

```bash
rm -rf services/mock-api/data/    # SQLite DB (transactions, accounts, budgets)
rm -rf local_data/chroma/          # Vector database
python run_local.py                # Rebuilds everything from scratch
```
