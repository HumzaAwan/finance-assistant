# AI-Powered Personal Finance Assistant — Project Overview

> Full audit and improvement history: `BACKEND_AUDIT_ROADMAP.md`

---

## 1. High-level architecture

```mermaid
graph TB
    subgraph UI["Frontend  :8501"]
        CHAT["💬 Chat tab\nStarter prompts · session memory\nPOST /chat or /chat/stream"]
        DASH["📊 Dashboard tab\nHealth · metrics · accounts\nbudget vs actual · intent chart"]
    end

    subgraph Agent["Agent API  :8000"]
        direction TB
        API["FastAPI\n/chat · /chat/stream\n/health · /metrics"]
        GRAPH["LangGraph\nstateful multi-node graph"]
        REDIS["FakeRedis / Redis\nsession memory per session_id"]
        RETRIEVER["Hybrid RAG Retriever\nBM25 + Chroma + RRF"]
        CHROMA[("Chroma\nlocal_data/chroma/\n11 documents · 800-char chunks")]
        OLLAMA["Ollama  :11434\nllama3.2 (chat)\nnomic-embed-text (embed)"]
        PROM1["Prometheus /metrics\nrequests · latency · RAG hits"]

        API --> GRAPH
        GRAPH --> REDIS
        GRAPH --> RETRIEVER
        RETRIEVER --> CHROMA
        RETRIEVER --> OLLAMA
        GRAPH --> OLLAMA
    end

    subgraph Mock["Mock Banking API  :8001"]
        direction TB
        ROUTES["FastAPI routes\n/transactions · /accounts\n/budgets · /metrics"]
        DB[("SQLite\n2 users · 600 transactions\n4 accounts · 12 budget targets")]
        PROM2["Prometheus middleware\nper-endpoint counters + latency"]

        ROUTES --> DB
    end

    UI -->|"POST /chat\nPOST /chat/stream"| API
    UI -->|"GET /accounts\nGET /budgets\nGET /metrics\nGET /summary"| Mock
    GRAPH -->|"GET /transactions\nGET /budgets"| Mock
```

---

## 2. LangGraph agent — full workflow

```mermaid
flowchart TD
    START([User message arrives\nPOST /chat or /chat/stream]) --> SNAP

    SNAP["agent_app.py\nload memory_snapshot from Redis\nbuild AgentState\nGRAPH.invoke or GRAPH.ainvoke"]

    SNAP --> IR

    subgraph Graph["LangGraph Graph"]
        IR["intent_router\n──────────────────\nChatOllama classification\nheuristic fallback if LLM down\nledger-language boost\nsafe_invoke_or_none + tenacity retry"]

        IR -->|"transaction_query\ninsight_request\nfinancial_advice"| TN

        TN["transactions_node\n──────────────────\nChatOllama extracts:\nstart_date · end_date · category · limit\nGET /transactions on mock API\nGET /budgets for advice+insight intents\nwiden query on empty results"]

        TN --> IN

        IN["insights_node\n──────────────────\ntotals · by_category\navg daily · biggest transaction\ncalendar week comparison\nbudget_comparison: pct_used\nover_budget_categories list"]

        IN -->|"financial_advice"| RN
        IN -->|"transaction_query\ninsight_request"| RES

        RN["rag_node\n──────────────────\nmap intent + keywords → topic whitelist\ncall RAGRetriever.retrieve\nBM25 + vector + RRF\ntop-4 chunks with score ≥ 0.20"]

        RN --> RES

        IR -->|"general"| RES

        RES["response_node\n──────────────────\nassemble system prompt\nhuman payload:\n  • insights JSON\n  • transaction preview ≤35 rows\n  • RAG chunk lines\n  • memory_snapshot last-5 turns\n\nIF streaming_mode=True:\n  → return streaming_context dict\n  → skip LLM call\n\nIF streaming_mode=False:\n  → safe_invoke_or_none ChatOllama\n  → fallback: _offline_summary"]
    end

    RES -->|"streaming_mode=False"| WRITE["Write user+assistant turns\nto Redis\nReturn JSON response"]

    RES -->|"streaming_mode=True\n/chat/stream endpoint"| STREAM["llm.astream prompt_context\nyield data:{token:...} SSE events\nWrite Redis after final token\nyield data:[DONE]"]

    WRITE --> END1([Response to client])
    STREAM --> END2([SSE stream to client])
```

---

## 3. RAG pipeline — ingest and retrieval

```mermaid
flowchart LR
    subgraph Ingest["Ingest  ·  rag/ingest.py  ·  run once"]
        direction TB
        D1["emergency_fund_sizing.md"]
        D2["50_30_20_rule.md"]
        D3["debt_payoff_strategies.md"]
        D4["investment_basics.md"]
        D5["credit_score_factors.md"]
        D6["tax_saving_accounts.md"]
        D7["spending_benchmarks.md"]
        D8["subscription_audit.md"]
        D9["+ 3 pre-existing docs"]

        PASS1["Pass 1\nMarkdownHeaderTextSplitter\n## → section metadata\n### → subsection metadata"]
        PASS2["Pass 2\nRecursiveCharacterTextSplitter\nchunk_size=800\noverlap=160 (20%)"]
        EMBED["OllamaEmbeddings\nnomic-embed-text\nbatch=48 chunks"]
        STORE[("Chroma collection\nfinance_knowledge\nid · embedding · text · metadata\nsource · topic · section")]

        D1 & D2 & D3 & D4 --> PASS1
        D5 & D6 & D7 & D8 & D9 --> PASS1
        PASS1 --> PASS2 --> EMBED --> STORE
    end

    subgraph Startup["Startup  ·  RAGRetriever.__init__"]
        FETCH["collection.get all docs"]
        BM25["BM25Okapi index\ntokenized corpus\nin memory"]
        FETCH --> BM25
        STORE -.-> FETCH
    end

    subgraph Query["Query  ·  RAGRetriever.retrieve"]
        direction TB
        QIN["query string"]
        FILT["Topic filter\nintent + keyword → whitelist\nwhere topic IN list"]

        VEC["Dense search\nembed_query → Chroma.query\nn_candidates = top_k × 3\nranked list A with distances"]
        KW["Sparse search\nBM25Okapi.get_scores\ntopicfiltered\nranked list B"]

        RRF["Reciprocal Rank Fusion\nk = 60\nscore d = Σ 1 / 60 + rank d + 1\nmerged ranked list"]

        MISS["Fetch BM25-only IDs\nfrom Chroma individually"]
        THR["Score threshold ≥ 0.20\nreturn top-4 chunks\ncontent · source · topic · score"]

        QIN --> FILT
        FILT --> VEC & KW
        VEC & KW --> RRF
        RRF --> MISS --> THR
    end
```

---

## 4. Mock Banking API — data model

```mermaid
erDiagram
    TransactionRecord {
        string id PK
        string user_id
        float  amount
        string category
        string description
        string merchant
        datetime timestamp
    }

    AccountRecord {
        string id PK
        string user_id
        string name
        string account_type
        float  balance
        string currency
        datetime last_updated
    }

    BudgetRecord {
        string id PK
        string user_id
        string category
        float  monthly_limit
        datetime updated_at
    }

    TransactionRecord }|--|| User : "belongs to"
    AccountRecord     }|--|| User : "belongs to"
    BudgetRecord      }|--|| User : "belongs to"
```

**Seed data:** 2 users (`user_001`, `user_002`) · 300 transactions each · 180-day window · 50+ merchants · 7 categories · 2 accounts per user · 6 budget targets per user

---

## 5. API endpoints

### Agent API  `http://127.0.0.1:8000`

```mermaid
graph LR
    subgraph AgentEndpoints["Agent API  :8000"]
        C1["POST /chat\nrate limit 30/min\nPydantic validation\nPrometheus counter+timer"]
        C2["POST /chat/stream\nrate limit 20/min\nSSE — data:{token:...}\ndata:[DONE]"]
        C3["GET /chat/history/{session_id}"]
        C4["DELETE /chat/history/{session_id}"]
        C5["GET /health\nstatus + ollama reachability"]
        C6["GET /metrics\nPrometheus text format"]
    end
```

### Mock Banking API  `http://127.0.0.1:8001`

```mermaid
graph LR
    subgraph MockEndpoints["Mock Banking API  :8001"]
        M1["GET /transactions\nuser_id · start_date · end_date\ncategory · limit 1-250"]
        M2["GET /transactions/summary\nperiod=weekly|monthly|all\naggregated metrics"]
        M3["GET /transactions/{id}"]
        M4["GET /accounts\nuser_id filter"]
        M5["GET /accounts/{id}"]
        M6["GET /accounts/{id}/balance\nbalance · currency · last_updated"]
        M7["GET /budgets/{user_id}\nall category limits"]
        M8["PUT /budgets/{user_id}\nupsert targets — body: list of BudgetCategory"]
        M9["GET /health"]
        M10["GET /metrics\nPrometheus — per endpoint"]
    end
```

---

## 6. Streamlit dashboard

```mermaid
flowchart TD
    subgraph Tabs["Streamlit  :8501"]
        CHAT_TAB["💬 Chat tab\n─────────────\nStarter prompt buttons\nChat message history\nIntent badge on responses\nPOST /chat (180s timeout)"]

        DASH_TAB["📊 Dashboard tab\n─────────────\n🔄 Refresh button  (cache TTL 10s)"]

        H["Service Health\n● Agent API status\n● Mock API status\n● Ollama reachability"]
        M["Chat Metrics\nTotal requests\nRAG hit rate\nAvg latency · p95 latency"]
        IC["Intent Distribution\nBar chart by intent label\nfrom Prometheus counter"]
        EP["Mock API Endpoint Usage\nBar chart per route\nfrom Prometheus middleware"]
        AC["Account Balances\nMetric cards per account\nfrom GET /accounts"]
        BV["Budget vs Actual\nSide-by-side bar chart\n🟢 OK  🟡 >80%  🔴 Over 100%\nDataframe with status column"]
        RAW["🔬 Raw Prometheus\nCollapsible expander\nText from /metrics both services"]

        DASH_TAB --> H & M & IC & EP & AC & BV & RAW
    end
```

---

## 7. Resilience and reliability

```mermaid
flowchart TD
    LLM_FAIL["LLM call fails\nor times out"]
    RETRY["tenacity retry\n3 attempts\n2-8s exponential backoff\nsafe_invoke_or_none"]
    HEUR["Heuristic intent\nfallback"]
    OFFLINE["_offline_summary\nrule-based response"]

    NULL_DATE["LLM returns\nstart_date='null'"]
    FILTER["Drop null-like strings\nin transactions_node\n+ mock API route guard"]

    EMPTY_TX["Empty transaction\nresult set"]
    WIDEN["Widen query:\ndrop date + category filters\nretry with limit=180"]

    RAG_LOW["RAG chunk\nscore < 0.20"]
    DROP["Chunk dropped\nbefore LLM prompt"]

    OLLAMA_DOWN["Ollama unreachable\nat startup"]
    WARN["check_ollama_health\nlogs warning\nservice still starts"]

    LLM_FAIL --> RETRY
    RETRY -->|"all retries failed"| HEUR & OFFLINE
    NULL_DATE --> FILTER
    EMPTY_TX --> WIDEN
    RAG_LOW --> DROP
    OLLAMA_DOWN --> WARN
```

---

## 8. Prometheus observability

### Agent API metrics

| Metric | Type | Label | What it measures |
|---|---|---|---|
| `agent_chat_requests_total` | Counter | `intent` | Requests per classified intent |
| `agent_chat_duration_seconds` | Histogram | — | End-to-end latency (p50, p95, p99) |
| `agent_rag_hits_total` | Counter | — | RAG queries returning ≥1 chunk |

### Mock API metrics (HTTP middleware)

| Metric | Type | Label | What it measures |
|---|---|---|---|
| `mock_api_requests_total` | Counter | `endpoint` | Hits per route prefix |
| `mock_api_duration_seconds` | Histogram | `endpoint` | Latency per route |

View: `http://localhost:8000/metrics` · `http://localhost:8001/metrics`

Optional Grafana queries:
```promql
rate(agent_chat_requests_total[5m])
histogram_quantile(0.95, agent_chat_duration_seconds_bucket)
agent_rag_hits_total / ignoring(intent) agent_chat_requests_total
```

---

## 9. Configuration reference

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.2` | Chat LLM |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Per-request LLM timeout |
| `BANKING_API_URL` | `http://127.0.0.1:8001` | Mock banking API |
| `AGENT_API_URL` | `http://127.0.0.1:8000` | Agent API (used by Streamlit) |
| `CHROMA_MODE` | `persist` | `persist` (embedded) or `http` |
| `CHROMA_COLLECTION` | `finance_knowledge` | Chroma collection name |
| `REDIS_USE_FAKEREDIS` | `true` | In-process fake Redis |
| `REDIS_URL` | `redis://127.0.0.1:6379` | Real Redis (if fakeredis=false) |
| `DEFAULT_USER_ID` | `user_001` | Default user for seeded data |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## 10. Technology stack

| Technology | Role |
|---|---|
| **Python 3.11–3.13** | All services |
| **FastAPI + uvicorn** | Agent API (8000) + Mock Banking API (8001) |
| **LangGraph** | Multi-node stateful agent graph with conditional routing |
| **LangChain** | `ChatOllama`, `OllamaEmbeddings`, `MarkdownHeaderTextSplitter` |
| **Ollama** | Local LLM (`llama3.2`) + embeddings (`nomic-embed-text`) |
| **Chroma** | Persistent vector store (embedded SQLite) |
| **rank-bm25** | `BM25Okapi` sparse index for hybrid retrieval |
| **tenacity** | Retry with exponential backoff on all LLM calls |
| **slowapi** | Rate limiting (30/min chat, 20/min stream) |
| **prometheus-client** | `/metrics` endpoints + HTTP middleware on both services |
| **Redis / FakeRedis** | Session-scoped conversation memory |
| **SQLAlchemy + SQLite** | Mock bank persistence (transactions, accounts, budgets) |
| **Faker** | Deterministic synthetic transaction data |
| **Streamlit** | Chat UI + live observability dashboard |
| **httpx** | HTTP client (agent → mock API) |
| **pandas** | Dashboard data processing |
| **pydantic v2** | Request/response validation schemas |
| **python-dotenv** | `.env` config loading |

---

## 11. Running the project

```bash
cd "finance-assistant/"

python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows

pip install --upgrade pip
pip install -r requirements-local.txt

python run_local.py                # Full start (seed + ingest + all services)
python run_local.py --skip-ingest  # Fast restart (skip RAG re-embed)
python run_local.py --free-ports   # Windows: kill stale port occupants first
```

| URL | Service |
|---|---|
| http://127.0.0.1:8501 | Streamlit — Chat + Dashboard |
| http://127.0.0.1:8000/docs | Agent API Swagger |
| http://127.0.0.1:8001/docs | Mock Banking API Swagger |
| http://127.0.0.1:8000/metrics | Agent Prometheus |
| http://127.0.0.1:8001/metrics | Mock API Prometheus |

---

## 12. Out of scope (by design)

- No real bank connectors (PSD2 / Open Banking)
- No production auth (API keys, OAuth)
- Mock data is finite — aggregates reflect only the seeded slice
- Local LLMs can still hallucinate; the insights-first prompt design minimises but does not eliminate numeric errors
