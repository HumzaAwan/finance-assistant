# AI-Powered Personal Finance Assistant — Project Overview

> Full audit and improvement history: `BACKEND_AUDIT_ROADMAP.md`  
> Mermaid architecture diagrams: `ARCHITECTURE_DIAGRAMS.md`

---

## 1. High-level architecture

```mermaid
graph TB
    subgraph UI["Frontend  :8501"]
        CHAT["💬 Chat tab\nStarter prompts · session memory\nPOST /chat or /chat/stream"]
        DASH["📊 Dashboard tab\nHealth · metrics · accounts\nbudget vs actual · intent chart\n🏥 Financial Health Gauge\nML Reliability Metrics"]
    end

    subgraph Agent["Agent API  :8000"]
        direction TB
        API["FastAPI\n/chat · /chat/stream\n/health · /metrics"]
        GRAPH["Async LangGraph\n7-node stateful graph\nainvoke / astream"]
        REDIS["FakeRedis / Redis\nsession memory per session_id"]
        RETRIEVER["Hybrid RAG Retriever\nBM25 + Chroma + RRF\n+ Cross-Encoder Reranker"]
        CHROMA[("Chroma\nlocal_data/chroma/\n16 documents · 800-char chunks")]
        OLLAMA["Ollama  :11434\nllama3.2 (chat)\nnomic-embed-text (embed)"]
        PROM1["Prometheus /metrics\nrequests · latency · RAG hits\nstructured output · compliance\nanomalies detected"]

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

    subgraph Eval["Evaluation"]
        EV["eval/run_eval.py\n20 queries · 6 intents\nRelevance · Groundedness\nCompliance · JSON report"]
    end

    UI -->|"POST /chat\nPOST /chat/stream"| API
    UI -->|"GET /accounts\nGET /budgets\nGET /metrics\nGET /summary"| Mock
    GRAPH -->|"GET /transactions\nGET /budgets\nGET /accounts\n(async httpx)"| Mock
    EV -->|"POST /chat"| API
```

---

## 2. LangGraph agent — full workflow

```mermaid
flowchart TD
    START([User message arrives\nPOST /chat or /chat/stream]) --> SNAP

    SNAP["agent_app.py\nload memory_snapshot from Redis\nbuild AgentState\nawait GRAPH.ainvoke"]

    SNAP --> IR

    subgraph Graph["Async LangGraph Graph — 7 nodes"]
        IR["intent_router  async\n──────────────────\n6 labels: transaction_query ·\ninsight_request · financial_advice ·\nfinancial_health · anomaly_check · general\nChatOllama classify + heuristic fallback\nsafe_invoke_or_none + tenacity retry"]

        IR -->|"transaction_query\ninsight_request\nfinancial_advice\nfinancial_health\nanomaly_check"| TN

        TN["transactions_node  async\n──────────────────\nPrimary: with_structured_output(TxQueryParams)\nFallback: regex json.loads\nPrometheus: structured_output_success/fallback\nasync httpx.AsyncClient → /transactions\n+ /budgets (advice, insight, health)\n+ /accounts (health, anomaly)\nwiden query on empty results"]

        TN --> IN

        IN["insights_node  async\n──────────────────\ntotals · by_category\navg daily · biggest transaction\ncalendar week comparison\nbudget_comparison pct_used\nover_budget_categories list"]

        IN -->|"financial_advice"| RN
        IN -->|"financial_health"| FH
        IN -->|"anomaly_check"| AN
        IN -->|"transaction_query\ninsight_request"| RES

        RN["rag_node  async\n──────────────────\nmap intent + keywords → topic whitelist\n16 topics (11 US + 5 UK)\ncall RAGRetriever.retrieve\nBM25 + vector + RRF + reranker\ntop-4 chunks"]

        FH["financial_health_node  async\n──────────────────\n5 components scored 0–100:\n• Savings rate  25pts\n• Debt-to-income  20pts\n• Emergency fund  25pts\n• Budget adherence  20pts\n• Spending stability  10pts\nGrade: Excellent/Good/Fair/Needs attention\nTop improvement tip"]

        AN["anomalies_node  async\n──────────────────\nRule 1: Amount z-score per merchant\nRule 2: Duplicate within 24h\nRule 3: Unusual time 01:00–05:00\nRule 4: First-time high-risk MCC\nPrometheus: anomalies_detected_total{rule}"]

        RN --> RES
        FH --> RES
        AN --> RES

        IR -->|"general"| RES

        RES["response_node  async\n──────────────────\nFCA compliance guardrail:\n  • detects regulated advice patterns\n  • returns fixed FCA response\n  • fires compliance_triggered_total\nfinancial_health → format score\nanomaly_check → format flags\nfinancial_advice → append disclaimer\nIF streaming_mode=True:\n  → return streaming_context dict\n  → skip LLM call\nIF streaming_mode=False:\n  → safe_invoke_or_none ChatOllama\n  → fallback: _offline_summary"]
    end

    RES -->|"streaming_mode=False"| WRITE["Write user+assistant turns\nto Redis · Return JSON response\n{response, intent, compliance_triggered}"]

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
        US["US Finance docs (11)\nemergency_fund_sizing\n50_30_20_rule\ndebt_payoff_strategies\ninvestment_basics\ncredit_score_factors\ntax_saving_accounts\nspending_benchmarks\nsubscription_audit\nbudgeting_strategies\nsaving_techniques\nfinancial_literacy"]
        UK["🇬🇧 UK Finance docs (5)\nuk_isa_guide\nuk_pension_basics\nuk_tax_basics\nuk_open_banking\nuk_credit_scores"]

        PASS1["Pass 1\nMarkdownHeaderTextSplitter\n## → section  ### → subsection"]
        PASS2["Pass 2\nRecursiveCharacterTextSplitter\nchunk_size=800  overlap=160"]
        EMBED["OllamaEmbeddings\nnomic-embed-text\nbatch=48"]
        STORE[("Chroma collection\nfinance_knowledge\nid · embedding · text · metadata\nsource · topic · section")]

        US & UK --> PASS1 --> PASS2 --> EMBED --> STORE
    end

    subgraph Startup["Startup  ·  RAGRetriever.__init__"]
        FETCH["collection.get all docs"]
        BM25["BM25Okapi index\nin memory"]
        RERANK["CrossEncoder\nms-marco-MiniLM-L-6-v2\n(optional, graceful skip)"]
        FETCH --> BM25
        STORE -.-> FETCH
    end

    subgraph Query["Query  ·  RAGRetriever.retrieve"]
        direction TB
        QIN["query string"]
        FILT["Topic filter\nintent + keyword → whitelist\n16 topics incl. UK"]
        VEC["Dense search\nembed_query → Chroma.query\nn_candidates = top_k × 3"]
        KW["Sparse search\nBM25Okapi.get_scores\ntopic-filtered"]
        RRF["Reciprocal Rank Fusion  k=60\npool = top_k × 4 for reranker"]
        CE["Cross-Encoder reranker\nscore (query, chunk) pairs\nsort descending"]
        THR["Score threshold ≥ 0.20\nreturn top-K\ncontent · source · topic · score · reranker_score"]

        QIN --> FILT --> VEC & KW --> RRF --> CE --> THR
    end
```

---

## 4. Financial Health Score — component model

```mermaid
flowchart LR
    subgraph Components["5 Scoring Components"]
        C1["Savings Rate\n25 pts\n≥20% income saved → full score\nlinear scale to 0"]
        C2["Debt-to-Income Ratio\n20 pts\n≤15% DTI → 20\n≤36% → 10\n>50% → 0"]
        C3["Emergency Fund\n25 pts\nsavings / avg monthly essential spend\n≥6 months → full\nlinear ≥1 month"]
        C4["Budget Adherence\n20 pts\ncategories within budget / total\n× 20"]
        C5["Spending Stability\n10 pts\nCV of weekly essential spend\n8-week window\n<10% CV → full score"]
    end

    Components --> TOTAL["Overall Score  0–100\nGrade:\n≥80 Excellent · 60–79 Good\n40–59 Fair · <40 Needs attention"]
    TOTAL --> TIP["Top Improvement\nWeakest component ratio\n+ one-sentence action"]
```

---

## 5. Anomaly Detection — 4-rule engine

```mermaid
flowchart TD
    TXS["Last 90 days transactions"] --> R1 & R2 & R3 & R4

    R1["Rule 1 — Amount Z-Score\nper-merchant mean/std\nflag if z > 2.5\nhigh: z > 3.5 · medium: z > 2.5"]
    R2["Rule 2 — Duplicate\nsame merchant + amount\nwithin 24-hour window\nconfidence: high"]
    R3["Rule 3 — Unusual Time\n01:00–05:00 UTC\nconfidence: medium"]
    R4["Rule 4 — High-Risk MCC\nfirst-time: gambling · crypto\nforex · payday lending\nconfidence: high"]

    R1 & R2 & R3 & R4 --> DEDUP["Deduplicate by\ntransaction_id + rule"]
    DEDUP --> FLAGS["Flagged transactions\n{id · merchant · amount · date\nrule · confidence · reason}"]
    FLAGS --> PROM["anomalies_detected_total\n{rule_triggered}"]
```

---

## 6. FCA Compliance Guardrail

```mermaid
flowchart TD
    Q["User query"] --> CHECK["Regex pattern match:\n'should I invest'\n'recommend me a fund'\n'what should I do with my pension'\n'is it worth buying [stock/fund/etf]'\n..."]

    CHECK -->|"regulated advice\ndetected"| BLOCK["Return fixed FCA response\n'I can provide general financial\ninformation but am not authorised\nto give regulated financial advice…'\n\ncompliance_triggered_total.inc()\nlog: compliance_triggered=true"]

    CHECK -->|"safe query"| PROCEED["Proceed to LLM"]

    PROCEED --> ADVICE_CHECK["intent == financial_advice?"]
    ADVICE_CHECK -->|"yes"| DISC["Append disclaimer:\n'This is general financial\ninformation only and does not\nconstitute regulated financial advice.'"]
    ADVICE_CHECK -->|"no"| CLEAN["Return response as-is"]
```

---

## 7. Structured LLM output — two-path extraction

```mermaid
flowchart TD
    CALL["Extract query params from user message"] --> PRIMARY

    subgraph PRIMARY["Primary  (P3-4)"]
        SO["llm.with_structured_output\nTxQueryParams  method=json_mode\nPydantic v2 model validation\nstart_date · end_date · category · limit · user_id"]
    end

    PRIMARY -->|"success"| OK["structured_output_success_total.inc()"]
    PRIMARY -->|"exception / invalid"| FALL

    subgraph FALL["Fallback  (original)"]
        REG["safe_invoke_or_none\n→ regex-clean\n→ json.loads()"]
    end

    FALL --> FB["structured_output_fallback_total.inc()"]

    OK & FB --> MERGE["Merge with _fallback_date_window heuristics\n→ async httpx.AsyncClient GET /transactions"]
```

---

## 8. Mock Banking API — data model

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

## 9. API endpoints

### Agent API  `http://127.0.0.1:8000`

| Method | Path | Rate limit | Description |
|---|---|---|---|
| `POST` | `/chat` | 30/min | Async chat; returns `{response, intent, session_id, compliance_triggered}` |
| `POST` | `/chat/stream` | 20/min | SSE streaming; emits `data: {"token": "..."}` then `data: [DONE]` |
| `GET` | `/chat/history/{session_id}` | — | Full Redis conversation history |
| `DELETE` | `/chat/history/{session_id}` | — | Clear session |
| `GET` | `/health` | — | Ollama reachability + reranker status (`reranker_enabled`, `reranker_model`) |
| `GET` | `/metrics` | — | Prometheus text format |

### Mock Banking API  `http://127.0.0.1:8001`

| Method | Path | Description |
|---|---|---|
| `GET` | `/transactions` | Filter by user_id, start/end date, category, limit 1–250 |
| `GET` | `/transactions/summary` | `period=weekly\|monthly\|all` |
| `GET` | `/transactions/{id}` | Single transaction |
| `GET` | `/accounts` | List accounts by user_id |
| `GET` | `/accounts/{id}/balance` | Balance payload |
| `GET` | `/budgets/{user_id}` | All category budget limits |
| `PUT` | `/budgets/{user_id}` | Upsert budget categories |
| `GET` | `/health` | — |
| `GET` | `/metrics` | Prometheus per-endpoint |

---

## 10. Streamlit dashboard

```mermaid
flowchart TD
    subgraph Tabs["Streamlit  :8501"]
        CHAT_TAB["💬 Chat tab\n─────────────\n6 starter prompt buttons\n(incl. health score + ISA + anomalies)\nChat message history\nIntent badge on responses\nPOST /chat (180s timeout)"]

        DASH_TAB["📊 Dashboard tab\n─────────────\n🔄 Refresh  (cache TTL 10s)"]

        H["Service Health\n● Agent API + Ollama status\n● Mock API status"]
        M["Chat Metrics\nTotal requests · RAG hit rate\nAvg latency · p95 latency\nRequests by intent (bar chart)"]
        ML["ML Reliability Metrics\nStructured output rate\nFCA guardrail activations\nAnomalies detected\nReranker status"]
        FHG["🏥 Financial Health Gauge\nOn-demand score computation\nNumerical gauge (0–100) + grade\nFull component breakdown"]
        EP["Mock API Endpoint Usage\nBar chart per route"]
        AC["Account Balances\nMetric cards per account"]
        BV["Budget vs Actual\nSide-by-side bar chart\n🟢 OK  🟡 >80%  🔴 Over 100%"]
        RAW["🔬 Raw Prometheus\nCollapsible expander"]

        DASH_TAB --> H & M & ML & FHG & EP & AC & BV & RAW
    end
```

---

## 11. Prometheus metrics — full reference

### Agent API

| Metric | Type | Labels | What it measures |
|---|---|---|---|
| `agent_chat_requests_total` | Counter | `intent` | Requests per classified intent |
| `agent_chat_duration_seconds` | Histogram | — | End-to-end latency |
| `agent_rag_hits_total` | Counter | — | RAG queries returning ≥1 chunk |
| `structured_output_success_total` | Counter | — | Successful `TxQueryParams` structured output extractions |
| `structured_output_fallback_total` | Counter | — | Fallbacks to regex extraction |
| `compliance_triggered_total` | Counter | — | FCA regulated-advice guardrail activations |
| `anomalies_detected_total` | Counter | `rule_triggered` | Flagged transactions per detection rule |

### Mock API (HTTP middleware)

| Metric | Type | Label | What it measures |
|---|---|---|---|
| `mock_api_requests_total` | Counter | `endpoint` | Hits per route prefix |
| `mock_api_duration_seconds` | Histogram | `endpoint` | Latency per route |

Example Prometheus queries:
```promql
rate(agent_chat_requests_total[5m])
histogram_quantile(0.95, rate(agent_chat_duration_seconds_bucket[5m]))
structured_output_success_total / (structured_output_success_total + structured_output_fallback_total)
rate(anomalies_detected_total[1h])
```

---

## 12. UK RAG Knowledge Base

Five UK-specific documents added to `services/agent/rag/documents/`:

| File | Topic tag | Content |
|---|---|---|
| `uk_isa_guide.md` | `uk_isa_guide` | Cash ISA, Stocks & Shares ISA, LISA (25% bonus), JISA, £20,000 annual allowance 2025/26 |
| `uk_pension_basics.md` | `uk_pension_basics` | Auto-enrolment 8%, SIPP £60k allowance, State Pension £221.20/wk, salary sacrifice |
| `uk_tax_basics.md` | `uk_tax_basics` | Income tax bands 2025/26, NI Class 1, CGT £3,000 allowance, Self Assessment deadlines |
| `uk_open_banking.md` | `uk_open_banking` | PSD2 mandates, consumer data rights, revocation, how this project relates |
| `uk_credit_scores.md` | `uk_credit_scores` | Experian/Equifax/TransUnion scales, factors, overdraft impact, improvement actions |

All documents use `##` and `###` headers for optimal chunk splitting with `MarkdownHeaderTextSplitter`.

---

## 13. Evaluation framework

```
eval/
├── run_eval.py          # Standalone evaluation script
└── results/
    └── eval_YYYY-MM-DD.json
```

`run_eval.py` runs 20 representative queries against the live `/chat` endpoint covering all 6 intent types. Scores each response on:

| Dimension | Method | Scale |
|---|---|---|
| **Relevance** | Keyword overlap between query and response | 0–1 |
| **Groundedness** | RAG chunk terms appearing in response | 0–1 |
| **Compliance** | `financial_advice` responses contain the FCA disclaimer | 0 or 1 |

```bash
python eval/run_eval.py [--url http://127.0.0.1:8000] [--user-id user_001]
```

---

## 14. Resilience and reliability

```mermaid
flowchart TD
    LLM_FAIL["LLM call fails\nor times out"]
    RETRY["tenacity retry\n3 attempts · 2-8s backoff\nsafe_invoke_or_none"]
    HEUR["Heuristic intent\nfallback"]
    OFFLINE["_offline_summary\nrule-based response"]

    SO_FAIL["with_structured_output\nfails / returns invalid"]
    FB["Regex fallback extraction\nstructured_output_fallback_total.inc()"]

    NULL_DATE["LLM returns 'null' date"]
    FILTER["Drop null-like strings\nin transactions_node"]

    EMPTY_TX["Empty transaction\nresult set"]
    WIDEN["Widen query:\ndrop date + category filters\nretry with limit=180"]

    RAG_LOW["RAG chunk\nscore < 0.20"]
    DROP["Chunk dropped\nbefore prompt"]

    RERANK_FAIL["Reranker init fails\n(model download / OOM)"]
    SKIP["Reranker disabled\ngracefully\n_reranker_enabled = False"]

    LLM_FAIL --> RETRY --> HEUR & OFFLINE
    SO_FAIL --> FB
    NULL_DATE --> FILTER
    EMPTY_TX --> WIDEN
    RAG_LOW --> DROP
    RERANK_FAIL --> SKIP
```

---

## 15. Configuration reference

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.2` | Chat LLM |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `BANKING_API_URL` | `http://127.0.0.1:8001` | Mock banking API |
| `AGENT_API_URL` | `http://127.0.0.1:8000` | Agent API (used by Streamlit) |
| `CHROMA_MODE` | `persist` | `persist` (embedded SQLite) or `http` |
| `CHROMA_COLLECTION` | `finance_knowledge` | Chroma collection name |
| `REDIS_USE_FAKEREDIS` | `true` | In-process fake Redis |
| `REDIS_URL` | `redis://127.0.0.1:6379` | Real Redis (if fakeredis=false) |
| `DEFAULT_USER_ID` | `user_001` | Default user for seeded data |
| `LOG_LEVEL` | `INFO` | Logging level |
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranker (skipped if sentence-transformers absent) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face reranker model |
| `HTTP_TIMEOUT_SECONDS` | `30` | Timeout for agent → mock API requests |

---

## 16. Technology stack

| Technology | Version | Role |
|---|---|---|
| **Python** | 3.11–3.13 | All services |
| **FastAPI + uvicorn** | ≥0.115 | Agent API (:8000) + Mock Banking API (:8001) |
| **LangGraph** | ≥1.1 | Async 7-node stateful agent graph |
| **LangChain** | ≥1.2 | `ChatOllama`, `OllamaEmbeddings`, `MarkdownHeaderTextSplitter` |
| **Ollama** | — | Local LLM (`llama3.2`) + embeddings (`nomic-embed-text`) |
| **Chroma** | ≥1.5.9 | Persistent vector store |
| **rank-bm25** | ≥0.2.2 | `BM25Okapi` sparse index for hybrid retrieval |
| **sentence-transformers** | ≥3.0 (optional) | Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) |
| **tenacity** | ≥8.2 | Retry with exponential backoff on all LLM calls |
| **slowapi** | ≥0.1.9 | Rate limiting (30/min chat, 20/min stream) |
| **prometheus-client** | ≥0.20 | `/metrics` on both services; 7 agent counters/histograms |
| **httpx** | ≥0.27 | Async HTTP client (agent → mock API) |
| **Redis / FakeRedis** | ≥5.0 | Session-scoped conversation memory |
| **SQLAlchemy + SQLite** | ≥2.0 | Mock bank persistence |
| **Faker** | ≥24.0 | Deterministic synthetic transaction data |
| **Streamlit** | ≥1.41 | Chat UI + live observability + health score gauge |
| **pandas** | ≥2.0 | Dashboard data processing |
| **pydantic v2** | ≥2.10 | Request/response validation; `TxQueryParams` structured output |
| **pytest** | ≥8.0 | Unit tests (4 test modules, 40+ tests) |

---

## 17. Out of scope (by design)

- No real bank connectors (PSD2 / Open Banking) — mock API mirrors the data shape
- No production auth (API keys, OAuth, mTLS)
- Mock data is finite — aggregates reflect only the seeded 180-day slice
- Local LLMs can hallucinate; insights-first prompt design minimises but does not eliminate numeric errors
- FCA compliance guardrail is pattern-based, not legally complete — demonstrates regulatory awareness, not legal authorisation

---

## 18. Bug Fix Round 2 — June 2026

Seven engineering bugs were identified through structured live testing (20 representative queries) and fixed in full, each with a dedicated regression test. All 59 tests in `services/agent/tests/test_bug_fixes.py` pass.

### BF2-1 — Stale state between requests

**Symptom:** Different queries returned identical responses in the same session.

The root cause was not a mutable default in `AgentState` (which is a TypedDict with no defaults), but the absence of an explicit invariant check. Fixed by adding `_assert_clean_state()` — called before every `GRAPH.ainvoke()` — which raises `AssertionError` immediately if any result field (`transaction_data`, `insights`, `rag_context`, `anomalies`, `health_score`) is non-None at graph entry. A defensive `list(history)` copy was also added so the memory snapshot is never a shared reference.

### BF2-2 — User question lost in LLM prompt

**Symptom:** The LLM replied "you didn't ask a specific question" for specific queries like "How much did I spend on dining last month?".

`response_node` was assembling `human_payload` with a rules block first, then the question buried after the JSON context. The LLM pattern-matched to the context and lost the question. Fixed by restructuring the payload:

```
User question: {question}               ← FIRST line
[blank line]
AUTHORITATIVE_NUMBERS_RULE: ...
INSIGHT JSON: ...
TRANSACTION PREVIEW JSON: ...
RAG CONTEXT: ...

Answer the user's question directly and specifically. Do not summarise the data unless asked.
                                         ← LAST line
```

### BF2-3 — Intent router missing UK finance keywords

**Symptom:** "Cash ISA vs Stocks and Shares ISA" routed to `transaction_query`. "Spending summary" routed to `financial_health`.

Two separate causes: the heuristic keyword list lacked UK product terms, and heuristic upgrades only fired when the LLM returned `"general"` — a wrong LLM classification (e.g. `"transaction_query"`) could not be corrected. Fixed by:

1. Expanding all keyword lists with full UK finance vocabulary (ISA, LISA, JISA, SIPP, pension, ETF, compound interest, open banking, etc. for `financial_advice`; compare, breakdown, overspending, etc. for `insight_request`)
2. Adding `_apply_strong_overrides()` — unconditionally corrects the LLM label when a strong domain signal is detected in the message, regardless of what the LLM returned

### BF2-4 — Financial health score non-deterministic

**Symptom:** Same user, same session: score was 48 on one call, 38 on the next.

Each of the five component functions was calling `datetime.now(timezone.utc)` independently. Sub-second timing differences between calls crossed a date boundary or gave slightly different cutoffs. Fixed by:

- Calling `anchor = date.today()` **once** at the top of `financial_health_node`
- Passing `anchor` as a parameter to every component function — no component touches the system clock independently
- Extending the income lookback from 30 to 90 days when the 30-day window returns no income (handles sparse pay cycles gracefully instead of returning 0)
- Adding `_SCORE_CACHE` keyed on `(user_id, anchor_date_iso)` — same user + same day returns the cached result without recomputing

### BF2-5 — Anomaly detection never fires

**Part A — Seed data:** The 600 seeded transactions had no anomalies (consistent amounts, no duplicates, no 3am transactions, no gambling merchants). Added `seed_anomalies()` injecting four deterministic records for `user_001` via `session.merge()` (idempotent, runs on every startup):

| Anomaly | Details |
|---|---|
| Duplicate charge | Netflix £14.99 twice, 4 hours apart |
| Amount outlier | Green Bowl £295.00 (~6× the typical food spend of ~£50) |
| Unusual time | City Diner £42.50 at 03:17 UTC |
| High-risk merchant | BetKing Casino £50.00 (first-time gambling) |

**Part B — Detection logic:**
- **Z-score self-contamination fix:** The outlier was included in its own baseline, inflating mean/std and hiding itself. Switched to **leave-one-out** baseline — the transaction under test is excluded from the mean/std calculation
- `anomaly_check` intent now fetches 120–250 transactions (was defaulting to 20 from LLM extraction — far too few for reliable z-score baselines across merchants)
- Duplicate window made configurable via `ANOMALY_DUPLICATE_WINDOW_HOURS` env var (default 24h); must match both merchant name AND amount
- Per-transaction evaluation log added at DEBUG level

### BF2-6 — FCA guardrail incomplete

**Symptom:** "Should I invest my pension" → guardrail fired correctly. "Which stocks would you recommend" → LLM soft-refusal instead of fixed guardrail message.

Added 11 new `re.compile(..., re.IGNORECASE)` patterns covering stock-recommendation phrasing:

```python
re.compile(r"\bwhich stocks?\b", re.IGNORECASE)
re.compile(r"\brecommend i buy\b", re.IGNORECASE)
re.compile(r"\bwhat should i invest\b", re.IGNORECASE)
re.compile(r"\bbest fund\b", re.IGNORECASE)
re.compile(r"\bwhich fund\b", re.IGNORECASE)
re.compile(r"\bshould i put my money in\b", re.IGNORECASE)
re.compile(r"\bworth investing in\b", re.IGNORECASE)
re.compile(r"\bgood investment\b", re.IGNORECASE)
re.compile(r"\bbuy shares?\b", re.IGNORECASE)
re.compile(r"\bwhere should i invest\b", re.IGNORECASE)
re.compile(r"\bwhat to invest in\b", re.IGNORECASE)
```

The intercept returns **only** the fixed FCA message — no LLM call, no disclaimer appended, Prometheus counter incremented, structured log with `query_snippet` field.

### BF2-7 — Garbage input returns hallucinated response

**Symptom:** "asdfjkl spending money help" was routed to `financial_advice` and returned a detailed Open Banking explanation.

The intent router had no confidence threshold. Fixed with two fast checks that run **before** any LLM call:

- `_is_gibberish(text)` — True if zero tokens appear in `_ENGLISH_VOCAB` (~250 common English + finance words)
- `_is_low_signal(text)` — True if fewer than 2 recognisable tokens (catches `"asdfjkl xyz 123"`)

Both return `{"intent": "unclear_intent"}` immediately. `response_node` handles this intent with a constrained one-sentence clarification prompt: `"The user's message was unclear. Ask them one specific clarifying question about their finances. Do not generate financial data or advice."` — capped to one sentence.

`unclear_intent` is a proper first-class intent in `LABELS`, `route_intent`, and `response_node`.
