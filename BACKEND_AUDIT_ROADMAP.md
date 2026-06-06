# AI-Powered Personal Finance Assistant — Backend Audit & Roadmap

> **Last updated:** June 2026 — All P1, P2, and P3 items implemented. Roadmap complete.

---

## Summary

| Area | Initial Finding | Status |
|---|---|---|
| RAG Pipeline | 3 prose blobs, chunk_size=500, no hybrid search | ✅ Rebuilt |
| LangGraph Agent | Stale memory snapshot, bad intent routing for advice | ✅ Fixed & Extended |
| Agent API (FastAPI) | No validation, no streaming, no rate limiting | ✅ Production-grade |
| Ollama Integration | No retries, no startup health check, no timeout | ✅ Resilient |
| Mock Banking API | 150 transactions, 1 user, no accounts/budgets | ✅ Full dataset |

---

## Bugs Fixed (6 / 6)

| ID | File | Bug | Status |
|---|---|---|---|
| B1 | `rag/retriever.py` | Chroma client re-instantiated on every call | ✅ Fixed |
| B2 | `rag/retriever.py` | No relevance score threshold — low-quality snippets passed to LLM | ✅ Fixed |
| B3 | `rag/rag_tools.py` | Dead-code tool directly instantiated `RAGRetriever` | ✅ Fixed |
| B4 | `graph/state.py` + `agent_app.py` | Stale memory: nodes read history before new message was persisted | ✅ Fixed |
| B5 | `graph/nodes/response_node.py` | n8n `route_hint` metadata injected into LLM prompt | ✅ Fixed |
| B6 | `graph/agent_graph.py` | `financial_advice` intent bypassed transactions and insights nodes | ✅ Fixed |

---

## P1 — Critical Improvements

| ID | Item | Status |
|---|---|---|
| P1-1 | Singleton Chroma client in `RAGRetriever` | ✅ Done (B1 fix) |
| P1-2 | Score threshold filter (min_score=0.25) | ✅ Done (B2 fix) |
| P1-3 | Intent routing fix for `financial_advice` | ✅ Done (B6 fix) |
| P1-4 | Rebuild RAG documents (8 structured files, chunk_size=800, overlap=160) | ✅ Done |
| P1-5 | Ollama health check on startup + `tenacity` retry (3×, 2s backoff) | ✅ Done |

### P1-4 Details — New RAG Corpus
Eight structured topic files replace three prose blobs. Each uses `##`/`###` headers so `MarkdownHeaderTextSplitter` preserves section context in chunk metadata. 800-char chunks with 160-char overlap ensure rules like the 50/30/20 formula land in one chunk.

| File | Topic Tag | Key Content |
|---|---|---|
| `emergency_fund_sizing.md` | `emergency_fund_sizing` | 3–12 month rules, HYSA, tiered build plan |
| `50_30_20_rule.md` | `50_30_20_rule` | Category breakdown with dollar examples |
| `debt_payoff_strategies.md` | `debt_payoff_strategies` | Snowball vs Avalanche with comparison table |
| `investment_basics.md` | `investment_basics` | Compound interest, index funds, DCA |
| `credit_score_factors.md` | `credit_score_factors` | FICO weights, fastest improvement actions |
| `tax_saving_accounts.md` | `tax_saving_accounts` | 401k / IRA / HSA 2025 limits + strategy |
| `spending_benchmarks.md` | `spending_benchmarks` | BLS averages vs recommended %s by category |
| `subscription_audit.md` | `subscription_audit` | Audit process, cancellation tips, savings table |

### P1-5 Details — Ollama Resilience
- `deps.check_ollama_health()` called in FastAPI lifespan; pings `/api/tags` and logs model list. Service starts even if Ollama is down.
- `utils/llm_utils.py::safe_invoke()` wraps every LLM call with `tenacity` (3 attempts, 2–8 s exponential backoff).
- `OLLAMA_TIMEOUT_SECONDS` env var (default 120) plumbed into `ChatOllama`.

---

## P2 — Important Improvements

| ID | Item | Status |
|---|---|---|
| P2-1 | Topic metadata filtering in Chroma queries via `rag_node` intent mapping | ✅ Done |
| P2-2 | `/accounts`, `/accounts/{id}/balance`, `/budgets/{user_id}` endpoints on mock API | ✅ Done |
| P2-3 | Pydantic Field validators + slowapi rate limiting (30 req/min `/chat`) | ✅ Done |
| P2-4 | SSE streaming endpoint `/chat/stream` (token-by-token via `astream`) | ✅ Done |
| P2-5 | Seed expanded to 300+ txns × 2 users × 180 days, 50+ merchants | ✅ Done |

### P2-1 Details — Topic Filtering
`rag_node.py` maps intent → topic whitelist then narrows by keyword matching (e.g. "401k" → `tax_saving_accounts`). The topic list is passed as `{"topic": {"$in": [...]}}` where filter to Chroma. When `financial_advice` queries contain no specific keyword the full advice topic list is used as fallback.

### P2-2 Details — New Mock API Endpoints
```
GET  /accounts?user_id=            → list[Account]
GET  /accounts/{id}                → Account
GET  /accounts/{id}/balance        → {balance, currency}
GET  /budgets/{user_id}            → UserBudget
PUT  /budgets/{user_id}            → UserBudget  (upsert)
GET  /metrics                      → Prometheus text
```
`AccountRecord` and `BudgetRecord` SQLAlchemy models added to `db.py`. Seed creates 2 accounts (checking + savings) and 6 default budget targets per user.

### P2-3 Details — Validation & Rate Limiting
```python
class ChatPayload(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(..., pattern=r"^[a-zA-Z0-9_\-\.]+$", max_length=128)
    user_id: str = Field(..., min_length=1, max_length=64)
    route_hint: str | None = Field(None, max_length=256)
```
slowapi limiter: `@limiter.limit("30/minute")` on `/chat`, `20/minute` on `/chat/stream`.

### P2-4 Details — SSE Streaming
Two-stage pipeline:
1. `GRAPH.invoke(..., streaming_mode=True)` — response_node skips LLM call and returns assembled prompt context in `state["streaming_context"]`.
2. Streaming endpoint calls `llm.astream(prompt_context)` and emits `data: {"token": "..."}` lines, terminated with `data: [DONE]`.
Memory is written to Redis after the full stream completes.

### P2-5 Details — Expanded Mock Data
- 300 transactions per user (was 150), 180-day horizon (was 90).
- User `user_002` added (income $5,500/month, seed 99).
- 50+ unique merchant names across 7 categories.
- Paychecks and recurring subscriptions seeded deterministically; weekend spending uplift applied for realism.

---

## P3 — Nice-to-Have Improvements

| ID | Item | Status |
|---|---|---|
| P3-1 | BM25 keyword search + RRF fusion in `RAGRetriever` | ✅ Done |
| P3-2 | Cross-encoder reranker | ⚠️ Deferred — needs separate model |
| P3-3 | Budget targets — mock endpoints + `insights_node` comparison | ✅ Done |
| P3-4 | Structured LLM output via `with_structured_output` | ⚠️ Deferred — current approach stable |
| P3-5 | Prometheus `/metrics` on both services | ✅ Done |
| P3-6 | Async LangGraph nodes (`ainvoke` / `astream`) | ⚠️ Deferred — see note |

### P3-1 Details — BM25 Hybrid Search
At `RAGRetriever` init: full corpus fetched from Chroma → `BM25Okapi` index built in memory.

At query time:
1. Dense vector search → ranked ID list A (top K×3 candidates).
2. BM25 keyword search → ranked ID list B (topic-filtered).
3. Reciprocal Rank Fusion (k=60): `score(d) = Σ 1/(60 + rank(d) + 1)`.
4. Score-threshold filter, return top-K.

IDs that appear only in the BM25 list but not in vector results are fetched from Chroma individually.

### P3-3 Details — Budget Comparison
- `transactions_node` calls `GET /budgets/{user_id}` for `insight_request` / `financial_advice` intents and stores results in `transaction_data["budgets"]`.
- `insights_node._budget_comparison()` maps actual spend-by-category to budget limits, computes `pct_used`, flags categories as `"ok"` / `"warning"` (>80%) / `"over"` (>100%).
- `over_budget_categories` list surfaced in insights payload so `response_node` can include it in the LLM prompt.

### P3-5 Details — Prometheus Metrics
Agent service tracks:
- `agent_chat_requests_total{intent}` — request counter by classified intent.
- `agent_chat_duration_seconds` — end-to-end latency histogram.
- `agent_rag_hits_total` — RAG retrievals returning ≥1 chunk.

Mock API uses an HTTP middleware that instruments every route with:
- `mock_api_requests_total{endpoint}` — by path prefix.
- `mock_api_duration_seconds{endpoint}` — latency histogram.

### P3-2 — Deferred: Cross-Encoder Reranker
A true cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) processes query+chunk jointly for higher-quality relevance scoring. Requires `sentence-transformers` (~500 MB) and GPU or fast CPU. The BM25+RRF hybrid (P3-1) provides most of the gain in a local Ollama environment. Add reranker when you deploy to a machine with at least 8 GB RAM to spare.

### P3-4 — Deferred: Structured Output
`transactions_node` currently uses regex-cleaned `json.loads()` with empty-dict fallback. Migrate to `llm.with_structured_output(TxQueryParams, method="json_mode")` when upgrading to a model that reliably supports JSON mode (llama3.2 3B is marginal; llama3.1 8B or above recommended).

### P3-6 — Deferred: Async Graph Nodes
Convert all nodes to `async def`, replace `httpx.Client` → `httpx.AsyncClient`, and call `await GRAPH.ainvoke(state)` instead of `await asyncio.to_thread(GRAPH.invoke, state)`. This eliminates the thread-pool overhead but requires all five node files to be updated simultaneously. Implement when readying for a production deployment.

---

## What This Project Demonstrates on a CV

**Senior engineering talking points:**

1. **End-to-end RAG pipeline** — document ingestion, two-pass Markdown + character chunking, nomic-embed-text embeddings, Chroma vector store, BM25 hybrid search with Reciprocal Rank Fusion, and topic-metadata filtering. You can explain every design decision.

2. **LangGraph orchestration** — typed `AgentState` (TypedDict), conditional edge routing by intent, `memory_snapshot` pattern for consistent in-flight state, and a two-stage streaming mode that decouples context assembly from token generation.

3. **Production API patterns** — Pydantic field validation, slowapi rate limiting, SSE streaming via `StreamingResponse`, Prometheus instrumentation, structured JSON logging, and lifespan startup health checks.

4. **Resilience engineering** — `tenacity` retry with exponential backoff wrapping every LLM call; graceful degradation path in `_offline_summary`; BM25 index gracefully disabled if `rank-bm25` is missing.

5. **Domain realism** — 600 synthetic transactions across 2 users, 180-day window, 50+ merchants, bi-monthly paychecks, recurring subscriptions, weekend spending patterns, budget targets with over-budget alerting in insights.

**One-liner for your CV:**
> "Built a local-first AI personal finance assistant: LangGraph agent with hybrid RAG (BM25 + vector + RRF), SSE streaming, budget tracking, and full Prometheus observability — running entirely on Ollama with no cloud dependencies."
