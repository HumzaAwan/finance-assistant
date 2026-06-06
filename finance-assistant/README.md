# AI-Powered Personal Finance Assistant

A **locally runnable** AI finance assistant — Streamlit chat + live dashboard, async LangGraph agent with 7 nodes, hybrid RAG (BM25 + Chroma + RRF + cross-encoder reranker), financial health scoring, anomaly detection, FCA compliance guardrail, UK fintech knowledge base, SSE streaming, and end-to-end evaluation framework. No Docker, no cloud API key required.

Full architecture → **`PROJECT_OVERVIEW.md`** · Audit trail → **`BACKEND_AUDIT_ROADMAP.md`** · Diagrams → **`ARCHITECTURE_DIAGRAMS.md`**

---

## System Architecture

```mermaid
graph TB
    subgraph Browser["Browser"]
        ST["Streamlit :8501\n💬 Chat  |  📊 Dashboard\n🏥 Health Score Gauge"]
    end

    subgraph AgentSvc["Agent API  :8000  (FastAPI + async LangGraph)"]
        direction TB
        EP_CHAT["POST /chat\nPOST /chat/stream (SSE)"]
        EP_META["GET /health  GET /metrics"]
        LG["Async LangGraph Graph\n7 nodes"]
        MEM["FakeRedis / Redis\nSession Memory"]
        RAG["Hybrid RAG\nBM25 + Chroma + RRF\n+ Cross-Encoder Reranker"]
        OLL["Ollama :11434\nllama3.2 · nomic-embed-text"]

        EP_CHAT --> LG
        LG --> MEM
        LG --> RAG
        LG --> OLL
        RAG --> OLL
    end

    subgraph MockSvc["Mock Banking API  :8001  (FastAPI + SQLite)"]
        TXN["GET /transactions\nGET /transactions/summary"]
        ACC["GET /accounts"]
        BUD["GET /budgets  PUT /budgets"]
        DB[("SQLite\n600 txns · 2 users")]

        TXN & ACC & BUD --> DB
    end

    subgraph Eval["Evaluation"]
        EV["eval/run_eval.py\n20 queries · 6 intents\nRelevance · Groundedness\nCompliance scoring"]
    end

    ST -->|"POST /chat"| EP_CHAT
    ST -->|"/accounts /budgets /metrics"| MockSvc
    LG -->|"async httpx"| MockSvc
    EV -->|"POST /chat"| EP_CHAT
```

---

## LangGraph Agent Flow (7 nodes, all async)

```mermaid
flowchart TD
    A([User Message]) --> B["intent_router\nasync · 6 labels\nChatOllama + heuristic fallback"]

    B -->|"transaction_query\ninsight_request\nfinancial_advice\nfinancial_health\nanomaly_check"| C["transactions_node\nasync · structured output\nTxQueryParams via with_structured_output\nfallback: regex extraction\nasync httpx.AsyncClient"]

    B -->|"general"| G

    C --> D["insights_node\nasync · pure-Python\ntotals · by-category · week-over-week\nbudget vs actual comparison"]

    D -->|"financial_advice"| E["rag_node\nasync · UK + US topics\nBM25 + vector + RRF\n+ cross-encoder reranker"]
    D -->|"financial_health"| FH["financial_health_node\nasync · 5-component score\n0–100 · grade · improvement tip"]
    D -->|"anomaly_check"| AN["anomalies_node\nasync · 4-rule detection\nz-score · duplicate · time · MCC"]
    D -->|"transaction_query\ninsight_request"| G

    E --> G
    FH --> G
    AN --> G

    G["response_node\nasync · FCA compliance guardrail\nblocks regulated advice\nappends disclaimer to financial_advice\nChatorllama synthesis"]

    G --> H(["Final response"])
```

---

## Hybrid RAG Pipeline with Reranker

```mermaid
flowchart LR
    subgraph Ingest["Ingest  (run once)"]
        DOCS["16 Markdown docs\n11 US + 5 UK"]
        MHS["MarkdownHeaderTextSplitter\n## / ### → section metadata"]
        RCS["RecursiveCharacterTextSplitter\nchunk_size=800  overlap=160"]
        EMB["OllamaEmbeddings\nnomic-embed-text  batch=48"]
        CHR[("Chroma\nlocal_data/chroma/")]

        DOCS --> MHS --> RCS --> EMB --> CHR
    end

    subgraph Retrieve["Retrieve  (per query)"]
        Q["User Query"]
        VEC["Dense: Chroma similarity\nranked list A"]
        BM["Sparse: BM25Okapi\nranked list B"]
        RRF["RRF fusion  k=60"]
        CE["Cross-Encoder Reranker\nms-marco-MiniLM-L-6-v2\n(RERANKER_ENABLED=true)"]
        THR["Top-K  score ≥ 0.20"]

        Q --> VEC & BM --> RRF --> CE --> THR
    end

    CHR -.->|"BM25 index\nat startup"| BM
    CHR --> VEC
```

---

## Prerequisites

- Python **3.11–3.13**
- [Ollama](https://ollama.com) running locally

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Optional — cross-encoder reranker (~80 MB download on first run):
```bash
pip install sentence-transformers
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

**Transactions & Insights**
- "List my recent spending"
- "What did I spend on food this month?"
- "Compare my spending this week vs last week"
- "Which category am I overspending in?"

**Financial Health Score**
- "What is my overall financial health score?"
- "Am I on track financially? What should I improve?"
- "Give me a full financial situation assessment"

**Anomaly Detection**
- "Are there any suspicious transactions?"
- "Show me unusual activity in my account"
- "Have I been charged twice for anything?"

**Financial Advice (triggers RAG · UK knowledge base)**
- "How much should I have in my emergency fund?"
- "Explain the UK ISA allowance — which type should I use?"
- "How does salary sacrifice work for pension contributions?"
- "What is the 50/30/20 rule?"
- "How do UK credit scores work?"

---

## Running the evaluation suite

```bash
# Requires the full stack to be running
python eval/run_eval.py

# Or target a specific user / server
python eval/run_eval.py --url http://127.0.0.1:8000 --user-id user_001
```

Outputs a summary table to stdout and saves `eval/results/eval_YYYY-MM-DD.json`.

---

## Running the tests

```bash
cd services/agent
pytest tests/ -v
```

Includes `tests/test_bug_fixes.py` — 59 regression tests covering all 7 Bug Fix Round 2 items.

---

## Engineering quality — Bug Fix Round 2 (June 2026)

Seven confirmed engineering bugs were found through 20-query live system testing and fixed:

| # | Symptom | Root cause fixed |
|---|---|---|
| 1 | Different queries returned identical responses | Added `_assert_clean_state()` guard before every `ainvoke`; defensive history copy |
| 2 | LLM replied "you didn't ask a specific question" | Restructured `human_payload`: question first, JSON context after, directive last |
| 3 | "Cash ISA vs S&S ISA" routed to `transaction_query` | Strong-signal keyword override (`_apply_strong_overrides`) fires regardless of LLM label |
| 4 | Financial health score changed between identical calls | `anchor_date` pinned once at node entry; 90-day income fallback; per-session cache |
| 5 | Anomaly detection returned "no suspicious transactions" | Leave-one-out z-score; 4 synthetic anomalies injected into seed; `anomaly_check` now fetches 250 transactions |
| 6 | "Which stocks would you recommend" bypassed guardrail | 11 new FCA regex patterns; guardrail returns fixed message with zero LLM calls |
| 7 | Gibberish input returned detailed financial advice | Gibberish/low-signal check before LLM; `unclear_intent` route with constrained one-sentence clarifier |

---

## Resetting local data

```bash
rm -rf services/mock-api/data/    # SQLite DB
rm -rf local_data/chroma/          # Vector database
python run_local.py                # Rebuilds from scratch
```
