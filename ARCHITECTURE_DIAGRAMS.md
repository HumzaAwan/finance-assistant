# Architecture Diagrams — AI-Powered Personal Finance Assistant

> **KPMG Submission Reference** · Generated: June 2026  
> Full-stack AI system: LangGraph agent · Hybrid RAG with reranking · Financial health scoring · Anomaly detection · FCA compliance · UK fintech knowledge base · End-to-end evaluation framework

---

## 1. System Architecture Overview

```mermaid
graph TB
    subgraph Client["Client Layer"]
        ST["🖥 Streamlit UI\n:8501\nChat · Dashboard · Health Gauge"]
    end

    subgraph AgentService["Agent Service :8000"]
        FA["FastAPI\n/chat · /chat/stream\n/health · /metrics"]
        LG["LangGraph Graph\n(async, 7 nodes)"]
        PROM1["Prometheus\nMetrics Exporter"]
    end

    subgraph GraphNodes["LangGraph Node Pipeline"]
        N1["intent_router\n(async)"]
        N2["transactions_node\n(async + structured output)"]
        N3["insights_node\n(async)"]
        N4["rag_node\n(async)"]
        N5["financial_health_node\n(async) ✨NEW"]
        N6["anomalies_node\n(async) ✨NEW"]
        N7["response_node\n(async + FCA guardrail) ✨NEW"]
    end

    subgraph RAGPipeline["Hybrid RAG Pipeline"]
        BM25["BM25 Okapi\nKeyword Search"]
        CHROMA["Chroma\nVector Store"]
        RRF["RRF Fusion\nk=60"]
        RERANK["Cross-Encoder\nReranker ✨NEW\nms-marco-MiniLM-L-6-v2"]
    end

    subgraph KnowledgeBase["RAG Knowledge Base (16 docs)"]
        US["US Finance\n11 documents\n(50/30/20, emergency fund,\ndebt payoff, etc.)"]
        UK["🇬🇧 UK Finance ✨NEW\n5 documents\n(ISA, pension, tax,\nOpen Banking, credit)"]
    end

    subgraph MockBanking["Mock Banking API :8001"]
        MAPI["FastAPI\n/transactions\n/accounts\n/budgets"]
        SQLITE[("SQLite\nfinance.db\n600 transactions\n2 users")]
        PROM2["Prometheus\nMetrics Exporter"]
    end

    subgraph Infra["Infrastructure"]
        OLLAMA["Ollama :11434\nllama3.2 (LLM)\nnomic-embed-text (embed)"]
        REDIS["Redis / FakeRedis\nSession Memory\nTTL: 86400s"]
    end

    subgraph Eval["Evaluation Framework ✨NEW"]
        EVALSCRIPT["eval/run_eval.py\n20 queries · 6 intents\nRelevance · Groundedness\nCompliance scoring"]
        RESULTS["eval/results/\neval_YYYY-MM-DD.json"]
    end

    ST -->|"POST /chat\nGET /metrics"| FA
    ST -->|"GET /transactions\nGET /accounts\nGET /budgets"| MAPI
    FA --> LG
    LG --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N3 --> N5
    N3 --> N6
    N4 --> N7
    N5 --> N7
    N6 --> N7
    N4 --> BM25
    N4 --> CHROMA
    BM25 --> RRF
    CHROMA --> RRF
    RRF --> RERANK
    RERANK --> N7
    BM25 -.->|indexes| US
    BM25 -.->|indexes| UK
    CHROMA -.->|embeds| US
    CHROMA -.->|embeds| UK
    N2 -->|"GET /transactions\nGET /budgets"| MAPI
    N5 -->|"GET /transactions\nGET /accounts"| MAPI
    N6 -->|"GET /transactions\n(90 days)"| MAPI
    MAPI --> SQLITE
    LG -->|read/write history| REDIS
    FA --> OLLAMA
    EVALSCRIPT -->|"POST /chat"| FA
    EVALSCRIPT --> RESULTS
    FA --> PROM1
    MAPI --> PROM2

    style N5 fill:#d4edda,stroke:#28a745
    style N6 fill:#d4edda,stroke:#28a745
    style N7 fill:#fff3cd,stroke:#ffc107
    style RERANK fill:#d4edda,stroke:#28a745
    style UK fill:#cce5ff,stroke:#004085
    style EVAL fill:#f8d7da,stroke:#721c24
```

---

## 2. LangGraph Full Node Flow

```mermaid
stateDiagram-v2
    [*] --> intent_router : AgentState\n{messages, user_id, session_id}

    intent_router --> transactions_node : transaction_query\ninsight_request\nfinancial_advice\nfinancial_health\nanomaly_check

    intent_router --> response_node : general\nunclear_intent ✦

    transactions_node --> insights_node : transaction_data populated

    insights_node --> rag_node : financial_advice
    insights_node --> response_node : transaction_query\ninsight_request
    insights_node --> financial_health_node : financial_health
    insights_node --> anomalies_node : anomaly_check

    rag_node --> response_node : rag_context populated
    financial_health_node --> response_node : health_score populated
    anomalies_node --> response_node : anomalies populated

    response_node --> [*] : final_response\n(FCA guardrail · unclear_intent handler)

    note right of intent_router
        LLM classify + heuristic fallback
        7 intent labels (+ unclear_intent ✦)
        Gibberish / low-signal check BEFORE LLM call
        _apply_strong_overrides() for UK domain signals
        async def
    end note

    note right of transactions_node
        Structured output (TxQueryParams)
        with_structured_output → fallback regex
        async httpx.AsyncClient
        anomaly_check: min 120 transactions fetched ✦
        Prometheus: structured_output_success/fallback
    end note

    note right of rag_node
        BM25 + Chroma + RRF + Reranker
        UK + US knowledge base
        Topic metadata filtering
    end note

    note right of financial_health_node
        5-component score (0–100)
        anchor_date pinned ONCE → all 5 components share it ✦
        Income 90-day fallback if 30-day window empty ✦
        _SCORE_CACHE keyed on (user_id, anchor_date) ✦
        Grade: Excellent/Good/Fair/Needs attention
    end note

    note right of anomalies_node
        4 detection rules
        Z-score: leave-one-out baseline ✦
        Duplicate: configurable window (ANOMALY_DUPLICATE_WINDOW_HOURS) ✦
        Unusual time · High-risk MCC
        Prometheus: anomalies_detected_total
    end note

    note right of response_node
        unclear_intent → one-sentence clarification ✦
        FCA guardrail: 19 patterns (expanded) ✦
        question = first line · directive = last line ✦
        Compliance disclaimer appended for financial_advice
        Prometheus: compliance_triggered_total
    end note
```

> **✦** = changed in Bug Fix Round 2 (June 2026)

---

## 3. Hybrid RAG Pipeline with Cross-Encoder Reranker

```mermaid
flowchart LR
    Q["User Query"] --> EMB["Embed Query\nnomic-embed-text"]
    Q --> TOK["Tokenize\nWhitespace + punct"]

    EMB --> CHROMA["Chroma\nVector Search\n(top k×3 candidates)\ncosine similarity"]
    TOK --> BM25["BM25 Okapi\nKeyword Search\n(top k×3 candidates)"]

    subgraph TopicFilter["Topic Metadata Filter"]
        TF["Intent → Topic Whitelist\n16 topics: ISA, pension, tax,\nbudget, emergency_fund..."]
    end

    CHROMA --> TF
    BM25 --> TF
    TF --> RRF

    subgraph RRFBlock["Reciprocal Rank Fusion (k=60)"]
        RRF["score(d) = Σ 1/(k + rank_i)\nFusion across both ranked lists"]
    end

    RRF --> CANDS["Fused Candidates\n(top k×2)"]

    subgraph Reranker["✨ Cross-Encoder Reranker (NEW)"]
        CE["cross-encoder/ms-marco-MiniLM-L-6-v2\nScore (query, chunk_text) pairs\nSingle forward pass per pair"]
        THRESH["Score Threshold Filter\nmin_score=0.20"]
        TOPK["Top-K Selection\nk=4 (configurable)"]
    end

    CANDS --> CE
    CE --> THRESH
    THRESH --> TOPK

    TOPK --> OUT["Retrieved Chunks\n[{content, source, topic, score, reranker_score}]"]

    subgraph HealthEndpoint["Health Endpoint"]
        HE["/health\nreranker_enabled: bool\nreranker_model: str"]
    end

    OUT --> HE

    style Reranker fill:#d4edda,stroke:#28a745
    style RRFBlock fill:#e2d9f3,stroke:#6f42c1
```

---

## 4. Financial Health Score — Component Architecture

```mermaid
flowchart TD
    ANCHOR["anchor = date.today()\ncalled ONCE per invocation ✦"] --> COMP
    INPUT["transaction_data + account_data + budget_data\nfetched ONCE, shared across all components ✦"] --> COMP

    subgraph COMP["5-Component Scoring Engine"]
        C1["Component 1: Savings Rate\nWeight: 25 pts\n(income - spend) / income\n30-day window; extends to 90 if no income found ✦\n≥20% → 25 pts, linear to 0"]
        C2["Component 2: Debt-to-Income Ratio\nWeight: 20 pts\nmonthly_debt / monthly_net_income\n≤15% → 20 pts · ≤36% → 10 pts · >50% → 0"]
        C3["Component 3: Emergency Fund\nWeight: 25 pts\nsavings_balance / avg_monthly_essential_spend\n≥6 months → 25 pts, linear <1 → 0"]
        C4["Component 4: Budget Adherence\nWeight: 20 pts\ncategories within budget / total budgeted\n20 × pct_within_budget"]
        C5["Component 5: Spending Stability\nWeight: 10 pts\nStd dev / mean of weekly essential spend\n8-week window · <10% CV → 10 pts · >40% → 0"]
    end

    ANCHOR --> C1
    ANCHOR --> C2
    ANCHOR --> C3
    ANCHOR --> C5

    C1 --> AGG["Score Aggregation\noverall_score = Σ component_scores"]
    C2 --> AGG
    C3 --> AGG
    C4 --> AGG
    C5 --> AGG

    AGG --> CACHE["_SCORE_CACHE ✦\nkey: (user_id, anchor_date_iso)\nReturns cached result on repeat call\nGuarantees identical score same day"]
    AGG --> GRADE["Grade Assignment\n≥80 → Excellent 🟢\n60–79 → Good 🟡\n40–59 → Fair 🟠\n<40 → Needs Attention 🔴"]
    AGG --> TOP["Top Improvement\nLowest-scoring component\n+ one-sentence action"]

    GRADE --> OUTPUT["HealthScoreResult\noverall_score: int\ngrade: str\ncomponent_scores: dict\ntop_improvement: dict\nanchor_date: str ✦"]
    TOP --> OUTPUT

    OUTPUT --> STATE["AgentState\nhealth_score: dict"]
    STATE --> RESP["response_node\n→ formatted for user"]
    STATE --> DASH["Streamlit Dashboard\nGauge chart (0–100)"]

    style COMP fill:#e8f4f8,stroke:#1a73e8
    style CACHE fill:#d4edda,stroke:#28a745
    style ANCHOR fill:#d4edda,stroke:#28a745
```

> **✦** = Bug Fix Round 2: anchor_date pinned once eliminates non-determinism from independent `datetime.now()` calls in each component.

---

## 5. Anomaly Detection Pipeline

```mermaid
flowchart TD
    TXS["120–250 Transactions ✦\n(anomaly_check intent: min 120 fetched\nfor reliable z-score baselines)"] --> RULES

    subgraph SEED["Synthetic Anomalies in Seed Data ✦"]
        SA1["Netflix £14.99 × 2\n(duplicate, 4h apart)"]
        SA2["Green Bowl £295\n(z-score outlier, ~6× mean)"]
        SA3["City Diner at 03:17 UTC\n(unusual time)"]
        SA4["BetKing Casino £50\n(first-time gambling)"]
    end

    SEED -.->|"injected via seed_anomalies()\nidempotent session.merge()"| TXS

    subgraph RULES["4-Rule Detection Engine"]
        R1["Rule 1: Amount Z-Score ✦\nLeave-one-out baseline:\ntx excluded from its own mean/std\nFlag if z > 2.5\nhigh: z > 3.5 · medium: z > 2.5"]
        R2["Rule 2: Duplicate Detection ✦\nSame merchant AND amount ✦\nConfigurable window (ANOMALY_DUPLICATE_WINDOW_HOURS)\ndefault 24h · Confidence: high"]
        R3["Rule 3: Unusual Time\n01:00–05:00 UTC\nTimestamp stored as UTC in seed ✦\nConfidence: medium"]
        R4["Rule 4: High-Risk MCC\nFirst-time merchant category:\ngambling · crypto · forex\npayday lending · Confidence: high"]
    end

    TXS --> R1
    TXS --> R2
    TXS --> R3
    TXS --> R4

    R1 --> DEDUP["Deduplicate & Merge\n(transaction may trigger multiple rules)"]
    R2 --> DEDUP
    R3 --> DEDUP
    R4 --> DEDUP

    DEDUP --> FLAGS["Flagged Transactions\n[{transaction_id, merchant, amount, date,\nrule_triggered, confidence, reason}]"]

    FLAGS --> LOG["Per-transaction evaluation log ✦\nDEBUG: {merchant, amount, z_score, rules_fired}"]
    FLAGS --> PROM["Prometheus Counter\nanomalies_detected_total\n{rule_triggered: str}"]
    FLAGS --> STATE2["AgentState\nanomalies: list[AnomalyFlag]"]
    STATE2 --> RESP2["response_node\n→ structured user-facing\nanomaly report"]

    style RULES fill:#fff3cd,stroke:#ffc107
    style PROM fill:#f8d7da,stroke:#dc3545
    style SEED fill:#d4edda,stroke:#28a745
    style LOG fill:#e2d9f3,stroke:#6f42c1
```

> **✦** = Bug Fix Round 2: leave-one-out z-score prevents self-contamination; synthetic anomalies ensure detection always fires on test data.

---

## 6. FCA Compliance Guardrail

```mermaid
flowchart TD
    QUERY["User Query"] --> CLASSIFY["Compliance Classifier\n19 regex patterns (re.IGNORECASE) ✦\nOriginal 8 + 11 new stock/fund patterns:\n'should I invest' · 'recommend me a fund'\n'which stocks' · 'best fund' · 'which fund' ✦\n'worth investing in' · 'good investment' ✦\n'buy shares' · 'where should i invest' ✦\n'what to invest in' · 'recommend i buy' ✦"]

    CLASSIFY -->|"regulated_advice\ndetected"| BLOCK["BLOCK LLM Call ✦\nReturn ONLY fixed FCA response:\n'I can provide general financial\ninformation and education, but\nI am not authorised to give\nregulated financial advice...'\nNo disclaimer appended ✦"]

    CLASSIFY -->|"safe / general\nfinancial info"| PROCEED["Proceed to LLM\nGenerate normal response"]

    BLOCK --> LOG["Structured Log ✦\ncompliance_triggered=true\nquery_snippet=first 80 chars ✦"]
    BLOCK --> PROM3["Prometheus\ncompliance_triggered_total.inc()"]

    PROCEED --> INTENT_CHECK["Check intent == financial_advice?"]
    INTENT_CHECK -->|"yes"| DISCLAIMER["Append disclaimer:\n'This is general financial information\nonly and does not constitute\nregulated financial advice.'"]
    INTENT_CHECK -->|"no"| CLEAN["Return response as-is"]

    DISCLAIMER --> FINAL["final_response"]
    CLEAN --> FINAL
    BLOCK --> FINAL

    style BLOCK fill:#f8d7da,stroke:#dc3545
    style DISCLAIMER fill:#fff3cd,stroke:#ffc107
    style PROM3 fill:#f8d7da,stroke:#dc3545
```

> **✦** = Bug Fix Round 2: 11 additional patterns added; guardrail returns **only** the fixed message with zero LLM calls; `query_snippet` field added to compliance log.

---

## 7. Structured LLM Output — Two-Path Architecture

```mermaid
flowchart TD
    LLM_CALL["LLM Call for Query Parameter Extraction"] --> PRIMARY

    subgraph PRIMARY["Primary Path (P3-4 NEW)"]
        SO["llm.with_structured_output\nTxQueryParams, method='json_mode'\nPydantic v2 model validation"]
    end

    PRIMARY -->|"success + valid"| SUCCESS["TxQueryParams object\nstart_date, end_date\ncategory, limit, user_id"]
    PRIMARY -->|"exception or\ninvalid output"| FALLBACK

    subgraph FALLBACK["Fallback Path (original)"]
        REGEX["Regex-clean LLM text\n→ json.loads()"]
        PARSE["Parse dict\n→ manual validation"]
    end

    FALLBACK --> FALLBACK_RESULT["dict with extracted params"]

    SUCCESS --> PROM_OK["structured_output_success_total.inc()"]
    FALLBACK_RESULT --> PROM_FAIL["structured_output_fallback_total.inc()"]

    PROM_OK --> MERGE["Merge with heuristic date window\n_fallback_date_window()"]
    PROM_FAIL --> MERGE

    MERGE --> API["GET /transactions\nwith validated params"]

    subgraph MODEL["TxQueryParams (Pydantic v2)"]
        FIELDS["start_date: date | None\nend_date: date | None\ncategory: CategoryEnum | None\nlimit: int = 20\nuser_id: str"]
    end

    style PRIMARY fill:#d4edda,stroke:#28a745
    style FALLBACK fill:#fff3cd,stroke:#ffc107
```

---

## 8. Evaluation Framework

```mermaid
flowchart TD
    SCRIPT["eval/run_eval.py\n(standalone, hits live /chat)"] --> QUERIES

    subgraph QUERIES["20 Queries across 6 intents"]
        TQ["transaction_query ×4\n'How much did I spend on food?'\n'Show me transport costs'\n..."]
        IR["insight_request ×4\n'Summarise my spending this month'\n'Top categories breakdown'\n..."]
        FA["financial_advice ×4\n'How do I build an emergency fund?'\n'Explain UK ISA allowances'\n..."]
        FH["financial_health ×3\n'What is my financial health score?'\n'Give me an overall assessment'\n..."]
        AC["anomaly_check ×3\n'Any suspicious transactions?'\n'Unusual activity alerts'\n..."]
        GN["general ×2\n'Hello'\n'What can you do?'"]
    end

    QUERIES --> EXEC["Execute via POST /chat\nRecord: latency_ms, intent,\nrag_chunks, response_length"]

    subgraph SCORING["3-Dimension Scoring"]
        S1["Relevance (0–1)\nKeyword overlap heuristic:\nquery terms in response"]
        S2["Groundedness (0–1)\nRAG: response terms in\nretrieved chunk content"]
        S3["Compliance (0 or 1)\nfinancial_advice responses\nmust contain disclaimer"]
    end

    EXEC --> S1
    EXEC --> S2
    EXEC --> S3

    S1 --> REPORT["Summary Table (stdout)\n+ JSON report"]
    S2 --> REPORT
    S3 --> REPORT

    REPORT --> FILE["eval/results/\neval_YYYY-MM-DD.json\n{query, intent, latency_ms,\nrelevance, groundedness,\ncompliance, response}"]
    REPORT --> STDOUT["Stdout Summary\nMean latency · Mean relevance\nMean groundedness\nCompliance pass rate"]

    style SCRIPT fill:#f8d7da,stroke:#721c24
    style SCORING fill:#e2d9f3,stroke:#6f42c1
```

---

## 9. Request Lifecycle — Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant ST as Streamlit :8501
    participant AG as Agent API :8000
    participant LG as LangGraph
    participant IR as intent_router
    participant TN as transactions_node
    participant IN as insights_node
    participant FH as financial_health_node
    participant AN as anomalies_node
    participant RN as rag_node
    participant RES as response_node
    participant MOCK as Mock API :8001
    participant OLL as Ollama :11434
    participant RED as Redis/FakeRedis
    participant RAG as RAG (BM25+Chroma+Reranker)

    User->>ST: "What is my financial health score?"
    ST->>AG: POST /chat {message, session_id, user_id}
    AG->>RED: get_history(session_id, last_n=5)
    RED-->>AG: [history turns]
    AG->>LG: ainvoke(AgentState)

    LG->>IR: async intent_router(state)
    IR->>OLL: classify intent
    OLL-->>IR: "financial_health"
    IR-->>LG: {intent: "financial_health"}

    LG->>TN: async transactions_node(state)
    TN->>OLL: with_structured_output(TxQueryParams)
    OLL-->>TN: TxQueryParams or fallback
    TN->>MOCK: GET /transactions?user_id=...&limit=250
    MOCK-->>TN: {transactions: [...], count: N}
    TN->>MOCK: GET /accounts?user_id=...
    MOCK-->>TN: {accounts: [...]}
    TN-->>LG: {transaction_data: {...}}

    LG->>IN: async insights_node(state)
    IN-->>LG: {insights: {...}}

    LG->>FH: async financial_health_node(state)
    FH->>MOCK: GET /accounts (savings balance)
    MOCK-->>FH: account balances
    FH-->>LG: {health_score: {overall:72, grade:"Good", ...}}

    LG->>RES: async response_node(state)
    RES->>RES: FCA compliance check (no regulated advice here)
    RES->>OLL: generate response with health score context
    OLL-->>RES: formatted health score response
    RES-->>LG: {final_response: "Your financial health score is 72/100 (Good)..."}

    LG-->>AG: final state
    AG->>RED: add_message(user), add_message(assistant)
    AG->>AG: Prometheus: CHAT_REQUESTS.inc(), CHAT_LATENCY.observe()
    AG-->>ST: {response, intent, session_id}
    ST-->>User: Display response + health gauge
```

---

## 10. Data Model (ER Diagram)

```mermaid
erDiagram
    USER {
        string user_id PK
    }

    TRANSACTION {
        string id PK
        string user_id FK
        float amount
        string category
        string description
        string merchant
        datetime timestamp
    }

    ACCOUNT {
        string id PK
        string user_id FK
        string name
        string account_type
        float balance
        string currency
        datetime last_updated
    }

    BUDGET {
        string id PK
        string user_id FK
        string category
        float monthly_limit
        datetime updated_at
    }

    HEALTH_SCORE {
        string user_id FK
        int overall_score
        string grade
        float savings_rate_score
        float dti_score
        float emergency_fund_score
        float budget_adherence_score
        float spending_stability_score
        string top_improvement
        datetime computed_at
    }

    ANOMALY_FLAG {
        string transaction_id FK
        string rule_triggered
        string confidence
        string reason
        datetime flagged_at
    }

    SESSION_MEMORY {
        string session_id PK
        string user_id
        list messages
        int ttl_seconds
    }

    RAG_CHUNK {
        string id PK
        string source
        string topic
        string section
        string content
        vector embedding
        float reranker_score
    }

    USER ||--o{ TRANSACTION : "has"
    USER ||--o{ ACCOUNT : "has"
    USER ||--o{ BUDGET : "sets"
    USER ||--|| HEALTH_SCORE : "has computed"
    USER ||--o{ SESSION_MEMORY : "has"
    TRANSACTION ||--o| ANOMALY_FLAG : "may trigger"
    RAG_CHUNK }|--|| RAG_CHUNK : "BM25 + vector indexed"
```

---

## 11. UK Fintech Knowledge Base — Document Taxonomy

```mermaid
mindmap
    root((RAG Knowledge Base\n16 documents))
        US Finance
            50/30/20 Rule
            Budgeting Strategies
            Emergency Fund Sizing
            Debt Payoff Strategies
            Investment Basics
            Credit Score Factors
            Tax Saving Accounts\n401k/IRA/HSA
            Spending Benchmarks\nBLS averages
            Subscription Audit
            Saving Techniques
            Financial Literacy
        🇬🇧 UK Finance NEW
            uk_isa_guide
                Cash ISA
                Stocks and Shares ISA
                Lifetime ISA LISA
                Junior ISA
                £20000 annual allowance 2025/26
            uk_pension_basics
                Auto-enrolment 8% minimum
                SIPP £60000 annual allowance
                State pension £221.20/week
                Salary sacrifice
            uk_tax_basics
                Income tax bands 2025/26
                National Insurance Class 1
                CGT allowance £3000
                Self-assessment deadlines
            uk_open_banking
                PSD2 mandates
                Data banks must share
                Consumer rights
                Project relevance
            uk_credit_scores
                Experian 0-999
                Equifax 0-1000
                TransUnion 0-710
                Improvement actions
```

---

## 12. Prometheus Metrics Map

```mermaid
graph LR
    subgraph AgentMetrics["Agent API Metrics"]
        M1["agent_chat_requests_total\n{intent: str}\nCounter"]
        M2["agent_chat_duration_seconds\nHistogram"]
        M3["agent_rag_hits_total\nCounter"]
        M4["structured_output_success_total\nCounter ✨NEW"]
        M5["structured_output_fallback_total\nCounter ✨NEW"]
        M6["compliance_triggered_total\nCounter ✨NEW"]
        M7["anomalies_detected_total\n{rule_triggered: str}\nCounter ✨NEW"]
    end

    subgraph MockMetrics["Mock API Metrics"]
        M8["mock_api_requests_total\n{endpoint: str}\nCounter"]
        M9["mock_api_duration_seconds\n{endpoint: str}\nHistogram"]
    end

    subgraph Dashboard["Streamlit Dashboard"]
        D1["Chat Requests chart"]
        D2["RAG Hit Rate gauge"]
        D3["P95 Latency"]
        D4["Intent Distribution bar"]
        D5["Financial Health Gauge ✨NEW"]
        D6["Compliance Alerts ✨NEW"]
    end

    M1 --> D1
    M1 --> D4
    M3 --> D2
    M2 --> D3
    M4 --> D6
    M5 --> D6
    M6 --> D6

    style M4 fill:#d4edda,stroke:#28a745
    style M5 fill:#d4edda,stroke:#28a745
    style M6 fill:#d4edda,stroke:#28a745
    style M7 fill:#d4edda,stroke:#28a745
    style D5 fill:#d4edda,stroke:#28a745
    style D6 fill:#d4edda,stroke:#28a745
```

---

## 13. Deployment Topology

```mermaid
graph TB
    subgraph LocalDev["Local Development (run_local.py)"]
        subgraph Ports["Port Map"]
            P1["8000 — Agent API"]
            P2["8001 — Mock Banking API"]
            P3["8501 — Streamlit"]
            P4["11434 — Ollama (external)"]
            P5["6379 — Redis (optional, FakeRedis default)"]
            P6["8002 — Chroma HTTP (optional)"]
        end

        subgraph Startup["Startup Sequence"]
            S1["1. seed.py → finance.db\n(600 transactions, 2 users)"]
            S2["2. mock-api → :8001\n(wait /health)"]
            S3["3. ingest.py → Chroma\n(16 docs, nomic-embed-text)"]
            S4["4. agent-api → :8000\n(wait /health)"]
            S5["5. streamlit → :8501"]
        end

        S1 --> S2 --> S3 --> S4 --> S5
    end

    subgraph Storage["Persistent Storage"]
        FS["local_data/chroma/\n(Chroma vector store)"]
        DB["services/mock-api/data/finance.db\n(SQLite)"]
        MEM["In-process FakeRedis\n(session memory)"]
    end

    subgraph Optional["Optional"]
        RREDIS["Real Redis :6379"]
        CHTTP["Chroma HTTP :8002"]
        N8N["n8n Webhook Router\n(finance_workflow.json)"]
    end

    LocalDev -.->|"persist"| Storage
    LocalDev -.->|"REDIS_USE_FAKEREDIS=false"| RREDIS
    LocalDev -.->|"CHROMA_MODE=http"| CHTTP
```

---

## Summary Table — What's New

| Feature | Category | Prometheus Counter | Impact |
|---|---|---|---|
| Cross-encoder reranker | ML Engineering | — | Better RAG precision |
| Structured LLM output | ML Engineering | `structured_output_success_total` `structured_output_fallback_total` | Measurable reliability |
| Async LangGraph nodes | Backend | — | True async, `ainvoke` |
| Financial health score | Fintech Domain | — | Novel scoring product |
| Anomaly detection | Fintech Domain | `anomalies_detected_total{rule}` | Risk management |
| FCA compliance guardrail | Regulatory | `compliance_triggered_total` | UK regulatory awareness |
| Evaluation framework | ML Ops | — | Measured system quality |
| UK RAG knowledge base | Domain Knowledge | — | UK fintech relevance |

---

## Bug Fix Round 2 — Diagrams Delta (June 2026)

### Diagram 2 changes: `unclear_intent` graph path

```mermaid
flowchart LR
    IR["intent_router"]

    subgraph GibberishGuard["Gibberish / Low-Signal Check (pre-LLM)"]
        GC["_is_gibberish(text)\nzero tokens in _ENGLISH_VOCAB\n→ unclear_intent immediately"]
        LS["_is_low_signal(text)\n< 2 recognisable English words\n→ unclear_intent immediately"]
    end

    subgraph StrongOverride["Strong-Signal Override (post-LLM)"]
        SO["_apply_strong_overrides()\nUK domain signals override\nwrong LLM classification:\nISA/SIPP/pension → financial_advice\nfraud/suspicious → anomaly_check\nhealth score → financial_health"]
    end

    IR --> GibberishGuard
    GibberishGuard -->|"gibberish / low-signal"| UI["unclear_intent\n→ response_node directly"]
    GibberishGuard -->|"real input"| LLM_CALL["LLM classify\n7 labels"]
    LLM_CALL --> StrongOverride
    StrongOverride --> ROUTE["Route to appropriate\npipeline node"]

    UI --> RN["response_node\nConstrained prompt:\n'Ask one clarifying question.\nDo not generate financial data.'\nMax one sentence"]

    style GibberishGuard fill:#f8d7da,stroke:#dc3545
    style StrongOverride fill:#fff3cd,stroke:#ffc107
    style UI fill:#f8d7da,stroke:#dc3545
```

### Bug Fix Round 2 — Changes at a glance

| Bug | Diagram affected | Change |
|---|---|---|
| BF2-1 Stale state | — | `_assert_clean_state()` guard; no diagram change |
| BF2-2 Question lost in prompt | — | Prompt structure (code change only) |
| BF2-3 Missing keywords | Diagram 2 | `_apply_strong_overrides()` + expanded keyword lists |
| BF2-4 Non-deterministic score | Diagram 4 | `anchor_date` pinned once; `_SCORE_CACHE` |
| BF2-5 Anomaly detection | Diagram 5 | Leave-one-out z-score; synthetic anomaly seed |
| BF2-6 FCA guardrail | Diagram 6 | 11 new patterns; ONLY fixed message returned |
| BF2-7 Garbage input | Diagram 2 + new | Gibberish guard; `unclear_intent` path |
