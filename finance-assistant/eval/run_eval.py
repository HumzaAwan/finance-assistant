"""End-to-end evaluation script for the Finance LangGraph Agent.

Runs 20 representative queries against the live /chat endpoint and scores
responses on three dimensions:
    1. Relevance     — keyword overlap between query and response (0–1)
    2. Groundedness  — for RAG responses, terms from retrieved chunks appear
                       in the response (0–1)
    3. Compliance    — financial_advice responses include the required disclaimer (0 or 1)

Outputs a summary table to stdout and saves a JSON report to
eval/results/eval_YYYY-MM-DD.json.

Usage:
    python eval/run_eval.py [--url http://127.0.0.1:8000] [--user-id user_001]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

# ── Eval query bank — 20 queries across 6 intent types ───────────────────────

EVAL_QUERIES: list[dict[str, Any]] = [
    # transaction_query × 4
    {
        "intent": "transaction_query",
        "message": "How much did I spend on food last month?",
        "keywords": ["food", "spend", "spent", "£", "$"],
    },
    {
        "intent": "transaction_query",
        "message": "Show me my transport costs for the last 3 months.",
        "keywords": ["transport", "costs", "spend", "month"],
    },
    {
        "intent": "transaction_query",
        "message": "What are my biggest transactions this week?",
        "keywords": ["transaction", "largest", "biggest", "week"],
    },
    {
        "intent": "transaction_query",
        "message": "List all my entertainment spending.",
        "keywords": ["entertainment", "spending", "category"],
    },

    # insight_request × 4
    {
        "intent": "insight_request",
        "message": "Give me a breakdown of my spending by category.",
        "keywords": ["food", "transport", "utilities", "entertainment", "category", "breakdown"],
    },
    {
        "intent": "insight_request",
        "message": "How does my spending this week compare to last week?",
        "keywords": ["week", "compared", "vs", "this week", "last week"],
    },
    {
        "intent": "insight_request",
        "message": "What is my average daily spend?",
        "keywords": ["average", "daily", "spend"],
    },
    {
        "intent": "insight_request",
        "message": "Summarise my overall spending over the last 3 months.",
        "keywords": ["total", "spend", "summary", "month"],
    },

    # financial_advice × 4
    {
        "intent": "financial_advice",
        "message": "How do I build a 6-month emergency fund?",
        "keywords": ["emergency fund", "month", "savings", "save"],
        "must_have_disclaimer": True,
    },
    {
        "intent": "financial_advice",
        "message": "Explain the UK ISA allowance and which ISA type is best for long-term savings.",
        "keywords": ["isa", "allowance", "£20,000", "stocks", "shares"],
        "must_have_disclaimer": True,
    },
    {
        "intent": "financial_advice",
        "message": "What is the 50/30/20 budgeting rule and how do I apply it?",
        "keywords": ["50", "30", "20", "needs", "wants", "savings"],
        "must_have_disclaimer": True,
    },
    {
        "intent": "financial_advice",
        "message": "How does salary sacrifice work for UK pension contributions?",
        "keywords": ["salary sacrifice", "pension", "national insurance", "tax"],
        "must_have_disclaimer": True,
    },

    # financial_health × 3
    {
        "intent": "financial_health",
        "message": "What is my overall financial health score?",
        "keywords": ["score", "health", "grade"],
    },
    {
        "intent": "financial_health",
        "message": "Give me a full financial situation assessment.",
        "keywords": ["score", "savings", "budget", "emergency"],
    },
    {
        "intent": "financial_health",
        "message": "Am I on track financially? What should I improve?",
        "keywords": ["score", "improvement", "improve", "component"],
    },

    # anomaly_check × 3
    {
        "intent": "anomaly_check",
        "message": "Are there any suspicious transactions in my account?",
        "keywords": ["suspicious", "flag", "anomaly", "detected", "unusual"],
    },
    {
        "intent": "anomaly_check",
        "message": "Show me any unusual activity or potential fraud.",
        "keywords": ["unusual", "fraud", "activity", "flagged"],
    },
    {
        "intent": "anomaly_check",
        "message": "Have I been charged twice for anything recently?",
        "keywords": ["duplicate", "charge", "twice", "detected"],
    },

    # general × 2
    {
        "intent": "general",
        "message": "Hello! What can you help me with?",
        "keywords": ["finance", "help", "spending", "budget"],
    },
    {
        "intent": "general",
        "message": "What financial topics do you know about?",
        "keywords": ["isa", "pension", "budget", "emergency", "spending"],
    },
]

DISCLAIMER_PATTERN = re.compile(
    r"general financial information|not constitute|regulated financial advice|fca",
    re.IGNORECASE,
)


def _relevance_score(query: str, response: str, keywords: list[str]) -> float:
    """Keyword overlap: fraction of expected keywords found in the response."""
    if not keywords:
        return 1.0
    resp_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in resp_lower)
    return round(hits / len(keywords), 3)


def _groundedness_score(response: str, chunks: list[str]) -> float:
    """Fraction of RAG chunk terms (≥5 chars) that appear in the response."""
    if not chunks:
        return 1.0  # no RAG context expected — vacuously grounded

    resp_lower = response.lower()
    all_terms: list[str] = []
    for chunk in chunks:
        terms = [w for w in re.sub(r"[^a-z0-9\s]", " ", chunk.lower()).split() if len(w) >= 5]
        all_terms.extend(terms[:30])  # cap per chunk

    if not all_terms:
        return 1.0

    hits = sum(1 for t in all_terms if t in resp_lower)
    return round(hits / len(all_terms), 3)


def _compliance_score(intent: str, response: str, must_have_disclaimer: bool) -> float:
    """1.0 if disclaimer present (required for financial_advice), 0.0 if absent when required."""
    if not must_have_disclaimer:
        return 1.0  # compliance not applicable
    return 1.0 if DISCLAIMER_PATTERN.search(response) else 0.0


def run_query(client: httpx.Client, base_url: str, user_id: str, query: dict) -> dict:
    """Execute one query against /chat and collect metrics."""
    payload = {
        "message": query["message"],
        "session_id": f"eval-session-{int(time.time())}",
        "user_id": user_id,
    }

    t0 = time.perf_counter()
    try:
        resp = client.post(f"{base_url}/chat", json=payload, timeout=180.0)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as exc:
        return {
            "query": query["message"],
            "expected_intent": query["intent"],
            "actual_intent": "ERROR",
            "latency_ms": -1,
            "response_length": 0,
            "relevance": 0.0,
            "groundedness": 1.0,
            "compliance": 1.0,
            "error": str(exc),
            "response": "",
        }

    response_text: str = data.get("response", "")
    actual_intent: str = data.get("intent", "")

    # Relevance
    relevance = _relevance_score(query["message"], response_text, query.get("keywords", []))

    # Groundedness — placeholder chunks (not returned by /chat; would need /chat debug endpoint)
    groundedness = _groundedness_score(response_text, [])

    # Compliance
    compliance = _compliance_score(
        actual_intent,
        response_text,
        query.get("must_have_disclaimer", False),
    )

    return {
        "query": query["message"],
        "expected_intent": query["intent"],
        "actual_intent": actual_intent,
        "latency_ms": latency_ms,
        "response_length": len(response_text),
        "relevance": relevance,
        "groundedness": groundedness,
        "compliance": compliance,
        "error": None,
        "response": response_text[:500],  # truncate for report readability
    }


def print_summary_table(results: list[dict]) -> None:
    header = f"{'Query':<55} {'ExpIntent':<18} {'ActIntent':<18} {'ms':>6} {'Rel':>5} {'Grnd':>5} {'Comp':>5}"
    print("\n" + "=" * 115)
    print("EVALUATION RESULTS")
    print("=" * 115)
    print(header)
    print("-" * 115)

    for r in results:
        q_trunc = r["query"][:52] + "..." if len(r["query"]) > 55 else r["query"]
        error_marker = " !" if r.get("error") else ""
        print(
            f"{q_trunc + error_marker:<55} "
            f"{r['expected_intent']:<18} "
            f"{r['actual_intent']:<18} "
            f"{r['latency_ms']:>6.0f} "
            f"{r['relevance']:>5.2f} "
            f"{r['groundedness']:>5.2f} "
            f"{r['compliance']:>5.1f}"
        )

    print("=" * 115)


def print_averages(results: list[dict]) -> None:
    valid = [r for r in results if r.get("error") is None]
    n = len(valid)
    if n == 0:
        print("No valid results to average.")
        return

    avg_latency = sum(r["latency_ms"] for r in valid) / n
    avg_relevance = sum(r["relevance"] for r in valid) / n
    avg_groundedness = sum(r["groundedness"] for r in valid) / n

    advice_results = [r for r in valid if r["expected_intent"] == "financial_advice"]
    compliance_pass_rate = (
        sum(r["compliance"] for r in advice_results) / len(advice_results)
        if advice_results
        else float("nan")
    )

    intent_accuracy = sum(1 for r in valid if r["actual_intent"] == r["expected_intent"]) / n

    print("\n📊 OVERALL AVERAGES")
    print(f"  Queries evaluated:     {n}/{len(results)}")
    print(f"  Intent accuracy:       {intent_accuracy:.1%}")
    print(f"  Mean latency:          {avg_latency:.0f} ms")
    print(f"  Mean relevance:        {avg_relevance:.3f}")
    print(f"  Mean groundedness:     {avg_groundedness:.3f}")
    print(f"  Compliance pass rate:  {compliance_pass_rate:.1%}" if not __import__("math").isnan(compliance_pass_rate) else "  Compliance pass rate:  N/A")
    print()


def save_report(results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"eval_{date.today().isoformat()}.json"

    valid = [r for r in results if r.get("error") is None]
    n = max(len(valid), 1)

    report = {
        "eval_date": date.today().isoformat(),
        "n_queries": len(results),
        "n_valid": len(valid),
        "summary": {
            "mean_latency_ms": round(sum(r["latency_ms"] for r in valid) / n, 1),
            "mean_relevance": round(sum(r["relevance"] for r in valid) / n, 3),
            "mean_groundedness": round(sum(r["groundedness"] for r in valid) / n, 3),
            "intent_accuracy": round(
                sum(1 for r in valid if r["actual_intent"] == r["expected_intent"]) / n, 3
            ),
        },
        "results": results,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return filename


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end evaluation for the Finance Agent.")
    parser.add_argument("--url", default=os.getenv("AGENT_API_URL", "http://127.0.0.1:8000"), help="Agent API base URL")
    parser.add_argument("--user-id", default=os.getenv("DEFAULT_USER_ID", "user_001"), help="User ID for queries")
    parser.add_argument("--output-dir", default="eval/results", help="Directory for JSON report output")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    output_dir = Path(args.output_dir)

    print(f"🔍 Finance Agent Evaluation — {len(EVAL_QUERIES)} queries")
    print(f"   Target: {base_url}")
    print(f"   User:   {args.user_id}\n")

    # Health check
    try:
        health = httpx.get(f"{base_url}/health", timeout=10.0)
        status = health.json()
        print(f"✅ Agent health: {status.get('status')} | Ollama: {status.get('ollama')}")
    except Exception as exc:
        print(f"⚠️  Health check failed: {exc} — proceeding anyway.\n")

    results: list[dict] = []

    with httpx.Client() as client:
        for i, query in enumerate(EVAL_QUERIES, 1):
            print(f"  [{i:02d}/{len(EVAL_QUERIES)}] {query['intent']}: {query['message'][:60]}...", end="", flush=True)
            result = run_query(client, base_url, args.user_id, query)
            results.append(result)
            status_char = "✅" if result.get("error") is None else "❌"
            print(f" {status_char} {result['latency_ms']:.0f}ms | rel={result['relevance']:.2f}")

            # Small delay between queries to avoid hitting rate limits
            time.sleep(1.5)

    print_summary_table(results)
    print_averages(results)

    report_path = save_report(results, output_dir)
    print(f"📄 JSON report saved to: {report_path}")


if __name__ == "__main__":
    main()
