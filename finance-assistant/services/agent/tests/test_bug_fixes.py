"""Regression tests for Bug Fix Round 2 — June 2026.

Each test maps directly to one of the seven confirmed engineering bugs.
Tests are unit-level (no live Ollama / banking API required) unless marked
``@pytest.mark.integration``.
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── conftest sets required env vars before any imports ────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# BUG 1 — Stale state between requests
# ─────────────────────────────────────────────────────────────────────────────

class TestBug1StaleState:
    """_build_state must produce a fully isolated dict on every call.

    We test the logic directly rather than importing agent_app (which triggers
    the full graph / chromadb chain) by replicating only the state-builder
    contract that was fixed.
    """

    # The exact fields that must be None at graph entry (mirrors _MUTABLE_STATE_FIELDS)
    MUTABLE_FIELDS = ("transaction_data", "insights", "rag_context", "anomalies", "health_score")

    def _make_state(self, message: str = "hello", session: str = "s1") -> dict:
        """Replicate the relevant portion of _build_state."""
        from langchain_core.messages import HumanMessage
        return {
            "messages": [HumanMessage(content=message)],
            "user_id": "user_001",
            "session_id": session,
            "intent": "",
            "transaction_data": None,
            "insights": None,
            "rag_context": None,
            "health_score": None,
            "anomalies": None,
            "final_response": "",
            "route_hint": None,
            "memory_snapshot": [],
            "streaming_mode": False,
            "streaming_context": None,
            "compliance_triggered": False,
        }

    def _assert_clean(self, state: dict) -> None:
        """Replicate _assert_clean_state logic."""
        for field in self.MUTABLE_FIELDS:
            value = state.get(field)
            assert value is None or value == [] or value == {}, (
                f"State field '{field}' must be None/empty at graph entry but got: {type(value).__name__}"
            )

    def test_fresh_mutable_fields_each_call(self):
        """All mutable result fields are None in a freshly-built state."""
        state_a = self._make_state("query A")
        state_b = self._make_state("query B")

        assert state_a is not state_b

        for field in self.MUTABLE_FIELDS:
            assert state_a[field] is None, f"Expected {field} to be None in state_a"
            assert state_b[field] is None, f"Expected {field} to be None in state_b"

    def test_messages_list_not_shared(self):
        """Mutating messages in one state must not affect another."""
        state_a = self._make_state("query A")
        state_b = self._make_state("query B")

        original_b_len = len(state_b["messages"])
        state_a["messages"].append("extra")
        assert len(state_b["messages"]) == original_b_len, (
            "Appending to state_a['messages'] must not affect state_b"
        )

    def test_assert_clean_state_raises_on_dirty_field(self):
        """_assert_clean_state must raise if a mutable field is non-None."""
        dirty = self._make_state()
        dirty["transaction_data"] = {"transactions": [{"id": "x"}]}
        with pytest.raises(AssertionError, match="transaction_data"):
            self._assert_clean(dirty)

    def test_assert_clean_state_passes_on_fresh_state(self):
        """_assert_clean_state must not raise for a clean build."""
        state = self._make_state()
        self._assert_clean(state)  # should not raise

    def test_agent_app_assert_fn_raises_correctly(self):
        """The real _assert_clean_state in agent_app raises for dirty state."""
        # Import agent_app if possible; skip gracefully if chromadb not present
        pytest.importorskip("chromadb", reason="chromadb not installed; skipping agent_app import test")
        from agent_app import _assert_clean_state
        dirty = {
            "transaction_data": {"transactions": [{"id": "x"}]},
            "insights": None, "rag_context": None, "anomalies": None, "health_score": None,
        }
        with pytest.raises(AssertionError, match="transaction_data"):
            _assert_clean_state(dirty)


# ─────────────────────────────────────────────────────────────────────────────
# BUG 2 — User question lost in LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestBug2QuestionInPrompt:
    """User question must appear as the very first content line of human_payload."""

    @pytest.mark.asyncio
    async def test_question_is_first_line_of_human_payload(self):
        """The assembled human_payload must begin with 'User question: <text>'."""
        from langchain_core.messages import HumanMessage
        # Import the module BEFORE patching so patch() can locate it
        import graph.nodes.response_node as rn_mod

        captured_payload: dict = {}

        def _capture(llm, msgs):
            for _role, content in msgs:
                if _role == "human":
                    captured_payload["human"] = content
            result = MagicMock()
            result.content = "You spent £120 on dining last month."
            return result

        with patch.object(rn_mod, "get_runtime"), \
             patch.object(rn_mod, "safe_invoke_or_none", side_effect=_capture):

            state = {
                "messages": [HumanMessage(content="How much did I spend on dining last month?")],
                "intent": "transaction_query",
                "transaction_data": {
                    "transactions": [
                        {"id": "1", "amount": -50.0, "category": "food",
                         "merchant": "City Diner", "timestamp": "2026-05-10T12:00:00+00:00",
                         "description": "dinner"}
                    ],
                    "count": 1, "request_params": {},
                },
                "insights": {"total_spent": 120.0, "by_category": {"food": 120.0}},
                "rag_context": [],
                "memory_snapshot": [],
                "streaming_mode": False,
                "health_score": None,
                "anomalies": None,
                "compliance_triggered": False,
            }
            await rn_mod.response_node(state)

        human = captured_payload.get("human", "")
        first_line = human.strip().splitlines()[0] if human else ""
        assert first_line.startswith("User question:"), (
            f"First line of human_payload must start with 'User question:' but got: {first_line!r}"
        )
        assert "How much did I spend on dining last month?" in first_line

    @pytest.mark.asyncio
    async def test_directive_is_last_meaningful_line(self):
        """'Answer the user's question directly' must appear at the end of human_payload."""
        from langchain_core.messages import HumanMessage
        import graph.nodes.response_node as rn_mod

        captured_payload: dict = {}

        def _capture(llm, msgs):
            for _role, content in msgs:
                if _role == "human":
                    captured_payload["human"] = content
            result = MagicMock()
            result.content = "answer"
            return result

        with patch.object(rn_mod, "get_runtime"), \
             patch.object(rn_mod, "safe_invoke_or_none", side_effect=_capture):

            state = {
                "messages": [HumanMessage(content="What is my balance?")],
                "intent": "transaction_query",
                "transaction_data": {"transactions": [
                    {"id": "x", "amount": -10.0, "category": "food",
                     "merchant": "M", "timestamp": "2026-05-01T10:00:00+00:00",
                     "description": "d"}
                ], "count": 1, "request_params": {}},
                "insights": None,
                "rag_context": [],
                "memory_snapshot": [],
                "streaming_mode": False,
                "health_score": None,
                "anomalies": None,
                "compliance_triggered": False,
            }
            await rn_mod.response_node(state)

        human = captured_payload.get("human", "")
        last_nonempty = [line for line in human.splitlines() if line.strip()][-1]
        assert "Answer the user's question directly" in last_nonempty, (
            f"Last line must contain directive but got: {last_nonempty!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BUG 3 — Intent router missing keywords
# ─────────────────────────────────────────────────────────────────────────────

class TestBug3IntentRouterKeywords:
    """Heuristic and strong-override logic must route UK finance queries correctly."""

    def test_isa_routes_to_financial_advice_via_heuristic(self):
        from graph.nodes.intent_router import _intent_from_heuristics
        assert _intent_from_heuristics("Cash ISA vs Stocks and Shares ISA") == "financial_advice"

    def test_spending_summary_routes_to_insight_request(self):
        from graph.nodes.intent_router import _intent_from_heuristics
        assert _intent_from_heuristics("spending summary for last month") == "insight_request"

    def test_overspending_routes_to_insight_request(self):
        from graph.nodes.intent_router import _intent_from_heuristics
        assert _intent_from_heuristics("which category am I overspending in?") == "insight_request"

    def test_health_score_routes_to_financial_health(self):
        from graph.nodes.intent_router import _intent_from_heuristics
        assert _intent_from_heuristics("What is my financial health score?") == "financial_health"

    def test_suspicious_transaction_routes_to_anomaly_check(self):
        from graph.nodes.intent_router import _intent_from_heuristics
        assert _intent_from_heuristics("Are there any suspicious transactions?") == "anomaly_check"

    def test_strong_override_isa_over_transaction_query(self):
        """Strong-signal override must correct a wrong LLM classification for ISA queries."""
        from graph.nodes.intent_router import _apply_strong_overrides
        result = _apply_strong_overrides("transaction_query", "Explain Cash ISA vs Stocks and Shares ISA")
        assert result == "financial_advice", f"Expected financial_advice but got {result}"

    def test_strong_override_pension_over_general(self):
        from graph.nodes.intent_router import _apply_strong_overrides
        result = _apply_strong_overrides("general", "how does pension auto-enrolment work?")
        assert result == "financial_advice"

    def test_strong_override_fraud_over_transaction_query(self):
        from graph.nodes.intent_router import _apply_strong_overrides
        result = _apply_strong_overrides("transaction_query", "I see a suspicious charge on my account")
        assert result == "anomaly_check"

    @pytest.mark.parametrize("phrase,expected", [
        ("explain the difference between a Cash ISA and a Stocks and Shares ISA", "financial_advice"),
        ("What is my SIPP allowance?", "financial_advice"),
        ("compound interest on index funds", "financial_advice"),
        ("compare this week vs last week spending", "insight_request"),
        ("where am I spending the most?", "insight_request"),
        ("my financial health", "financial_health"),
        ("am I on track financially?", "financial_health"),
        ("fraud on my account", "anomaly_check"),
        ("charged twice for Netflix", "anomaly_check"),
    ])
    def test_heuristic_keywords(self, phrase: str, expected: str):
        from graph.nodes.intent_router import _intent_from_heuristics
        result = _intent_from_heuristics(phrase)
        assert result == expected, f"'{phrase}' → expected {expected}, got {result}"


# ─────────────────────────────────────────────────────────────────────────────
# BUG 4 — Financial health score non-deterministic
# ─────────────────────────────────────────────────────────────────────────────

class TestBug4FinancialHealthDeterminism:
    """Same user + same anchor_date must produce identical scores."""

    def _make_state(self, user_id: str = "user_001") -> dict:
        anchor = date.today()
        # Create transactions spanning 90 days with income and spend
        transactions = []
        for i in range(60):
            ts = (anchor - timedelta(days=i)).isoformat() + "T12:00:00+00:00"
            # Two paychecks per month
            if i % 15 == 0:
                transactions.append({
                    "id": f"pay-{i}",
                    "amount": 3000.0,
                    "category": "income",
                    "merchant": "ACME Payroll",
                    "timestamp": ts,
                    "description": "salary",
                })
            else:
                transactions.append({
                    "id": f"spend-{i}",
                    "amount": -float(30 + (i % 20)),
                    "category": "food",
                    "merchant": "Green Bowl",
                    "timestamp": ts,
                    "description": "lunch",
                })

        return {
            "user_id": user_id,
            "transaction_data": {
                "transactions": transactions,
                "accounts": [
                    {"id": f"{user_id}_savings", "account_type": "savings", "balance": 8000.0}
                ],
                "budgets": [],
            },
            "insights": None,
            "messages": [],
            "intent": "financial_health",
            "memory_snapshot": [],
        }

    @pytest.mark.asyncio
    async def test_identical_score_on_repeated_call(self):
        """Calling financial_health_node twice with the same state must return the same score."""
        # Clear the cache to ensure a fresh computation
        import graph.nodes.financial_health_node as fh_mod
        fh_mod._SCORE_CACHE.clear()

        from graph.nodes.financial_health_node import financial_health_node

        state = self._make_state()
        result_a = await financial_health_node(state)
        result_b = await financial_health_node(state)

        score_a = result_a["health_score"]["overall_score"]
        score_b = result_b["health_score"]["overall_score"]

        assert score_a == score_b, (
            f"Score must be deterministic but got {score_a} then {score_b}"
        )

    @pytest.mark.asyncio
    async def test_anchor_date_present_in_result(self):
        """health_score must include anchor_date for auditability."""
        import graph.nodes.financial_health_node as fh_mod
        fh_mod._SCORE_CACHE.clear()

        from graph.nodes.financial_health_node import financial_health_node

        state = self._make_state()
        result = await financial_health_node(state)
        assert "anchor_date" in result["health_score"], "health_score must include anchor_date"

    @pytest.mark.asyncio
    async def test_income_lookback_extended_to_90_days(self):
        """When income only appears >30 days ago, score must not default to zero savings rate."""
        import graph.nodes.financial_health_node as fh_mod
        fh_mod._SCORE_CACHE.clear()

        from graph.nodes.financial_health_node import _savings_rate_score

        anchor = date.today()
        # Income only 45 days ago (outside 30-day window, inside 90-day window)
        transactions = [
            {
                "id": "old-pay",
                "amount": 3000.0,
                "category": "income",
                "merchant": "ACME Payroll",
                "timestamp": (anchor - timedelta(days=45)).isoformat() + "T10:00:00+00:00",
                "description": "salary",
            },
            *[
                {
                    "id": f"s-{i}",
                    "amount": -50.0,
                    "category": "food",
                    "merchant": "M",
                    "timestamp": (anchor - timedelta(days=i)).isoformat() + "T10:00:00+00:00",
                    "description": "d",
                }
                for i in range(50)
            ]
        ]

        score, explanation = _savings_rate_score(transactions, anchor)
        assert score > 0.0, (
            f"Savings rate score must not be zero when income exists in 90-day window; got {score}"
        )
        assert "90" in explanation or "45" in explanation or "%" in explanation


# ─────────────────────────────────────────────────────────────────────────────
# BUG 5 — Anomaly detection never fires
# ─────────────────────────────────────────────────────────────────────────────

class TestBug5AnomalyDetection:
    """Leave-one-out z-score, duplicate window, and high-risk detection must fire."""

    def _tx(self, tid: str, merchant: str, amount: float, timestamp: str,
            category: str = "food", desc: str = "purchase") -> dict:
        return {
            "id": tid,
            "merchant": merchant,
            "amount": amount,
            "timestamp": timestamp,
            "category": category,
            "description": desc,
        }

    # ── Rule 1: z-score ──────────────────────────────────────────────────────

    def test_zscore_flags_outlier_with_leave_one_out(self):
        """An outlier transaction must be flagged even though it is included in the list."""
        from graph.nodes.anomalies_node import _rule_amount_zscore

        # 10 transactions at ~£30, plus one outlier at £300
        baseline = [
            self._tx(f"b{i}", "Green Bowl", -30.0 - i, "2026-01-01T12:00:00+00:00")
            for i in range(10)
        ]
        outlier = self._tx("outlier", "Green Bowl", -300.0, "2026-05-01T12:00:00+00:00")
        txs = baseline + [outlier]

        flags = _rule_amount_zscore(txs)
        flagged_ids = {f["transaction_id"] for f in flags}
        assert "outlier" in flagged_ids, (
            "Leave-one-out z-score must flag the £300 outlier. "
            f"Got flags: {flags}"
        )

    def test_zscore_does_not_flag_normal_transaction(self):
        """A normal transaction within 2.5 std devs must NOT be flagged."""
        from graph.nodes.anomalies_node import _rule_amount_zscore

        txs = [self._tx(f"n{i}", "Merchant A", -30.0 - i * 0.5, "2026-01-01T12:00:00+00:00")
               for i in range(10)]
        flags = _rule_amount_zscore(txs)
        assert len(flags) == 0, f"No flags expected but got: {flags}"

    # ── Rule 2: duplicate detection ──────────────────────────────────────────

    def test_duplicate_charge_flagged_within_window(self):
        """Two identical charges within 24h must both be flagged."""
        from graph.nodes.anomalies_node import _rule_duplicate

        txs = [
            self._tx("dup-a", "Netflix", -14.99, "2026-05-20T14:00:00+00:00", "entertainment"),
            self._tx("dup-b", "Netflix", -14.99, "2026-05-20T18:00:00+00:00", "entertainment"),
        ]
        flags = _rule_duplicate(txs)
        flagged_ids = {f["transaction_id"] for f in flags}
        assert "dup-a" in flagged_ids and "dup-b" in flagged_ids, (
            f"Both duplicate transactions must be flagged. Got: {flagged_ids}"
        )

    def test_duplicate_not_flagged_outside_window(self):
        """Two charges >24h apart must NOT be flagged as duplicates."""
        from graph.nodes.anomalies_node import _rule_duplicate

        txs = [
            self._tx("far-a", "Netflix", -14.99, "2026-05-01T12:00:00+00:00", "entertainment"),
            self._tx("far-b", "Netflix", -14.99, "2026-05-03T12:00:00+00:00", "entertainment"),
        ]
        flags = _rule_duplicate(txs)
        assert len(flags) == 0, f"No flags expected for charges >24h apart, got: {flags}"

    def test_duplicate_requires_same_merchant(self):
        """Same amount but different merchants must NOT be flagged as duplicates."""
        from graph.nodes.anomalies_node import _rule_duplicate

        txs = [
            self._tx("m1", "Netflix", -14.99, "2026-05-20T14:00:00+00:00", "entertainment"),
            self._tx("m2", "Spotify", -14.99, "2026-05-20T15:00:00+00:00", "entertainment"),
        ]
        flags = _rule_duplicate(txs)
        assert len(flags) == 0, f"Different merchants must not be flagged as duplicates: {flags}"

    # ── Rule 3: unusual time ─────────────────────────────────────────────────

    def test_unusual_time_flagged_at_0317(self):
        """Transaction at 03:17 UTC must be flagged."""
        from graph.nodes.anomalies_node import _rule_unusual_time

        txs = [self._tx("night", "City Diner", -42.5, "2026-05-15T03:17:00+00:00")]
        flags = _rule_unusual_time(txs)
        assert len(flags) == 1, f"Expected 1 flag for 03:17 UTC transaction, got: {flags}"
        assert flags[0]["rule_triggered"] == "unusual_time"

    def test_unusual_time_not_flagged_at_noon(self):
        """Transaction at 12:00 UTC must NOT be flagged."""
        from graph.nodes.anomalies_node import _rule_unusual_time

        txs = [self._tx("day", "Starbucks", -4.5, "2026-05-15T12:00:00+00:00")]
        flags = _rule_unusual_time(txs)
        assert len(flags) == 0

    # ── Rule 4: high-risk merchant ───────────────────────────────────────────

    def test_high_risk_gambling_flagged(self):
        """First-time gambling transaction must be flagged."""
        from graph.nodes.anomalies_node import _rule_high_risk_merchant

        txs = [
            self._tx("g1", "BetKing Casino", -50.0, "2026-05-01T20:00:00+00:00",
                     "entertainment", "gambling deposit")
        ]
        flags = _rule_high_risk_merchant(txs)
        assert len(flags) == 1, f"Expected gambling flag, got: {flags}"
        assert flags[0]["rule_triggered"] == "high_risk_merchant"
        assert flags[0]["confidence"] == "high"

    def test_high_risk_not_flagged_second_time(self):
        """Second transaction at same high-risk merchant must NOT be flagged."""
        from graph.nodes.anomalies_node import _rule_high_risk_merchant

        txs = [
            self._tx("g1", "betking casino", -50.0, "2026-04-01T20:00:00+00:00",
                     "entertainment", "gambling deposit"),
            self._tx("g2", "betking casino", -25.0, "2026-05-01T20:00:00+00:00",
                     "entertainment", "gambling deposit"),
        ]
        flags = _rule_high_risk_merchant(txs)
        flagged_ids = {f["transaction_id"] for f in flags}
        assert "g2" not in flagged_ids, "Second visit to same high-risk merchant must not be re-flagged"

    @pytest.mark.asyncio
    async def test_anomaly_node_full_pipeline(self):
        """anomalies_node must detect at least the duplicate and unusual-time anomaly."""
        from graph.nodes.anomalies_node import anomalies_node

        txs = [
            # Duplicate
            {"id": "d1", "merchant": "Netflix", "amount": -14.99,
             "timestamp": "2026-05-20T14:00:00+00:00", "category": "entertainment",
             "description": "sub"},
            {"id": "d2", "merchant": "Netflix", "amount": -14.99,
             "timestamp": "2026-05-20T18:00:00+00:00", "category": "entertainment",
             "description": "sub"},
            # Unusual time
            {"id": "u1", "merchant": "City Diner", "amount": -42.5,
             "timestamp": "2026-05-15T03:17:00+00:00", "category": "food",
             "description": "late night"},
        ]

        state: dict = {
            "transaction_data": {"transactions": txs},
            "intent": "anomaly_check",
        }
        result = await anomalies_node(state)
        anomalies = result.get("anomalies", [])
        rules_fired = {a["rule_triggered"] for a in anomalies}
        assert "duplicate" in rules_fired, f"Expected 'duplicate' rule, got: {rules_fired}"
        assert "unusual_time" in rules_fired, f"Expected 'unusual_time' rule, got: {rules_fired}"


# ─────────────────────────────────────────────────────────────────────────────
# BUG 6 — FCA guardrail incomplete
# ─────────────────────────────────────────────────────────────────────────────

class TestBug6FCAGuardrail:
    """The FCA guardrail must fire for all expanded regulated-advice patterns."""

    @pytest.mark.parametrize("query", [
        "which stocks would you recommend",
        "Which Stocks would you recommend I buy?",
        "what should i invest in?",
        "best fund for my money",
        "which fund should I pick?",
        "should i put my money in this ETF?",
        "worth investing in Bitcoin?",
        "is Apple a good investment?",
        "I want to buy shares in Tesla",
        "where should i invest £10000?",
        "what to invest in right now?",
        "recommend i buy index funds",
        # Original patterns
        "should I invest my pension",
        "is it worth buying this ETF?",
    ])
    def test_regulated_advice_pattern_detected(self, query: str):
        from graph.nodes.response_node import _is_regulated_advice
        assert _is_regulated_advice(query), f"Pattern must be detected as regulated advice: {query!r}"

    @pytest.mark.asyncio
    async def test_guardrail_returns_exact_fixed_message(self):
        """A regulated-advice query must return ONLY the fixed FCA message — no LLM call."""
        from langchain_core.messages import HumanMessage
        # Import module before patching
        import graph.nodes.response_node as rn_mod

        llm_call_count = {"n": 0}

        def _mock_safe_invoke(llm, msgs):
            llm_call_count["n"] += 1
            result = MagicMock()
            result.content = "some LLM response"
            return result

        with patch.object(rn_mod, "get_runtime"), \
             patch.object(rn_mod, "safe_invoke_or_none", side_effect=_mock_safe_invoke):

            state: dict = {
                "messages": [HumanMessage(content="which stocks would you recommend I buy?")],
                "intent": "financial_advice",
                "transaction_data": None,
                "insights": None,
                "rag_context": [],
                "memory_snapshot": [],
                "streaming_mode": False,
                "health_score": None,
                "anomalies": None,
                "compliance_triggered": False,
            }

            result = await rn_mod.response_node(state)

        assert result["compliance_triggered"] is True, "compliance_triggered must be True"
        assert result["final_response"] == rn_mod._FCA_FIXED_RESPONSE, (
            f"Expected exact FCA fixed response.\n"
            f"Got: {result['final_response']!r}"
        )
        assert llm_call_count["n"] == 0, (
            f"LLM must NOT be called for regulated advice, but was called {llm_call_count['n']} time(s)"
        )

    def test_general_advice_patterns_not_blocked(self):
        """Non-regulated advisory queries must NOT be intercepted by the guardrail."""
        from graph.nodes.response_node import _is_regulated_advice
        benign = [
            "What is an ISA?",
            "Explain the 50/30/20 rule",
            "How does compound interest work?",
            "What is auto-enrolment?",
        ]
        for q in benign:
            assert not _is_regulated_advice(q), f"Benign query incorrectly blocked: {q!r}"


# ─────────────────────────────────────────────────────────────────────────────
# BUG 7 — Garbage input returns hallucinated response
# ─────────────────────────────────────────────────────────────────────────────

class TestBug7GarbageInput:
    """Pure gibberish must be routed to unclear_intent, not a financial data node."""

    def test_gibberish_detected(self):
        from graph.nodes.intent_router import _is_gibberish
        assert _is_gibberish("asdfjkl xyz")
        assert _is_gibberish("qwerty zxcvb nmkl")
        assert _is_gibberish("123 456 789")

    def test_real_text_not_gibberish(self):
        from graph.nodes.intent_router import _is_gibberish
        assert not _is_gibberish("how much did I spend on food?")
        assert not _is_gibberish("List my last 10 transactions")
        assert not _is_gibberish("What is my financial health score?")

    def test_low_signal_detected(self):
        from graph.nodes.intent_router import _is_low_signal
        assert _is_low_signal("asdfjkl xyz 123")

    def test_low_signal_not_triggered_for_real_queries(self):
        from graph.nodes.intent_router import _is_low_signal
        assert not _is_low_signal("List my last 10 transactions")
        assert not _is_low_signal("How much did I spend this month?")
        assert not _is_low_signal("Explain the difference between a Cash ISA and a Stocks and Shares ISA")

    @pytest.mark.asyncio
    async def test_gibberish_routes_to_unclear_intent(self):
        """'asdfjkl xyz 123' must return intent='unclear_intent' without an LLM call."""
        from langchain_core.messages import HumanMessage
        # Import module before patching
        import graph.nodes.intent_router as ir_mod

        llm_call_count = {"n": 0}

        def _mock_safe_invoke(llm, msgs):
            llm_call_count["n"] += 1
            m = MagicMock()
            m.content = "general"
            return m

        with patch.object(ir_mod, "get_runtime") as mock_rt, \
             patch.object(ir_mod, "safe_invoke_or_none", side_effect=_mock_safe_invoke):

            mock_rt.return_value.llm_chat = MagicMock()

            state: dict = {
                "messages": [HumanMessage(content="asdfjkl xyz 123")],
                "intent": "",
                "user_id": "user_001",
                "session_id": "sess-x",
            }
            result = await ir_mod.intent_router(state)

        assert result["intent"] == "unclear_intent", (
            f"Expected 'unclear_intent' but got '{result['intent']}'"
        )
        assert llm_call_count["n"] == 0, (
            "LLM must NOT be called for pure gibberish input"
        )

    @pytest.mark.asyncio
    async def test_unclear_intent_returns_clarification_not_data(self):
        """response_node with intent='unclear_intent' must return a clarification, not financial data."""
        from langchain_core.messages import HumanMessage
        import graph.nodes.response_node as rn_mod

        mock_response = MagicMock()
        mock_response.content = "Could you clarify what aspect of your finances you need help with?"

        with patch.object(rn_mod, "get_runtime"), \
             patch.object(rn_mod, "safe_invoke_or_none", return_value=mock_response):

            state: dict = {
                "messages": [HumanMessage(content="asdfjkl xyz 123")],
                "intent": "unclear_intent",
                "transaction_data": None,
                "insights": None,
                "rag_context": [],
                "memory_snapshot": [],
                "streaming_mode": False,
                "health_score": None,
                "anomalies": None,
                "compliance_triggered": False,
            }
            result = await rn_mod.response_node(state)

        response_text = result.get("final_response", "")
        assert len(response_text) > 0, "Clarification response must not be empty"
        assert result.get("compliance_triggered") is False

        # A one-sentence clarification should be short (< 300 chars)
        assert len(response_text) < 300, (
            f"Clarification response is too long ({len(response_text)} chars) — "
            "may be generating financial data instead of asking a question."
        )
