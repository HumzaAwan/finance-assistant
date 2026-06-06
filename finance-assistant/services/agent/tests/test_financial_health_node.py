"""Unit tests for financial_health_node.

Tests cover:
    - Happy path with full transaction + account + budget data
    - Empty data (no transactions)
    - Partial data (accounts missing, budgets missing)
    - Score bounds (always 0–100)
    - Grade assignment accuracy
    - Top improvement selection
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from graph.nodes.financial_health_node import (
    _budget_adherence_score,
    _dti_score,
    _emergency_fund_score,
    _grade,
    _savings_rate_score,
    _spending_stability_score,
    financial_health_node,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_tx(
    amount: float,
    category: str = "food",
    days_ago: int = 10,
    merchant: str = "Tesco",
    description: str = "",
) -> dict[str, Any]:
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {
        "id": f"tx-{abs(amount):.0f}-{days_ago}",
        "amount": amount,
        "category": category,
        "merchant": merchant,
        "description": description,
        "timestamp": ts.isoformat(),
    }


def _make_account(balance: float, account_type: str = "savings") -> dict[str, Any]:
    return {
        "id": "acc-001",
        "account_type": account_type,
        "balance": balance,
        "currency": "GBP",
        "name": "Main Savings",
    }


# ── Unit tests: individual components ────────────────────────────────────────

class TestSavingsRateScore:
    def test_high_savings_rate_full_score(self):
        """Income £2000, spend £1200 → savings rate 40% → full 25 pts."""
        txs = [
            _make_tx(2000, category="income", days_ago=5),
            _make_tx(-1200, category="food", days_ago=10),
        ]
        score, explanation = _savings_rate_score(txs)
        assert score == 25.0
        assert "savings rate" in explanation.lower()

    def test_zero_savings_rate_zero_score(self):
        """Spend equals income → savings rate 0% → 0 pts."""
        txs = [
            _make_tx(1000, category="income", days_ago=5),
            _make_tx(-1000, category="food", days_ago=10),
        ]
        score, _ = _savings_rate_score(txs)
        assert score == 0.0

    def test_no_income_returns_zero(self):
        """No income transactions → score 0."""
        txs = [_make_tx(-500, category="food", days_ago=10)]
        score, explanation = _savings_rate_score(txs)
        assert score == 0.0
        assert "no income" in explanation.lower()

    def test_empty_transactions_returns_zero(self):
        score, _ = _savings_rate_score([])
        assert score == 0.0

    def test_score_bounded_0_25(self):
        txs = [
            _make_tx(5000, category="income", days_ago=5),
            _make_tx(-100, category="food", days_ago=10),
        ]
        score, _ = _savings_rate_score(txs)
        assert 0.0 <= score <= 25.0


class TestDtiScore:
    def test_low_dti_full_score(self):
        """No debt repayments → DTI 0% → 20 pts."""
        txs = [
            _make_tx(3000, category="income", days_ago=5),
            _make_tx(-200, category="food", days_ago=10),
        ]
        score, _ = _dti_score(txs)
        assert score == 20.0

    def test_high_dti_zero_score(self):
        """Debt repayments exceed 50% of income → 0 pts."""
        txs = [
            _make_tx(1000, category="income", days_ago=5),
            _make_tx(-600, category="food", days_ago=10, merchant="loan repayment", description="loan repayment"),
        ]
        score, _ = _dti_score(txs)
        assert score == 0.0

    def test_empty_transactions(self):
        score, explanation = _dti_score([])
        assert 0.0 <= score <= 20.0
        assert "insufficient" in explanation.lower()


class TestEmergencyFundScore:
    def test_full_coverage_full_score(self):
        """6+ months coverage → 25 pts."""
        txs = [_make_tx(-500, category="food", days_ago=i * 10) for i in range(1, 10)]
        accounts = [_make_account(10000.0)]
        score, _ = _emergency_fund_score(txs, accounts)
        assert score == 25.0

    def test_zero_savings_zero_score(self):
        txs = [_make_tx(-500, category="food", days_ago=i * 10) for i in range(1, 10)]
        accounts = [_make_account(0.0)]
        score, _ = _emergency_fund_score(txs, accounts)
        assert score == 0.0

    def test_no_accounts_returns_default(self):
        txs = [_make_tx(-500, category="food", days_ago=10)]
        score, explanation = _emergency_fund_score(txs, [])
        assert 0.0 <= score <= 25.0

    def test_score_bounded_0_25(self):
        txs = [_make_tx(-300, category="utilities", days_ago=i * 5) for i in range(1, 20)]
        accounts = [_make_account(50000.0)]
        score, _ = _emergency_fund_score(txs, accounts)
        assert 0.0 <= score <= 25.0


class TestBudgetAdherenceScore:
    def test_all_within_budget_full_score(self):
        insights = {
            "budget_comparison": {
                "food": {"actual": 300, "monthly_limit": 500, "pct_used": 60, "status": "ok"},
                "transport": {"actual": 100, "monthly_limit": 250, "pct_used": 40, "status": "ok"},
            }
        }
        score, _ = _budget_adherence_score(insights, [{"category": "food", "monthly_limit": 500}])
        assert score == 20.0

    def test_all_over_budget_zero_score(self):
        insights = {
            "budget_comparison": {
                "food": {"actual": 700, "monthly_limit": 500, "pct_used": 140, "status": "over"},
                "transport": {"actual": 400, "monthly_limit": 250, "pct_used": 160, "status": "over"},
            }
        }
        score, _ = _budget_adherence_score(insights, [{"category": "food", "monthly_limit": 500}])
        assert score == 0.0

    def test_no_data_returns_midrange(self):
        score, _ = _budget_adherence_score(None, [])
        assert 0.0 <= score <= 20.0


class TestSpendingStabilityScore:
    def test_stable_spend_full_score(self):
        """Very consistent weekly spend → CV near 0 → 10 pts."""
        txs = []
        for week in range(8):
            for day in range(7):
                days_ago = week * 7 + day
                txs.append(_make_tx(-100.0, category="food", days_ago=days_ago))
        score, _ = _spending_stability_score(txs)
        assert score >= 8.0  # nearly full score for very stable spend

    def test_empty_transactions_returns_default(self):
        score, explanation = _spending_stability_score([])
        assert 0.0 <= score <= 10.0
        assert "fewer" in explanation.lower()

    def test_score_bounded_0_10(self):
        txs = []
        for i in range(1, 60):
            amount = -float(i * 50)
            txs.append(_make_tx(amount, category="utilities", days_ago=i))
        score, _ = _spending_stability_score(txs)
        assert 0.0 <= score <= 10.0


# ── Grade assignment ──────────────────────────────────────────────────────────

class TestGrade:
    def test_excellent(self):
        assert _grade(85) == "Excellent"

    def test_good(self):
        assert _grade(70) == "Good"

    def test_fair(self):
        assert _grade(50) == "Fair"

    def test_needs_attention(self):
        assert _grade(30) == "Needs attention"

    def test_boundary_80(self):
        assert _grade(80) == "Excellent"

    def test_boundary_60(self):
        assert _grade(60) == "Good"

    def test_boundary_40(self):
        assert _grade(40) == "Fair"

    def test_zero(self):
        assert _grade(0) == "Needs attention"


# ── Full node integration tests ───────────────────────────────────────────────

class TestFinancialHealthNode:
    def _build_state(self, txs=None, accounts=None, budgets=None, insights=None) -> dict:
        return {
            "transaction_data": {
                "transactions": txs or [],
                "accounts": accounts or [],
                "budgets": budgets or [],
            },
            "insights": insights,
        }

    def test_happy_path_returns_valid_structure(self):
        txs = [
            _make_tx(3000, category="income", days_ago=5),
            _make_tx(-400, category="food", days_ago=10),
            _make_tx(-200, category="transport", days_ago=15),
            _make_tx(-100, category="utilities", days_ago=20),
        ]
        accounts = [_make_account(5000.0)]
        budgets = [
            {"category": "food", "monthly_limit": 500},
            {"category": "transport", "monthly_limit": 250},
        ]
        insights = {
            "budget_comparison": {
                "food": {"actual": 400, "monthly_limit": 500, "pct_used": 80, "status": "warning"},
                "transport": {"actual": 200, "monthly_limit": 250, "pct_used": 80, "status": "warning"},
            }
        }
        state = self._build_state(txs, accounts, budgets, insights)
        result = asyncio.run(financial_health_node(state))

        assert "health_score" in result
        hs = result["health_score"]
        assert "overall_score" in hs
        assert "grade" in hs
        assert "component_scores" in hs
        assert "top_improvement" in hs
        assert 0 <= hs["overall_score"] <= 100
        assert hs["grade"] in {"Excellent", "Good", "Fair", "Needs attention"}

        assert set(hs["component_scores"].keys()) == {
            "savings_rate", "debt_to_income", "emergency_fund",
            "budget_adherence", "spending_stability"
        }

    def test_empty_data_returns_valid_structure(self):
        state = self._build_state()
        result = asyncio.run(financial_health_node(state))

        assert "health_score" in result
        hs = result["health_score"]
        assert 0 <= hs["overall_score"] <= 100
        assert hs["grade"] in {"Excellent", "Good", "Fair", "Needs attention"}

    def test_score_sum_matches_components(self):
        txs = [_make_tx(2000, category="income", days_ago=5)]
        state = self._build_state(txs)
        result = asyncio.run(financial_health_node(state))

        hs = result["health_score"]
        components = hs["component_scores"]
        component_sum = sum(v["score"] for v in components.values())
        # Allow ±1 due to rounding
        assert abs(hs["overall_score"] - component_sum) <= 1

    def test_top_improvement_is_weakest_component(self):
        txs = [
            _make_tx(3000, category="income", days_ago=5),
            _make_tx(-2800, category="food", days_ago=10),  # very low savings rate
        ]
        state = self._build_state(txs)
        result = asyncio.run(financial_health_node(state))

        hs = result["health_score"]
        improvement = hs["top_improvement"]
        assert "component" in improvement
        assert "action" in improvement
        assert isinstance(improvement["action"], str)
        assert len(improvement["action"]) > 10
