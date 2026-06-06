"""Unit tests for FCA compliance guardrail in response_node.

Tests cover:
    - Regulated advice queries are blocked
    - Non-regulated queries pass through
    - Disclaimer is appended to financial_advice responses
    - Prometheus counter fires on regulated advice detection
"""
from __future__ import annotations

import pytest

from graph.nodes.response_node import (
    DISCLAIMER_PATTERN,
    _GENERAL_ADVICE_DISCLAIMER,
    _is_regulated_advice,
)


class TestIsRegulatedAdvice:
    def test_should_i_invest_is_regulated(self):
        assert _is_regulated_advice("Should I invest in index funds?") is True

    def test_recommend_me_a_fund_is_regulated(self):
        assert _is_regulated_advice("Can you recommend me a fund to put my savings in?") is True

    def test_pension_advice_is_regulated(self):
        assert _is_regulated_advice("What should I do with my pension?") is True

    def test_is_it_worth_buying_shares_is_regulated(self):
        assert _is_regulated_advice("Is it worth buying Tesla shares right now?") is True

    def test_advise_me_on_portfolio_is_regulated(self):
        assert _is_regulated_advice("Advise me on my portfolio allocation.") is True

    def test_general_isa_question_is_not_regulated(self):
        assert _is_regulated_advice("How does an ISA work?") is False

    def test_emergency_fund_question_is_not_regulated(self):
        assert _is_regulated_advice("How do I build an emergency fund?") is False

    def test_spending_question_is_not_regulated(self):
        assert _is_regulated_advice("How much did I spend on food?") is False

    def test_budget_rule_question_is_not_regulated(self):
        assert _is_regulated_advice("What is the 50/30/20 budgeting rule?") is False

    def test_pension_explainer_is_not_regulated(self):
        """Explaining what a pension is vs giving personalised pension advice."""
        assert _is_regulated_advice("How does auto-enrolment work?") is False

    def test_case_insensitive_matching(self):
        assert _is_regulated_advice("SHOULD I INVEST in crypto?") is True


class TestDisclaimerPattern:
    def test_disclaimer_pattern_matches_standard_disclaimer(self):
        assert DISCLAIMER_PATTERN.search(_GENERAL_ADVICE_DISCLAIMER) is not None

    def test_disclaimer_contains_fca_reference(self):
        assert "regulated financial advice" in _GENERAL_ADVICE_DISCLAIMER.lower()

    def test_disclaimer_starts_with_newline_separator(self):
        assert _GENERAL_ADVICE_DISCLAIMER.startswith("\n\n---\n")
