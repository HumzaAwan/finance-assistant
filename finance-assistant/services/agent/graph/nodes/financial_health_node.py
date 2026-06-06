"""Financial health scoring node.

Computes a 0–100 score across five weighted components derived from real
transaction and account data. Designed to demonstrate domain-specific ML
engineering for a UK fintech context.

Bug 4 fix: anchor_date is pinned ONCE at the top of financial_health_node and
passed to every component function. No component calls datetime.now() or
date.today() independently, ensuring the score is identical for the same user
and the same calendar date regardless of execution timing.
"""
from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from graph.state import AgentState

_log = logging.getLogger("agent.graph.nodes.financial_health_node")

# Per-session cache: (user_id, anchor_date_iso) → health_score dict.
# Avoids recomputing for identical inputs within the same interpreter process.
_SCORE_CACHE: dict[tuple[str, str], dict] = {}

# Categories considered "essential" spend for emergency fund and stability calculations.
ESSENTIAL_CATEGORIES: frozenset[str] = frozenset({"utilities", "food", "transport"})

# Account types considered savings/liquid emergency reserves.
SAVINGS_ACCOUNT_TYPES: frozenset[str] = frozenset({"savings", "current", "checking"})

GRADE_THRESHOLDS = [
    (80, "Excellent"),
    (60, "Good"),
    (40, "Fair"),
    (0, "Needs attention"),
]


def _grade(score: float) -> str:
    for threshold, label in GRADE_THRESHOLDS:
        if score >= threshold:
            return label
    return "Needs attention"


def _cutoff_dt(anchor: date, days: int) -> datetime:
    """Return a timezone-aware cutoff datetime = anchor − days."""
    return datetime(anchor.year, anchor.month, anchor.day, tzinfo=timezone.utc) - timedelta(days=days)


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


# ── Component 1: Savings Rate ─────────────────────────────────────────────────

def _savings_rate_score(transactions: list[dict], anchor: date) -> tuple[float, str]:
    """25 pts max. savings_rate = (income − spend) / income.

    First tries the 30-day window; if no income is found, extends to 90 days
    before defaulting to 0 so that sparse pay-cycle data is handled gracefully.
    """
    for days in (30, 90):
        cutoff = _cutoff_dt(anchor, days)
        total_income = 0.0
        total_spend = 0.0

        for tx in transactions:
            ts = _parse_ts(tx.get("timestamp"))
            if ts is None or ts < cutoff:
                continue
            amount = float(tx.get("amount", 0))
            if amount > 0:
                total_income += amount
            else:
                total_spend += abs(amount)

        if total_income > 0:
            break  # found income data — use this window
    else:
        return 0.0, f"No income found in last 90 days — cannot compute savings rate."

    savings_rate = max(0.0, (total_income - total_spend) / total_income)
    score = min(25.0, 25.0 * (savings_rate / 0.20)) if savings_rate < 0.20 else 25.0

    explanation = (
        f"Savings rate {savings_rate:.1%} over last {days} days "
        f"(income £{total_income:.0f}, spend £{total_spend:.0f}). "
        f"Target: ≥20%."
    )
    return round(score, 1), explanation


# ── Component 2: Debt-to-Income Ratio ────────────────────────────────────────

def _dti_score(transactions: list[dict], anchor: date) -> tuple[float, str]:
    """20 pts max. Approximated from transactions labelled as debt repayments."""
    cutoff = _cutoff_dt(anchor, 30)

    debt_keywords = frozenset({
        "loan", "mortgage", "credit card", "repayment", "finance",
        "klarna", "afterpay", "clearpay", "overdraft",
    })

    monthly_debt = 0.0
    monthly_income = 0.0

    for tx in transactions:
        ts = _parse_ts(tx.get("timestamp"))
        if ts is None or ts < cutoff:
            continue

        amount = float(tx.get("amount", 0))
        desc = str(tx.get("description", "")).lower() + " " + str(tx.get("merchant", "")).lower()

        if amount > 0:
            monthly_income += amount
        elif any(kw in desc for kw in debt_keywords):
            monthly_debt += abs(amount)

    if monthly_income <= 0:
        return 10.0, "Insufficient income data for DTI calculation; defaulting to mid-range score."

    dti = monthly_debt / monthly_income

    if dti <= 0.15:
        score = 20.0
        band = "excellent (≤15%)"
    elif dti <= 0.36:
        score = 10.0 + 10.0 * (0.36 - dti) / (0.36 - 0.15)
        band = "moderate (15–36%)"
    elif dti <= 0.50:
        score = 10.0 * (0.50 - dti) / (0.50 - 0.36)
        band = "high (36–50%)"
    else:
        score = 0.0
        band = "very high (>50%)"

    explanation = (
        f"Debt-to-income ratio {dti:.1%} ({band}). "
        f"Monthly debt repayments £{monthly_debt:.0f} vs income £{monthly_income:.0f}."
    )
    return round(score, 1), explanation


# ── Component 3: Emergency Fund Coverage ─────────────────────────────────────

def _emergency_fund_score(
    transactions: list[dict], accounts: list[dict], anchor: date
) -> tuple[float, str]:
    """25 pts max. coverage = savings_balance / avg_monthly_essential_spend."""
    savings_balance = sum(
        float(a.get("balance", 0))
        for a in accounts
        if str(a.get("account_type", "")).lower() in SAVINGS_ACCOUNT_TYPES
    )

    cutoff = _cutoff_dt(anchor, 90)
    essential_spend = 0.0

    for tx in transactions:
        category = str(tx.get("category", "")).lower()
        if category not in ESSENTIAL_CATEGORIES:
            continue
        ts = _parse_ts(tx.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        amount = float(tx.get("amount", 0))
        if amount < 0:
            essential_spend += abs(amount)

    avg_monthly_essential = essential_spend / 3.0  # 90-day window → 3 months

    if avg_monthly_essential <= 0:
        return 12.5, "No essential spend data found. Emergency fund score defaulted."

    coverage = savings_balance / avg_monthly_essential

    if coverage >= 6:
        score = 25.0
    elif coverage >= 1:
        score = 25.0 * (coverage - 1) / (6 - 1)
    else:
        score = 0.0

    explanation = (
        f"Emergency fund covers {coverage:.1f} months of essential spend. "
        f"Savings balance £{savings_balance:.0f}, avg monthly essential spend £{avg_monthly_essential:.0f}. "
        f"Target: ≥6 months."
    )
    return round(score, 1), explanation


# ── Component 4: Budget Adherence ────────────────────────────────────────────

def _budget_adherence_score(insights: dict | None, budgets: list[dict]) -> tuple[float, str]:
    """20 pts max. pct_categories_within_budget × 20."""
    if not budgets or not insights:
        return 10.0, "Insufficient budget data for adherence calculation."

    budget_comparison = insights.get("budget_comparison") or {}
    if not budget_comparison:
        return 10.0, "Budget comparison not available."

    total_cats = len(budget_comparison)
    within_budget = sum(1 for v in budget_comparison.values() if v.get("status") in {"ok", "warning"})
    over_budget = total_cats - within_budget

    pct_within = within_budget / total_cats if total_cats > 0 else 0.0
    score = 20.0 * pct_within

    explanation = (
        f"{within_budget}/{total_cats} categories within budget ({pct_within:.0%}). "
        f"{over_budget} categories over budget."
    )
    return round(score, 1), explanation


# ── Component 5: Spending Stability ──────────────────────────────────────────

def _spending_stability_score(transactions: list[dict], anchor: date) -> tuple[float, str]:
    """10 pts max. CV of weekly essential spend over last 8 weeks."""
    cutoff = _cutoff_dt(anchor, 56)  # 8 weeks = 56 days; anchor is fixed

    weekly: defaultdict[tuple[int, int], float] = defaultdict(float)

    for tx in transactions:
        category = str(tx.get("category", "")).lower()
        if category not in ESSENTIAL_CATEGORIES:
            continue
        ts = _parse_ts(tx.get("timestamp"))
        if ts is None or ts < cutoff:
            continue
        amount = float(tx.get("amount", 0))
        if amount < 0:
            iso = ts.isocalendar()
            weekly[(iso[0], iso[1])] += abs(amount)

    values = list(weekly.values())

    if len(values) < 2:
        return 5.0, "Fewer than 2 weeks of data; stability score defaulted to mid-range."

    mean = statistics.mean(values)
    std = statistics.stdev(values)
    cv = std / mean if mean > 0 else 0.0

    if cv < 0.10:
        score = 10.0
    elif cv <= 0.40:
        score = 10.0 * (0.40 - cv) / (0.40 - 0.10)
    else:
        score = 0.0

    explanation = (
        f"Weekly essential spend CV = {cv:.2f} (std £{std:.0f} / mean £{mean:.0f}). "
        f"Target: CV < 0.10 for full score."
    )
    return round(score, 1), explanation


# ── Aggregation ───────────────────────────────────────────────────────────────

async def financial_health_node(state: AgentState) -> dict:
    """Compute the 5-component financial health score and write it to state.

    anchor_date is pinned once here so all five components use the same date
    window and the score is deterministic for a given user + calendar day.
    """
    user_id: str = str(state.get("user_id") or "unknown")

    # Pin the calculation anchor to today — called ONCE for the whole invocation.
    anchor: date = date.today()
    cache_key: tuple[str, str] = (user_id, anchor.isoformat())

    # Return cached result if the same user is scored again in the same session.
    if cache_key in _SCORE_CACHE:
        _log.info({
            "event": "financial_health_cache_hit",
            "user_id": user_id,
            "anchor_date": anchor.isoformat(),
        })
        return {"health_score": _SCORE_CACHE[cache_key]}

    bundle = state.get("transaction_data") or {}
    # ALL transactions fetched by transactions_node — passed once to every component.
    transactions: list[dict] = bundle.get("transactions") or []
    accounts: list[dict] = bundle.get("accounts") or []
    budgets: list[dict] = bundle.get("budgets") or []
    insights: dict | None = state.get("insights")

    # Compute all five components using the shared anchor date.
    sr_score, sr_explanation = _savings_rate_score(transactions, anchor)
    dti_score_val, dti_explanation = _dti_score(transactions, anchor)
    ef_score, ef_explanation = _emergency_fund_score(transactions, accounts, anchor)
    ba_score, ba_explanation = _budget_adherence_score(insights, budgets)
    ss_score, ss_explanation = _spending_stability_score(transactions, anchor)

    component_scores: dict[str, Any] = {
        "savings_rate": {
            "score": sr_score,
            "max": 25,
            "explanation": sr_explanation,
        },
        "debt_to_income": {
            "score": dti_score_val,
            "max": 20,
            "explanation": dti_explanation,
        },
        "emergency_fund": {
            "score": ef_score,
            "max": 25,
            "explanation": ef_explanation,
        },
        "budget_adherence": {
            "score": ba_score,
            "max": 20,
            "explanation": ba_explanation,
        },
        "spending_stability": {
            "score": ss_score,
            "max": 10,
            "explanation": ss_explanation,
        },
    }

    overall_score = int(round(sr_score + dti_score_val + ef_score + ba_score + ss_score))
    grade = _grade(overall_score)

    weakest_key = min(
        component_scores,
        key=lambda k: component_scores[k]["score"] / component_scores[k]["max"],
    )
    weakest = component_scores[weakest_key]

    improvement_actions = {
        "savings_rate": "Aim to save at least 20% of your monthly income by reducing discretionary spend.",
        "debt_to_income": "Focus on paying down high-interest debt to bring your DTI below 36%.",
        "emergency_fund": "Build your emergency fund to cover 3–6 months of essential expenses in a Cash ISA or instant-access account.",
        "budget_adherence": "Review your over-budget categories and set realistic spending caps using the envelope method.",
        "spending_stability": "Smooth out irregular essential spend by setting up direct debits and monthly billing cycles.",
    }

    top_improvement: dict[str, Any] = {
        "component": weakest_key,
        "current_score": weakest["score"],
        "max_score": weakest["max"],
        "action": improvement_actions.get(weakest_key, "Review and improve this component."),
    }

    health_score: dict[str, Any] = {
        "overall_score": overall_score,
        "grade": grade,
        "component_scores": component_scores,
        "top_improvement": top_improvement,
        "anchor_date": anchor.isoformat(),
    }

    # Cache the result for this (user_id, anchor_date) pair.
    _SCORE_CACHE[cache_key] = health_score

    _log.info({
        "event": "financial_health_computed",
        "user_id": user_id,
        "anchor_date": anchor.isoformat(),
        "overall_score": overall_score,
        "grade": grade,
        "weakest_component": weakest_key,
    })

    return {"health_score": health_score}
