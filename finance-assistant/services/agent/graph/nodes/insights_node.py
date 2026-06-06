from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from graph.state import AgentState

_log = logging.getLogger("agent.graph.nodes.insights_node")


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _week_bucket(dt: datetime) -> tuple[int, int]:
    iso = dt.isocalendar()
    return int(iso[0]), int(iso[1])


def _utc_date_only(value: Any) -> date | None:
    parsed = _parse_dt(value)
    return parsed.astimezone(timezone.utc).date() if parsed else None


def calculate_week_comparison(expenses: list[dict]) -> float:
    totals: dict[tuple[int, int], float] = defaultdict(float)
    for row in expenses:
        parsed = _parse_dt(row.get("timestamp"))
        if not parsed:
            continue
        totals[_week_bucket(parsed)] += abs(float(row.get("amount", 0.0)))

    sorted_keys = sorted(totals)
    if len(sorted_keys) < 2:
        return 0.0

    current_key = sorted_keys[-1]
    prev_key = sorted_keys[-2]
    curr = totals[current_key]
    prev_tot = totals[prev_key]
    if prev_tot == 0:
        return 0.0
    return (curr - prev_tot) / prev_tot * 100.0


def _calendar_week_comparison(expenses: list[dict]) -> dict[str, Any]:
    """Split debits by ISO Monday weeks in UTC — current week partial through today."""
    today = datetime.now(timezone.utc).date()
    monday_this = today - timedelta(days=today.weekday())
    prior_monday = monday_this - timedelta(days=7)
    prior_sunday = monday_this - timedelta(days=1)

    current_week = 0.0
    prior_week = 0.0
    n_curr = 0
    n_prior = 0

    for row in expenses:
        d = _utc_date_only(row.get("timestamp"))
        if d is None:
            continue
        amt = abs(float(row.get("amount", 0.0)))
        if prior_monday <= d <= prior_sunday:
            prior_week += amt
            n_prior += 1
        if monday_this <= d <= today:
            current_week += amt
            n_curr += 1

    pct: float | None
    if prior_week > 0:
        pct = round((current_week - prior_week) / prior_week * 100.0, 2)
    else:
        pct = None

    return {
        "definition": "UTC calendar; prior week is prior Monday–Sunday; current week runs Monday through reference_today (partial).",
        "reference_today_utc": today.isoformat(),
        "current_week_partial_spend": round(current_week, 2),
        "prior_calendar_week_spend": round(prior_week, 2),
        "pct_delta_current_partial_vs_prior_full_week_pct": pct,
        "expense_row_counts": {"current_week": n_curr, "prior_week": n_prior},
    }


def aggregate_insights(transactions: list[dict]) -> dict | None:
    if not transactions:
        return None

    expenses: list[dict] = []
    by_category: defaultdict[str, float] = defaultdict(float)
    total_spent = 0.0

    for tx in transactions:
        category = str(tx.get("category", "misc"))
        amount_value = float(tx.get("amount", 0))

        if category == "income" and amount_value > 0:
            continue
        if amount_value >= 0:
            continue

        abs_amount = abs(amount_value)
        expenses.append(tx)
        by_category[category] += abs_amount
        total_spent += abs_amount

    top_category = max(by_category.items(), default=("", 0.0), key=lambda item: item[1])[0]

    timestamps = [_parse_dt(e.get("timestamp")) for e in expenses]
    timestamps = [t for t in timestamps if t is not None]
    unique_days = {t.date().isoformat() for t in timestamps}
    denominator = len(unique_days) if unique_days else 1

    avg_daily_spend = total_spent / denominator

    biggest_transaction = None
    if expenses:
        worst = min(expenses, key=lambda row: float(row.get("amount", 0.0)))
        biggest_transaction = {
            "amount": float(worst.get("amount", 0)),
            "description": str(worst.get("description", "")),
            "merchant": str(worst.get("merchant", "")),
        }

    wow = calculate_week_comparison(expenses)
    cal_week = _calendar_week_comparison(expenses)

    return {
        "total_spent": round(total_spent, 2),
        "scope_note": "total_spent aggregates every expense row in the fetched batch; use calendar_week_comparison for this week vs last week.",
        "by_category": dict(by_category),
        "top_category": top_category,
        "avg_daily_spend": round(avg_daily_spend, 2),
        "biggest_transaction": biggest_transaction,
        "week_over_week_change": round(wow, 2),
        "calendar_week_comparison": cal_week,
    }


def _budget_comparison(by_category: dict[str, float], budgets: list[dict]) -> dict:
    """Compare actual monthly-equivalent spending against budget targets."""
    if not budgets:
        return {}

    budget_map = {b["category"]: float(b["monthly_limit"]) for b in budgets}
    comparison: dict[str, dict] = {}

    for cat, actual in sorted(by_category.items()):
        limit = budget_map.get(cat)
        if limit is None:
            continue
        pct = round(actual / limit * 100, 1) if limit > 0 else None
        comparison[cat] = {
            "actual": round(actual, 2),
            "monthly_limit": limit,
            "pct_used": pct,
            "status": "over" if (pct or 0) > 100 else "warning" if (pct or 0) > 80 else "ok",
        }

    return comparison


async def insights_node(state: AgentState) -> dict:
    bundle = state.get("transaction_data") or {}
    txs = list(bundle.get("transactions") or [])

    insights_payload = aggregate_insights(txs)

    if not insights_payload:
        _log.info({"event": "insights_empty_dataset"})
        return {"insights": None}

    budgets = bundle.get("budgets") or []
    if budgets:
        comparison = _budget_comparison(insights_payload.get("by_category", {}), budgets)
        if comparison:
            insights_payload["budget_comparison"] = comparison
            over_budget = [c for c, v in comparison.items() if v["status"] == "over"]
            if over_budget:
                insights_payload["over_budget_categories"] = over_budget

    _log.info({"event": "insights_ready", "top_category": insights_payload.get("top_category"), "budget_cats": len(budgets)})
    return {"insights": insights_payload}
