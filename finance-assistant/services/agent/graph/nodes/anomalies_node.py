"""Anomaly detection node.

Applies four rule-based detectors to the last 90 days of transactions and
returns a list of flagged transactions with rule, confidence, and reason.

Rules:
    1. Amount z-score: per-merchant leave-one-out historical anomaly
    2. Duplicate detection: same merchant + amount within configurable window
    3. Unusual transaction time: domestic card transactions 01:00–05:00 UTC
    4. High-risk MCC (merchant category): first-time use of risk-flagged categories

Bug 5B fixes:
    - Z-score uses leave-one-out baseline (transaction excluded from its own mean/std)
      so an outlier cannot inflate its own baseline and escape detection.
    - Duplicate window is configurable via ANOMALY_DUPLICATE_WINDOW_HOURS env var.
    - Unusual-time rule documents that timestamps are treated as UTC (seed stores UTC).
    - Per-transaction evaluation log line added for every transaction evaluated.
"""
from __future__ import annotations

import logging
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from prometheus_client import Counter

from graph.state import AgentState

_log = logging.getLogger("agent.graph.nodes.anomalies_node")

ANOMALIES_DETECTED = Counter(
    "anomalies_detected_total",
    "Anomalous transactions detected by the anomaly detection engine",
    ["rule_triggered"],
)

# Merchant categories considered high-risk for first-time use.
HIGH_RISK_CATEGORIES: frozenset[str] = frozenset({
    "gambling", "casino", "betting", "lottery",
    "crypto", "cryptocurrency", "bitcoin", "exchange",
    "forex", "foreign exchange", "fx trading",
    "payday", "payday loan", "short-term loan",
})

# Night hours (01:00–05:00) flagged for unusual domestic card use.
_NIGHT_START_HOUR = 1
_NIGHT_END_HOUR = 5

# Z-score thresholds.
_Z_HIGH = 3.5
_Z_MEDIUM = 2.5

# Duplicate detection window — configurable via environment variable.
_DUPLICATE_WINDOW_SECONDS: int = int(
    float(os.getenv("ANOMALY_DUPLICATE_WINDOW_HOURS", "24")) * 3600
)


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _flag(tx: dict, rule: str, confidence: str, reason: str) -> dict:
    return {
        "transaction_id": tx.get("id", ""),
        "merchant": tx.get("merchant", ""),
        "amount": float(tx.get("amount", 0)),
        "date": str(tx.get("timestamp", ""))[:10],
        "rule_triggered": rule,
        "confidence": confidence,
        "reason": reason,
    }


# ── Rule 1: Amount z-score (leave-one-out) ────────────────────────────────────

def _rule_amount_zscore(transactions: list[dict]) -> list[dict]:
    """Flag transactions where amount deviates >2.5 std devs from merchant history.

    Uses leave-one-out baseline: the transaction under test is excluded from its
    own mean/std calculation so a genuine outlier cannot inflate the baseline and
    hide itself.
    """
    # Build per-merchant amount list: (tx_id, abs_amount)
    merchant_txs: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)

    for tx in transactions:
        amount = float(tx.get("amount", 0))
        if amount < 0:  # expenses only
            merchant = str(tx.get("merchant", "UNKNOWN")).upper()
            merchant_txs[merchant].append((str(tx.get("id", "")), abs(amount)))

    flags: list[dict] = []
    for tx in transactions:
        amount = float(tx.get("amount", 0))
        if amount >= 0:
            continue
        merchant = str(tx.get("merchant", "UNKNOWN")).upper()
        tx_id = str(tx.get("id", ""))
        all_amounts = merchant_txs[merchant]

        # Exclude current transaction from baseline (leave-one-out)
        baseline = [a for tid, a in all_amounts if tid != tx_id]
        if len(baseline) < 3:
            _log.debug({
                "event": "anomaly_evaluated",
                "merchant": tx.get("merchant"),
                "amount": abs(amount),
                "rules_fired": [],
                "skip_reason": "insufficient_history",
            })
            continue

        mean = statistics.mean(baseline)
        stdev = statistics.stdev(baseline)
        if stdev == 0:
            continue

        z = (abs(amount) - mean) / stdev

        rules_fired: list[str] = []
        if z > _Z_HIGH:
            confidence = "high"
            rules_fired.append("amount_anomaly")
        elif z > _Z_MEDIUM:
            confidence = "medium"
            rules_fired.append("amount_anomaly")
        else:
            _log.debug({
                "event": "anomaly_evaluated",
                "merchant": tx.get("merchant"),
                "amount": abs(amount),
                "z_score": round(z, 2),
                "rules_fired": [],
            })
            continue

        _log.info({
            "event": "anomaly_evaluated",
            "merchant": tx.get("merchant"),
            "amount": abs(amount),
            "z_score": round(z, 2),
            "mean_baseline": round(mean, 2),
            "stdev_baseline": round(stdev, 2),
            "rules_fired": rules_fired,
        })

        reason = (
            f"Transaction of £{abs(amount):.2f} at {tx.get('merchant', '')} is "
            f"{z:.1f} standard deviations above the usual amount "
            f"(historical mean £{mean:.2f}, std £{stdev:.2f})."
        )
        flags.append(_flag(tx, "amount_anomaly", confidence, reason))

    return flags


# ── Rule 2: Duplicate detection ───────────────────────────────────────────────

def _rule_duplicate(transactions: list[dict]) -> list[dict]:
    """Flag pairs of transactions with same merchant AND amount within the duplicate window.

    Window is configurable via ANOMALY_DUPLICATE_WINDOW_HOURS (default 24h).
    Both merchant name and amount must match exactly (cents-level precision).
    """
    flags: list[dict] = []
    expense_txs = [tx for tx in transactions if float(tx.get("amount", 0)) < 0]

    for i, tx_a in enumerate(expense_txs):
        ts_a = _parse_ts(tx_a.get("timestamp"))
        if ts_a is None:
            continue
        amt_a = float(tx_a.get("amount", 0))
        merchant_a = str(tx_a.get("merchant", "")).upper()

        for tx_b in expense_txs[i + 1:]:
            if tx_a.get("id") == tx_b.get("id"):
                continue
            ts_b = _parse_ts(tx_b.get("timestamp"))
            if ts_b is None:
                continue
            if abs((ts_a - ts_b).total_seconds()) > _DUPLICATE_WINDOW_SECONDS:
                continue
            if float(tx_b.get("amount", 0)) != amt_a:
                continue
            if str(tx_b.get("merchant", "")).upper() != merchant_a:
                continue

            reason = (
                f"Duplicate charge: £{abs(amt_a):.2f} at {tx_a.get('merchant', '')} "
                f"appears twice within {_DUPLICATE_WINDOW_SECONDS // 3600}h "
                f"({tx_a.get('timestamp', '')[:16]} and "
                f"{tx_b.get('timestamp', '')[:16]})."
            )
            _log.info({
                "event": "anomaly_evaluated",
                "merchant": tx_a.get("merchant"),
                "amount": abs(amt_a),
                "rules_fired": ["duplicate"],
            })
            flags.append(_flag(tx_a, "duplicate", "high", reason))
            flags.append(_flag(tx_b, "duplicate", "high", reason))
            break

    return flags


# ── Rule 3: Unusual transaction time ─────────────────────────────────────────

def _rule_unusual_time(transactions: list[dict]) -> list[dict]:
    """Flag expense transactions occurring between 01:00 and 05:00 UTC.

    Timestamps are stored as UTC in the seed database. The rule operates on UTC
    hours directly. If the deployment locale is non-UTC, convert the timestamp
    to the desired timezone before extracting the hour.
    """
    flags: list[dict] = []
    for tx in transactions:
        amount = float(tx.get("amount", 0))
        if amount >= 0:
            continue
        ts = _parse_ts(tx.get("timestamp"))
        if ts is None:
            continue
        hour = ts.hour  # UTC hour (seed stores UTC)
        fired = _NIGHT_START_HOUR <= hour < _NIGHT_END_HOUR
        _log.debug({
            "event": "anomaly_evaluated",
            "merchant": tx.get("merchant"),
            "amount": abs(amount),
            "hour_utc": hour,
            "rules_fired": ["unusual_time"] if fired else [],
        })
        if fired:
            reason = (
                f"Transaction of £{abs(amount):.2f} at {tx.get('merchant', '')} "
                f"occurred at {ts.strftime('%H:%M')} UTC, which is an unusual hour "
                f"(between 01:00 and 05:00)."
            )
            flags.append(_flag(tx, "unusual_time", "medium", reason))

    return flags


# ── Rule 4: First-time high-risk merchant category ───────────────────────────

def _rule_high_risk_merchant(transactions: list[dict]) -> list[dict]:
    """Flag first-ever transactions at merchants with high-risk category keywords."""
    seen_merchants: set[str] = set()
    flags: list[dict] = []

    sorted_txs = sorted(
        [tx for tx in transactions if float(tx.get("amount", 0)) < 0],
        key=lambda tx: str(tx.get("timestamp", "")),
    )

    for tx in sorted_txs:
        merchant = str(tx.get("merchant", "")).lower()
        category = str(tx.get("category", "")).lower()
        desc = str(tx.get("description", "")).lower()

        combined = f"{merchant} {category} {desc}"
        is_high_risk = any(kw in combined for kw in HIGH_RISK_CATEGORIES)

        fired: list[str] = []
        if is_high_risk and merchant not in seen_merchants:
            fired.append("high_risk_merchant")

        _log.debug({
            "event": "anomaly_evaluated",
            "merchant": tx.get("merchant"),
            "amount": abs(float(tx.get("amount", 0))),
            "is_high_risk": is_high_risk,
            "rules_fired": fired,
        })

        if not is_high_risk:
            seen_merchants.add(merchant)
            continue

        if merchant in seen_merchants:
            continue

        reason = (
            f"First-time transaction of £{abs(float(tx.get('amount', 0))):.2f} at "
            f"'{tx.get('merchant', '')}', which appears to be a high-risk merchant "
            f"category (gambling, crypto, forex, or payday lending)."
        )
        flags.append(_flag(tx, "high_risk_merchant", "high", reason))
        seen_merchants.add(merchant)

    return flags


# ── Node ──────────────────────────────────────────────────────────────────────

async def anomalies_node(state: AgentState) -> dict:
    """Run all four anomaly detection rules and return flagged transactions."""
    bundle = state.get("transaction_data") or {}
    transactions: list[dict] = bundle.get("transactions") or []

    if not transactions:
        _log.info({"event": "anomalies_no_transactions"})
        return {"anomalies": []}

    _log.info({
        "event": "anomaly_detection_start",
        "transaction_count": len(transactions),
        "duplicate_window_hours": _DUPLICATE_WINDOW_SECONDS // 3600,
    })

    zscore_flags = _rule_amount_zscore(transactions)
    duplicate_flags = _rule_duplicate(transactions)
    time_flags = _rule_unusual_time(transactions)
    risk_flags = _rule_high_risk_merchant(transactions)

    all_flags: list[dict] = []
    all_flags.extend(zscore_flags)
    all_flags.extend(duplicate_flags)
    all_flags.extend(time_flags)
    all_flags.extend(risk_flags)

    # Update Prometheus counters
    for rule, count in [
        ("amount_anomaly", len(zscore_flags)),
        ("duplicate", len(duplicate_flags)),
        ("unusual_time", len(time_flags)),
        ("high_risk_merchant", len(risk_flags)),
    ]:
        if count:
            ANOMALIES_DETECTED.labels(rule_triggered=rule).inc(count)

    # Deduplicate by transaction_id + rule
    seen: set[tuple[str, str]] = set()
    unique_flags: list[dict] = []
    for flag in all_flags:
        key = (flag["transaction_id"], flag["rule_triggered"])
        if key not in seen:
            seen.add(key)
            unique_flags.append(flag)

    _log.info({
        "event": "anomaly_detection_complete",
        "total_flags": len(unique_flags),
        "by_rule": {
            "amount_anomaly": len(zscore_flags),
            "duplicate": len(duplicate_flags),
            "unusual_time": len(time_flags),
            "high_risk_merchant": len(risk_flags),
        },
    })

    return {"anomalies": unique_flags}
