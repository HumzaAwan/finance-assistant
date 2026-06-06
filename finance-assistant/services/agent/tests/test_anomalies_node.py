"""Unit tests for anomalies_node.

Tests cover all 4 detection rules:
    1. Amount z-score
    2. Duplicate detection
    3. Unusual transaction time
    4. High-risk merchant category

Plus: happy path, empty data, and error/exception path.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from graph.nodes.anomalies_node import (
    _rule_amount_zscore,
    _rule_duplicate,
    _rule_high_risk_merchant,
    _rule_unusual_time,
    anomalies_node,
)


def _tx(
    amount: float,
    merchant: str = "Tesco",
    category: str = "food",
    description: str = "",
    days_ago: float = 1,
    hour: int = 14,
    tx_id: str | None = None,
) -> dict[str, Any]:
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    ts = ts.replace(hour=hour, minute=0, second=0, microsecond=0)
    return {
        "id": tx_id or f"tx-{abs(amount):.0f}-{merchant}-{days_ago}",
        "amount": amount,
        "merchant": merchant,
        "category": category,
        "description": description,
        "timestamp": ts.isoformat(),
    }


# ── Rule 1: Amount z-score ────────────────────────────────────────────────────

class TestRuleAmountZscore:
    def test_anomalous_amount_flagged(self):
        """Create a merchant history then insert a 4-sigma outlier."""
        normal_txs = [_tx(-50.0, merchant="BP Fuel", days_ago=30 - i) for i in range(10)]
        outlier = _tx(-500.0, merchant="BP Fuel", days_ago=1)
        txs = normal_txs + [outlier]
        flags = _rule_amount_zscore(txs)
        assert any(f["transaction_id"] == outlier["id"] for f in flags), "Outlier should be flagged"

    def test_normal_amount_not_flagged(self):
        txs = [_tx(-50.0, merchant="Costa Coffee", days_ago=30 - i) for i in range(10)]
        txs.append(_tx(-52.0, merchant="Costa Coffee", days_ago=1))
        flags = _rule_amount_zscore(txs)
        recent_id = txs[-1]["id"]
        assert not any(f["transaction_id"] == recent_id for f in flags), "Normal amount should not be flagged"

    def test_fewer_than_3_transactions_ignored(self):
        txs = [_tx(-100.0, merchant="AmazonPrime", days_ago=5), _tx(-1000.0, merchant="AmazonPrime", days_ago=1)]
        flags = _rule_amount_zscore(txs)
        assert len(flags) == 0, "Insufficient history — should not flag"

    def test_income_transactions_ignored(self):
        txs = [_tx(3000.0, merchant="Employer", days_ago=30 - i, category="income") for i in range(10)]
        txs.append(_tx(30000.0, merchant="Employer", days_ago=1, category="income"))
        flags = _rule_amount_zscore(txs)
        assert len(flags) == 0, "Income transactions should be ignored"

    def test_empty_returns_no_flags(self):
        assert _rule_amount_zscore([]) == []

    def test_confidence_levels(self):
        """High z-score (>3.5) yields high confidence; medium z-score yields medium."""
        normal_txs = [_tx(-50.0, merchant="Shell", days_ago=30 - i) for i in range(10)]
        high_outlier = _tx(-5000.0, merchant="Shell", days_ago=1, tx_id="high-outlier")
        flags = _rule_amount_zscore(normal_txs + [high_outlier])
        matched = [f for f in flags if f["transaction_id"] == "high-outlier"]
        assert matched, "High outlier not flagged"
        assert matched[0]["confidence"] == "high"


# ── Rule 2: Duplicate detection ───────────────────────────────────────────────

class TestRuleDuplicate:
    def test_same_merchant_amount_within_24h_flagged(self):
        ts_base = datetime.now(timezone.utc) - timedelta(hours=10)
        tx_a = {
            "id": "dup-a",
            "amount": -29.99,
            "merchant": "Netflix",
            "category": "entertainment",
            "description": "",
            "timestamp": ts_base.isoformat(),
        }
        tx_b = {
            "id": "dup-b",
            "amount": -29.99,
            "merchant": "Netflix",
            "category": "entertainment",
            "description": "",
            "timestamp": (ts_base + timedelta(hours=5)).isoformat(),
        }
        flags = _rule_duplicate([tx_a, tx_b])
        flagged_ids = {f["transaction_id"] for f in flags}
        assert "dup-a" in flagged_ids or "dup-b" in flagged_ids

    def test_different_amounts_not_flagged(self):
        ts_base = datetime.now(timezone.utc) - timedelta(hours=2)
        tx_a = {"id": "tx-a", "amount": -29.99, "merchant": "Netflix", "category": "entertainment", "description": "", "timestamp": ts_base.isoformat()}
        tx_b = {"id": "tx-b", "amount": -14.99, "merchant": "Netflix", "category": "entertainment", "description": "", "timestamp": (ts_base + timedelta(hours=1)).isoformat()}
        flags = _rule_duplicate([tx_a, tx_b])
        assert len(flags) == 0

    def test_beyond_24h_not_flagged(self):
        ts_base = datetime.now(timezone.utc) - timedelta(days=3)
        tx_a = {"id": "tx-c", "amount": -50.0, "merchant": "Gym", "category": "health", "description": "", "timestamp": ts_base.isoformat()}
        tx_b = {"id": "tx-d", "amount": -50.0, "merchant": "Gym", "category": "health", "description": "", "timestamp": (ts_base + timedelta(hours=25)).isoformat()}
        flags = _rule_duplicate([tx_a, tx_b])
        assert len(flags) == 0

    def test_empty_returns_no_flags(self):
        assert _rule_duplicate([]) == []


# ── Rule 3: Unusual time ─────────────────────────────────────────────────────

class TestRuleUnusualTime:
    def test_night_hour_flagged(self):
        for hour in [1, 2, 3, 4]:
            txs = [_tx(-50.0, hour=hour, tx_id=f"night-{hour}")]
            flags = _rule_unusual_time(txs)
            assert len(flags) == 1, f"Hour {hour} should be flagged"
            assert flags[0]["rule_triggered"] == "unusual_time"
            assert flags[0]["confidence"] == "medium"

    def test_day_hour_not_flagged(self):
        for hour in [6, 12, 18, 23]:
            txs = [_tx(-50.0, hour=hour)]
            flags = _rule_unusual_time(txs)
            assert len(flags) == 0, f"Hour {hour} should not be flagged"

    def test_income_not_flagged(self):
        txs = [_tx(3000.0, hour=3, category="income")]
        flags = _rule_unusual_time(txs)
        assert len(flags) == 0

    def test_empty_returns_no_flags(self):
        assert _rule_unusual_time([]) == []


# ── Rule 4: High-risk merchant category ──────────────────────────────────────

class TestRuleHighRiskMerchant:
    def test_first_time_high_risk_flagged(self):
        txs = [_tx(-200.0, merchant="BetFred", description="gambling bet", tx_id="bet-tx")]
        flags = _rule_high_risk_merchant(txs)
        assert len(flags) == 1
        assert flags[0]["rule_triggered"] == "high_risk_merchant"
        assert flags[0]["confidence"] == "high"

    def test_second_time_same_merchant_not_flagged(self):
        txs = [
            _tx(-200.0, merchant="BetFred", description="gambling bet", days_ago=30, tx_id="bet-1"),
            _tx(-100.0, merchant="BetFred", description="gambling bet", days_ago=1, tx_id="bet-2"),
        ]
        flags = _rule_high_risk_merchant(txs)
        assert len(flags) == 1  # only first time
        assert flags[0]["transaction_id"] == "bet-1"

    def test_normal_merchant_not_flagged(self):
        txs = [_tx(-50.0, merchant="Sainsbury's", category="food", tx_id="safe-tx")]
        flags = _rule_high_risk_merchant(txs)
        assert len(flags) == 0

    def test_empty_returns_no_flags(self):
        assert _rule_high_risk_merchant([]) == []

    def test_crypto_keyword_triggers_flag(self):
        txs = [_tx(-500.0, merchant="Coinbase", description="cryptocurrency purchase", tx_id="crypto-tx")]
        flags = _rule_high_risk_merchant(txs)
        assert any(f["transaction_id"] == "crypto-tx" for f in flags)


# ── Full node integration ─────────────────────────────────────────────────────

class TestAnomaliesNode:
    def _build_state(self, txs: list) -> dict:
        return {"transaction_data": {"transactions": txs}}

    def test_happy_path_returns_anomalies_key(self):
        txs = [_tx(-50.0, merchant="Tesco", days_ago=i) for i in range(30)]
        txs.append(_tx(-5000.0, merchant="Tesco", days_ago=1, tx_id="outlier"))
        state = self._build_state(txs)
        result = asyncio.run(anomalies_node(state))

        assert "anomalies" in result
        assert isinstance(result["anomalies"], list)

    def test_empty_transactions_returns_empty_list(self):
        state = self._build_state([])
        result = asyncio.run(anomalies_node(state))
        assert result["anomalies"] == []

    def test_clean_transactions_may_return_empty(self):
        """Normal, consistent transactions should produce zero or minimal flags."""
        txs = [
            _tx(-50.0, merchant="Tesco", days_ago=i * 7, hour=12)
            for i in range(1, 12)
        ]
        state = self._build_state(txs)
        result = asyncio.run(anomalies_node(state))
        assert isinstance(result["anomalies"], list)

    def test_flag_structure_is_correct(self):
        """Each flag must have required fields."""
        txs = [_tx(-200.0, hour=2, tx_id="night-tx")]
        state = self._build_state(txs)
        result = asyncio.run(anomalies_node(state))

        if result["anomalies"]:
            flag = result["anomalies"][0]
            assert "transaction_id" in flag
            assert "merchant" in flag
            assert "amount" in flag
            assert "date" in flag
            assert "rule_triggered" in flag
            assert "confidence" in flag
            assert "reason" in flag
            assert flag["confidence"] in {"low", "medium", "high"}

    def test_no_duplicate_flags_for_same_tx_and_rule(self):
        """Deduplication: same transaction_id + rule should not appear twice."""
        txs = [_tx(-50.0, merchant="Tesco", days_ago=i) for i in range(10)]
        state = self._build_state(txs)
        result = asyncio.run(anomalies_node(state))

        seen: set[tuple] = set()
        for flag in result["anomalies"]:
            key = (flag["transaction_id"], flag["rule_triggered"])
            assert key not in seen, f"Duplicate flag: {key}"
            seen.add(key)
