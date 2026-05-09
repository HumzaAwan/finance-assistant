from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from database.db import TransactionRecord, get_session
from models.transaction import Transaction

router = APIRouter()
log = logging.getLogger("mock_api.routes.transactions")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "user_001")


def _evt(payload: dict) -> str:
    return json.dumps(payload, default=str)


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _list_filter_bounds(start_date: str | None, end_date: str | None) -> tuple[datetime | None, datetime | None]:
    """Inclusive calendar-day semantics for YYYY-MM-DD query params (UTC)."""

    start_bound: datetime | None = None
    end_exclusive: datetime | None = None

    if start_date:
        raw = start_date.strip()
        if "T" in raw:
            start_bound = parse_dt(raw)
        else:
            d = date.fromisoformat(raw[:10])
            start_bound = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

    if end_date:
        raw = end_date.strip()
        if "T" in raw:
            d = parse_dt(raw).astimezone(timezone.utc).date()
        else:
            d = date.fromisoformat(raw[:10])
        nx = d + timedelta(days=1)
        end_exclusive = datetime(nx.year, nx.month, nx.day, tzinfo=timezone.utc)

    return start_bound, end_exclusive


@router.get("/transactions/summary")
async def transaction_summary(
    user_id: str = Query(DEFAULT_USER_ID),
    period: str = Query("weekly", pattern=r"^(weekly|monthly|all)$"),
):
    log.info("%s", _evt({"event": "summary_request", "user_id": user_id, "period": period}))
    return await asyncio.to_thread(_summarize_blocking, user_id, period)


def _summarize_blocking(user_id: str, period: str) -> dict:
    session = get_session()
    try:
        now = datetime.now(timezone.utc)
        if period == "weekly":
            start = now - timedelta(days=7)
        elif period == "monthly":
            start = now - timedelta(days=30)
        else:
            start = datetime.min.replace(tzinfo=timezone.utc)

        txs = session.scalars(
            select(TransactionRecord)
            .where(TransactionRecord.user_id == user_id, TransactionRecord.timestamp >= start)
            .order_by(TransactionRecord.timestamp.asc())
        ).all()

        totals: dict[str, float] = defaultdict(float)
        for row in txs:
            if row.category == "income":
                totals["income"] += row.amount
            elif row.amount < 0:
                totals[row.category] += abs(row.amount)

        expense_total = sum(amt for cat, amt in totals.items() if cat != "income")
        expense_cats = {c: amt for c, amt in totals.items() if c != "income"}
        top_category = max(expense_cats, key=expense_cats.get, default="")

        denom_days_set = {
            tx.timestamp.astimezone(timezone.utc).date()
            for tx in txs
            if tx.category != "income" and tx.amount < 0
        }
        denom_days = len(denom_days_set) if denom_days_set else max(int((now - start).days) or 1, 1)

        avg_daily = expense_total / denom_days if denom_days else 0.0

        biggest_tx = session.scalar(
            select(TransactionRecord)
            .where(
                TransactionRecord.user_id == user_id,
                TransactionRecord.category != "income",
                TransactionRecord.amount < 0,
            )
            .order_by(TransactionRecord.amount.asc())
            .limit(1)
        )

        last_week_start = now - timedelta(days=7)
        prev_week_start = now - timedelta(days=14)

        avg_cur_week = session.scalar(
            select(func.avg(TransactionRecord.amount)).where(
                TransactionRecord.user_id == user_id,
                TransactionRecord.timestamp >= last_week_start,
                TransactionRecord.category != "income",
                TransactionRecord.amount < 0,
            )
        )

        avg_prev_week = session.scalar(
            select(func.avg(TransactionRecord.amount)).where(
                TransactionRecord.user_id == user_id,
                TransactionRecord.timestamp >= prev_week_start,
                TransactionRecord.timestamp < last_week_start,
                TransactionRecord.category != "income",
                TransactionRecord.amount < 0,
            )
        )

        wow = 0.0
        if avg_prev_week:
            denom = abs(avg_prev_week) if avg_prev_week else 1e-9
            wow = ((avg_cur_week or 0.0) - avg_prev_week) / denom * 100

        biggest_payload = None
        if biggest_tx:
            biggest_payload = {
                "amount": biggest_tx.amount,
                "description": biggest_tx.description,
                "merchant": biggest_tx.merchant,
                "timestamp": biggest_tx.timestamp.replace(tzinfo=timezone.utc).isoformat(),
            }

        payload = {
            "user_id": user_id,
            "period": period,
            "metrics": {
                "total_spent": round(expense_total, 2),
                "by_category": dict(totals),
                "top_category": top_category,
                "avg_daily_spend": round(abs(avg_daily), 2),
                "biggest_transaction": biggest_payload,
                "week_over_week_change": round(wow, 2),
            },
        }

        metrics = payload["metrics"]
        log.info(
            "%s",
            _evt(
                {
                    "event": "summary_complete",
                    "txn_count": len(txs),
                    "metrics": metrics,
                }
            ),
        )
        return payload
    finally:
        session.close()

@router.get("/transactions")
async def list_transactions(
    user_id: str = Query(DEFAULT_USER_ID),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=250),
):
    log.info("%s", _evt({"event": "transactions_list_request", "user_id": user_id}))

    def _list():
        sess = get_session()
        stmt = select(TransactionRecord).where(TransactionRecord.user_id == user_id)
        start_bound, end_exclusive = _list_filter_bounds(start_date, end_date)
        if start_bound is not None:
            stmt = stmt.where(TransactionRecord.timestamp >= start_bound)
        if end_exclusive is not None:
            stmt = stmt.where(TransactionRecord.timestamp < end_exclusive)
        if category:
            allowed = {"food", "transport", "utilities", "entertainment", "health", "shopping", "income"}
            if category not in allowed:
                raise HTTPException(status_code=400, detail="Invalid category filter")
            stmt = stmt.where(TransactionRecord.category == category)
        stmt = stmt.order_by(TransactionRecord.timestamp.desc()).limit(limit)
        try:
            rows = sess.scalars(stmt).all()
            ordered = []
            for row in reversed(rows):
                ordered.append(Transaction.model_validate(row, from_attributes=True).model_dump(mode="json"))
            return ordered
        finally:
            sess.close()

    txs = await asyncio.to_thread(_list)
    resp = {"user_id": user_id, "count": len(txs), "transactions": txs}
    log.info("%s", _evt({"event": "transactions_list_ready", **{"count": len(txs)}}))
    return resp


@router.get("/transactions/{transaction_id}", response_model=Transaction)
async def get_transaction(transaction_id: str):
    log.info("%s", _evt({"event": "txn_lookup", "id": transaction_id}))

    def _fetch():
        sess = get_session()
        try:
            row = sess.get(TransactionRecord, transaction_id)
            if not row:
                return None
            return Transaction.model_validate(row, from_attributes=True)
        finally:
            sess.close()

    record = await asyncio.to_thread(_fetch)
    if not record:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return record
