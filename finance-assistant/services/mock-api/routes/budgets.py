from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Query
from sqlalchemy import select

from database.db import BudgetRecord, get_session
from models.account import BudgetCategory, UserBudget

router = APIRouter()
log = logging.getLogger("mock_api.routes.budgets")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "user_001")


def _evt(payload: dict) -> str:
    return json.dumps(payload, default=str)


@router.get("/budgets/{user_id}", response_model=UserBudget)
async def get_budgets(user_id: str):
    """Return all category budget targets for a user."""
    log.info("%s", _evt({"event": "budgets_get", "user_id": user_id}))

    def _fetch():
        sess = get_session()
        try:
            rows = sess.scalars(
                select(BudgetRecord).where(BudgetRecord.user_id == user_id)
            ).all()
            return rows
        finally:
            sess.close()

    rows = await asyncio.to_thread(_fetch)
    categories = [BudgetCategory(category=r.category, monthly_limit=r.monthly_limit) for r in rows]
    updated = max((r.updated_at for r in rows), default=datetime.now(timezone.utc))
    return UserBudget(user_id=user_id, categories=categories, updated_at=updated)


@router.put("/budgets/{user_id}", response_model=UserBudget)
async def set_budgets(user_id: str, categories: list[BudgetCategory]):
    """Upsert category budget targets for a user."""
    log.info("%s", _evt({"event": "budgets_set", "user_id": user_id, "categories": len(categories)}))
    now = datetime.now(timezone.utc)

    def _upsert():
        sess = get_session()
        try:
            for cat in categories:
                existing = sess.scalar(
                    select(BudgetRecord)
                    .where(BudgetRecord.user_id == user_id, BudgetRecord.category == cat.category)
                )
                if existing:
                    existing.monthly_limit = cat.monthly_limit
                    existing.updated_at = now
                else:
                    sess.add(BudgetRecord(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        category=cat.category,
                        monthly_limit=cat.monthly_limit,
                        updated_at=now,
                    ))
            sess.commit()
        finally:
            sess.close()

    await asyncio.to_thread(_upsert)
    return UserBudget(user_id=user_id, categories=categories, updated_at=now)
