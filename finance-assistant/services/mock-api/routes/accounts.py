from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from database.db import AccountRecord, get_session
from models.account import Account

router = APIRouter()
log = logging.getLogger("mock_api.routes.accounts")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "user_001")


def _evt(payload: dict) -> str:
    return json.dumps(payload, default=str)


@router.get("/accounts", response_model=list[Account])
async def list_accounts(user_id: str = Query(DEFAULT_USER_ID)):
    """Return all accounts for a user."""
    log.info("%s", _evt({"event": "accounts_list", "user_id": user_id}))

    def _fetch():
        sess = get_session()
        try:
            rows = sess.scalars(
                select(AccountRecord)
                .where(AccountRecord.user_id == user_id)
                .order_by(AccountRecord.name)
            ).all()
            return [Account.model_validate(r, from_attributes=True) for r in rows]
        finally:
            sess.close()

    accounts = await asyncio.to_thread(_fetch)
    log.info("%s", _evt({"event": "accounts_list_ready", "count": len(accounts)}))
    return accounts


@router.get("/accounts/{account_id}", response_model=Account)
async def get_account(account_id: str):
    """Return a single account by ID."""
    log.info("%s", _evt({"event": "account_lookup", "id": account_id}))

    def _fetch():
        sess = get_session()
        try:
            row = sess.get(AccountRecord, account_id)
            if not row:
                return None
            return Account.model_validate(row, from_attributes=True)
        finally:
            sess.close()

    account = await asyncio.to_thread(_fetch)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    return account


@router.get("/accounts/{account_id}/balance")
async def get_balance(account_id: str):
    """Return current balance for a single account."""
    log.info("%s", _evt({"event": "balance_lookup", "id": account_id}))

    def _fetch():
        sess = get_session()
        try:
            row = sess.get(AccountRecord, account_id)
            return row
        finally:
            sess.close()

    row = await asyncio.to_thread(_fetch)
    if not row:
        raise HTTPException(status_code=404, detail="Account not found")
    return {
        "account_id": account_id,
        "name": row.name,
        "account_type": row.account_type,
        "balance": row.balance,
        "currency": row.currency,
        "last_updated": row.last_updated.isoformat(),
    }
