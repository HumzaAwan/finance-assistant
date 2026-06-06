from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, Float, String, create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

_BASE = Path(__file__).resolve().parent.parent

_default_data = (_BASE / "data").resolve()

DATA_DIR = Path(os.getenv("MOCK_API_DATA_DIR", str(_default_data))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
_db_file = DATA_DIR / "finance.db"

DATABASE_URL = URL.create("sqlite+pysqlite", database=str(_db_file.resolve()))

_engine = None
SessionLocal = None


class Base(DeclarativeBase):
    pass


class TransactionRecord(Base):
    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[str] = mapped_column(String, index=True, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    merchant: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)


class AccountRecord(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    account_type: Mapped[str] = mapped_column(String, nullable=False)
    balance: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    currency: Mapped[str] = mapped_column(String, nullable=False, default="USD")
    last_updated: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class BudgetRecord(Base):
    """Per-user, per-category monthly spending target."""
    __tablename__ = "budgets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    monthly_limit: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


_log = logging.getLogger("mock_api.database.db")


def get_engine():
    global _engine, SessionLocal
    if _engine is None:
        _engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=_engine)
        SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
        _log.info({"event": "db_engine_initialized", "path": str(_db_file)})
    return _engine


def get_session():
    if SessionLocal is None:
        get_engine()
    return SessionLocal()


get_engine()
