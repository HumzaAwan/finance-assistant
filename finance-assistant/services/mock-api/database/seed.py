from __future__ import annotations

import json
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone

from faker import Faker
from sqlalchemy import func, select

from database.db import DATA_DIR, TransactionRecord, get_session

log = logging.getLogger("mock_api.database.seed")
JSON_PATH = DATA_DIR / "mock_transactions.json"

USER_ID = "user_001"
TARGET_COUNT = 150


def transaction_count(session) -> int:
    return (
        session.scalar(
            select(func.count()).select_from(TransactionRecord).where(TransactionRecord.user_id == USER_ID)
        )
        or 0
    )


def already_seeded() -> bool:
    session = get_session()
    try:
        count = transaction_count(session)
        if count >= TARGET_COUNT:
            log.info({"event": "seed_skip", "reason": "sufficient_transactions", "count": count})
            return True
        return False
    finally:
        session.close()


def make_ts(rng: random.Random, faker: Faker, day_anchor: datetime) -> datetime:
    minute = rng.randint(0, 1439)
    noise = faker.random_int(min=0, max=120)
    return (day_anchor + timedelta(minutes=minute + noise)).replace(second=0, microsecond=0, tzinfo=timezone.utc)


def generate_transactions(now: datetime) -> list[dict]:
    faker = Faker()
    rng = random.Random(42)
    horizon_start = (now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)) - timedelta(days=90)
    rows: list[dict] = []

    paycheck_offsets = {5, 20, 35, 52, 70, 88}
    for offset in paycheck_offsets:
        payday = horizon_start + timedelta(days=offset)
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "user_id": USER_ID,
                "amount": round(3000.0 + rng.uniform(-25.0, 25.0), 2),
                "category": "income",
                "description": rng.choice(["Payroll deposit", "Salary credit"]),
                "merchant": rng.choice(["ACME Payroll", "HR Pay Services"]),
                "timestamp": make_ts(rng, faker, payday).isoformat(),
            }
        )

    while len(rows) < TARGET_COUNT:
        offset = rng.randint(0, 89)
        day_anchor = (horizon_start + timedelta(days=offset)).replace(tzinfo=timezone.utc)

        dow = day_anchor.weekday()
        weekend = dow >= 5
        ts = make_ts(rng, faker, day_anchor)

        if rng.random() < 0.04 and dow in {2, 3}:
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "user_id": USER_ID,
                    "amount": -round(rng.uniform(55.0, 195.0), 2),
                    "category": "utilities",
                    "description": rng.choice(["Electric payment", "Water & sewer", "Gas bill"]),
                    "merchant": rng.choice(["PowerCo", "City Utilities"]),
                    "timestamp": ts.isoformat(),
                }
            )
            continue

        r = rng.uniform(0.0, 1.0)
        if r < 0.36:
            amt = rng.uniform(18.0, 92.0) if weekend else rng.uniform(10.0, 52.0)
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "user_id": USER_ID,
                    "amount": -round(amt, 2),
                    "category": "food",
                    "description": faker.sentence(nb_words=4).rstrip("."),
                    "merchant": rng.choice(["Green Bowl", "Market Fresh", "City Diner"]),
                    "timestamp": ts.isoformat(),
                }
            )
        elif r < 0.53:
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "user_id": USER_ID,
                    "amount": -round(rng.uniform(5.0, 49.8), 2),
                    "category": "transport",
                    "description": rng.choice(["Transit pass", "Rideshare fare", "Fuel top-up"]),
                    "merchant": rng.choice(["Metro Mobility", "QuickFuel"]),
                    "timestamp": ts.isoformat(),
                }
            )
        elif r < 0.61:
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "user_id": USER_ID,
                    "amount": -round(rng.uniform(52.0, 199.8), 2),
                    "category": "utilities",
                    "description": rng.choice(["Utility bundle", "Phone & internet"]),
                    "merchant": rng.choice(["TeleLink", "PowerCo"]),
                    "timestamp": ts.isoformat(),
                }
            )
        elif r < 0.75:
            amt = rng.uniform(22.0, 112.0) if weekend else rng.uniform(11.0, 78.0)
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "user_id": USER_ID,
                    "amount": -round(amt, 2),
                    "category": "entertainment",
                    "description": rng.choice(["Movie night", "Streaming bundle", "Concert seating"]),
                    "merchant": rng.choice(["CineHaus", "StreamVerse"]),
                    "timestamp": ts.isoformat(),
                }
            )
        elif r < 0.87:
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "user_id": USER_ID,
                    "amount": -round(rng.uniform(20.0, 205.0), 2),
                    "category": "health",
                    "description": rng.choice(["Pharmacy refill", "Clinic copay"]),
                    "merchant": rng.choice(["WellNest Clinic", "PharmaQuick"]),
                    "timestamp": ts.isoformat(),
                }
            )
        else:
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "user_id": USER_ID,
                    "amount": -round(rng.uniform(30.0, 297.5), 2),
                    "category": "shopping",
                    "description": rng.choice(["Clothing buys", "Home goods"]),
                    "merchant": rng.choice(["Thread & Co.", "Urban Supply"]),
                    "timestamp": ts.isoformat(),
                }
            )

        if len(rows) > TARGET_COUNT * 50:
            break

    rng.shuffle(rows)
    rows.sort(key=lambda x: x["timestamp"])
    rows = rows[:TARGET_COUNT]

    incomes = sum(1 for r in rows if r["category"] == "income")
    deficit = TARGET_COUNT - len(rows)
    if deficit > 0 or incomes < 6:
        need = deficit if deficit > 0 else 0
        need = max(need, max(0, 6 - incomes))
        horizon_end = horizon_start + timedelta(days=90)
        for _ in range(need):
            anchor = horizon_start + timedelta(days=rng.randint(0, 89))
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "user_id": USER_ID,
                    "amount": round(3000.0 + rng.uniform(-10.0, 10.0), 2),
                    "category": "income",
                    "description": rng.choice(["Payroll reconciliation", "Market adjustment"]),
                    "merchant": "ACME Payroll",
                    "timestamp": make_ts(rng, faker, anchor).isoformat(),
                }
            )

    rows.sort(key=lambda x: x["timestamp"])
    return rows[:TARGET_COUNT]


def run_seed():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    if already_seeded():
        return

    log.info({"event": "seed_generate", "target": TARGET_COUNT})
    now = datetime.now(timezone.utc)
    payloads = generate_transactions(now)
    payloads = payloads[:TARGET_COUNT]

    session = get_session()
    try:
        for p in payloads:
            session.merge(
                TransactionRecord(
                    id=p["id"],
                    user_id=p["user_id"],
                    amount=p["amount"],
                    category=p["category"],
                    description=p["description"],
                    merchant=p["merchant"],
                    timestamp=datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00")),
                )
            )
        session.commit()
    finally:
        session.close()

    export: list[dict] = []
    sess2 = get_session()
    try:
        full_rows = sess2.scalars(
            select(TransactionRecord).where(TransactionRecord.user_id == USER_ID).order_by(TransactionRecord.timestamp)
        ).all()
        export = [
            {
                "id": r.id,
                "user_id": r.user_id,
                "amount": r.amount,
                "category": r.category,
                "description": r.description,
                "merchant": r.merchant,
                "timestamp": r.timestamp.replace(tzinfo=timezone.utc).isoformat(),
            }
            for r in full_rows
        ]
        with JSON_PATH.open("w", encoding="utf-8") as fh:
            json.dump(export, fh, indent=2)
    finally:
        sess2.close()

    log.info({"event": "seed_complete", "written": len(payloads), "export_rows": len(export), "json": str(JSON_PATH)})


if __name__ == "__main__":
    run_seed()
