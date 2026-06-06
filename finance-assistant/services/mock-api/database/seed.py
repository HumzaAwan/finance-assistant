from __future__ import annotations

import hashlib
import json
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone

from faker import Faker
from sqlalchemy import func, select

from database.db import AccountRecord, BudgetRecord, DATA_DIR, TransactionRecord, get_session

log = logging.getLogger("mock_api.database.seed")
JSON_PATH = DATA_DIR / "mock_transactions.json"

# Two synthetic users for multi-user demo scenarios.
USERS = {
    "user_001": {"seed": 42, "income": 3000.0, "target": 300, "horizon_days": 180},
    "user_002": {"seed": 99, "income": 5500.0, "target": 300, "horizon_days": 180},
}

MERCHANTS: dict[str, list[str]] = {
    "food": [
        "Green Bowl", "Market Fresh", "City Diner", "Sushi Spot", "Taco Bell",
        "Morning Brew", "Pizza Palace", "Corner Bakery", "Thai Garden", "Burger Barn",
    ],
    "transport": [
        "Metro Mobility", "QuickFuel", "Lyft", "Uber", "CityBike",
        "Parking Authority", "Rail & Ride", "ExpressBus", "TollRoad Inc",
    ],
    "utilities": [
        "PowerCo", "City Utilities", "TeleLink", "AquaCity", "GasMart",
        "BrightFiber Internet", "HomeShield Insurance",
    ],
    "entertainment": [
        "CineHaus", "StreamVerse", "GameZone", "BookNook", "Eventbrite",
        "SportsArena", "MusicHive", "ArcadeWorld", "Netflix", "Spotify",
    ],
    "health": [
        "WellNest Clinic", "PharmaQuick", "FitLife Gym", "MindEase Therapy",
        "DentalCare Plus", "VisionCenter", "LabCorp",
    ],
    "shopping": [
        "Thread & Co.", "Urban Supply", "HomeGoods Depot", "ElectroMart",
        "BookBazaar", "PetEmporium", "GardenWorld", "Thrift Haven",
    ],
    "income": ["ACME Payroll", "HR Pay Services", "Freelance Platform"],
}

CATEGORY_WEIGHTS = {
    "food": 0.36,
    "transport": 0.17,
    "utilities": 0.08,
    "entertainment": 0.14,
    "health": 0.12,
    "shopping": 0.13,
}

CATEGORY_AMOUNT_RANGES = {
    "food": (10.0, 92.0),
    "transport": (5.0, 55.0),
    "utilities": (52.0, 200.0),
    "entertainment": (11.0, 130.0),
    "health": (20.0, 210.0),
    "shopping": (30.0, 300.0),
}

DEFAULT_BUDGETS = {
    "food": 500.0,
    "transport": 250.0,
    "utilities": 300.0,
    "entertainment": 200.0,
    "health": 150.0,
    "shopping": 200.0,
}


def _anomaly_id(name: str) -> str:
    """Deterministic UUID derived from a fixed name — safe to merge on re-run."""
    return str(uuid.UUID(hashlib.md5(f"anomaly-user_001-{name}".encode()).hexdigest()))


# Fixed IDs for the four synthetic anomalies so session.merge() is idempotent.
_ANOMALY_IDS = {
    "duplicate_a": _anomaly_id("duplicate_charge_a"),
    "duplicate_b": _anomaly_id("duplicate_charge_b"),
    "amount_outlier": _anomaly_id("amount_zscore_outlier"),
    "unusual_time": _anomaly_id("unusual_time_0317"),
    "high_risk": _anomaly_id("high_risk_gambling"),
}


def seed_anomalies(session, now: datetime) -> None:
    """Inject four deterministic anomaly transactions for user_001.

    Called unconditionally in run_seed() so anomalies are always present even
    when the main dataset was seeded in a previous run. Uses session.merge() so
    re-runs are safe (no duplicate rows).

    Anomalies injected:
        1. Duplicate charge — Netflix £14.99 twice within 4 hours (last 14 days)
        2. Amount z-score outlier — Green Bowl £295.00 (~6× the usual food spend)
        3. Unusual time — City Diner £42.50 at 03:17 UTC
        4. High-risk merchant — BetKing Casino £50.00 (first-time gambling)
    """
    # Check if anomalies already exist; skip if all four are present.
    existing_ids = {
        row.id
        for row in session.scalars(
            select(TransactionRecord).where(
                TransactionRecord.id.in_(list(_ANOMALY_IDS.values()))
            )
        ).all()
    }
    if len(existing_ids) == len(_ANOMALY_IDS):
        log.info({"event": "anomalies_already_seeded"})
        return

    # Anchor all anomaly timestamps relative to *now* so they remain within
    # the anomaly detection window (last 90 days) however long ago the main
    # seed ran.
    recent = now - timedelta(days=7)   # 1 week ago
    duplicate_base = now - timedelta(days=3)
    unusual_day = now - timedelta(days=10)
    high_risk_day = now - timedelta(days=5)

    anomalies = [
        # 1 — Duplicate: Netflix £14.99 twice, 4 hours apart
        TransactionRecord(
            id=_ANOMALY_IDS["duplicate_a"],
            user_id="user_001",
            amount=-14.99,
            category="entertainment",
            description="Monthly subscription",
            merchant="Netflix",
            timestamp=duplicate_base.replace(hour=14, minute=0, second=0, microsecond=0),
        ),
        TransactionRecord(
            id=_ANOMALY_IDS["duplicate_b"],
            user_id="user_001",
            amount=-14.99,
            category="entertainment",
            description="Monthly subscription",
            merchant="Netflix",
            timestamp=duplicate_base.replace(hour=18, minute=0, second=0, microsecond=0),
        ),
        # 2 — Amount outlier: Green Bowl at £295.00 (~6× typical food spend of ~£50)
        TransactionRecord(
            id=_ANOMALY_IDS["amount_outlier"],
            user_id="user_001",
            amount=-295.00,
            category="food",
            description="Large catering order",
            merchant="Green Bowl",
            timestamp=recent.replace(hour=13, minute=0, second=0, microsecond=0),
        ),
        # 3 — Unusual time: City Diner at 03:17 UTC
        TransactionRecord(
            id=_ANOMALY_IDS["unusual_time"],
            user_id="user_001",
            amount=-42.50,
            category="food",
            description="Late night purchase",
            merchant="City Diner",
            timestamp=unusual_day.replace(hour=3, minute=17, second=0, microsecond=0),
        ),
        # 4 — High-risk merchant: first-time gambling transaction
        TransactionRecord(
            id=_ANOMALY_IDS["high_risk"],
            user_id="user_001",
            amount=-50.00,
            category="entertainment",
            description="gambling deposit",
            merchant="BetKing Casino",
            timestamp=high_risk_day.replace(hour=20, minute=0, second=0, microsecond=0),
        ),
    ]

    for record in anomalies:
        session.merge(record)

    log.info({
        "event": "anomalies_seeded",
        "user_id": "user_001",
        "count": len(anomalies),
        "ids": list(_ANOMALY_IDS.values()),
    })


def transaction_count(session, user_id: str) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(TransactionRecord)
            .where(TransactionRecord.user_id == user_id)
        )
        or 0
    )


def already_seeded() -> bool:
    session = get_session()
    try:
        for user_id, cfg in USERS.items():
            count = transaction_count(session, user_id)
            if count < cfg["target"]:
                log.info({"event": "seed_needed", "user_id": user_id, "count": count})
                return False
        log.info({"event": "seed_skip", "reason": "all_users_sufficient"})
        return True
    finally:
        session.close()


def make_ts(rng: random.Random, faker: Faker, day_anchor: datetime) -> datetime:
    minute = rng.randint(0, 1439)
    noise = faker.random_int(min=0, max=120)
    return (day_anchor + timedelta(minutes=minute + noise)).replace(
        second=0, microsecond=0, tzinfo=timezone.utc
    )


def generate_transactions(user_id: str, cfg: dict, now: datetime) -> list[dict]:
    faker = Faker()
    rng = random.Random(cfg["seed"])
    horizon_start = now.astimezone(timezone.utc) - timedelta(days=cfg["horizon_days"])
    target = cfg["target"]
    income = cfg["income"]
    rows: list[dict] = []

    # Seed paychecks (bi-monthly)
    paycheck_offsets = list(range(5, cfg["horizon_days"], 15))[:12]
    for offset in paycheck_offsets:
        payday = horizon_start + timedelta(days=offset)
        rows.append({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "amount": round(income + rng.uniform(-30.0, 30.0), 2),
            "category": "income",
            "description": rng.choice(["Payroll deposit", "Salary credit"]),
            "merchant": rng.choice(MERCHANTS["income"]),
            "timestamp": make_ts(rng, faker, payday).isoformat(),
        })

    # Seed recurring subscriptions (monthly)
    for sub_offset in range(0, cfg["horizon_days"], 30):
        anchor = horizon_start + timedelta(days=sub_offset + 2)
        rows.append({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "amount": -rng.choice([9.99, 13.99, 15.49, 17.99]),
            "category": "entertainment",
            "description": "Monthly subscription",
            "merchant": rng.choice(["Netflix", "Spotify", "StreamVerse"]),
            "timestamp": make_ts(rng, faker, anchor).isoformat(),
        })

    # Seed expense transactions up to target
    attempts = 0
    while len(rows) < target and attempts < target * 100:
        attempts += 1
        offset = rng.randint(0, cfg["horizon_days"] - 1)
        day_anchor = (horizon_start + timedelta(days=offset)).replace(tzinfo=timezone.utc)
        ts = make_ts(rng, faker, day_anchor)

        # Choose category by weighted random
        r = rng.random()
        cumulative = 0.0
        chosen = "food"
        for cat, weight in CATEGORY_WEIGHT_LIST:
            cumulative += weight
            if r < cumulative:
                chosen = cat
                break

        lo, hi = CATEGORY_AMOUNT_RANGES[chosen]
        # Weekend uplift on discretionary categories
        weekend = day_anchor.weekday() >= 5
        if weekend and chosen in {"food", "entertainment"}:
            hi = min(hi * 1.4, hi + 60)

        rows.append({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "amount": -round(rng.uniform(lo, hi), 2),
            "category": chosen,
            "description": faker.sentence(nb_words=4).rstrip("."),
            "merchant": rng.choice(MERCHANTS[chosen]),
            "timestamp": ts.isoformat(),
        })

    rows.sort(key=lambda x: x["timestamp"])
    return rows[:target]


# Pre-compute weighted category list for efficient sampling
CATEGORY_WEIGHT_LIST = list(CATEGORY_WEIGHTS.items())


def seed_accounts(session, user_id: str, user_seed: int, now: datetime) -> None:
    """Seed two accounts (checking + savings) for a user if not present."""
    rng = random.Random(user_seed + 1000)
    existing = session.scalar(
        select(func.count())
        .select_from(AccountRecord)
        .where(AccountRecord.user_id == user_id)
    )
    if existing:
        return

    accounts = [
        AccountRecord(
            id=f"{user_id}_checking",
            user_id=user_id,
            name="Main Checking",
            account_type="checking",
            balance=round(rng.uniform(1_500, 5_000), 2),
            currency="USD",
            last_updated=now,
        ),
        AccountRecord(
            id=f"{user_id}_savings",
            user_id=user_id,
            name="Emergency Savings",
            account_type="savings",
            balance=round(rng.uniform(3_000, 12_000), 2),
            currency="USD",
            last_updated=now,
        ),
    ]
    for acc in accounts:
        session.add(acc)
    log.info({"event": "accounts_seeded", "user_id": user_id})


def seed_budgets(session, user_id: str, now: datetime) -> None:
    """Seed default monthly budget targets for a user if not present."""
    existing = session.scalar(
        select(func.count())
        .select_from(BudgetRecord)
        .where(BudgetRecord.user_id == user_id)
    )
    if existing:
        return

    for category, limit in DEFAULT_BUDGETS.items():
        session.add(BudgetRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            category=category,
            monthly_limit=limit,
            updated_at=now,
        ))
    log.info({"event": "budgets_seeded", "user_id": user_id})


def run_seed():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    now = datetime.now(timezone.utc)

    # Always inject anomalies regardless of whether the main seed has run,
    # so they remain within the detection window even on an already-seeded DB.
    anomaly_session = get_session()
    try:
        seed_anomalies(anomaly_session, now)
        anomaly_session.commit()
    finally:
        anomaly_session.close()

    if already_seeded():
        return
    session = get_session()
    all_payloads: list[dict] = []

    try:
        for user_id, cfg in USERS.items():
            count = transaction_count(session, user_id)
            if count >= cfg["target"]:
                continue

            log.info({"event": "seed_generate", "user_id": user_id, "target": cfg["target"]})
            payloads = generate_transactions(user_id, cfg, now)

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
            all_payloads.extend(payloads)

            seed_accounts(session, user_id, cfg["seed"], now)
            seed_budgets(session, user_id, now)

        session.commit()
    finally:
        session.close()

    # Export JSON snapshot
    sess2 = get_session()
    try:
        export: list[dict] = []
        for user_id in USERS:
            full_rows = sess2.scalars(
                select(TransactionRecord)
                .where(TransactionRecord.user_id == user_id)
                .order_by(TransactionRecord.timestamp)
            ).all()
            export.extend({
                "id": r.id,
                "user_id": r.user_id,
                "amount": r.amount,
                "category": r.category,
                "description": r.description,
                "merchant": r.merchant,
                "timestamp": r.timestamp.replace(tzinfo=timezone.utc).isoformat(),
            } for r in full_rows)

        with JSON_PATH.open("w", encoding="utf-8") as fh:
            json.dump(export, fh, indent=2)
    finally:
        sess2.close()

    log.info({"event": "seed_complete", "total_rows": len(all_payloads), "users": list(USERS.keys())})


if __name__ == "__main__":
    run_seed()
