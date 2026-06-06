from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Account(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "acc_001",
                "user_id": "user_001",
                "name": "Main Checking",
                "account_type": "checking",
                "balance": 3_240.50,
                "currency": "USD",
                "last_updated": "2026-06-01T10:00:00+00:00",
            }
        },
    )

    id: str
    user_id: str
    name: str
    account_type: str = Field(description="checking | savings | credit")
    balance: float
    currency: str = "USD"
    last_updated: datetime = Field(description="ISO 8601")


class BudgetCategory(BaseModel):
    """Single-category monthly budget target."""
    category: str
    monthly_limit: float


class UserBudget(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: str
    categories: list[BudgetCategory]
    updated_at: datetime
