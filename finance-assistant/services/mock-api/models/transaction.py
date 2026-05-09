from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "user_001",
                "amount": -42.50,
                "category": "food",
                "description": "Lunch meeting",
                "merchant": "Cafe Verde",
                "timestamp": "2026-03-01T12:34:56",
            }
        },
    )

    id: str
    user_id: str
    amount: float
    category: str
    description: str
    merchant: str
    timestamp: datetime = Field(description="ISO 8601")
