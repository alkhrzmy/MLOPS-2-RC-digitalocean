from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    product: str
    quantity: str
    confidence: float | None = None
