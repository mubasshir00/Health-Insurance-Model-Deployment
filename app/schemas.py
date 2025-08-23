from pydantic import BaseModel, Field, HttpUrl
from typing import Any, Dict, Optional


class PredictRequest(BaseModel):
    id: str = Field(..., description="Correlation id from caller")
    inputs: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    id: str
    result: Dict[str, Any]
    latency_ms: int


class PredictAsyncRequest(PredictRequest):
    callback_url: HttpUrl


class Ack(BaseModel):
    status: str
    id: str