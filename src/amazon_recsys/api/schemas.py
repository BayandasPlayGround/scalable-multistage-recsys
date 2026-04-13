from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    status: str
    source: str
    version: str | None = None


class RecommendationRequest(BaseModel):
    user_id: str | None = None
    history_items: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def validate_inputs(self) -> "RecommendationRequest":
        if self.user_id is None and not self.history_items:
            raise ValueError("Provide either user_id or history_items.")
        return self


class RecommendationItemResponse(BaseModel):
    item_id: str
    title: str
    source_category: str
    price: float | None
    average_rating: float | None
    retrieval_score: float | None = None
    score: float | None = None
    candidate_sources: str | None = None


class RecommendationResponse(BaseModel):
    top_k: int
    source: str
    active_bundle_version: str | None = None
    items: list[RecommendationItemResponse]


class HistoryItemResponse(BaseModel):
    ordered_at: str
    item_id: str
    title: str
    source_category: str
    review_rating: float | None
    verified_purchase: int | None
    price: float | None
    average_rating: float | None


class HistoryResponse(BaseModel):
    user_id: str
    items: list[HistoryItemResponse]


class AvailableUserResponse(BaseModel):
    user_id: str
    interaction_count: int
    history_length: int
    last_ordered_at: str | None = None


class AvailableUsersResponse(BaseModel):
    total: int
    items: list[AvailableUserResponse]


class ConfigResponse(BaseModel):
    config: dict[str, Any]


class ModelSummaryResponse(BaseModel):
    ready: bool
    source: str
    version: str
    run_name: str
    run_profile: str
    model_backend: str
    retriever_variants: list[str]
    created_at: str


class EvaluationSummaryResponse(BaseModel):
    source: str
    summary: dict[str, Any]
