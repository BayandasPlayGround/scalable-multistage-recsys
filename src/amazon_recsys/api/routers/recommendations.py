from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException

from amazon_recsys.api.dependencies import get_recommendation_service
from amazon_recsys.api.schemas import (
    AvailableUserResponse,
    AvailableUsersResponse,
    HistoryItemResponse,
    HistoryResponse,
    RecommendationItemResponse,
    RecommendationRequest,
    RecommendationResponse,
    UserProfileResponse,
    UserProfileSummaryResponse,
)
from amazon_recsys.application.services import BundleRecommendationService


router = APIRouter(tags=["recommendations"])


@router.get("/users", response_model=AvailableUsersResponse)
def available_users(
    limit: int = 100,
    min_history: int = 1,
    query: str | None = None,
    service: BundleRecommendationService = Depends(get_recommendation_service),
) -> AvailableUsersResponse:
    try:
        items = service.list_available_users(limit=limit, min_history=min_history, query=query)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AvailableUsersResponse(
        total=len(items),
        items=[AvailableUserResponse(**asdict(item)) for item in items],
    )


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(
    request: RecommendationRequest,
    service: BundleRecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    try:
        items = service.recommend(
            user_id=request.user_id,
            history_items=request.history_items,
            top_k=request.top_k,
        )
        model = service.get_active_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RecommendationResponse(
        top_k=request.top_k or len(items),
        source=model["source"],
        active_bundle_version=model["version"],
        items=[RecommendationItemResponse(**asdict(item)) for item in items],
    )


@router.get("/users/{user_id}/history", response_model=HistoryResponse)
def user_history(
    user_id: str,
    service: BundleRecommendationService = Depends(get_recommendation_service),
) -> HistoryResponse:
    try:
        items = service.get_user_history(user_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return HistoryResponse(
        user_id=user_id,
        items=[HistoryItemResponse(**asdict(item)) for item in items],
    )


@router.get("/users/{user_id}/profile", response_model=UserProfileResponse)
def user_profile(
    user_id: str,
    history_limit: int = 20,
    recommendation_limit: int = 5,
    service: BundleRecommendationService = Depends(get_recommendation_service),
) -> UserProfileResponse:
    try:
        payload = service.get_user_profile(
            user_id,
            history_limit=history_limit,
            recommendation_limit=recommendation_limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return UserProfileResponse(
        profile=UserProfileSummaryResponse(**asdict(payload["profile"])),
        history=[HistoryItemResponse(**asdict(item)) for item in payload["history"]],
        recommendations=[RecommendationItemResponse(**asdict(item)) for item in payload["recommendations"]],
    )
