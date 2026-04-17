from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException

from amazon_recsys.api.schemas import ConfigResponse, EvaluationSummaryResponse, ModelSummaryResponse
from amazon_recsys.application.services import BundleRecommendationService
from amazon_recsys.config.settings import AppSettings
from amazon_recsys.presentation.dependencies import get_recommendation_service, get_settings


router = APIRouter(tags=["models"])


@router.get("/config", response_model=ConfigResponse)
def config(settings: AppSettings = Depends(get_settings)) -> ConfigResponse:
    return ConfigResponse(config=settings.safe_config())


@router.get("/models/active", response_model=ModelSummaryResponse)
def active_model(
    service: BundleRecommendationService = Depends(get_recommendation_service),
) -> ModelSummaryResponse:
    try:
        return ModelSummaryResponse(**asdict(service.get_active_model()))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/evaluate/summary", response_model=EvaluationSummaryResponse)
def evaluate_summary(
    service: BundleRecommendationService = Depends(get_recommendation_service),
) -> EvaluationSummaryResponse:
    try:
        model = service.get_active_model()
        return EvaluationSummaryResponse(
            source=model.source,
            summary=service.get_evaluation_summary().to_dict(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
