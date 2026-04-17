from dataclasses import asdict

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from amazon_recsys.api.schemas import HealthResponse, ReadyResponse
from amazon_recsys.application.services import BundleRecommendationService
from amazon_recsys.config.settings import AppSettings
from amazon_recsys.presentation.dependencies import get_recommendation_service, get_settings


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: AppSettings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        app_name=settings.app_name,
        version=settings.app_version,
    )


@router.get("/ready", response_model=ReadyResponse)
def ready(
    service: BundleRecommendationService = Depends(get_recommendation_service),
) -> JSONResponse:
    payload = ReadyResponse(**asdict(service.readiness()))
    status_code = 200 if payload.ready else 503
    return JSONResponse(status_code=status_code, content=payload.model_dump())
