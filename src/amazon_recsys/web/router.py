from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from amazon_recsys.api.dependencies import get_recommendation_service
from amazon_recsys.application.services import BundleRecommendationService


TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
router = APIRouter(include_in_schema=False)


def _split_history_items(raw_value: str | None) -> list[str] | None:
    if not raw_value:
        return None
    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    return items or None


@router.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    user_id: str | None = None,
    history_items: str | None = None,
    top_k: int = 7,
    service: BundleRecommendationService = Depends(get_recommendation_service),
) -> HTMLResponse:
    parsed_history = _split_history_items(history_items)
    recommendations = []
    history = []
    error = None
    try:
        active_model = service.get_active_model()
        evaluation_summary = service.get_evaluation_summary()
        if user_id or parsed_history:
            recommendations = service.recommend(user_id=user_id, history_items=parsed_history, top_k=top_k)
            if user_id:
                history = service.get_user_history(user_id)
    except Exception as exc:
        active_model = service.readiness()
        evaluation_summary = {}
        error = str(exc)
    return TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "active_model": active_model,
            "evaluation_summary": evaluation_summary,
            "recommendations": recommendations,
            "history": history,
            "user_id": user_id or "",
            "history_items": history_items or "",
            "top_k": top_k,
            "error": error,
        },
    )
