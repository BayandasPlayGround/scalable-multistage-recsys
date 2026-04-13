from __future__ import annotations

import os
import signal
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from amazon_recsys.api.dependencies import get_recommendation_service
from amazon_recsys.application.services import BundleRecommendationService


TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
router = APIRouter(include_in_schema=False)
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _split_history_items(raw_value: str | None) -> list[str] | None:
    if not raw_value:
        return None
    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    return items or None


def _terminate_local_process() -> None:
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        os._exit(0)


def _is_loopback_request(request: Request) -> bool:
    hostname = (request.url.hostname or "").strip().lower()
    return hostname in LOOPBACK_HOSTS


def _can_shutdown_from_request(request: Request, environment: str) -> bool:
    return environment == "local" or _is_loopback_request(request)


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
    active_model = service.readiness()
    evaluation_summary = {}
    available_users = []
    try:
        active_model = service.get_active_model()
        evaluation_summary = service.get_evaluation_summary()
        available_users = service.list_available_users(limit=250, min_history=3)
    except Exception as exc:
        error = str(exc)

    if user_id or parsed_history:
        try:
            recommendations = service.recommend(user_id=user_id, history_items=parsed_history, top_k=top_k)
            if user_id:
                history = service.get_user_history(user_id)
        except Exception as exc:
            error = str(exc)
    return TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "active_model": active_model,
            "evaluation_summary": evaluation_summary,
            "recommendations": recommendations,
            "history": history,
            "available_users": available_users,
            "is_local_environment": service.settings.environment == "local",
            "can_shutdown_local_server": _can_shutdown_from_request(request, service.settings.environment),
            "environment_name": service.settings.environment,
            "user_id": user_id or "",
            "history_items": history_items or "",
            "top_k": top_k,
            "error": error,
        },
    )


@router.get("/favicon.ico")
def favicon() -> RedirectResponse:
    return RedirectResponse(url="/static/favicon.svg", status_code=307)


@router.post("/local/shutdown")
def local_shutdown(request: Request) -> JSONResponse:
    settings = request.app.state.container.settings
    if not _can_shutdown_from_request(request, settings.environment):
        return JSONResponse(status_code=403, content={"status": "forbidden", "detail": "Local shutdown is available only from localhost or the local environment."})

    timer = threading.Timer(0.35, _terminate_local_process)
    timer.daemon = True
    timer.start()
    return JSONResponse({"status": "shutting_down", "detail": "Local server shutdown requested."})
