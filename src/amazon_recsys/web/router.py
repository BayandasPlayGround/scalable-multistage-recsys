from __future__ import annotations

import os
import re
import signal
import threading
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from amazon_recsys.application.services import BundleRecommendationService
from amazon_recsys.domain.entities import EvaluationSummary
from amazon_recsys.presentation.dependencies import get_recommendation_service


TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
router = APIRouter(include_in_schema=False)
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _split_history_items(raw_value: str | None) -> list[str] | None:
    if not raw_value:
        return None
    items = [item.strip() for item in re.split(r"[\s,+]+", raw_value) if item.strip()]
    return items or None


def _terminate_local_process() -> None:
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except OSError:
        os._exit(0)


def _is_loopback_request(request: Request) -> bool:
    hostname = (request.url.hostname or "").strip().lower()
    return hostname in LOOPBACK_HOSTS


def _can_shutdown_from_request(request: Request, environment: str) -> bool:
    return environment == "local" or _is_loopback_request(request)


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _format_window_label(value: str | None) -> str:
    timestamp = _parse_iso_timestamp(value)
    if timestamp is None:
        return "n/a"
    return timestamp.strftime("%d %b")


def _sparkline_points(series: list[float | int | None], *, width: int = 152, height: int = 44, padding: int = 4) -> str:
    numeric = [float(value) if value is not None else None for value in series]
    finite = [value for value in numeric if value is not None]
    if not finite:
        return ""
    if len(numeric) == 1:
        x_start = float(padding)
        x_end = float(width - padding)
        y_value = height / 2
        return f"{x_start:.1f},{y_value:.1f} {x_end:.1f},{y_value:.1f}"
    minimum = min(finite)
    maximum = max(finite)
    spread = maximum - minimum if maximum != minimum else 1.0
    points: list[str] = []
    usable_width = width - (padding * 2)
    usable_height = height - (padding * 2)
    for index, raw_value in enumerate(numeric):
        if raw_value is None:
            continue
        x_value = padding + (usable_width * index / max(len(numeric) - 1, 1))
        y_ratio = (raw_value - minimum) / spread if spread else 0.5
        y_value = height - padding - (usable_height * y_ratio)
        points.append(f"{x_value:.1f},{y_value:.1f}")
    return " ".join(points)


def _build_monitoring_trends(summaries: list[dict]) -> list[dict]:
    if not summaries:
        return []
    latest = summaries[-1]

    def _metric_value(summary: dict, key: str) -> float | None:
        concept = summary.get("concept_drift") or {}
        metrics = concept.get("metrics") or {}
        value = metrics.get(key)
        return float(value) if value is not None else None

    drift_peak_values: list[float] = []
    for summary in summaries:
        feature_drifts = summary.get("feature_drifts") or []
        metric_values = [float(item.get("metric_value", 0.0)) for item in feature_drifts if item.get("metric_value") is not None]
        drift_peak_values.append(max(metric_values) if metric_values else 0.0)

    concept = latest.get("concept_drift") or {}
    monitored_k = int(concept.get("monitored_k", 10))
    metric_suffix = str(monitored_k)
    trend_specs = [
        {
            "title": "Performance Drop",
            "series": [float((summary.get("concept_drift") or {}).get("performance_drop", 0.0) or 0.0) for summary in summaries],
            "latest": float(concept.get("performance_drop", 0.0) or 0.0),
            "format": "percent",
            "tone": "alert",
        },
        {
            "title": f"Hit Rate @{metric_suffix}",
            "series": [_metric_value(summary, f"hit_rate_at_{metric_suffix}") for summary in summaries],
            "latest": _metric_value(latest, f"hit_rate_at_{metric_suffix}"),
            "format": "score",
            "tone": "neutral",
        },
        {
            "title": "Inference Requests",
            "series": [int(summary.get("inference_count", 0)) for summary in summaries],
            "latest": int(latest.get("inference_count", 0)),
            "format": "count",
            "tone": "neutral",
        },
        {
            "title": "Peak Drift Metric",
            "series": drift_peak_values,
            "latest": drift_peak_values[-1] if drift_peak_values else 0.0,
            "format": "score",
            "tone": "alert",
        },
    ]

    labels = [_format_window_label(summary.get("window_end")) for summary in summaries]
    cards: list[dict] = []
    for spec in trend_specs:
        series = spec["series"]
        if not any(value is not None for value in series):
            continue
        latest_value = spec["latest"]
        previous_value = next((value for value in reversed(series[:-1]) if value is not None), None) if len(series) > 1 else None
        delta = None
        if latest_value is not None and previous_value is not None:
            delta = float(latest_value) - float(previous_value)
        cards.append(
            {
                "title": spec["title"],
                "labels": labels,
                "series": series,
                "latest": latest_value,
                "delta": delta,
                "format": spec["format"],
                "tone": spec["tone"],
                "points": _sparkline_points(series),
            }
        )
    return cards


@router.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    user_id: str | None = None,
    selected_user_id: str | None = None,
    history_items: str | None = None,
    top_k: int = 7,
    service: BundleRecommendationService = Depends(get_recommendation_service),
) -> HTMLResponse:
    parsed_history = _split_history_items(history_items)
    effective_selected_user_id = selected_user_id or user_id
    recommendations = []
    history = []
    selected_user_profile = None
    selected_user_history = []
    selected_user_recommendations = []
    error = None
    active_model = service.readiness()
    evaluation_summary = EvaluationSummary()
    monitoring_summary = {}
    monitoring_history = []
    monitoring_trends = []
    available_users = []
    try:
        active_model = service.get_active_model()
        evaluation_summary = service.get_evaluation_summary()
        monitoring_service = request.app.state.container.monitoring_service
        monitoring = monitoring_service.latest_summary()
        monitoring_summary = monitoring.to_dict() if monitoring is not None else {}
        monitoring_history = [item.to_dict() for item in monitoring_service.recent_summaries(limit=8)]
        monitoring_trends = _build_monitoring_trends(monitoring_history)
        available_users = service.list_available_users(limit=250, min_history=3)
    except FileNotFoundError as exc:
        error = str(exc)

    if effective_selected_user_id:
        try:
            user_payload = service.get_user_profile(
                effective_selected_user_id,
                history_limit=20,
                recommendation_limit=min(top_k, 6),
            )
            selected_user_profile = user_payload.profile
            selected_user_history = user_payload.history
            selected_user_recommendations = user_payload.recommendations
        except (FileNotFoundError, KeyError) as exc:
            error = str(exc)

    if user_id or parsed_history:
        try:
            recommendations = service.recommend(user_id=user_id, history_items=parsed_history, top_k=top_k)
            if user_id:
                history = service.get_user_history(user_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            error = str(exc)
    return TEMPLATES.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "active_model": active_model,
            "evaluation_summary": evaluation_summary,
            "monitoring_summary": monitoring_summary,
            "monitoring_history": monitoring_history,
            "monitoring_previous_summary": monitoring_history[-2] if len(monitoring_history) > 1 else {},
            "monitoring_trends": monitoring_trends,
            "recommendations": recommendations,
            "history": history,
            "available_users": available_users,
            "selected_user_id": effective_selected_user_id or "",
            "selected_user_profile": selected_user_profile,
            "selected_user_history": selected_user_history,
            "selected_user_recommendations": selected_user_recommendations,
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
