from fastapi import APIRouter, Depends, HTTPException

from amazon_recsys.api.schemas import (
    CandidateDiagnosticsHistoryResponse,
    CandidateDiagnosticsResponse,
    MonitoringHistoryResponse,
    MonitoringSummaryResponse,
)
from amazon_recsys.monitoring.service import MonitoringService
from amazon_recsys.presentation.dependencies import get_monitoring_service


router = APIRouter(tags=["monitoring"])


@router.get("/monitoring/drift/summary", response_model=MonitoringSummaryResponse)
def drift_summary(
    service: MonitoringService = Depends(get_monitoring_service),
) -> MonitoringSummaryResponse:
    try:
        summary = service.latest_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if summary is None:
        return MonitoringSummaryResponse(available=False, summary={})
    return MonitoringSummaryResponse(
        available=True,
        bundle_version=summary.bundle_version,
        status=summary.status,
        summary=summary.to_dict(),
    )


@router.get("/monitoring/drift/history", response_model=MonitoringHistoryResponse)
def drift_history(
    limit: int = 8,
    service: MonitoringService = Depends(get_monitoring_service),
) -> MonitoringHistoryResponse:
    try:
        items = service.recent_summaries(limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    payload = [item.to_dict() for item in items]
    return MonitoringHistoryResponse(total=len(payload), items=payload)


@router.get("/monitoring/candidate-recall/summary", response_model=CandidateDiagnosticsResponse)
def candidate_recall_summary(
    service: MonitoringService = Depends(get_monitoring_service),
) -> CandidateDiagnosticsResponse:
    try:
        summary = service.latest_candidate_diagnostics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if summary is None:
        return CandidateDiagnosticsResponse(available=False, summary={})
    return CandidateDiagnosticsResponse(
        available=True,
        bundle_version=str(summary.get("bundle_version")) if summary.get("bundle_version") is not None else None,
        summary=summary,
    )


@router.get("/monitoring/candidate-recall/history", response_model=CandidateDiagnosticsHistoryResponse)
def candidate_recall_history(
    limit: int = 8,
    service: MonitoringService = Depends(get_monitoring_service),
) -> CandidateDiagnosticsHistoryResponse:
    try:
        items = service.recent_candidate_diagnostics(limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return CandidateDiagnosticsHistoryResponse(total=len(items), items=items)
