from fastapi import APIRouter, Depends, HTTPException

from amazon_recsys.api.schemas import MonitoringHistoryResponse, MonitoringSummaryResponse
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
