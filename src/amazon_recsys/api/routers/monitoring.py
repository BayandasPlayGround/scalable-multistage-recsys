from fastapi import APIRouter, Depends, HTTPException

from amazon_recsys.api.dependencies import get_monitoring_service
from amazon_recsys.api.schemas import MonitoringSummaryResponse
from amazon_recsys.monitoring.service import MonitoringService


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
