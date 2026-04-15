from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from amazon_recsys.application.services import BundleRecommendationService
from amazon_recsys.config.settings import AppSettings, get_settings
from amazon_recsys.infrastructure.artifacts import LocalArtifactStore
from amazon_recsys.monitoring.service import MonitoringService
from amazon_recsys.monitoring.store import LocalMonitoringStore
from amazon_recsys.ml.pipelines import PackageTrainingPipeline
from amazon_recsys.observability.logging import configure_logging


@dataclass(slots=True)
class Container:
    settings: AppSettings
    artifact_store: LocalArtifactStore
    monitoring_store: LocalMonitoringStore
    monitoring_service: MonitoringService
    training_pipeline: PackageTrainingPipeline
    recommendation_service: BundleRecommendationService


def build_container(settings: AppSettings | None = None) -> Container:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    artifact_store = LocalArtifactStore(resolved_settings)
    monitoring_store = LocalMonitoringStore(resolved_settings)
    monitoring_service = MonitoringService(
        settings=resolved_settings,
        artifact_store=artifact_store,
        monitoring_store=monitoring_store,
    )
    training_pipeline = PackageTrainingPipeline(resolved_settings)
    recommendation_service = BundleRecommendationService(
        artifact_store=artifact_store,
        settings=resolved_settings,
        monitoring_service=monitoring_service,
    )
    return Container(
        settings=resolved_settings,
        artifact_store=artifact_store,
        monitoring_store=monitoring_store,
        monitoring_service=monitoring_service,
        training_pipeline=training_pipeline,
        recommendation_service=recommendation_service,
    )


@lru_cache
def get_container() -> Container:
    return build_container()


def reset_container_cache() -> None:
    get_container.cache_clear()
    get_settings.cache_clear()
