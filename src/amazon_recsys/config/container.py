from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from amazon_recsys.application.services import BundleRecommendationService
from amazon_recsys.config.settings import AppSettings, get_settings
from amazon_recsys.infrastructure.artifacts import LocalArtifactStore
from amazon_recsys.ml.pipelines import PackageTrainingPipeline
from amazon_recsys.observability.logging import configure_logging


@dataclass(slots=True)
class Container:
    settings: AppSettings
    artifact_store: LocalArtifactStore
    training_pipeline: PackageTrainingPipeline
    recommendation_service: BundleRecommendationService


def build_container(settings: AppSettings | None = None) -> Container:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    artifact_store = LocalArtifactStore(resolved_settings)
    training_pipeline = PackageTrainingPipeline(resolved_settings)
    recommendation_service = BundleRecommendationService(
        artifact_store=artifact_store,
        settings=resolved_settings,
    )
    return Container(
        settings=resolved_settings,
        artifact_store=artifact_store,
        training_pipeline=training_pipeline,
        recommendation_service=recommendation_service,
    )


@lru_cache
def get_container() -> Container:
    return build_container()


def reset_container_cache() -> None:
    get_container.cache_clear()
    get_settings.cache_clear()
