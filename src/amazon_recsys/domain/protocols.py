from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from amazon_recsys.domain.entities import (
    ActiveBundlePointer,
    ActiveModelSummary,
    AvailableUser,
    BundleManifest,
    EvaluationSummary,
    HistoryItem,
    MonitoringSummary,
    OutcomeIngestResult,
    RecommendationItem,
    ReadyState,
    ReferenceProfile,
    RuntimeBundle,
    UserProfilePayload,
)

if TYPE_CHECKING:
    from amazon_recsys.ml.pipelines import TrainingSession


class ArtifactStore(Protocol):
    def save_bundle(self, session: TrainingSession, version: str | None = None) -> BundleManifest: ...

    def write_manifest(self, manifest: BundleManifest) -> None: ...

    def activate_bundle(self, version: str) -> ActiveBundlePointer: ...

    def read_active_manifest(self) -> BundleManifest | None: ...

    def load_bundle(self, manifest: BundleManifest) -> RuntimeBundle: ...

    def load_active_bundle(self) -> RuntimeBundle | None: ...


class MonitoringStore(Protocol):
    def save_reference_profile(self, profile: ReferenceProfile) -> Path: ...

    def load_reference_profile(self, bundle_version: str) -> ReferenceProfile: ...

    def append_inference_records(self, records: list[object]) -> int: ...

    def append_outcome_records(self, records: list[object]) -> int: ...

    def ingest_outcomes(self, source: Path) -> OutcomeIngestResult: ...

    def load_inference_frame(
        self,
        *,
        bundle_version: str | None = None,
        window_start: str | None = None,
        window_end: str | None = None,
    ): ...

    def load_outcome_frame(
        self,
        *,
        window_start: str | None = None,
        window_end: str | None = None,
    ): ...

    def save_monitoring_summary(
        self,
        summary: MonitoringSummary,
        feature_frame,
        concept_frame,
        reference_profile: ReferenceProfile,
    ) -> dict[str, Path]: ...

    def load_latest_summary(self, bundle_version: str) -> MonitoringSummary | None: ...

    def list_summaries(self, bundle_version: str) -> list[MonitoringSummary]: ...


class RecommendationTelemetry(Protocol):
    def record_recommendation(
        self,
        *,
        bundle: RuntimeBundle,
        user_id: str | None,
        history_items: list[str] | None,
        top_k: int,
        items: list[RecommendationItem],
    ) -> int: ...


class TrainingPipeline(Protocol):
    def run(self, force_rebuild: bool = False) -> TrainingSession: ...

    def evaluate(self, force_rebuild: bool = False) -> EvaluationSummary: ...


class RecommendationService(Protocol):
    def readiness(self) -> ReadyState: ...

    def recommend(
        self,
        *,
        user_id: str | None = None,
        history_items: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RecommendationItem]: ...

    def get_user_history(self, user_id: str, limit: int = 15) -> list[HistoryItem]: ...

    def list_available_users(
        self,
        *,
        limit: int = 100,
        min_history: int = 1,
        query: str | None = None,
    ) -> list[AvailableUser]: ...

    def get_user_profile(
        self,
        user_id: str,
        *,
        history_limit: int = 20,
        recommendation_limit: int = 5,
    ) -> UserProfilePayload: ...

    def get_active_model(self) -> ActiveModelSummary: ...

    def get_evaluation_summary(self) -> EvaluationSummary: ...
