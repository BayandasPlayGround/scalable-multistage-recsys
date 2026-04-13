from __future__ import annotations

from typing import Protocol

from amazon_recsys.domain.entities import ActiveBundlePointer, BundleManifest, HistoryItem, RecommendationItem, RuntimeBundle


class ArtifactStore(Protocol):
    def save_bundle(self, session: object, version: str | None = None) -> BundleManifest: ...

    def activate_bundle(self, version: str) -> ActiveBundlePointer: ...

    def read_active_manifest(self) -> BundleManifest | None: ...

    def load_active_bundle(self) -> RuntimeBundle | None: ...


class TrainingPipeline(Protocol):
    def run(self, force_rebuild: bool = False) -> object: ...

    def evaluate(self, force_rebuild: bool = False) -> dict[str, object]: ...


class RecommendationService(Protocol):
    def recommend(
        self,
        *,
        user_id: str | None = None,
        history_items: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RecommendationItem]: ...

    def get_user_history(self, user_id: str, limit: int = 15) -> list[HistoryItem]: ...

    def get_active_model(self) -> dict[str, object]: ...

    def get_evaluation_summary(self) -> dict[str, object]: ...
