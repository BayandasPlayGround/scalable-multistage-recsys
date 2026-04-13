from __future__ import annotations

from typing import Any

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import BundleManifest, HistoryItem, RecommendationItem, RuntimeBundle, utcnow_iso
from amazon_recsys.infrastructure.artifacts import LocalArtifactStore
from amazon_recsys.ml.legacy import load_legacy_pipeline


def _mock_bundle(settings: AppSettings) -> RuntimeBundle:
    manifest = BundleManifest(
        version="mock",
        created_at=utcnow_iso(),
        manifest_path="mock",
        bundle_dir=str(settings.resolved_bundle_root / "mock"),
        runtime_bundle_path=str(settings.resolved_bundle_root / "mock" / "runtime_bundle.pkl"),
        evaluation_summary_path=None,
        run_name=settings.training.run_name,
        run_profile=settings.training.run_profile,
        model_backend="mock",
        retriever_variants=["mock"],
        notes={"reason": "No active bundle was configured."},
    )
    return RuntimeBundle(
        manifest=manifest,
        evaluation_summary={
            "source": "mock",
            "message": "No active bundle is available yet. Train and activate a bundle to replace mock responses.",
            "metric_files": [],
        },
        is_mock=True,
    )


class BundleRecommendationService:
    def __init__(self, artifact_store: LocalArtifactStore, settings: AppSettings) -> None:
        self.artifact_store = artifact_store
        self.settings = settings
        self._bundle_cache: RuntimeBundle | None = None
        self._bundle_version: str | None = None

    def refresh(self) -> None:
        self._bundle_cache = None
        self._bundle_version = None

    def _load_bundle(self, force: bool = False) -> RuntimeBundle:
        if force:
            self.refresh()
        manifest = self.artifact_store.read_active_manifest()
        if manifest is None:
            if self.settings.use_mock_bundle_if_missing:
                if self._bundle_cache is None or self._bundle_version != "mock":
                    self._bundle_cache = _mock_bundle(self.settings)
                    self._bundle_version = "mock"
                return self._bundle_cache
            raise FileNotFoundError("No active bundle is configured.")
        if self._bundle_cache is not None and self._bundle_version == manifest.version:
            return self._bundle_cache
        bundle = self.artifact_store.load_bundle(manifest)
        self._bundle_cache = bundle
        self._bundle_version = manifest.version
        return bundle

    def readiness(self) -> dict[str, Any]:
        try:
            bundle = self._load_bundle()
        except FileNotFoundError:
            return {"ready": False, "status": "not_ready", "source": "none", "version": None}
        return {
            "ready": True,
            "status": "ready",
            "source": "mock" if bundle.is_mock else "bundle",
            "version": bundle.manifest.version,
        }

    def recommend(
        self,
        *,
        user_id: str | None = None,
        history_items: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RecommendationItem]:
        bundle = self._load_bundle()
        effective_top_k = top_k or self.settings.serving.default_top_k
        if bundle.is_mock:
            seed_item = history_items[0] if history_items else "mock-item"
            return [
                RecommendationItem(
                    item_id=f"mock-{index}",
                    title=f"Mock recommendation {index} for {seed_item}",
                    source_category="mock",
                    price=float(index * 10),
                    average_rating=4.0,
                    retrieval_score=1.0 / index,
                    score=0.9 / index,
                    candidate_sources="mock",
                )
                for index in range(1, effective_top_k + 1)
            ]
        legacy = load_legacy_pipeline()
        frame = legacy.recommend(
            bundle.prepared,
            bundle.split_artifacts,
            bundle.retrievers,
            ranker=bundle.ranker,
            user_id=user_id,
            history_items=history_items,
            top_k=effective_top_k,
        )
        if frame.empty:
            return self._popularity_backfill(
                bundle=bundle,
                user_id=user_id,
                history_items=history_items,
                top_k=effective_top_k,
            )
        return [
            RecommendationItem(
                item_id=str(row["parent_asin"]),
                title=str(row.get("title", "")),
                source_category=str(row.get("source_category", "")),
                price=float(row["price"]) if row.get("price") is not None else None,
                average_rating=float(row["average_rating"]) if row.get("average_rating") is not None else None,
                retrieval_score=float(row["retrieval_score"]) if row.get("retrieval_score") is not None else None,
                score=float(row["score"]) if "score" in row and row.get("score") is not None else None,
                candidate_sources=str(row.get("candidate_sources", "")) or None,
            )
            for row in frame.to_dict(orient="records")
        ]

    def _popularity_backfill(
        self,
        *,
        bundle: RuntimeBundle,
        user_id: str | None,
        history_items: list[str] | None,
        top_k: int,
    ) -> list[RecommendationItem]:
        seen_items = set(history_items or [])
        if not seen_items and user_id is not None:
            try:
                seen_items = {item.item_id for item in self.get_user_history(user_id, limit=100)}
            except KeyError:
                seen_items = set()
        item_frame = bundle.prepared.item_features.copy()
        if "train_positive_count" in item_frame.columns:
            sort_columns = ["train_positive_count", "rating_number"]
        else:
            sort_columns = ["rating_number"]
        item_frame = item_frame[~item_frame["parent_asin"].isin(seen_items)].sort_values(sort_columns, ascending=False).head(top_k)
        return [
            RecommendationItem(
                item_id=str(row["parent_asin"]),
                title=str(row.get("title", "")),
                source_category=str(row.get("source_category", "")),
                price=float(row["price"]) if row.get("price") is not None else None,
                average_rating=float(row["average_rating"]) if row.get("average_rating") is not None else None,
                retrieval_score=None,
                score=None,
                candidate_sources="popularity_fallback",
            )
            for row in item_frame.to_dict(orient="records")
        ]

    def get_user_history(self, user_id: str, limit: int = 15) -> list[HistoryItem]:
        bundle = self._load_bundle()
        if bundle.is_mock:
            return []
        legacy = load_legacy_pipeline()
        history = legacy.get_user_order_history(
            bundle.prepared,
            bundle.split_artifacts,
            user_id=user_id,
            split="test",
            limit=limit,
        )
        return [
            HistoryItem(
                ordered_at=str(row["ordered_at"]),
                item_id=str(row["parent_asin"]),
                title=str(row.get("title", "")),
                source_category=str(row.get("source_category", "")),
                review_rating=float(row["review_rating"]) if row.get("review_rating") is not None else None,
                verified_purchase=int(row["verified_purchase"]) if row.get("verified_purchase") is not None else None,
                price=float(row["price"]) if row.get("price") is not None else None,
                average_rating=float(row["average_rating"]) if row.get("average_rating") is not None else None,
            )
            for row in history.to_dict(orient="records")
        ]

    def get_active_model(self) -> dict[str, Any]:
        bundle = self._load_bundle()
        return {
            "ready": True,
            "source": "mock" if bundle.is_mock else "bundle",
            "version": bundle.manifest.version,
            "run_name": bundle.manifest.run_name,
            "run_profile": bundle.manifest.run_profile,
            "model_backend": bundle.manifest.model_backend,
            "retriever_variants": bundle.manifest.retriever_variants,
            "created_at": bundle.manifest.created_at,
        }

    def get_evaluation_summary(self) -> dict[str, Any]:
        bundle = self._load_bundle()
        return bundle.evaluation_summary
