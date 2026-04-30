from __future__ import annotations

import logging
from collections import Counter
from statistics import mean

import pandas as pd

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import (
    ActiveModelSummary,
    AvailableUser,
    BundleManifest,
    EvaluationSummary,
    HistoryItem,
    RecommendationItem,
    ReadyState,
    RuntimeBundle,
    UserProfile,
    UserProfilePayload,
    utcnow_iso,
)
from amazon_recsys.domain.protocols import ArtifactStore, RecommendationTelemetry
from amazon_recsys.ml import core


LOGGER = logging.getLogger(__name__)


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
        evaluation_summary=EvaluationSummary(
            source="mock",
            message="No active bundle is available yet. Train and activate a bundle to replace mock responses.",
            metric_files=[],
        ),
        is_mock=True,
    )


class BundleRecommendationService:
    def __init__(
        self,
        artifact_store: ArtifactStore,
        settings: AppSettings,
        monitoring_service: RecommendationTelemetry | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.settings = settings
        self.monitoring_service = monitoring_service
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

    def readiness(self) -> ReadyState:
        try:
            manifest = self.artifact_store.read_active_manifest()
        except FileNotFoundError:
            return ReadyState(ready=False, status="not_ready", source="none", version=None)
        if manifest is None:
            if self.settings.use_mock_bundle_if_missing:
                return ReadyState(ready=True, status="ready", source="mock", version="mock")
            return ReadyState(ready=False, status="not_ready", source="none", version=None)
        return ReadyState(
            ready=True,
            status="ready",
            source="bundle",
            version=manifest.version,
        )

    def recommend(
        self,
        *,
        user_id: str | None = None,
        history_items: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RecommendationItem]:
        bundle = self._load_bundle()
        effective_top_k = top_k or self.settings.serving.default_top_k
        items = self._compute_recommendations(
            bundle=bundle,
            user_id=user_id,
            history_items=history_items,
            top_k=effective_top_k,
        )
        if self.monitoring_service is not None:
            try:
                self.monitoring_service.record_recommendation(
                    bundle=bundle,
                    user_id=user_id,
                    history_items=history_items,
                    top_k=effective_top_k,
                    items=items,
                )
            except OSError:
                LOGGER.exception("Failed to record recommendation inference for monitoring.")
        return items

    def _compute_recommendations(
        self,
        *,
        bundle: RuntimeBundle,
        user_id: str | None = None,
        history_items: list[str] | None = None,
        top_k: int,
    ) -> list[RecommendationItem]:
        effective_top_k = max(1, int(top_k))
        if bundle.is_mock:
            seed_item = history_items[0] if history_items else "mock-item"
            items = [
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
            return items
        effective_history_items = history_items
        if user_id is not None and effective_history_items is None:
            effective_history_items = self._serving_history_item_ids(bundle, user_id)
        frame = core.recommend(
            bundle.prepared,
            bundle.split_artifacts,
            bundle.retrievers,
            ranker=bundle.ranker,
            user_id=user_id,
            history_items=effective_history_items,
            top_k=effective_top_k,
        )
        if frame.empty:
            items = self._popularity_backfill(
                bundle=bundle,
                user_id=user_id,
                history_items=history_items,
                top_k=effective_top_k,
            )
        else:
            items = [
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
        return items

    def _serving_history_frame(self, bundle: RuntimeBundle, user_id: str, limit: int | None = None) -> pd.DataFrame | None:
        if bundle.serving_index is None or bundle.serving_index.user_history.empty:
            return None
        history = bundle.serving_index.user_history
        user_history = history[history["user_id"].astype(str) == str(user_id)].copy()
        if user_history.empty:
            return None
        if "timestamp_dt" in user_history.columns:
            user_history = user_history.sort_values("timestamp_dt")
        if limit is not None and len(user_history) > limit:
            user_history = user_history.tail(limit).copy()
        return user_history.reset_index(drop=True)

    def _serving_history_item_ids(self, bundle: RuntimeBundle, user_id: str) -> list[str] | None:
        user_history = self._serving_history_frame(bundle, user_id, limit=None)
        if user_history is None or user_history.empty:
            return None
        return [str(value) for value in user_history["parent_asin"].tolist()]

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
        history = self._serving_history_frame(bundle, user_id, limit=limit)
        if history is None:
            history = core.get_user_order_history(
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

    def get_user_profile(
        self,
        user_id: str,
        *,
        history_limit: int = 20,
        recommendation_limit: int = 5,
    ) -> UserProfilePayload:
        bundle = self._load_bundle()
        if bundle.is_mock:
            raise KeyError(f"Unknown user_id: {user_id}")

        history = self.get_user_history(user_id, limit=history_limit)
        if not history:
            raise KeyError(f"Unknown user_id: {user_id}")

        available_user = next(
            (item for item in self.list_available_users(limit=5000, min_history=0, query=user_id) if item.user_id == user_id),
            None,
        )

        ratings = [item.review_rating for item in history if item.review_rating is not None]
        verified = [item.verified_purchase for item in history if item.verified_purchase is not None]
        category_counts = Counter(item.source_category for item in history if item.source_category)
        profile = UserProfile(
            user_id=user_id,
            interaction_count=available_user.interaction_count if available_user is not None else len(history),
            history_length=available_user.history_length if available_user is not None else len(history),
            last_ordered_at=available_user.last_ordered_at if available_user is not None else history[0].ordered_at,
            average_review_rating=mean(ratings) if ratings else None,
            verified_purchase_rate=(sum(verified) / len(verified)) if verified else None,
            top_categories=[name for name, _ in category_counts.most_common(3)],
        )
        recommendations = self._compute_recommendations(
            bundle=bundle,
            user_id=user_id,
            history_items=None,
            top_k=recommendation_limit,
        )
        return UserProfilePayload(profile=profile, history=history, recommendations=recommendations)

    def list_available_users(
        self,
        *,
        limit: int = 100,
        min_history: int = 1,
        query: str | None = None,
    ) -> list[AvailableUser]:
        bundle = self._load_bundle()
        if bundle.is_mock:
            return []

        if bundle.serving_index is not None and not bundle.serving_index.user_summary.empty:
            summary = bundle.serving_index.user_summary.copy()
            summary["user_id"] = summary["user_id"].astype(str)
            summary["history_length"] = summary["history_length"].astype(int)
            summary["interaction_count"] = summary["interaction_count"].fillna(0).astype(int)
            summary = summary[summary["history_length"] >= int(min_history)].copy()
            if query:
                needle = query.strip().lower()
                if needle:
                    summary = summary[summary["user_id"].str.lower().str.contains(needle, regex=False)].copy()
            if summary.empty:
                return []
            summary = summary.sort_values(
                ["interaction_count", "history_length", "user_id"],
                ascending=[False, False, True],
            ).head(limit)
            return [
                AvailableUser(
                    user_id=str(row["user_id"]),
                    interaction_count=int(row["interaction_count"]),
                    history_length=int(row["history_length"]),
                    last_ordered_at=str(row["last_ordered_at"]) if pd.notna(row.get("last_ordered_at")) else None,
                )
                for row in summary.to_dict(orient="records")
            ]

        examples = bundle.split_artifacts.test_examples.copy()
        if examples.empty:
            return []

        examples["user_id"] = examples["user_id"].astype(str)
        examples["history_length"] = examples["history_length"].astype(int)
        examples = examples[examples["history_length"] >= int(min_history)].copy()
        if examples.empty:
            return []

        interactions = bundle.prepared.interactions.copy()
        interactions["user_id"] = interactions["user_id"].astype(str)

        interaction_counts = (
            interactions.groupby("user_id", as_index=False)
            .size()
            .rename(columns={"size": "interaction_count"})
        )
        if "timestamp_dt" in interactions.columns:
            last_orders = (
                interactions.groupby("user_id", as_index=False)["timestamp_dt"]
                .max()
                .rename(columns={"timestamp_dt": "last_ordered_at"})
            )
        else:
            last_orders = interactions.groupby("user_id", as_index=False)["timestamp"].max()
            last_orders["last_ordered_at"] = last_orders["timestamp"].astype(str)
            last_orders = last_orders.drop(columns=["timestamp"])

        summary = (
            examples.groupby("user_id", as_index=False)["history_length"]
            .max()
            .merge(interaction_counts, on="user_id", how="left")
            .merge(last_orders, on="user_id", how="left")
        )
        summary["interaction_count"] = summary["interaction_count"].fillna(0).astype(int)
        if query:
            needle = query.strip().lower()
            if needle:
                summary = summary[summary["user_id"].str.lower().str.contains(needle, regex=False)].copy()
        if summary.empty:
            return []

        summary = summary.sort_values(
            ["interaction_count", "history_length", "user_id"],
            ascending=[False, False, True],
        ).head(limit)

        available_users: list[AvailableUser] = []
        for row in summary.to_dict(orient="records"):
            last_ordered = row.get("last_ordered_at")
            if hasattr(last_ordered, "strftime"):
                last_ordered_value = last_ordered.strftime("%Y-%m-%d")
            elif last_ordered is None:
                last_ordered_value = None
            else:
                last_ordered_value = str(last_ordered)
            available_users.append(
                AvailableUser(
                    user_id=str(row["user_id"]),
                    interaction_count=int(row["interaction_count"]),
                    history_length=int(row["history_length"]),
                    last_ordered_at=last_ordered_value,
                )
            )
        return available_users

    def get_active_model(self) -> ActiveModelSummary:
        manifest = self.artifact_store.read_active_manifest()
        if manifest is not None:
            return ActiveModelSummary(
                ready=True,
                source="bundle",
                version=manifest.version,
                run_name=manifest.run_name,
                run_profile=manifest.run_profile,
                model_backend=manifest.model_backend,
                retriever_variants=manifest.retriever_variants,
                created_at=manifest.created_at,
            )
        bundle = self._load_bundle()
        return ActiveModelSummary(
            ready=True,
            source="mock" if bundle.is_mock else "bundle",
            version=bundle.manifest.version,
            run_name=bundle.manifest.run_name,
            run_profile=bundle.manifest.run_profile,
            model_backend=bundle.manifest.model_backend,
            retriever_variants=bundle.manifest.retriever_variants,
            created_at=bundle.manifest.created_at,
        )

    def get_evaluation_summary(self) -> EvaluationSummary:
        bundle = self._load_bundle()
        return bundle.evaluation_summary
