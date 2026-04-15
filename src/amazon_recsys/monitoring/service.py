from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import InferenceLogRecord, MonitoringSummary, ReferenceProfile, RuntimeBundle, utcnow_iso
from amazon_recsys.infrastructure.artifacts import LocalArtifactStore
from amazon_recsys.monitoring.metrics import (
    MONITORED_K,
    compute_concept_drift,
    compute_feature_drifts,
    concept_result_frame,
    feature_results_frame,
    summary_status,
)
from amazon_recsys.monitoring.reference import build_reference_profile
from amazon_recsys.monitoring.store import LocalMonitoringStore
from amazon_recsys.monitoring.utils import ensure_utc_iso, hash_user_identifier
from amazon_recsys.observability.mlflow import MLflowTracker


class MonitoringService:
    def __init__(
        self,
        *,
        settings: AppSettings,
        artifact_store: LocalArtifactStore,
        monitoring_store: LocalMonitoringStore,
    ) -> None:
        self.settings = settings
        self.artifact_store = artifact_store
        self.monitoring_store = monitoring_store
        self.mlflow_tracker = MLflowTracker(settings)

    def _resolve_bundle_version(self, bundle_version: str | None = None) -> str:
        if bundle_version and bundle_version != "active":
            return str(bundle_version)
        manifest = self.artifact_store.read_active_manifest()
        if manifest is None:
            raise FileNotFoundError("No active bundle is configured.")
        return manifest.version

    def build_reference_profile(self, session: Any, bundle_version: str) -> Path:
        profile = build_reference_profile(self.settings, session, bundle_version=bundle_version, monitored_k=MONITORED_K)
        return self.monitoring_store.save_reference_profile(profile)

    def _reference_profile_for_bundle(self, bundle_version: str) -> ReferenceProfile:
        return self.monitoring_store.load_reference_profile(bundle_version)

    def _bundle_user_history_length(self, bundle: RuntimeBundle, user_id: str) -> int:
        interactions = bundle.prepared.interactions
        frame = interactions[interactions["user_id"].astype(str) == str(user_id)]
        return int(len(frame))

    def record_recommendation(
        self,
        *,
        bundle: RuntimeBundle,
        user_id: str | None,
        history_items: list[str] | None,
        top_k: int,
        items: list[Any],
    ) -> int:
        if not self.settings.monitoring.enabled or bundle.is_mock or not items:
            return 0

        provided_history = [str(item) for item in (history_items or []) if str(item).strip()]
        query_mode = "history_override" if provided_history else "known_user"
        known_user_lookup = set(bundle.prepared.interactions["user_id"].astype(str).unique())
        user_exists = bool(user_id and str(user_id) in known_user_lookup)
        request_history_length = len(provided_history) if provided_history else self._bundle_user_history_length(bundle, str(user_id)) if user_id else 0
        history_catalog = set(bundle.prepared.item_id_to_idx.keys())
        if provided_history:
            unseen_history_count = sum(1 for item in provided_history if item not in history_catalog)
            unseen_history_item_rate = float(unseen_history_count / max(len(provided_history), 1))
        else:
            unseen_history_item_rate = 0.0
        item_features = bundle.prepared.item_features.copy()
        popularity_column = "train_positive_count" if "train_positive_count" in item_features.columns else "rating_number"
        popularity_lookup = (
            item_features[["parent_asin", popularity_column]]
            .drop_duplicates("parent_asin")
            .set_index("parent_asin")[popularity_column]
            .to_dict()
        )

        request_id = uuid4().hex
        requested_at = utcnow_iso()
        user_key = hash_user_identifier(str(user_id)) if user_id else None
        records: list[InferenceLogRecord] = []
        for rank, item in enumerate(items, start=1):
            records.append(
                InferenceLogRecord(
                    requested_at=requested_at,
                    request_id=request_id,
                    bundle_version=bundle.manifest.version,
                    user_key=user_key,
                    query_mode=query_mode,
                    request_history_length=int(request_history_length),
                    top_k=int(top_k),
                    item_id=str(item.item_id),
                    rank=int(rank),
                    score=float(item.score) if item.score is not None else float(item.retrieval_score) if item.retrieval_score is not None else None,
                    candidate_sources=item.candidate_sources,
                    source_category=item.source_category,
                    price=float(item.price) if item.price is not None else None,
                    average_rating=float(item.average_rating) if item.average_rating is not None else None,
                    popularity_value=float(popularity_lookup.get(str(item.item_id))) if popularity_lookup.get(str(item.item_id)) is not None else None,
                    is_known_user=user_exists,
                    unseen_user=bool(user_id) and not user_exists,
                    unseen_history_item_rate=float(unseen_history_item_rate),
                )
            )
        return self.monitoring_store.append_inference_records(records)

    def ingest_outcomes(self, source: Path) -> dict[str, Any]:
        return self.monitoring_store.ingest_outcomes(Path(source))

    def _previous_summary(self, bundle_version: str, window_end: str) -> MonitoringSummary | None:
        current_end = pd.to_datetime(window_end, utc=True)
        previous = [
            summary
            for summary in self.monitoring_store.list_summaries(bundle_version)
            if pd.to_datetime(summary.window_end, utc=True) < current_end
        ]
        if not previous:
            return None
        return max(previous, key=lambda item: pd.to_datetime(item.window_end, utc=True))

    def run_monitoring(
        self,
        *,
        window_start: str,
        window_end: str,
        bundle_version: str = "active",
    ) -> MonitoringSummary:
        resolved_bundle_version = self._resolve_bundle_version(bundle_version)
        reference_profile = self._reference_profile_for_bundle(resolved_bundle_version)
        normalized_window_start = ensure_utc_iso(window_start)
        normalized_window_end = ensure_utc_iso(window_end)

        inference_frame = self.monitoring_store.load_inference_frame(
            bundle_version=resolved_bundle_version,
            window_start=normalized_window_start,
            window_end=normalized_window_end,
        )
        outcomes_end = ensure_utc_iso(pd.to_datetime(normalized_window_end, utc=True) + timedelta(days=int(self.settings.monitoring.attribution_horizon_days)))
        outcomes_frame = self.monitoring_store.load_outcome_frame(
            window_start=normalized_window_start,
            window_end=outcomes_end,
        )

        feature_drifts = compute_feature_drifts(reference_profile, inference_frame, self.settings.monitoring)
        previous_summary = self._previous_summary(resolved_bundle_version, normalized_window_end)
        concept_drift = compute_concept_drift(
            reference_profile,
            inference_frame,
            outcomes_frame,
            self.settings.monitoring,
            previous_summary=previous_summary,
            monitored_k=reference_profile.monitored_k,
        )
        status = summary_status(feature_drifts, concept_drift)
        summary = MonitoringSummary(
            bundle_version=resolved_bundle_version,
            reference_bundle_version=reference_profile.bundle_version,
            created_at=utcnow_iso(),
            window_start=normalized_window_start,
            window_end=normalized_window_end,
            status=status,
            inference_count=int(inference_frame["request_id"].nunique()) if not inference_frame.empty else 0,
            outcome_count=int(len(outcomes_frame)),
            feature_drifts=feature_drifts,
            concept_drift=concept_drift,
            top_drifting_features=[result.feature_name for result in feature_drifts[:3]],
        )
        feature_frame = feature_results_frame(feature_drifts)
        concept_frame = concept_result_frame(concept_drift)
        artifact_paths = self.monitoring_store.save_monitoring_summary(summary, feature_frame, concept_frame, reference_profile)
        mlflow_payload = self.mlflow_tracker.log_monitoring_summary(summary, artifact_paths)
        if mlflow_payload is not None:
            summary.mlflow = mlflow_payload
            self.monitoring_store.save_monitoring_summary(summary, feature_frame, concept_frame, reference_profile)
        return summary

    def monitor_backfill(self, *, days: int, bundle_version: str = "active") -> list[MonitoringSummary]:
        if int(days) <= 0:
            raise ValueError("days must be positive.")
        resolved_bundle_version = self._resolve_bundle_version(bundle_version)
        window_size = timedelta(days=int(self.settings.monitoring.window_days))
        matured_end = pd.Timestamp.now(tz="UTC") - timedelta(days=int(self.settings.monitoring.label_delay_days))
        window_start = matured_end - timedelta(days=int(days))
        results: list[MonitoringSummary] = []
        cursor = window_start
        while cursor < matured_end:
            next_cursor = min(cursor + window_size, matured_end)
            results.append(
                self.run_monitoring(
                    window_start=cursor.isoformat(),
                    window_end=next_cursor.isoformat(),
                    bundle_version=resolved_bundle_version,
                )
            )
            cursor = next_cursor
        return results

    def latest_summary(self, bundle_version: str = "active") -> MonitoringSummary | None:
        resolved_bundle_version = self._resolve_bundle_version(bundle_version)
        return self.monitoring_store.load_latest_summary(resolved_bundle_version)
