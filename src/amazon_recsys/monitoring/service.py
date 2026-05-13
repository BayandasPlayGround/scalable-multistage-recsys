from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import InferenceLogRecord, MonitoringSummary, OutcomeIngestResult, OutcomeLogRecord, OutcomeSimulationResult, RecommendationItem, ReferenceProfile, RuntimeBundle, utcnow_iso
from amazon_recsys.domain.protocols import ArtifactStore, MonitoringStore
from amazon_recsys.ml import core
from amazon_recsys.monitoring.metrics import (
    MONITORED_K,
    compute_concept_drift,
    compute_feature_drifts,
    concept_result_frame,
    feature_results_frame,
    summary_status,
)
from amazon_recsys.monitoring.reference import build_reference_profile
from amazon_recsys.monitoring.utils import ensure_utc_iso, hash_user_identifier
from amazon_recsys.observability.mlflow import MLflowTracker


class MonitoringService:
    def __init__(
        self,
        *,
        settings: AppSettings,
        artifact_store: ArtifactStore,
        monitoring_store: MonitoringStore,
        mlflow_tracker: MLflowTracker | None = None,
    ) -> None:
        self.settings = settings
        self.artifact_store = artifact_store
        self.monitoring_store = monitoring_store
        self.mlflow_tracker = mlflow_tracker or MLflowTracker(settings)

    def _resolve_bundle_version(self, bundle_version: str | None = None) -> str:
        if bundle_version and bundle_version != "active":
            return str(bundle_version)
        manifest = self.artifact_store.read_active_manifest()
        if manifest is None:
            raise FileNotFoundError("No active bundle is configured.")
        return manifest.version

    def build_reference_profile(self, session, bundle_version: str) -> Path:
        profile = build_reference_profile(self.settings, session, bundle_version=bundle_version, monitored_k=MONITORED_K)
        return self.monitoring_store.save_reference_profile(profile)

    def _reference_profile_for_bundle(self, bundle_version: str) -> ReferenceProfile:
        return self.monitoring_store.load_reference_profile(bundle_version)

    def _resolve_monitoring_window(
        self,
        *,
        window_start: str | None = None,
        window_end: str | None = None,
        days: int | None = None,
    ) -> tuple[str, str]:
        if window_start is not None or window_end is not None:
            if window_start is None or window_end is None:
                raise ValueError("Provide both window_start and window_end together.")
            return ensure_utc_iso(window_start), ensure_utc_iso(window_end)
        resolved_days = int(days) if days is not None else 1
        if resolved_days <= 0:
            raise ValueError("days must be positive.")
        window_end_ts = pd.Timestamp.now(tz="UTC")
        window_start_ts = window_end_ts - timedelta(days=resolved_days)
        return ensure_utc_iso(window_start_ts), ensure_utc_iso(window_end_ts)

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
        items: list[RecommendationItem],
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
        if bundle.serving_index is not None and not bundle.serving_index.user_summary.empty:
            user_summary = bundle.serving_index.user_summary.copy()
            user_summary["user_id"] = user_summary["user_id"].astype(str)
            user_summary = user_summary.set_index("user_id")
            user_exists = bool(user_id and str(user_id) in user_summary.index)
            if provided_history:
                request_history_length = len(provided_history)
            elif user_exists and "history_length" in user_summary.columns:
                request_history_length = int(user_summary.loc[str(user_id), "history_length"])
            else:
                request_history_length = 0

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

    def ingest_outcomes(self, source: Path) -> OutcomeIngestResult:
        return self.monitoring_store.ingest_outcomes(Path(source))

    def simulate_outcomes(
        self,
        *,
        bundle_version: str = "active",
        window_start: str | None = None,
        window_end: str | None = None,
        days: int | None = 1,
        delay_minutes: int = 60,
    ) -> OutcomeSimulationResult:
        resolved_bundle_version = self._resolve_bundle_version(bundle_version)
        normalized_window_start, normalized_window_end = self._resolve_monitoring_window(
            window_start=window_start,
            window_end=window_end,
            days=days,
        )
        inference_frame = self.monitoring_store.load_inference_frame(
            bundle_version=resolved_bundle_version,
            window_start=normalized_window_start,
            window_end=normalized_window_end,
        )
        if inference_frame.empty:
            raise ValueError(
                "No inference events were found in the selected window. "
                "Serve some recommendations first, then rerun monitoring with --simulate-outcomes."
            )

        linked = inference_frame.dropna(subset=["user_key"]).copy()
        linked = linked[linked["user_key"].astype(str).str.strip() != ""].copy()
        if linked.empty:
            raise ValueError(
                "Inference events were found, but none have a linked user_key. "
                "Use known-user recommendation requests for the automated local monitoring flow."
            )

        top_rank = linked.sort_values(["request_id", "rank"]).drop_duplicates("request_id").copy()
        records: list[OutcomeLogRecord] = []
        for row in top_rank.to_dict(orient="records"):
            occurred_at = ensure_utc_iso(
                pd.to_datetime(row["requested_at"], utc=True) + timedelta(minutes=int(delay_minutes))
            )
            records.append(
                OutcomeLogRecord(
                    occurred_at=occurred_at,
                    user_key=str(row["user_key"]),
                    item_id=str(row["item_id"]),
                    event_type="purchase",
                    rating=5.0,
                    value=None,
                    source="synthetic_inference_replay",
                )
            )

        ingested = self.monitoring_store.append_outcome_records(records)
        return OutcomeSimulationResult(
            source="synthetic_inference_replay",
            bundle_version=resolved_bundle_version,
            window_start=normalized_window_start,
            window_end=normalized_window_end,
            requests_seen=int(inference_frame["request_id"].nunique()),
            requests_with_user_key=int(top_rank["request_id"].nunique()),
            created=int(len(records)),
            ingested=ingested,
            event_type="purchase",
            rating=5.0,
        )

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
        simulate_outcomes: bool = False,
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
        if simulate_outcomes:
            self.simulate_outcomes(
                bundle_version=resolved_bundle_version,
                window_start=normalized_window_start,
                window_end=normalized_window_end,
                days=None,
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

    def monitor_backfill(self, *, days: int, bundle_version: str = "active", simulate_outcomes: bool = False) -> list[MonitoringSummary]:
        if int(days) <= 0:
            raise ValueError("days must be positive.")
        resolved_bundle_version = self._resolve_bundle_version(bundle_version)
        window_size = timedelta(days=int(self.settings.monitoring.window_days))
        if simulate_outcomes:
            inference_frame = self.monitoring_store.load_inference_frame(bundle_version=resolved_bundle_version)
            if inference_frame.empty:
                raise ValueError(
                    "No inference events were found for the active bundle. "
                    "Serve some recommendations first, then rerun monitor-backfill --simulate-outcomes."
                )
            inference_timestamps = pd.to_datetime(inference_frame["requested_at"], utc=True, errors="coerce").dropna()
            if inference_timestamps.empty:
                raise ValueError(
                    "Inference events were found, but no valid timestamps could be read from them."
                )
            matured_end = inference_timestamps.max() + pd.Timedelta(seconds=1)
        else:
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
                    simulate_outcomes=simulate_outcomes,
                )
            )
            cursor = next_cursor
        return results

    def latest_summary(self, bundle_version: str = "active") -> MonitoringSummary | None:
        resolved_bundle_version = self._resolve_bundle_version(bundle_version)
        return self.monitoring_store.load_latest_summary(resolved_bundle_version)

    def recent_summaries(self, bundle_version: str = "active", *, limit: int = 8) -> list[MonitoringSummary]:
        resolved_bundle_version = self._resolve_bundle_version(bundle_version)
        summaries = self.monitoring_store.list_summaries(resolved_bundle_version)
        ordered = sorted(summaries, key=lambda item: pd.to_datetime(item.window_end, utc=True))
        if limit <= 0:
            return ordered
        return ordered[-int(limit):]

    def _frame_records(self, frame: pd.DataFrame) -> list[dict[str, object]]:
        if frame.empty:
            return []
        safe = frame.astype(object).where(pd.notna(frame), None)
        return safe.to_dict(orient="records")

    def run_candidate_diagnostics(
        self,
        *,
        bundle_version: str = "active",
        split: str = "test",
        sample_size: int = 500,
        persist: bool = True,
    ) -> dict[str, object]:
        resolved_bundle_version = self._resolve_bundle_version(bundle_version)
        manifest = self.artifact_store.load_manifest(resolved_bundle_version)
        bundle = self.artifact_store.load_bundle(manifest)
        if bundle.prepared is None or bundle.split_artifacts is None:
            raise ValueError("The selected bundle does not include the artifacts required for candidate diagnostics.")
        if split == "test":
            examples = bundle.split_artifacts.test_examples
        elif split == "val":
            examples = bundle.split_artifacts.val_examples
        else:
            raise ValueError("split must be either 'val' or 'test'.")
        if sample_size and len(examples) > int(sample_size):
            examples = examples.sample(n=int(sample_size), random_state=bundle.prepared.config.seed).sort_values("example_id")

        diagnostics_payload = core.run_candidate_recovery_diagnostics(
            bundle.prepared,
            bundle.split_artifacts,
            bundle.retrievers,
            examples,
            split=split,
            bundle_version=manifest.version,
        )
        diagnostics = diagnostics_payload["diagnostics"]
        output_frames = diagnostics_payload["output_frames"]
        worst_slices = output_frames["candidate_recall_worst_slices"]
        overall_rows = diagnostics[(diagnostics["scope"] == "overall")].copy() if not diagnostics.empty else pd.DataFrame()
        source_rows = diagnostics[(diagnostics["scope"] == "candidate_source")].copy() if not diagnostics.empty else pd.DataFrame()
        cold_start_rows = diagnostics[(diagnostics["scope"] == "cold_start_user_type")].copy() if not diagnostics.empty else pd.DataFrame()
        summary: dict[str, object] = {
            "created_at": utcnow_iso(),
            "bundle_version": manifest.version,
            "split": split,
            "sample_size": int(len(examples)),
            "candidate_rows": int(diagnostics_payload["candidate_rows"]),
            "ranker_candidate_rows": int(diagnostics_payload["ranker_candidate_rows"]),
            "diagnostics": self._frame_records(diagnostics),
            "overall": self._frame_records(overall_rows),
            "source_summary": self._frame_records(source_rows),
            "cold_start_summary": self._frame_records(cold_start_rows),
            "worst_slices": self._frame_records(worst_slices),
        }
        if persist:
            artifact_paths = self.monitoring_store.save_candidate_diagnostics(summary, diagnostics, worst_slices)
            mlflow_payload = self.mlflow_tracker.log_candidate_diagnostics(summary, artifact_paths)
            if mlflow_payload is not None:
                summary["mlflow"] = mlflow_payload.to_dict()
                self.monitoring_store.save_candidate_diagnostics(summary, diagnostics, worst_slices)
        return summary

    def latest_candidate_diagnostics(self, bundle_version: str = "active") -> dict[str, object] | None:
        resolved_bundle_version = self._resolve_bundle_version(bundle_version)
        return self.monitoring_store.load_latest_candidate_diagnostics(resolved_bundle_version)

    def recent_candidate_diagnostics(self, bundle_version: str = "active", *, limit: int = 8) -> list[dict[str, object]]:
        resolved_bundle_version = self._resolve_bundle_version(bundle_version)
        summaries = self.monitoring_store.list_candidate_diagnostics(resolved_bundle_version)
        ordered = sorted(summaries, key=lambda item: pd.to_datetime(item.get("created_at"), utc=True))
        if limit <= 0:
            return ordered
        return ordered[-int(limit):]
