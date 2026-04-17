from __future__ import annotations

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import BundleManifest
from amazon_recsys.domain.protocols import ArtifactStore, MonitoringStore
from amazon_recsys.ml.pipelines import TrainingSession
from amazon_recsys.monitoring.reference import build_reference_profile
from amazon_recsys.observability.mlflow import MLflowTracker


class BundleExportService:
    def __init__(
        self,
        *,
        settings: AppSettings,
        artifact_store: ArtifactStore,
        monitoring_store: MonitoringStore,
        mlflow_tracker: MLflowTracker,
    ) -> None:
        self.settings = settings
        self.artifact_store = artifact_store
        self.monitoring_store = monitoring_store
        self.mlflow_tracker = mlflow_tracker

    def export_bundle(self, session: TrainingSession, version: str | None = None) -> BundleManifest:
        manifest = self.artifact_store.save_bundle(session, version=version)
        reference_profile = build_reference_profile(self.settings, session, bundle_version=manifest.version)
        reference_profile_path = self.monitoring_store.save_reference_profile(reference_profile)
        manifest.notes["reference_profile_path"] = str(reference_profile_path)
        manifest.notes["reference_bundle_version"] = manifest.version
        self.artifact_store.write_manifest(manifest)
        self.mlflow_tracker.log_bundle_export(session, manifest, extra_artifacts=[reference_profile_path])
        return manifest
