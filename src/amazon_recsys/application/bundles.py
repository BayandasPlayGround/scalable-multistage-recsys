from __future__ import annotations

import logging
import time

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import BundleManifest
from amazon_recsys.domain.protocols import ArtifactStore, MonitoringStore
from amazon_recsys.ml.pipelines import TrainingSession
from amazon_recsys.monitoring.reference import build_reference_profile
from amazon_recsys.observability.mlflow import MLflowTracker


LOGGER = logging.getLogger(__name__)


def _elapsed(start_time: float) -> str:
    seconds = time.perf_counter() - start_time
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


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
        export_start = time.perf_counter()
        LOGGER.info("Bundle export started: requested_version=%s", version or "<auto>")
        stage_start = time.perf_counter()
        LOGGER.info("Export stage 1/4: writing runtime bundle artifacts")
        manifest = self.artifact_store.save_bundle(session, version=version)
        LOGGER.info(
            "Export stage 1/4 complete in %s: version=%s bundle_dir=%s",
            _elapsed(stage_start),
            manifest.version,
            manifest.bundle_dir,
        )
        stage_start = time.perf_counter()
        LOGGER.info("Export stage 2/4: building monitoring reference profile")
        reference_profile = build_reference_profile(self.settings, session, bundle_version=manifest.version)
        reference_profile_path = self.monitoring_store.save_reference_profile(reference_profile)
        manifest.notes["reference_profile_path"] = str(reference_profile_path)
        manifest.notes["reference_bundle_version"] = manifest.version
        LOGGER.info("Export stage 2/4 complete in %s: reference_profile=%s", _elapsed(stage_start), reference_profile_path)
        stage_start = time.perf_counter()
        LOGGER.info("Export stage 3/4: writing bundle manifest")
        self.artifact_store.write_manifest(manifest)
        LOGGER.info("Export stage 3/4 complete in %s", _elapsed(stage_start))
        stage_start = time.perf_counter()
        LOGGER.info("Export stage 4/4: logging bundle artifacts to MLflow if enabled")
        self.mlflow_tracker.log_bundle_export(session, manifest, extra_artifacts=[reference_profile_path])
        LOGGER.info("Export stage 4/4 complete in %s", _elapsed(stage_start))
        LOGGER.info("Bundle export finished in %s: version=%s", _elapsed(export_start), manifest.version)
        return manifest
