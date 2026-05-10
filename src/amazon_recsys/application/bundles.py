from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import BundleManifest, utcnow_iso
from amazon_recsys.domain.protocols import ArtifactStore, MonitoringStore
from amazon_recsys.ml import core
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

    def _frame_records(self, frame: pd.DataFrame) -> list[dict[str, object]]:
        if frame.empty:
            return []
        safe = frame.astype(object).where(pd.notna(frame), None)
        return safe.to_dict(orient="records")

    def _persist_candidate_diagnostics(self, session: TrainingSession, manifest: BundleManifest) -> dict[str, object] | None:
        examples = session.split_artifacts.test_examples
        sample_size = int(session.pipeline_config.eval_user_cap or 500)
        if sample_size > 0 and len(examples) > sample_size:
            examples = examples.sample(n=sample_size, random_state=session.pipeline_config.seed).sort_values("example_id")
        diagnostics_payload = core.run_candidate_recovery_diagnostics(
            session.prepared,
            session.split_artifacts,
            session.retrievers,
            examples,
            split="test",
            bundle_version=manifest.version,
        )
        diagnostics = diagnostics_payload["diagnostics"]
        output_frames = diagnostics_payload["output_frames"]
        worst_slices = output_frames["candidate_recall_worst_slices"]
        overall_rows = diagnostics[diagnostics["scope"] == "overall"].copy() if not diagnostics.empty else pd.DataFrame()
        source_rows = diagnostics[diagnostics["scope"] == "candidate_source"].copy() if not diagnostics.empty else pd.DataFrame()
        cold_start_rows = diagnostics[diagnostics["scope"] == "cold_start_user_type"].copy() if not diagnostics.empty else pd.DataFrame()
        summary: dict[str, object] = {
            "created_at": utcnow_iso(),
            "bundle_version": manifest.version,
            "split": "test",
            "sample_size": int(len(examples)),
            "candidate_rows": int(diagnostics_payload["candidate_rows"]),
            "ranker_candidate_rows": int(diagnostics_payload["ranker_candidate_rows"]),
            "diagnostics": self._frame_records(diagnostics),
            "overall": self._frame_records(overall_rows),
            "source_summary": self._frame_records(source_rows),
            "cold_start_summary": self._frame_records(cold_start_rows),
            "worst_slices": self._frame_records(worst_slices),
        }
        artifact_paths = self.monitoring_store.save_candidate_diagnostics(summary, diagnostics, worst_slices)
        manifest.notes["candidate_diagnostics_path"] = str(artifact_paths["summary_path"])
        return summary

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
        LOGGER.info("Export stage 2/4: building monitoring reference profile and candidate diagnostics")
        reference_profile = build_reference_profile(self.settings, session, bundle_version=manifest.version)
        reference_profile_path = self.monitoring_store.save_reference_profile(reference_profile)
        manifest.notes["reference_profile_path"] = str(reference_profile_path)
        manifest.notes["reference_bundle_version"] = manifest.version
        candidate_diagnostics = self._persist_candidate_diagnostics(session, manifest)
        LOGGER.info(
            "Export stage 2/4 complete in %s: reference_profile=%s candidate_diagnostics=%s",
            _elapsed(stage_start),
            reference_profile_path,
            candidate_diagnostics is not None,
        )
        stage_start = time.perf_counter()
        LOGGER.info("Export stage 3/4: writing bundle manifest")
        self.artifact_store.write_manifest(manifest)
        LOGGER.info("Export stage 3/4 complete in %s", _elapsed(stage_start))
        stage_start = time.perf_counter()
        LOGGER.info("Export stage 4/4: logging bundle artifacts to MLflow if enabled")
        extra_artifacts: list[Path] = [reference_profile_path]
        candidate_diagnostics_path = manifest.notes.get("candidate_diagnostics_path")
        if candidate_diagnostics_path is not None:
            extra_artifacts.append(Path(str(candidate_diagnostics_path)))
        self.mlflow_tracker.log_bundle_export(session, manifest, extra_artifacts=extra_artifacts)
        LOGGER.info("Export stage 4/4 complete in %s", _elapsed(stage_start))
        LOGGER.info("Bundle export finished in %s: version=%s", _elapsed(export_start), manifest.version)
        return manifest
