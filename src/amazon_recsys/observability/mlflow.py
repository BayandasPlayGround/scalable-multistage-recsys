from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import BundleManifest, EvaluationSummary, MlflowRunInfo

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None

if TYPE_CHECKING:
    from amazon_recsys.domain.entities import MonitoringSummary
    from amazon_recsys.ml.core import PipelineConfig
    from amazon_recsys.ml.pipelines import TrainingSession


LOGGER = logging.getLogger(__name__)
_UNAVAILABLE_WARNING_EMITTED = False
_DIMENSION_COLUMNS = {"K", "split", "stage", "variant"}


def _stringify(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return ",".join(_stringify(item) for item in value)
    return str(value)


def _flatten_mapping(value: dict[str, object], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, item in value.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten_mapping(item, prefix=flat_key))
            continue
        flattened[flat_key] = _stringify(item)
    return flattened


def _numeric_value(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _step_value(row: pd.Series) -> int | None:
    if "K" not in row.index:
        return None
    step = _numeric_value(row.get("K"))
    return int(step) if step is not None else None


class MLflowTracker:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    @property
    def config(self):
        return self.settings.mlflow

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and mlflow is not None)

    def _warn_if_unavailable(self) -> None:
        global _UNAVAILABLE_WARNING_EMITTED
        if self.config.enabled and mlflow is None and not _UNAVAILABLE_WARNING_EMITTED:
            LOGGER.warning("MLflow tracking was enabled but the mlflow package could not be imported.")
            _UNAVAILABLE_WARNING_EMITTED = True

    def _configure(self) -> bool:
        if not self.enabled:
            self._warn_if_unavailable()
            return False
        assert mlflow is not None
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        return True

    def _configure_monitoring(self) -> bool:
        if not self.enabled:
            self._warn_if_unavailable()
            return False
        assert mlflow is not None
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(f"{self.config.experiment_name}-monitoring")
        return True

    def _run_name(self, phase: str) -> str:
        prefix = self.config.run_name_prefix.strip()
        base = f"{phase}-{self.settings.training.run_name}-{self.settings.training.run_profile}"
        return f"{prefix}-{base}" if prefix else base

    def _log_params(self, params: dict[str, object]) -> None:
        assert mlflow is not None
        for key, value in _flatten_mapping(params).items():
            mlflow.log_param(key, value[:500])

    def _log_metrics(self, metrics: dict[str, object]) -> None:
        assert mlflow is not None
        for key, value in metrics.items():
            numeric = _numeric_value(value)
            if numeric is not None:
                mlflow.log_metric(key, numeric)

    def _log_artifact_if_exists(self, path: Path, artifact_path: str | None = None) -> None:
        assert mlflow is not None
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def _log_artifacts_if_exists(self, path: Path, artifact_path: str | None = None) -> None:
        assert mlflow is not None
        if path.exists():
            mlflow.log_artifacts(str(path), artifact_path=artifact_path)

    def _log_metric_frames(self, eval_dir: Path) -> None:
        assert mlflow is not None
        for csv_path in sorted(eval_dir.glob("*_metrics.csv")):
            frame = pd.read_csv(csv_path)
            base_name = csv_path.stem.removesuffix("_metrics")
            for _, row in frame.iterrows():
                prefix = [base_name]
                for column in ("variant", "stage", "split"):
                    raw_value = row.get(column)
                    if raw_value is not None and not pd.isna(raw_value):
                        prefix.append(str(raw_value))
                metric_prefix = ".".join(prefix)
                step = _step_value(row)
                for column, value in row.items():
                    if column in _DIMENSION_COLUMNS:
                        continue
                    numeric = _numeric_value(value)
                    if numeric is None:
                        continue
                    metric_name = f"{metric_prefix}.{column}"
                    if step is None:
                        mlflow.log_metric(metric_name, numeric)
                    else:
                        mlflow.log_metric(metric_name, numeric, step=step)

    def _session_metrics(self, session: TrainingSession) -> dict[str, float]:
        prepared = session.prepared
        split_artifacts = session.split_artifacts
        interactions = prepared.interactions
        return {
            "dataset.interactions": float(len(interactions)),
            "dataset.users": float(interactions["user_id"].nunique()),
            "dataset.items": float(interactions["parent_asin"].nunique()),
            "dataset.hard_negatives": float(len(prepared.hard_negatives)),
            "splits.train_examples": float(len(split_artifacts.train_examples)),
            "splits.val_examples": float(len(split_artifacts.val_examples)),
            "splits.test_examples": float(len(split_artifacts.test_examples)),
            "retrievers.count": float(len(session.retrievers)),
        }

    def _session_tags(self, session: TrainingSession) -> dict[str, str]:
        return {
            "phase": "train",
            "environment": self.settings.environment,
            "model_backend": self.settings.ranking.backend,
            "run_profile": self.settings.training.run_profile,
            "run_name": self.settings.training.run_name,
            "retriever_variants": ",".join(sorted(session.retrievers.keys())),
        }

    def _summary_tags(self) -> dict[str, str]:
        return {
            "phase": "evaluate",
            "environment": self.settings.environment,
            "model_backend": self.settings.ranking.backend,
            "run_profile": self.settings.training.run_profile,
            "run_name": self.settings.training.run_name,
        }

    def _tracking_metadata(self, run_id: str) -> MlflowRunInfo:
        return MlflowRunInfo(
            run_id=run_id,
            experiment_name=self.config.experiment_name,
            tracking_uri=self.config.tracking_uri,
        )

    def log_training_session(self, session: TrainingSession) -> MlflowRunInfo | None:
        if not self._configure():
            return None
        assert mlflow is not None
        pipeline_config = session.pipeline_config
        with mlflow.start_run(run_name=self._run_name("train")) as run:
            mlflow.set_tags(self._session_tags(session))
            self._log_params(self.settings.safe_config())
            self._log_metrics(self._session_metrics(session))
            self._log_artifact_if_exists(Path(pipeline_config.artifact_root) / "config.json", artifact_path="training")
            self._log_artifacts_if_exists(Path(pipeline_config.eval_dir), artifact_path="evaluation")
            self._log_metric_frames(Path(pipeline_config.eval_dir))
            return self._tracking_metadata(run.info.run_id)

    def log_evaluation_summary(
        self,
        pipeline_config: PipelineConfig,
        summary: EvaluationSummary,
    ) -> MlflowRunInfo | None:
        if not self._configure():
            return None
        assert mlflow is not None
        with mlflow.start_run(run_name=self._run_name("evaluate")) as run:
            mlflow.set_tags(self._summary_tags())
            self._log_params(self.settings.safe_config())
            if summary.config_path is not None:
                self._log_artifact_if_exists(Path(summary.config_path), artifact_path="training")
            if summary.eval_dir is not None:
                self._log_artifacts_if_exists(Path(summary.eval_dir), artifact_path="evaluation")
            self._log_metric_frames(Path(pipeline_config.eval_dir))
            return self._tracking_metadata(run.info.run_id)

    def log_bundle_export(
        self,
        session: TrainingSession,
        manifest: BundleManifest,
        extra_artifacts: list[Path] | None = None,
    ) -> None:
        if not self._configure():
            return
        run_id = session.mlflow.run_id if session.mlflow is not None else None
        if not run_id:
            return
        assert mlflow is not None
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tags(
                {
                    "bundle.version": manifest.version,
                    "bundle.model_backend": manifest.model_backend,
                    "bundle.retriever_variants": ",".join(manifest.retriever_variants),
                }
            )
            self._log_artifacts_if_exists(Path(manifest.bundle_dir), artifact_path="bundle")
            for artifact_path in extra_artifacts or []:
                self._log_artifact_if_exists(Path(artifact_path), artifact_path="bundle")

    def log_monitoring_summary(
        self,
        summary: MonitoringSummary,
        artifact_paths: dict[str, Path],
    ) -> MlflowRunInfo | None:
        if not self._configure_monitoring():
            return None
        assert mlflow is not None
        with mlflow.start_run(run_name=f"monitor-{summary.bundle_version}-{summary.window_end}") as run:
            mlflow.set_tags(
                {
                    "phase": "monitoring",
                    "bundle_version": summary.bundle_version,
                    "reference_bundle_version": summary.reference_bundle_version,
                    "window_start": summary.window_start,
                    "window_end": summary.window_end,
                    "status": summary.status,
                }
            )
            for key, value in self.settings.monitoring.model_dump(mode="json").items():
                mlflow.log_param(f"monitoring.{key}", str(value)[:500])
            mlflow.log_metric("drift.data.alert_count", float(sum(result.status in {"warn", "alert"} for result in summary.feature_drifts)))
            for result in summary.feature_drifts:
                metric_name = f"drift.data.{result.feature_name}.{result.metric_type}"
                mlflow.log_metric(metric_name, float(result.metric_value))
            for metric_name, metric_value in summary.concept_drift.metrics.items():
                if metric_value is not None:
                    mlflow.log_metric(f"drift.concept.{metric_name}", float(metric_value))
            mlflow.log_metric("drift.concept.performance_drop", float(summary.concept_drift.performance_drop))
            for path in artifact_paths.values():
                self._log_artifact_if_exists(Path(path), artifact_path="monitoring")
            return MlflowRunInfo(
                run_id=run.info.run_id,
                experiment_name=f"{self.config.experiment_name}-monitoring",
                tracking_uri=self.config.tracking_uri,
            )
