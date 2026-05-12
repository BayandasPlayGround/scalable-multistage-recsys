from __future__ import annotations

import logging
import shutil
import time
from dataclasses import MISSING, dataclass
from pathlib import Path

import pandas as pd

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import EvaluationMetricPreview, EvaluationSummary, MlflowRunInfo
from amazon_recsys.ml import core
from amazon_recsys.ml.io_utils import set_artifact_write_mode
from amazon_recsys.observability.mlflow import MLflowTracker


LOGGER = logging.getLogger(__name__)


_SETTINGS_TO_CONFIG_FIELDS = {
    "categories": "categories",
    "run_name": "run_name",
    "run_profile": "run_profile",
    "seed": "seed",
    "k_core": "k_core",
    "dev_mode": "dev_mode",
    "dev_fraction": "dev_fraction",
    "show_progress": "show_progress",
    "train_positive_cap": "train_positive_cap",
    "split_eval_example_cap": "split_eval_example_cap",
    "metadata_download_if_missing": "metadata_download_if_missing",
    "enable_neural_retriever": "enable_neural_retriever",
    "neural_retriever_variant": "neural_retriever_variant",
    "dat_mimic_weight": "dat_mimic_weight",
    "dat_category_alignment_weight": "dat_category_alignment_weight",
    "blair_model_name": "blair_model_name",
    "blair_fallback_model": "blair_fallback_model",
    "blair_batch_size": "blair_batch_size",
    "blair_max_seq_length": "blair_max_seq_length",
    "blair_projection_dim": "blair_projection_dim",
    "blair_ann_trees": "blair_ann_trees",
    "blair_chunk_rows": "blair_chunk_rows",
    "retrieval_top_k": "retrieval_top_k",
    "candidate_union_top_k": "candidate_union_top_k",
    "candidate_union_batch_size": "candidate_union_batch_size",
    "cooccurrence_candidate_k": "cooccurrence_candidate_k",
    "latent_cf_candidate_k": "latent_cf_candidate_k",
    "content_candidate_k": "content_candidate_k",
    "neural_candidate_k": "neural_candidate_k",
    "category_backfill_enabled": "category_backfill_enabled",
    "recency_cooccurrence_enabled": "recency_cooccurrence_enabled",
    "candidate_source_balance_enabled": "candidate_source_balance_enabled",
    "vector_retriever_trigger_count": "vector_retriever_trigger_count",
    "eval_user_cap": "eval_user_cap",
    "min_free_disk_gb": "min_free_disk_gb",
    "ranker_backend": "ranker_backend",
    "ranker_candidate_top_k": "ranker_candidate_top_k",
    "ranker_train_example_cap": "ranker_train_example_cap",
    "ranker_val_example_cap": "ranker_val_example_cap",
    "ranker_negatives_per_positive": "ranker_negatives_per_positive",
    "ranker_hardneg_mix": "ranker_hardneg_mix",
    "xgb_learning_rate": "xgb_learning_rate",
    "xgb_n_estimators": "xgb_n_estimators",
    "xgb_max_depth": "xgb_max_depth",
    "xgb_subsample": "xgb_subsample",
    "xgb_colsample_bytree": "xgb_colsample_bytree",
}


_PROFILE_CONTROLLED_FIELDS = {
    "max_rows_per_category",
    "retriever_train_example_cap",
    "retriever_quality_min_history",
    "enable_neural_retriever",
    "neural_retriever_variant",
    "category_backfill_enabled",
    "recency_cooccurrence_enabled",
    "candidate_source_balance_enabled",
    "vector_retriever_trigger_count",
    "eval_user_cap",
    "candidate_union_top_k",
    "candidate_union_batch_size",
    "cooccurrence_candidate_k",
    "latent_cf_candidate_k",
    "content_candidate_k",
    "neural_candidate_k",
    "popularity_backfill_k",
    "ranker_candidate_top_k",
    "ranker_train_example_cap",
    "ranker_val_example_cap",
    "split_eval_example_cap",
}

_PROFILE_CONTROLLED_SETTING_FIELDS = {
    setting_field
    for setting_field, config_field in _SETTINGS_TO_CONFIG_FIELDS.items()
    if config_field in _PROFILE_CONTROLLED_FIELDS
}


def _elapsed(start_time: float) -> str:
    seconds = time.perf_counter() - start_time
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


def _directory_size_bytes(path: Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for item in root.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _gb(value: int | float) -> float:
    return float(value) / (1024 ** 3)


def _existing_parent(path: Path) -> Path:
    candidate = Path(path).resolve()
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(_existing_parent(path))
    return _gb(usage.free)


def _process_rss_mb() -> float | None:
    try:
        import psutil
    except ModuleNotFoundError:
        return None
    return psutil.Process().memory_info().rss / (1024 ** 2)


def _configure_run_file_logging(config: core.PipelineConfig) -> Path:
    log_dir = config.artifact_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training.log"
    root_logger = logging.getLogger()
    resolved = log_path.resolve()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == resolved:
            return log_path
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(root_logger.level or logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root_logger.addHandler(handler)
    LOGGER.info("Run file logging enabled: %s", log_path)
    return log_path


def _log_stage_heartbeat(config: core.PipelineConfig, label: str, *, stage_start: float | None = None) -> None:
    rss = _process_rss_mb()
    artifact_sizes = {
        "cache_gb": _gb(_directory_size_bytes(config.cache_dir)),
        "models_gb": _gb(_directory_size_bytes(config.model_dir)),
        "eval_gb": _gb(_directory_size_bytes(config.eval_dir)),
    }
    LOGGER.info(
        "Run heartbeat: stage=%s elapsed=%s disk_free_gb=%.2f rss_mb=%s cache_gb=%.2f models_gb=%.2f eval_gb=%.2f",
        label,
        _elapsed(stage_start) if stage_start is not None else "n/a",
        _disk_free_gb(config.artifact_root),
        f"{rss:.0f}" if rss is not None else "unavailable",
        artifact_sizes["cache_gb"],
        artifact_sizes["models_gb"],
        artifact_sizes["eval_gb"],
    )


def _run_preflight_checks(settings: AppSettings, config: core.PipelineConfig) -> None:
    threshold_gb = (
        float(config.min_free_disk_gb)
        if config.min_free_disk_gb is not None
        else (40.0 if config.run_profile == "quality-neural" else 5.0)
    )
    free_gb = _disk_free_gb(config.artifact_root)
    if free_gb < threshold_gb:
        raise RuntimeError(
            f"Preflight failed for run {config.run_name!r}: only {free_gb:.2f} GB free at "
            f"{_existing_parent(config.artifact_root)}, below required {threshold_gb:.2f} GB. "
            "Run `artifact-report`, prune stale artifacts, or move artifacts outside OneDrive before training."
        )
    run_name = config.run_name.lower()
    if "blair" in run_name and config.neural_retriever_variant != "blair_text":
        raise RuntimeError(
            f"Preflight failed: run_name={config.run_name!r} implies BLAIR, but "
            f"neural_retriever_variant={config.neural_retriever_variant!r}. Set "
            "AMAZON_RECSYS_NEURAL_RETRIEVER_VARIANT=blair_text."
        )
    if settings.gate.profile == "blair-v1" and not (
        config.enable_neural_retriever and config.neural_retriever_variant == "blair_text"
    ):
        raise RuntimeError(
            "Preflight failed: gate_profile='blair-v1' requires enable_neural_retriever=true "
            "and neural_retriever_variant='blair_text'."
        )
    LOGGER.info(
        "Preflight passed: disk_free_gb=%.2f threshold_gb=%.2f run_name=%s profile=%s neural=%s variant=%s gate=%s",
        free_gb,
        threshold_gb,
        config.run_name,
        config.run_profile,
        config.enable_neural_retriever,
        config.neural_retriever_variant,
        settings.gate.profile,
    )


def _metric_preview(csv_path: Path) -> EvaluationMetricPreview:
    frame = pd.read_csv(csv_path)
    return EvaluationMetricPreview(
        file=csv_path.name,
        rows=int(len(frame)),
        preview=frame.head(5).to_dict(orient="records"),
    )


def collect_evaluation_summary(pipeline_config: core.PipelineConfig) -> EvaluationSummary:
    eval_dir = Path(pipeline_config.eval_dir)
    metric_files = sorted(
        {
            *eval_dir.glob("*_metrics.csv"),
            *(eval_dir / name for name in [
                "candidate_recall_diagnostics.csv",
                "candidate_recall_by_category.csv",
                "candidate_recall_by_history_bucket.csv",
                "candidate_recall_by_source.csv",
                "candidate_recall_by_cold_start_type.csv",
                "candidate_recall_worst_slices.csv",
                "candidate_union_recall_by_category.csv",
                "candidate_union_recall_by_source.csv",
                "candidate_union_recall_by_history_bucket.csv",
                "candidate_union_recall_by_price_bucket.csv",
                "served_distribution_by_category_price.csv",
            ] if (eval_dir / name).exists()),
        }
    )
    return EvaluationSummary(
        eval_dir=str(eval_dir),
        metric_files=[_metric_preview(csv_path) for csv_path in metric_files],
        config_path=str(Path(pipeline_config.artifact_root) / "config.json"),
    )


def _explicit_setting_names(settings: AppSettings) -> set[str]:
    fields = set(getattr(settings, "model_fields_set", set()))
    fields.update(getattr(settings, "__pydantic_fields_set__", set()))
    for field_name in list(fields):
        if field_name not in _PROFILE_CONTROLLED_SETTING_FIELDS:
            continue
        field_def = settings.__class__.model_fields.get(field_name)
        if field_def is None:
            continue
        default = field_def.default
        if default is MISSING:
            continue
        if (
            field_name == "neural_retriever_variant"
            and "enable_neural_retriever" in fields
            and bool(getattr(settings, "enable_neural_retriever", False))
        ):
            continue
        if getattr(settings, field_name, object()) == default:
            fields.discard(field_name)
    return fields


def _pipeline_default(field_name: str) -> object:
    field_def = core.PipelineConfig.__dataclass_fields__[field_name]
    if field_def.default is not MISSING:
        return field_def.default
    if field_def.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field_def.default_factory()  # type: ignore[misc]
    raise KeyError(field_name)


def pipeline_config_from_settings(settings: AppSettings) -> core.PipelineConfig:
    set_artifact_write_mode(settings.artifact_write_mode)
    config = core.PipelineConfig(
        base_dir=settings.legacy_workspace_root,
        categories=settings.data.categories,
        run_name=settings.training.run_name,
        run_profile=settings.training.run_profile,
        seed=settings.training.seed,
        k_core=settings.training.k_core,
        dev_mode=settings.training.dev_mode,
        dev_fraction=settings.training.dev_fraction,
        show_progress=settings.training.show_progress,
        train_positive_cap=settings.training.train_positive_cap,
        split_eval_example_cap=settings.training.split_eval_example_cap,
        metadata_download_if_missing=settings.data.metadata_download_if_missing,
        enable_neural_retriever=settings.retrieval.enable_neural_retriever,
        neural_retriever_variant=settings.retrieval.neural_retriever_variant,
        dat_mimic_weight=settings.retrieval.dat_mimic_weight,
        dat_category_alignment_weight=settings.retrieval.dat_category_alignment_weight,
        blair_model_name=settings.retrieval.blair_model_name,
        blair_fallback_model=settings.retrieval.blair_fallback_model,
        blair_batch_size=settings.retrieval.blair_batch_size,
        blair_max_seq_length=settings.retrieval.blair_max_seq_length,
        blair_projection_dim=settings.retrieval.blair_projection_dim,
        blair_ann_trees=settings.retrieval.blair_ann_trees,
        blair_chunk_rows=settings.retrieval.blair_chunk_rows,
        retrieval_top_k=settings.retrieval.retrieval_top_k,
        candidate_union_top_k=settings.retrieval.candidate_union_top_k,
        candidate_union_batch_size=settings.retrieval.candidate_union_batch_size,
        cooccurrence_candidate_k=settings.retrieval.cooccurrence_candidate_k,
        latent_cf_candidate_k=settings.retrieval.latent_cf_candidate_k,
        content_candidate_k=settings.retrieval.content_candidate_k,
        neural_candidate_k=settings.retrieval.neural_candidate_k,
        category_backfill_enabled=settings.retrieval.category_backfill_enabled,
        recency_cooccurrence_enabled=settings.retrieval.recency_cooccurrence_enabled,
        candidate_source_balance_enabled=settings.retrieval.candidate_source_balance_enabled,
        vector_retriever_trigger_count=settings.retrieval.vector_retriever_trigger_count,
        eval_user_cap=settings.training.eval_user_cap,
        min_free_disk_gb=settings.training.min_free_disk_gb,
        ranker_backend=settings.ranking.backend,
        ranker_candidate_top_k=settings.ranking.ranker_candidate_top_k,
        ranker_train_example_cap=settings.ranking.ranker_train_example_cap,
        ranker_val_example_cap=settings.ranking.ranker_val_example_cap,
        ranker_negatives_per_positive=settings.ranking.ranker_negatives_per_positive,
        ranker_hardneg_mix=settings.ranking.ranker_hardneg_mix,
        xgb_learning_rate=settings.ranking.xgb_learning_rate,
        xgb_n_estimators=settings.ranking.xgb_n_estimators,
        xgb_max_depth=settings.ranking.xgb_max_depth,
        xgb_subsample=settings.ranking.xgb_subsample,
        xgb_colsample_bytree=settings.ranking.xgb_colsample_bytree,
    )
    explicit_settings = _explicit_setting_names(settings)
    explicit_config_values = {
        config_field: getattr(config, config_field)
        for setting_field, config_field in _SETTINGS_TO_CONFIG_FIELDS.items()
        if setting_field in explicit_settings and hasattr(config, config_field)
    }
    for field_name in _PROFILE_CONTROLLED_FIELDS:
        setting_field = next(
            (candidate for candidate, config_field in _SETTINGS_TO_CONFIG_FIELDS.items() if config_field == field_name),
            field_name,
        )
        if setting_field not in explicit_settings and hasattr(config, field_name):
            setattr(config, field_name, _pipeline_default(field_name))
    config = core.apply_run_profile(config)
    for config_field, value in explicit_config_values.items():
        setattr(config, config_field, value)
    _enforce_profile_candidate_budget_floor(config)
    return config


def _enforce_profile_candidate_budget_floor(config: core.PipelineConfig) -> None:
    floors_by_profile = {
        "quality": {
            "candidate_union_top_k": 500,
            "ranker_candidate_top_k": 200,
            "cooccurrence_candidate_k": 250,
            "latent_cf_candidate_k": 250,
            "content_candidate_k": 250,
            "popularity_backfill_k": 100,
        },
        "quality-neural": {
            "candidate_union_top_k": 650,
            "ranker_candidate_top_k": 250,
            "cooccurrence_candidate_k": 250,
            "latent_cf_candidate_k": 250,
            "content_candidate_k": 250,
            "neural_candidate_k": 250,
            "popularity_backfill_k": 100,
        },
        "full": {
            "candidate_union_top_k": 650,
            "ranker_candidate_top_k": 250,
            "cooccurrence_candidate_k": 250,
            "latent_cf_candidate_k": 250,
            "content_candidate_k": 250,
            "neural_candidate_k": 250,
            "popularity_backfill_k": 100,
        },
    }
    floors = floors_by_profile.get(config.run_profile)
    if not floors:
        return
    raised: list[str] = []
    for field_name, minimum_value in floors.items():
        current_value = int(getattr(config, field_name))
        if current_value < minimum_value:
            setattr(config, field_name, minimum_value)
            raised.append(f"{field_name}={current_value}->{minimum_value}")
    if raised:
        LOGGER.warning(
            "Candidate budget settings were below the %s profile floor and were raised: %s",
            config.run_profile,
            ", ".join(raised),
        )


@dataclass
class TrainingSession:
    settings: AppSettings
    pipeline_config: core.PipelineConfig
    prepared: core.PreparedArtifacts
    split_artifacts: core.SplitArtifacts
    retrievers: dict[str, core.RetrieverArtifacts]
    ranker: core.RankerArtifacts
    evaluation_summary: EvaluationSummary
    mlflow: MlflowRunInfo | None = None

    @property
    def mlflow_run_id(self) -> str | None:
        return self.mlflow.run_id if self.mlflow is not None else None

    @property
    def mlflow_experiment_name(self) -> str | None:
        return self.mlflow.experiment_name if self.mlflow is not None else None

    @property
    def mlflow_tracking_uri(self) -> str | None:
        return self.mlflow.tracking_uri if self.mlflow is not None else None


class PackageTrainingPipeline:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.mlflow_tracker = MLflowTracker(settings)

    def build_pipeline_config(self) -> core.PipelineConfig:
        return pipeline_config_from_settings(self.settings)

    def run(self, force_rebuild: bool = False) -> TrainingSession:
        run_start = time.perf_counter()
        config = self.build_pipeline_config()
        _configure_run_file_logging(config)
        try:
            return self._run_with_config(config, run_start=run_start, force_rebuild=force_rebuild)
        except Exception:
            LOGGER.exception(
                "Training run failed: run_name=%s profile=%s artifact_root=%s",
                config.run_name,
                config.run_profile,
                config.artifact_root,
            )
            raise

    def _run_with_config(
        self,
        config: core.PipelineConfig,
        *,
        run_start: float,
        force_rebuild: bool,
    ) -> TrainingSession:
        _run_preflight_checks(self.settings, config)
        _log_stage_heartbeat(config, "preflight")
        LOGGER.info(
            "Training run started: run_name=%s profile=%s categories=%s artifact_root=%s force_rebuild=%s",
            config.run_name,
            config.run_profile,
            ",".join(config.categories),
            config.artifact_root,
            force_rebuild,
        )
        LOGGER.info(
            "Training limits: dev_mode=%s dev_fraction=%s k_core=%s neural_retriever=%s neural_variant=%s retriever_cap=%s ranker_cap=%s eval_user_cap=%s",
            config.dev_mode,
            config.dev_fraction,
            config.k_core,
            config.enable_neural_retriever,
            config.neural_retriever_variant,
            config.retriever_train_example_cap,
            config.ranker_train_example_cap,
            config.eval_user_cap,
        )
        stage_start = time.perf_counter()
        LOGGER.info("Stage 1/6: preparing corpus and feature cache")
        prepared = core.prepare_corpus(config, force_rebuild=force_rebuild)
        LOGGER.info(
            "Stage 1/6 complete in %s: interactions=%s hard_negatives=%s items=%s",
            _elapsed(stage_start),
            f"{len(prepared.interactions):,}",
            f"{len(prepared.hard_negatives):,}",
            f"{len(prepared.item_features):,}",
        )
        _log_stage_heartbeat(config, "stage_1_prepare_corpus", stage_start=stage_start)
        stage_start = time.perf_counter()
        LOGGER.info("Stage 2/6: building train/validation/test splits")
        split_artifacts = core.make_splits(prepared)
        LOGGER.info(
            "Stage 2/6 complete in %s: train=%s val=%s test=%s users=%s",
            _elapsed(stage_start),
            f"{len(split_artifacts.train_examples):,}",
            f"{len(split_artifacts.val_examples):,}",
            f"{len(split_artifacts.test_examples):,}",
            f"{len(split_artifacts.user_id_to_idx):,}",
        )
        _log_stage_heartbeat(config, "stage_2_splits", stage_start=stage_start)
        stage_start = time.perf_counter()
        LOGGER.info("Stage 3/6: training retrievers")
        retrievers = core.train_retrievers(prepared, split_artifacts)
        LOGGER.info("Stage 3/6 complete in %s: retrievers=%s", _elapsed(stage_start), ",".join(sorted(retrievers)))
        _log_stage_heartbeat(config, "stage_3_retrievers", stage_start=stage_start)
        stage_start = time.perf_counter()
        LOGGER.info("Stage 4/6: training %s ranker", config.ranker_backend)
        ranker = core.train_ranker(prepared, split_artifacts, retrievers, backend=config.ranker_backend)
        LOGGER.info("Stage 4/6 complete in %s: backend=%s", _elapsed(stage_start), ranker.backend)
        _log_stage_heartbeat(config, "stage_4_ranker", stage_start=stage_start)
        stage_start = time.perf_counter()
        LOGGER.info("Stage 5/6: collecting evaluation summary")
        evaluation_summary = collect_evaluation_summary(config)
        LOGGER.info(
            "Stage 5/6 complete in %s: metric_files=%s",
            _elapsed(stage_start),
            len(evaluation_summary.metric_files),
        )
        _log_stage_heartbeat(config, "stage_5_evaluation_summary", stage_start=stage_start)
        gate_profile = self.settings.gate.profile
        if gate_profile and gate_profile != "off":
            # Local import keeps the module-level import graph unchanged.
            from amazon_recsys.ml.bundles import validate_acceptance_gates
            LOGGER.info("Validating acceptance gates: profile=%s eval_dir=%s", gate_profile, config.eval_dir)
            passes = validate_acceptance_gates(Path(config.eval_dir), gate_profile)
            for line in passes:
                LOGGER.info("Gate pass: %s", line)
        session = TrainingSession(
            settings=self.settings,
            pipeline_config=config,
            prepared=prepared,
            split_artifacts=split_artifacts,
            retrievers=retrievers,
            ranker=ranker,
            evaluation_summary=evaluation_summary,
        )
        stage_start = time.perf_counter()
        LOGGER.info("Stage 6/6: logging training metadata to MLflow if enabled")
        tracking_metadata = self.mlflow_tracker.log_training_session(session)
        if tracking_metadata is not None:
            session.mlflow = tracking_metadata
            LOGGER.info(
                "Stage 6/6 complete in %s: mlflow_run_id=%s experiment=%s",
                _elapsed(stage_start),
                tracking_metadata.run_id,
                tracking_metadata.experiment_name,
            )
        else:
            LOGGER.info("Stage 6/6 complete in %s: MLflow disabled", _elapsed(stage_start))
        _log_stage_heartbeat(config, "stage_6_mlflow", stage_start=stage_start)
        LOGGER.info("Training run finished in %s: run_name=%s", _elapsed(run_start), config.run_name)
        return session

    def evaluate(self, force_rebuild: bool = False) -> EvaluationSummary:
        run_start = time.perf_counter()
        config = self.build_pipeline_config()
        eval_dir = Path(config.eval_dir)
        LOGGER.info(
            "Evaluation requested: run_name=%s profile=%s eval_dir=%s force_rebuild=%s",
            config.run_name,
            config.run_profile,
            eval_dir,
            force_rebuild,
        )
        if not eval_dir.exists() or not any(eval_dir.glob("*_metrics.csv")):
            LOGGER.info("Evaluation metrics are missing; running training first")
            session = self.run(force_rebuild=force_rebuild)
            summary = EvaluationSummary.from_dict(session.evaluation_summary.to_dict())
            if session.mlflow is not None:
                summary.mlflow = session.mlflow
            return summary
        summary = collect_evaluation_summary(config)
        tracking_metadata = self.mlflow_tracker.log_evaluation_summary(config, summary)
        if tracking_metadata is not None:
            summary.mlflow = tracking_metadata
        LOGGER.info("Evaluation summary finished in %s: metric_files=%s", _elapsed(run_start), len(summary.metric_files))
        return summary
