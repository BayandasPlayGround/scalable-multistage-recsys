from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import EvaluationMetricPreview, EvaluationSummary, MlflowRunInfo
from amazon_recsys.ml import core
from amazon_recsys.observability.mlflow import MLflowTracker


LOGGER = logging.getLogger(__name__)


def _elapsed(start_time: float) -> str:
    seconds = time.perf_counter() - start_time
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


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


def pipeline_config_from_settings(settings: AppSettings) -> core.PipelineConfig:
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
        retrieval_top_k=settings.retrieval.retrieval_top_k,
        candidate_union_top_k=settings.retrieval.candidate_union_top_k,
        candidate_union_batch_size=settings.retrieval.candidate_union_batch_size,
        cooccurrence_candidate_k=settings.retrieval.cooccurrence_candidate_k,
        latent_cf_candidate_k=settings.retrieval.latent_cf_candidate_k,
        content_candidate_k=settings.retrieval.content_candidate_k,
        neural_candidate_k=settings.retrieval.neural_candidate_k,
        category_backfill_enabled=settings.retrieval.category_backfill_enabled,
        recency_cooccurrence_enabled=settings.retrieval.recency_cooccurrence_enabled,
        eval_user_cap=settings.training.eval_user_cap,
        ranker_backend=settings.ranking.backend,
        ranker_candidate_top_k=settings.ranking.ranker_candidate_top_k,
        ranker_train_example_cap=settings.ranking.ranker_train_example_cap,
        ranker_val_example_cap=settings.ranking.ranker_val_example_cap,
        ranker_negatives_per_positive=settings.ranking.ranker_negatives_per_positive,
        xgb_learning_rate=settings.ranking.xgb_learning_rate,
        xgb_n_estimators=settings.ranking.xgb_n_estimators,
        xgb_max_depth=settings.ranking.xgb_max_depth,
        xgb_subsample=settings.ranking.xgb_subsample,
        xgb_colsample_bytree=settings.ranking.xgb_colsample_bytree,
    )
    config = core.apply_run_profile(config)
    _enforce_profile_candidate_budget_floor(config)
    return config


def _enforce_profile_candidate_budget_floor(config: core.PipelineConfig) -> None:
    floors_by_profile = {
        "quality": {
            "candidate_union_top_k": 200,
            "ranker_candidate_top_k": 100,
            "cooccurrence_candidate_k": 100,
            "latent_cf_candidate_k": 150,
            "content_candidate_k": 100,
            "popularity_backfill_k": 50,
        },
        "quality-neural": {
            "candidate_union_top_k": 200,
            "ranker_candidate_top_k": 100,
            "cooccurrence_candidate_k": 100,
            "latent_cf_candidate_k": 150,
            "content_candidate_k": 100,
            "neural_candidate_k": 150,
            "popularity_backfill_k": 50,
        },
        "full": {
            "candidate_union_top_k": 300,
            "ranker_candidate_top_k": 200,
            "cooccurrence_candidate_k": 100,
            "latent_cf_candidate_k": 150,
            "content_candidate_k": 100,
            "neural_candidate_k": 150,
            "popularity_backfill_k": 50,
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
        LOGGER.info(
            "Training run started: run_name=%s profile=%s categories=%s artifact_root=%s force_rebuild=%s",
            config.run_name,
            config.run_profile,
            ",".join(config.categories),
            config.artifact_root,
            force_rebuild,
        )
        LOGGER.info(
            "Training limits: dev_mode=%s dev_fraction=%s k_core=%s neural_retriever=%s retriever_cap=%s ranker_cap=%s eval_user_cap=%s",
            config.dev_mode,
            config.dev_fraction,
            config.k_core,
            config.enable_neural_retriever,
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
        stage_start = time.perf_counter()
        LOGGER.info("Stage 3/6: training retrievers")
        retrievers = core.train_retrievers(prepared, split_artifacts)
        LOGGER.info("Stage 3/6 complete in %s: retrievers=%s", _elapsed(stage_start), ",".join(sorted(retrievers)))
        stage_start = time.perf_counter()
        LOGGER.info("Stage 4/6: training %s ranker", config.ranker_backend)
        ranker = core.train_ranker(prepared, split_artifacts, retrievers, backend=config.ranker_backend)
        LOGGER.info("Stage 4/6 complete in %s: backend=%s", _elapsed(stage_start), ranker.backend)
        stage_start = time.perf_counter()
        LOGGER.info("Stage 5/6: collecting evaluation summary")
        evaluation_summary = collect_evaluation_summary(config)
        LOGGER.info(
            "Stage 5/6 complete in %s: metric_files=%s",
            _elapsed(stage_start),
            len(evaluation_summary.metric_files),
        )
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
