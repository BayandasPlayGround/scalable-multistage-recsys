from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import EvaluationMetricPreview, EvaluationSummary, MlflowRunInfo
from amazon_recsys.ml import core
from amazon_recsys.observability.mlflow import MLflowTracker


def _metric_preview(csv_path: Path) -> EvaluationMetricPreview:
    frame = pd.read_csv(csv_path)
    return EvaluationMetricPreview(
        file=csv_path.name,
        rows=int(len(frame)),
        preview=frame.head(5).to_dict(orient="records"),
    )


def collect_evaluation_summary(pipeline_config: core.PipelineConfig) -> EvaluationSummary:
    eval_dir = Path(pipeline_config.eval_dir)
    metric_files = sorted(eval_dir.glob("*_metrics.csv"))
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
        metadata_download_if_missing=settings.data.metadata_download_if_missing,
        enable_neural_retriever=settings.retrieval.enable_neural_retriever,
        retrieval_top_k=settings.retrieval.retrieval_top_k,
        candidate_union_top_k=settings.retrieval.candidate_union_top_k,
        candidate_union_batch_size=settings.retrieval.candidate_union_batch_size,
        cooccurrence_candidate_k=settings.retrieval.cooccurrence_candidate_k,
        latent_cf_candidate_k=settings.retrieval.latent_cf_candidate_k,
        content_candidate_k=settings.retrieval.content_candidate_k,
        neural_candidate_k=settings.retrieval.neural_candidate_k,
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
    return core.apply_run_profile(config)


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
        config = self.build_pipeline_config()
        prepared = core.prepare_corpus(config, force_rebuild=force_rebuild)
        split_artifacts = core.make_splits(prepared)
        retrievers = core.train_retrievers(prepared, split_artifacts)
        ranker = core.train_ranker(prepared, split_artifacts, retrievers, backend=config.ranker_backend)
        evaluation_summary = collect_evaluation_summary(config)
        session = TrainingSession(
            settings=self.settings,
            pipeline_config=config,
            prepared=prepared,
            split_artifacts=split_artifacts,
            retrievers=retrievers,
            ranker=ranker,
            evaluation_summary=evaluation_summary,
        )
        tracking_metadata = self.mlflow_tracker.log_training_session(session)
        if tracking_metadata is not None:
            session.mlflow = tracking_metadata
        return session

    def evaluate(self, force_rebuild: bool = False) -> EvaluationSummary:
        config = self.build_pipeline_config()
        eval_dir = Path(config.eval_dir)
        if not eval_dir.exists() or not any(eval_dir.glob("*_metrics.csv")):
            session = self.run(force_rebuild=force_rebuild)
            summary = EvaluationSummary.from_dict(session.evaluation_summary.to_dict())
            if session.mlflow is not None:
                summary.mlflow = session.mlflow
            return summary
        summary = collect_evaluation_summary(config)
        tracking_metadata = self.mlflow_tracker.log_evaluation_summary(config, summary)
        if tracking_metadata is not None:
            summary.mlflow = tracking_metadata
        return summary
