from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.ml import core
from amazon_recsys.observability.mlflow import MLflowTracker


def _metric_preview(csv_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(csv_path)
    return {
        "file": csv_path.name,
        "rows": int(len(frame)),
        "preview": frame.head(5).to_dict(orient="records"),
    }


def collect_evaluation_summary(pipeline_config: Any) -> dict[str, Any]:
    eval_dir = Path(pipeline_config.eval_dir)
    metric_files = sorted(eval_dir.glob("*_metrics.csv"))
    return {
        "eval_dir": str(eval_dir),
        "metric_files": [_metric_preview(csv_path) for csv_path in metric_files],
        "config_path": str(Path(pipeline_config.artifact_root) / "config.json"),
    }


@dataclass
class TrainingSession:
    settings: AppSettings
    pipeline_config: Any
    prepared: Any
    split_artifacts: Any
    retrievers: dict[str, Any]
    ranker: Any
    evaluation_summary: dict[str, Any]
    mlflow_run_id: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_tracking_uri: str | None = None


class PackageTrainingPipeline:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.mlflow_tracker = MLflowTracker(settings)

    def build_pipeline_config(self) -> Any:
        config = core.PipelineConfig(
            base_dir=self.settings.legacy_workspace_root,
            categories=self.settings.data.categories,
            run_name=self.settings.training.run_name,
            run_profile=self.settings.training.run_profile,
            seed=self.settings.training.seed,
            k_core=self.settings.training.k_core,
            dev_mode=self.settings.training.dev_mode,
            dev_fraction=self.settings.training.dev_fraction,
            show_progress=self.settings.training.show_progress,
            metadata_download_if_missing=self.settings.data.metadata_download_if_missing,
            enable_neural_retriever=self.settings.retrieval.enable_neural_retriever,
            retrieval_top_k=self.settings.retrieval.retrieval_top_k,
            candidate_union_top_k=self.settings.retrieval.candidate_union_top_k,
            candidate_union_batch_size=self.settings.retrieval.candidate_union_batch_size,
            cooccurrence_candidate_k=self.settings.retrieval.cooccurrence_candidate_k,
            latent_cf_candidate_k=self.settings.retrieval.latent_cf_candidate_k,
            content_candidate_k=self.settings.retrieval.content_candidate_k,
            neural_candidate_k=self.settings.retrieval.neural_candidate_k,
            eval_user_cap=self.settings.training.eval_user_cap,
            ranker_backend=self.settings.ranking.backend,
            ranker_candidate_top_k=self.settings.ranking.ranker_candidate_top_k,
            ranker_train_example_cap=self.settings.ranking.ranker_train_example_cap,
            ranker_val_example_cap=self.settings.ranking.ranker_val_example_cap,
            ranker_negatives_per_positive=self.settings.ranking.ranker_negatives_per_positive,
            xgb_learning_rate=self.settings.ranking.xgb_learning_rate,
            xgb_n_estimators=self.settings.ranking.xgb_n_estimators,
            xgb_max_depth=self.settings.ranking.xgb_max_depth,
            xgb_subsample=self.settings.ranking.xgb_subsample,
            xgb_colsample_bytree=self.settings.ranking.xgb_colsample_bytree,
        )
        return core.apply_run_profile(config)

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
            session.mlflow_run_id = tracking_metadata["run_id"]
            session.mlflow_experiment_name = tracking_metadata["experiment_name"]
            session.mlflow_tracking_uri = tracking_metadata["tracking_uri"]
        return session

    def evaluate(self, force_rebuild: bool = False) -> dict[str, Any]:
        config = self.build_pipeline_config()
        eval_dir = Path(config.eval_dir)
        if not eval_dir.exists() or not any(eval_dir.glob("*_metrics.csv")):
            session = self.run(force_rebuild=force_rebuild)
            summary = dict(session.evaluation_summary)
            if session.mlflow_run_id is not None:
                summary["mlflow"] = {
                    "run_id": session.mlflow_run_id,
                    "experiment_name": session.mlflow_experiment_name,
                    "tracking_uri": session.mlflow_tracking_uri,
                }
            return summary
        summary = collect_evaluation_summary(config)
        tracking_metadata = self.mlflow_tracker.log_evaluation_summary(config, summary)
        if tracking_metadata is not None:
            summary = dict(summary)
            summary["mlflow"] = tracking_metadata
        return summary


LegacyTrainingSession = TrainingSession
LegacyTrainingPipeline = PackageTrainingPipeline
