from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_CATEGORIES = ("All_Beauty", "Automotive", "Industrial_and_Scientific")
VALID_RUN_PROFILES = {"debug", "quality", "quality-neural", "full"}
VALID_RANKER_BACKENDS = {"xgboost", "dlrm"}


def default_workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


class DataConfig(BaseModel):
    directory: Path
    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    metadata_download_if_missing: bool = True


class TrainingConfig(BaseModel):
    run_name: str = "default"
    run_profile: str = "debug"
    seed: int = 42
    k_core: int = 2
    dev_mode: bool = False
    dev_fraction: float = 0.2
    show_progress: bool = False
    eval_user_cap: int | None = 250
    train_positive_cap: int = 2_000_000
    split_eval_example_cap: int | None = None


class RetrievalConfig(BaseModel):
    enable_neural_retriever: bool = False
    retrieval_top_k: int = 50
    candidate_union_top_k: int = 75
    candidate_union_batch_size: int = 100
    cooccurrence_candidate_k: int = 50
    latent_cf_candidate_k: int = 50
    content_candidate_k: int = 50
    neural_candidate_k: int = 50
    category_backfill_enabled: bool = True
    recency_cooccurrence_enabled: bool = True


class RankingConfig(BaseModel):
    backend: str = "xgboost"
    ranker_candidate_top_k: int = 25
    ranker_train_example_cap: int = 1_000
    ranker_val_example_cap: int | None = 250
    ranker_negatives_per_positive: int = 5
    xgb_learning_rate: float = 0.05
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    default_top_k: int = 7
    active_bundle_path: Path
    artifact_root: Path
    use_mock_bundle_if_missing: bool = True


class MLflowConfig(BaseModel):
    enabled: bool = False
    tracking_uri: str
    experiment_name: str = "amazon-recsys"
    backend_root: Path
    run_name_prefix: str = ""


class MonitoringConfig(BaseModel):
    enabled: bool = False
    monitoring_root: Path
    window_days: int = 1
    label_delay_days: int = 2
    attribution_horizon_days: int = 7
    min_events_per_window: int = 500
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    js_warn: float = 0.10
    js_alert: float = 0.20
    performance_drop_warn: float = 0.10
    performance_drop_alert: float = 0.20


class AzureConfig(BaseModel):
    subscription_id: str = ""
    resource_group: str = "rg-amazon-recsys"
    location: str = "southafricanorth"
    ml_workspace: str = "aml-amazon-recsys"
    aks_cluster: str = "aks-amazon-recsys"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AMAZON_RECSYS_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Amazon RecSys Platform"
    app_version: str = "0.1.0"
    environment: str = "local"
    debug: bool = True
    log_level: str = "INFO"
    workspace_root: Path = Field(default_factory=default_workspace_root)

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    data_dir: Path = Path("amazon_review_data")
    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    metadata_download_if_missing: bool = True

    artifact_root: Path = Path("artifacts/amazon_recsys")
    active_bundle_path: Path = Path("artifacts/production/active_bundle.json")
    default_top_k: int = 7
    use_mock_bundle_if_missing: bool = True
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = "amazon-recsys"
    mlflow_backend_root: Path = Path("mlflow_runs")
    mlflow_run_name_prefix: str = ""
    monitoring_enabled: bool = False
    monitoring_root: Path = Path("artifacts/amazon_recsys/monitoring")
    monitoring_window_days: int = 1
    monitoring_label_delay_days: int = 2
    monitoring_attribution_horizon_days: int = 7
    monitoring_min_events_per_window: int = 500
    monitoring_psi_warn: float = 0.10
    monitoring_psi_alert: float = 0.25
    monitoring_js_warn: float = 0.10
    monitoring_js_alert: float = 0.20
    monitoring_performance_drop_warn: float = 0.10
    monitoring_performance_drop_alert: float = 0.20

    run_name: str = "default"
    run_profile: str = "debug"
    seed: int = 42
    k_core: int = 2
    dev_mode: bool = False
    dev_fraction: float = 0.2
    show_progress: bool = False
    eval_user_cap: int | None = 250
    train_positive_cap: int = 2_000_000
    split_eval_example_cap: int | None = None

    enable_neural_retriever: bool = False
    retrieval_top_k: int = 50
    candidate_union_top_k: int = 75
    candidate_union_batch_size: int = 100
    cooccurrence_candidate_k: int = 50
    latent_cf_candidate_k: int = 50
    content_candidate_k: int = 50
    neural_candidate_k: int = 50
    category_backfill_enabled: bool = True
    recency_cooccurrence_enabled: bool = True

    ranker_backend: str = "xgboost"
    ranker_candidate_top_k: int = 25
    ranker_train_example_cap: int = 1_000
    ranker_val_example_cap: int | None = 250
    ranker_negatives_per_positive: int = 5
    xgb_learning_rate: float = 0.05
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    azure_subscription_id: str = ""
    azure_resource_group: str = "rg-amazon-recsys"
    azure_location: str = "southafricanorth"
    azure_ml_workspace: str = "aml-amazon-recsys"
    azure_aks_cluster: str = "aks-amazon-recsys"

    @field_validator("workspace_root", mode="before")
    @classmethod
    def _coerce_workspace_root(cls, value: object) -> Path:
        return Path(value).expanduser().resolve() if value is not None else default_workspace_root()

    @field_validator("run_profile")
    @classmethod
    def _validate_run_profile(cls, value: str) -> str:
        if value not in VALID_RUN_PROFILES:
            raise ValueError(f"run_profile must be one of {sorted(VALID_RUN_PROFILES)}.")
        return value

    @field_validator("ranker_backend")
    @classmethod
    def _validate_ranker_backend(cls, value: str) -> str:
        if value not in VALID_RANKER_BACKENDS:
            raise ValueError(f"ranker_backend must be one of {sorted(VALID_RANKER_BACKENDS)}.")
        return value

    @field_validator("dev_fraction")
    @classmethod
    def _validate_dev_fraction(cls, value: float) -> float:
        if not 0 < float(value) <= 1:
            raise ValueError("dev_fraction must be in the interval (0, 1].")
        return float(value)

    def _resolve_relative_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.workspace_root / path).resolve()

    @property
    def code_root(self) -> Path:
        return default_workspace_root()

    @property
    def legacy_workspace_root(self) -> Path:
        raw_data_dir = Path(self.data_dir)
        primary_candidate = self._resolve_relative_path(raw_data_dir)
        if primary_candidate.exists():
            return self.workspace_root
        if not raw_data_dir.is_absolute():
            notebook_candidate = (self.workspace_root / "notebooks" / raw_data_dir).resolve()
            if notebook_candidate.exists():
                return (self.workspace_root / "notebooks").resolve()
        return self.workspace_root

    @property
    def resolved_data_dir(self) -> Path:
        raw_data_dir = Path(self.data_dir)
        if raw_data_dir.is_absolute():
            return raw_data_dir
        candidate = (self.legacy_workspace_root / raw_data_dir).resolve()
        if candidate.exists():
            return candidate
        notebook_candidate = (self.workspace_root / "notebooks" / raw_data_dir).resolve()
        if notebook_candidate.exists():
            return notebook_candidate
        return (self.workspace_root / raw_data_dir).resolve()

    @property
    def resolved_artifact_root(self) -> Path:
        return self._resolve_relative_path(Path(self.artifact_root))

    @property
    def resolved_bundle_root(self) -> Path:
        return (self.resolved_artifact_root / "bundles").resolve()

    @property
    def resolved_active_bundle_path(self) -> Path:
        return self._resolve_relative_path(Path(self.active_bundle_path))

    @property
    def resolved_mlflow_backend_root(self) -> Path:
        return self._resolve_relative_path(Path(self.mlflow_backend_root))

    @property
    def resolved_monitoring_root(self) -> Path:
        return self._resolve_relative_path(Path(self.monitoring_root))

    @property
    def resolved_mlflow_tracking_uri(self) -> str:
        raw = str(self.mlflow_tracking_uri).strip()
        if not raw:
            backend_root = self.resolved_mlflow_backend_root.resolve()
            try:
                return str(backend_root.relative_to(Path.cwd().resolve()))
            except ValueError:
                return backend_root.as_uri()
        if "://" in raw or raw in {"databricks", "uc"}:
            return raw
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate.resolve().as_uri()
        return raw

    @property
    def legacy_artifact_root(self) -> Path:
        return (self.legacy_workspace_root / "artifacts" / "amazon_recsys" / self.run_name).resolve()

    @property
    def data(self) -> DataConfig:
        return DataConfig(
            directory=self.resolved_data_dir,
            categories=self.categories,
            metadata_download_if_missing=self.metadata_download_if_missing,
        )

    @property
    def training(self) -> TrainingConfig:
        return TrainingConfig(
            run_name=self.run_name,
            run_profile=self.run_profile,
            seed=self.seed,
            k_core=self.k_core,
            dev_mode=self.dev_mode,
            dev_fraction=self.dev_fraction,
            show_progress=self.show_progress,
            eval_user_cap=self.eval_user_cap,
            train_positive_cap=self.train_positive_cap,
            split_eval_example_cap=self.split_eval_example_cap,
        )

    @property
    def retrieval(self) -> RetrievalConfig:
        return RetrievalConfig(
            enable_neural_retriever=self.enable_neural_retriever,
            retrieval_top_k=self.retrieval_top_k,
            candidate_union_top_k=self.candidate_union_top_k,
            candidate_union_batch_size=self.candidate_union_batch_size,
            cooccurrence_candidate_k=self.cooccurrence_candidate_k,
            latent_cf_candidate_k=self.latent_cf_candidate_k,
            content_candidate_k=self.content_candidate_k,
            neural_candidate_k=self.neural_candidate_k,
            category_backfill_enabled=self.category_backfill_enabled,
            recency_cooccurrence_enabled=self.recency_cooccurrence_enabled,
        )

    @property
    def ranking(self) -> RankingConfig:
        return RankingConfig(
            backend=self.ranker_backend,
            ranker_candidate_top_k=self.ranker_candidate_top_k,
            ranker_train_example_cap=self.ranker_train_example_cap,
            ranker_val_example_cap=self.ranker_val_example_cap,
            ranker_negatives_per_positive=self.ranker_negatives_per_positive,
            xgb_learning_rate=self.xgb_learning_rate,
            xgb_n_estimators=self.xgb_n_estimators,
            xgb_max_depth=self.xgb_max_depth,
            xgb_subsample=self.xgb_subsample,
            xgb_colsample_bytree=self.xgb_colsample_bytree,
        )

    @property
    def serving(self) -> ServingConfig:
        return ServingConfig(
            host=self.host,
            port=self.port,
            reload=self.reload,
            default_top_k=self.default_top_k,
            active_bundle_path=self.resolved_active_bundle_path,
            artifact_root=self.resolved_artifact_root,
            use_mock_bundle_if_missing=self.use_mock_bundle_if_missing,
        )

    @property
    def mlflow(self) -> MLflowConfig:
        return MLflowConfig(
            enabled=self.mlflow_enabled,
            tracking_uri=self.resolved_mlflow_tracking_uri,
            experiment_name=self.mlflow_experiment_name,
            backend_root=self.resolved_mlflow_backend_root,
            run_name_prefix=self.mlflow_run_name_prefix,
        )

    @property
    def monitoring(self) -> MonitoringConfig:
        return MonitoringConfig(
            enabled=self.monitoring_enabled,
            monitoring_root=self.resolved_monitoring_root,
            window_days=self.monitoring_window_days,
            label_delay_days=self.monitoring_label_delay_days,
            attribution_horizon_days=self.monitoring_attribution_horizon_days,
            min_events_per_window=self.monitoring_min_events_per_window,
            psi_warn=self.monitoring_psi_warn,
            psi_alert=self.monitoring_psi_alert,
            js_warn=self.monitoring_js_warn,
            js_alert=self.monitoring_js_alert,
            performance_drop_warn=self.monitoring_performance_drop_warn,
            performance_drop_alert=self.monitoring_performance_drop_alert,
        )

    @property
    def azure(self) -> AzureConfig:
        return AzureConfig(
            subscription_id=self.azure_subscription_id,
            resource_group=self.azure_resource_group,
            location=self.azure_location,
            ml_workspace=self.azure_ml_workspace,
            aks_cluster=self.azure_aks_cluster,
        )

    def ensure_runtime_directories(self) -> None:
        self.resolved_artifact_root.mkdir(parents=True, exist_ok=True)
        self.resolved_bundle_root.mkdir(parents=True, exist_ok=True)
        self.resolved_active_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        self.resolved_mlflow_backend_root.mkdir(parents=True, exist_ok=True)
        self.resolved_monitoring_root.mkdir(parents=True, exist_ok=True)

    def safe_config(self) -> dict[str, object]:
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment,
            "debug": self.debug,
            "workspace_root": str(self.workspace_root),
            "legacy_workspace_root": str(self.legacy_workspace_root),
            "data": self.data.model_dump(mode="json"),
            "training": self.training.model_dump(mode="json"),
            "retrieval": self.retrieval.model_dump(mode="json"),
            "ranking": self.ranking.model_dump(mode="json"),
            "serving": self.serving.model_dump(mode="json"),
            "mlflow": self.mlflow.model_dump(mode="json"),
            "monitoring": self.monitoring.model_dump(mode="json"),
            "azure": self.azure.model_dump(mode="json"),
            "legacy_artifact_root": str(self.legacy_artifact_root),
        }


@lru_cache
def get_settings() -> AppSettings:
    settings = AppSettings()
    settings.ensure_runtime_directories()
    return settings
