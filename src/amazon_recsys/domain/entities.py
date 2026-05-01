from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, TypeAlias

if TYPE_CHECKING:
    from amazon_recsys.ml.core import PipelineConfig, PreparedArtifacts, RankerArtifacts, RetrieverArtifacts, ServingIndex, SplitArtifacts


JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class NumericFeatureProfile(TypedDict):
    bin_edges: list[float]
    proportions: list[float]
    sample_size: int
    mean: float | None
    std: float | None


class CategoricalFeatureProfile(TypedDict):
    proportions: dict[str, float]
    sample_size: int


PreviewRow: TypeAlias = dict[str, object]

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class BundleManifest:
    version: str
    created_at: str
    manifest_path: str
    bundle_dir: str
    runtime_bundle_path: str
    evaluation_summary_path: str | None
    run_name: str
    run_profile: str
    model_backend: str
    bundle_format: str = "pickle"
    retriever_variants: list[str] = field(default_factory=list)
    notes: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BundleManifest":
        return cls(**payload)

    @property
    def manifest_file(self) -> Path:
        return Path(self.manifest_path)

    @property
    def runtime_bundle_file(self) -> Path:
        return Path(self.runtime_bundle_path)


@dataclass(slots=True)
class ActiveBundlePointer:
    version: str
    manifest_path: str
    activated_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ActiveBundlePointer":
        return cls(**payload)


@dataclass(slots=True)
class MlflowRunInfo:
    run_id: str
    experiment_name: str
    tracking_uri: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MlflowRunInfo":
        return cls(
            run_id=str(payload["run_id"]),
            experiment_name=str(payload["experiment_name"]),
            tracking_uri=str(payload["tracking_uri"]),
        )


@dataclass(slots=True)
class EvaluationMetricPreview:
    file: str
    rows: int
    preview: list[PreviewRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EvaluationMetricPreview":
        preview = payload.get("preview") or []
        return cls(
            file=str(payload["file"]),
            rows=int(payload["rows"]),
            preview=[dict(row) for row in preview],
        )


@dataclass(slots=True)
class EvaluationSummary:
    source: str | None = None
    message: str | None = None
    eval_dir: str | None = None
    metric_files: list[EvaluationMetricPreview] = field(default_factory=list)
    config_path: str | None = None
    mlflow: MlflowRunInfo | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "message": self.message,
            "eval_dir": self.eval_dir,
            "metric_files": [item.to_dict() for item in self.metric_files],
            "config_path": self.config_path,
        }
        if self.mlflow is not None:
            payload["mlflow"] = self.mlflow.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EvaluationSummary":
        metric_files = [
            EvaluationMetricPreview.from_dict(item)
            for item in payload.get("metric_files", [])
        ]
        mlflow_payload = payload.get("mlflow")
        return cls(
            source=str(payload["source"]) if payload.get("source") is not None else None,
            message=str(payload["message"]) if payload.get("message") is not None else None,
            eval_dir=str(payload["eval_dir"]) if payload.get("eval_dir") is not None else None,
            metric_files=metric_files,
            config_path=str(payload["config_path"]) if payload.get("config_path") is not None else None,
            mlflow=MlflowRunInfo.from_dict(mlflow_payload) if isinstance(mlflow_payload, dict) else None,
        )


@dataclass(slots=True)
class ReadyState:
    ready: bool
    status: str
    source: str
    version: str | None = None


@dataclass(slots=True)
class ActiveModelSummary:
    ready: bool
    source: str
    version: str
    run_name: str
    run_profile: str
    model_backend: str
    retriever_variants: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass(slots=True)
class RecommendationItem:
    item_id: str
    title: str
    source_category: str
    price: float | None
    average_rating: float | None
    retrieval_score: float | None = None
    score: float | None = None
    candidate_sources: str | None = None


@dataclass(slots=True)
class HistoryItem:
    ordered_at: str
    item_id: str
    title: str
    source_category: str
    review_rating: float | None
    verified_purchase: int | None
    price: float | None
    average_rating: float | None


@dataclass(slots=True)
class AvailableUser:
    user_id: str
    interaction_count: int
    history_length: int
    last_ordered_at: str | None = None


@dataclass(slots=True)
class UserProfile:
    user_id: str
    interaction_count: int
    history_length: int
    last_ordered_at: str | None = None
    average_review_rating: float | None = None
    verified_purchase_rate: float | None = None
    top_categories: list[str] = field(default_factory=list)


@dataclass(slots=True)
class UserProfilePayload:
    profile: UserProfile
    history: list[HistoryItem] = field(default_factory=list)
    recommendations: list[RecommendationItem] = field(default_factory=list)


@dataclass(slots=True)
class OutcomeIngestResult:
    source: str
    ingested: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class OutcomeSimulationResult:
    source: str
    bundle_version: str
    window_start: str
    window_end: str
    requests_seen: int
    requests_with_user_key: int
    created: int
    ingested: int
    event_type: str
    rating: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RuntimeBundle:
    manifest: BundleManifest
    prepared: PreparedArtifacts | None = None
    split_artifacts: SplitArtifacts | None = None
    retrievers: dict[str, RetrieverArtifacts] = field(default_factory=dict)
    ranker: RankerArtifacts | None = None
    serving_index: ServingIndex | None = None
    evaluation_summary: EvaluationSummary = field(default_factory=EvaluationSummary)
    is_mock: bool = False


@dataclass(slots=True)
class ReferenceProfile:
    bundle_version: str
    created_at: str
    monitored_k: int
    numeric_features: dict[str, NumericFeatureProfile] = field(default_factory=dict)
    categorical_features: dict[str, CategoricalFeatureProfile] = field(default_factory=dict)
    offline_metrics: dict[str, float | None] = field(default_factory=dict)
    sample_size: int = 0
    notes: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ReferenceProfile":
        return cls(**payload)


@dataclass(slots=True)
class InferenceLogRecord:
    requested_at: str
    request_id: str
    bundle_version: str
    user_key: str | None
    query_mode: str
    request_history_length: int
    top_k: int
    item_id: str
    rank: int
    score: float | None = None
    candidate_sources: str | None = None
    source_category: str | None = None
    price: float | None = None
    average_rating: float | None = None
    popularity_value: float | None = None
    is_known_user: bool = False
    unseen_user: bool = False
    unseen_history_item_rate: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class OutcomeLogRecord:
    occurred_at: str
    user_key: str
    item_id: str
    event_type: str
    rating: float | None = None
    value: float | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class FeatureDriftResult:
    feature_name: str
    metric_type: str
    metric_value: float
    warn_threshold: float
    alert_threshold: float
    status: str
    sample_size: int
    reference_value: JSONValue = None
    current_value: JSONValue = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FeatureDriftResult":
        return cls(**payload)


@dataclass(slots=True)
class ConceptDriftResult:
    status: str
    sample_size: int
    monitored_k: int
    metrics: dict[str, float | None] = field(default_factory=dict)
    baseline_metrics: dict[str, float | None] = field(default_factory=dict)
    previous_metrics: dict[str, float | None] = field(default_factory=dict)
    deltas: dict[str, float | None] = field(default_factory=dict)
    performance_drop: float = 0.0
    consecutive_degraded_windows: int = 0
    notes: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ConceptDriftResult":
        return cls(**payload)


@dataclass(slots=True)
class MonitoringSummary:
    bundle_version: str
    reference_bundle_version: str
    created_at: str
    window_start: str
    window_end: str
    status: str
    inference_count: int
    outcome_count: int
    feature_drifts: list[FeatureDriftResult] = field(default_factory=list)
    concept_drift: ConceptDriftResult = field(default_factory=lambda: ConceptDriftResult(status="insufficient_data", sample_size=0, monitored_k=10))
    top_drifting_features: list[str] = field(default_factory=list)
    mlflow: MlflowRunInfo | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["feature_drifts"] = [item.to_dict() for item in self.feature_drifts]
        payload["concept_drift"] = self.concept_drift.to_dict()
        payload["mlflow"] = self.mlflow.to_dict() if self.mlflow is not None else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MonitoringSummary":
        feature_drifts = [FeatureDriftResult.from_dict(item) for item in payload.get("feature_drifts", [])]
        concept_payload = payload.get("concept_drift") or {}
        concept_drift = ConceptDriftResult.from_dict(concept_payload) if concept_payload else ConceptDriftResult(status="insufficient_data", sample_size=0, monitored_k=10)
        mlflow_payload = payload.get("mlflow")
        return cls(
            bundle_version=str(payload["bundle_version"]),
            reference_bundle_version=str(payload.get("reference_bundle_version", payload["bundle_version"])),
            created_at=str(payload["created_at"]),
            window_start=str(payload["window_start"]),
            window_end=str(payload["window_end"]),
            status=str(payload["status"]),
            inference_count=int(payload.get("inference_count", 0)),
            outcome_count=int(payload.get("outcome_count", 0)),
            feature_drifts=feature_drifts,
            concept_drift=concept_drift,
            top_drifting_features=[str(item) for item in payload.get("top_drifting_features", [])],
            mlflow=MlflowRunInfo.from_dict(mlflow_payload) if isinstance(mlflow_payload, dict) else None,
        )
