from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    retriever_variants: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BundleManifest":
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActiveBundlePointer":
        return cls(**payload)


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


@dataclass
class RuntimeBundle:
    manifest: BundleManifest
    prepared: Any = None
    split_artifacts: Any = None
    retrievers: dict[str, Any] = field(default_factory=dict)
    ranker: Any = None
    evaluation_summary: dict[str, Any] = field(default_factory=dict)
    is_mock: bool = False


@dataclass(slots=True)
class ReferenceProfile:
    bundle_version: str
    created_at: str
    monitored_k: int
    numeric_features: dict[str, dict[str, Any]] = field(default_factory=dict)
    categorical_features: dict[str, dict[str, Any]] = field(default_factory=dict)
    offline_metrics: dict[str, float | None] = field(default_factory=dict)
    sample_size: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceProfile":
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OutcomeLogRecord:
    occurred_at: str
    user_key: str
    item_id: str
    event_type: str
    rating: float | None = None
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
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
    reference_value: Any = None
    current_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureDriftResult":
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConceptDriftResult":
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
    mlflow: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_drifts"] = [item.to_dict() for item in self.feature_drifts]
        payload["concept_drift"] = self.concept_drift.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MonitoringSummary":
        feature_drifts = [FeatureDriftResult.from_dict(item) for item in payload.get("feature_drifts", [])]
        concept_payload = payload.get("concept_drift") or {}
        concept_drift = ConceptDriftResult.from_dict(concept_payload) if concept_payload else ConceptDriftResult(status="insufficient_data", sample_size=0, monitored_k=10)
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
            mlflow=dict(payload.get("mlflow", {})),
        )
