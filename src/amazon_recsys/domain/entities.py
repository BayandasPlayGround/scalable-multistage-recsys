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


@dataclass
class RuntimeBundle:
    manifest: BundleManifest
    prepared: Any = None
    split_artifacts: Any = None
    retrievers: dict[str, Any] = field(default_factory=dict)
    ranker: Any = None
    evaluation_summary: dict[str, Any] = field(default_factory=dict)
    is_mock: bool = False
