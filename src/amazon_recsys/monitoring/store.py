from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import InferenceLogRecord, MonitoringSummary, OutcomeIngestResult, OutcomeLogRecord, ReferenceProfile
from amazon_recsys.monitoring.utils import ensure_utc_iso, hash_user_identifier, sanitize_filename


class LocalMonitoringStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.root = settings.monitoring.monitoring_root
        self.reference_dir = self.root / "reference_profiles"
        self.inference_log_path = self.root / "inference_logs.parquet"
        self.outcome_log_path = self.root / "outcomes.parquet"
        self.summaries_dir = self.root / "summaries"
        self.candidate_diagnostics_dir = self.root / "candidate_diagnostics"
        self.latest_dir = self.root / "latest"
        self.ensure_directories()

    def ensure_directories(self) -> None:
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self.latest_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._json_safe(payload), handle, indent=2)

    def _read_json(self, path: Path) -> dict[str, object]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _json_safe(self, value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except (TypeError, ValueError):
                return str(value)
        return value

    def _append_frame(self, path: Path, frame: pd.DataFrame, dedupe_columns: list[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read_parquet(path)
        if existing.empty:
            combined = frame.copy()
        else:
            combined = pd.concat([existing, frame], ignore_index=True)
        if dedupe_columns:
            combined = combined.drop_duplicates(subset=dedupe_columns, keep="last")
        combined.to_parquet(path, index=False)

    def save_reference_profile(self, profile: ReferenceProfile) -> Path:
        path = self.reference_dir / f"{sanitize_filename(profile.bundle_version)}.json"
        self._write_json(path, profile.to_dict())
        return path

    def load_reference_profile(self, bundle_version: str) -> ReferenceProfile:
        path = self.reference_dir / f"{sanitize_filename(bundle_version)}.json"
        if not path.exists():
            raise FileNotFoundError(f"Reference profile not found for bundle version {bundle_version!r}.")
        return ReferenceProfile.from_dict(self._read_json(path))

    def append_inference_records(self, records: list[InferenceLogRecord]) -> int:
        if not records:
            return 0
        frame = pd.DataFrame([record.to_dict() for record in records])
        self._append_frame(self.inference_log_path, frame, dedupe_columns=["request_id", "item_id", "rank"])
        return int(len(frame))

    def append_outcome_records(self, records: list[OutcomeLogRecord]) -> int:
        if not records:
            return 0
        frame = pd.DataFrame([record.to_dict() for record in records])
        self._append_frame(self.outcome_log_path, frame, dedupe_columns=["occurred_at", "user_key", "item_id", "event_type"])
        return int(len(frame))

    def _load_records_from_source(self, source: Path) -> pd.DataFrame:
        if not source.exists():
            raise FileNotFoundError(
                "Outcome source file was not found at "
                f"{source!s}. Create a CSV/JSON/Parquet file with columns "
                "'occurred_at', 'item_id', 'event_type', and either 'user_id' or 'user_key'. "
                "See docs/monitoring.md and docs/examples/outcomes.example.csv."
            )
        suffix = source.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(source)
        if suffix == ".parquet":
            return pd.read_parquet(source)
        if suffix in {".jsonl", ".ndjson"}:
            return pd.read_json(source, orient="records", lines=True)
        if suffix == ".json":
            return pd.read_json(source, orient="records")
        raise ValueError(f"Unsupported outcome source format: {source.suffix}")

    def ingest_outcomes(self, source: Path) -> OutcomeIngestResult:
        frame = self._load_records_from_source(source)
        if frame.empty:
            return OutcomeIngestResult(source=str(source), ingested=0)
        required = {"occurred_at", "item_id", "event_type"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Outcome source is missing required columns: {', '.join(missing)}")
        if "user_key" not in frame.columns and "user_id" not in frame.columns:
            raise ValueError("Outcome source must contain either 'user_key' or 'user_id'.")

        records: list[OutcomeLogRecord] = []
        for row in frame.to_dict(orient="records"):
            resolved_user_key = row.get("user_key")
            if resolved_user_key is None:
                resolved_user_key = hash_user_identifier(str(row.get("user_id", "")).strip())
            if not resolved_user_key:
                continue
            records.append(
                OutcomeLogRecord(
                    occurred_at=ensure_utc_iso(row["occurred_at"]),
                    user_key=str(resolved_user_key),
                    item_id=str(row["item_id"]),
                    event_type=str(row["event_type"]).strip().lower(),
                    rating=float(row["rating"]) if row.get("rating") is not None and row.get("rating") == row.get("rating") else None,
                    value=float(row["value"]) if row.get("value") is not None and row.get("value") == row.get("value") else None,
                    source=str(row.get("source", "external")).strip() if row.get("source") is not None else "external",
                )
            )
        ingested = self.append_outcome_records(records)
        return OutcomeIngestResult(source=str(source), ingested=ingested)

    def load_inference_frame(
        self,
        *,
        bundle_version: str | None = None,
        window_start: str | None = None,
        window_end: str | None = None,
    ) -> pd.DataFrame:
        frame = self._read_parquet(self.inference_log_path)
        if frame.empty:
            return frame
        if bundle_version is not None:
            frame = frame[frame["bundle_version"].astype(str) == str(bundle_version)].copy()
        if window_start is not None:
            start = pd.to_datetime(window_start, utc=True)
            frame = frame[pd.to_datetime(frame["requested_at"], utc=True, errors="coerce") >= start].copy()
        if window_end is not None:
            end = pd.to_datetime(window_end, utc=True)
            frame = frame[pd.to_datetime(frame["requested_at"], utc=True, errors="coerce") < end].copy()
        return frame.reset_index(drop=True)

    def load_outcome_frame(
        self,
        *,
        window_start: str | None = None,
        window_end: str | None = None,
    ) -> pd.DataFrame:
        frame = self._read_parquet(self.outcome_log_path)
        if frame.empty:
            return frame
        if window_start is not None:
            start = pd.to_datetime(window_start, utc=True)
            frame = frame[pd.to_datetime(frame["occurred_at"], utc=True, errors="coerce") >= start].copy()
        if window_end is not None:
            end = pd.to_datetime(window_end, utc=True)
            frame = frame[pd.to_datetime(frame["occurred_at"], utc=True, errors="coerce") < end].copy()
        return frame.reset_index(drop=True)

    def save_monitoring_summary(
        self,
        summary: MonitoringSummary,
        feature_frame: pd.DataFrame,
        concept_frame: pd.DataFrame,
        reference_profile: ReferenceProfile,
    ) -> dict[str, Path]:
        bundle_dir = self.summaries_dir / sanitize_filename(summary.bundle_version)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        summary_name = sanitize_filename(summary.window_end)
        summary_path = bundle_dir / f"{summary_name}.json"
        feature_path = bundle_dir / f"{summary_name}-feature_drift.csv"
        concept_path = bundle_dir / f"{summary_name}-concept_drift_metrics.csv"
        reference_path = bundle_dir / f"{summary_name}-reference_profile.json"

        self._write_json(summary_path, summary.to_dict())
        feature_frame.to_csv(feature_path, index=False)
        concept_frame.to_csv(concept_path, index=False)
        self._write_json(reference_path, reference_profile.to_dict())
        self._write_json(self.latest_dir / f"{sanitize_filename(summary.bundle_version)}.json", summary.to_dict())
        return {
            "summary_path": summary_path,
            "feature_path": feature_path,
            "concept_path": concept_path,
            "reference_path": reference_path,
        }

    def load_latest_summary(self, bundle_version: str) -> MonitoringSummary | None:
        path = self.latest_dir / f"{sanitize_filename(bundle_version)}.json"
        if not path.exists():
            return None
        return MonitoringSummary.from_dict(self._read_json(path))

    def list_summaries(self, bundle_version: str) -> list[MonitoringSummary]:
        bundle_dir = self.summaries_dir / sanitize_filename(bundle_version)
        if not bundle_dir.exists():
            return []
        summaries: list[MonitoringSummary] = []
        for path in sorted(bundle_dir.glob("*.json")):
            if path.name.endswith("-reference_profile.json"):
                continue
            summaries.append(MonitoringSummary.from_dict(self._read_json(path)))
        return summaries

    def save_candidate_diagnostics(
        self,
        summary: dict[str, object],
        diagnostic_frame: pd.DataFrame,
        worst_slice_frame: pd.DataFrame,
    ) -> dict[str, Path]:
        bundle_version = str(summary["bundle_version"])
        bundle_dir = self.candidate_diagnostics_dir / sanitize_filename(bundle_version)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        summary_name = sanitize_filename(str(summary.get("created_at", "candidate-diagnostics")))
        summary_path = bundle_dir / f"{summary_name}.json"
        diagnostics_path = bundle_dir / f"{summary_name}-candidate_recall.csv"
        worst_slices_path = bundle_dir / f"{summary_name}-worst_slices.csv"

        diagnostic_frame.to_csv(diagnostics_path, index=False)
        worst_slice_frame.to_csv(worst_slices_path, index=False)
        self._write_json(summary_path, summary)
        self._write_json(self.latest_dir / f"{sanitize_filename(bundle_version)}-candidate_recall.json", summary)
        return {
            "summary_path": summary_path,
            "diagnostics_path": diagnostics_path,
            "worst_slices_path": worst_slices_path,
        }

    def load_latest_candidate_diagnostics(self, bundle_version: str) -> dict[str, object] | None:
        path = self.latest_dir / f"{sanitize_filename(bundle_version)}-candidate_recall.json"
        if not path.exists():
            return None
        return self._read_json(path)

    def list_candidate_diagnostics(self, bundle_version: str) -> list[dict[str, object]]:
        bundle_dir = self.candidate_diagnostics_dir / sanitize_filename(bundle_version)
        if not bundle_dir.exists():
            return []
        summaries: list[dict[str, object]] = []
        for path in sorted(bundle_dir.glob("*.json")):
            summaries.append(self._read_json(path))
        return summaries
